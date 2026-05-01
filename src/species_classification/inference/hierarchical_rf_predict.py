#!/usr/bin/env python
# ********************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-16 13:02:21
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-30 13:03:38
# ********************************************

"""
Hierarchical Random Forest inference for bat species identification.

Architecture
------------
Stage 1 — Main RF (multiclass, 9 species):
    Predicts a species for every input chirp and produces a per-class
    agreement vector (``predict_proba``).

Stage 2 — Binary RFs (species-pair resolvers):
    Chirps whose Stage 1 prediction falls into a confused pair are
    re-examined by a dedicated binary RF trained only on that pair with
    balanced class representation.  Currently supported pairs:

    * ``coto_tabr`` — resolves Coto vs Tabr confusion
    * ``lano_tabr`` — resolves Lano vs Tabr confusion

    Routing rules:
        * Stage 1 predicts Coto or Tabr  →  consult coto_tabr binary RF
        * Stage 1 predicts Lano or Tabr  →  consult lano_tabr binary RF
        * Stage 1 predicts Tabr          →  consult *both* binary RFs;
                                            accept the one with higher
                                            rf_agreement

    All other Stage 1 predictions are accepted directly.

Output
------
A CSV written to ``--out-dir/predictions.csv`` with one row per input
chirp:

``file_id``              Integer fragment key.
``chirp_idx``            0-based chirp position within the fragment.
``rec_site``             Recording site (if present in input).
``stage1_pred``          Species predicted by the main RF.
``stage1_rf_agreement``  Fraction of main-RF trees that voted for
                         ``stage1_pred``.  Range [0, 1].
``final_pred``           Species after hierarchical resolution.
``final_rf_agreement``   Fraction of trees in the deciding RF that voted
                         for ``final_pred``.  Range [0, 1].
``routing``              Which RF made the final call:
                         ``'main'``, ``'coto_tabr'``, or ``'lano_tabr'``.

``rf_agreement`` is the fraction of trees in the relevant Random Forest
that voted for the predicted class.  It reflects classifier consensus,
not acoustic signal quality.  The SonoBat ``confidence`` column in the
input parquet is a separate quantity derived from pulse-level acoustics
(see ``sb_measures_postprocessing.py``).

If the input contains a ``species`` column (ground-truth labels), an
optional classification report and confusion matrix are written alongside
the predictions CSV and a summary is logged to stdout.

Typical Usage
-------------
::

    python hierarchical_rf_predict.py \\
        --input   /data/bats_2026-04-14T23_44_31.parquet \\
        --main-rf /data/random_forest/results_main \\
        --coto-tabr-rf /data/random_forest/results_coto_tabr \\
        --lano-tabr-rf /data/random_forest/results_lano_tabr \\
        --out-dir /data/random_forest/hierarchical_results
"""

import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

from logging_service import LoggingService
from sonobat_utils.utils import Utils

# Reuse the same non-feature column set as the training script so that
# feature extraction is identical at inference time.
_NON_FEATURE_COLS: frozenset[str] = frozenset([
    'file_id', 'chirp_idx', 'rec_site',
    'species', 'confidence',
    'TimeInFile',
    'is_last',                          # ← add this line
    'Filename', 'species_prob', 'species_2nd',
    'cntxt_sz', 'split', 'index',
    'Path', 'ParentDir', 'NextDirUp', 'Version', 'Filter',
    'Preemphasis', 'MaxSegLnght',
])

# Routing membership sets.
_COTO_TABR_SPECIES: frozenset[str] = frozenset(['Coto', 'Tabr'])
_LANO_TABR_SPECIES: frozenset[str] = frozenset(['Lano', 'Tabr'])

log = LoggingService()


# ---------------------------------------------------------------------------
# RFBundle — thin wrapper around a trained RF + its LabelEncoder
# ---------------------------------------------------------------------------

class RFBundle:
    """
    Load and hold a trained RandomForestClassifier together with its
    LabelEncoder from a results directory produced by
    ``species_pred_random_forest.py``.

    :param results_dir: Directory containing ``rf_model.joblib`` and
                        ``label_encoder.joblib``.
    :param name:        Human-readable name used in log messages.
    """

    def __init__(self, results_dir: str | Path, name: str) -> None:
        self.name        = name
        self.results_dir = Path(results_dir)
        rf_path = self.results_dir / 'rf_model.joblib'
        le_path = self.results_dir / 'label_encoder.joblib'

        for p in (rf_path, le_path):
            if not p.exists():
                log.err(f'[{name}] Required file not found: {p}')
                sys.exit(1)

        self.rf: object = joblib.load(rf_path)
        self.le: LabelEncoder = joblib.load(le_path)
        log.info(
            f'[{name}] Loaded RF ({self.rf.n_estimators} trees, '
            f'{len(self.le.classes_)} classes: '
            f'{", ".join(self.le.classes_)})'
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a feature matrix.

        :param X: 2-D float array, shape (n_chirps, n_features).
        :return:  Tuple of ``(predicted_species, rf_agreement)`` where
                  both are 1-D arrays of length ``n_chirps``.
                  ``predicted_species`` contains species code strings;
                  ``rf_agreement`` contains the max ``predict_proba``
                  value for each row.
        """
        proba      = self.rf.predict_proba(X)
        int_labels = proba.argmax(axis=1)
        species    = self.le.inverse_transform(int_labels)
        agreement  = proba.max(axis=1)
        return species, agreement


# ---------------------------------------------------------------------------
# HierarchicalPredictor
# ---------------------------------------------------------------------------

class HierarchicalPredictor:
    """
    Two-stage hierarchical Random Forest predictor for bat species.

    Stage 1 runs the main multiclass RF on every chirp.  Stage 2
    re-examines chirps whose Stage 1 prediction falls into a known
    confused pair, routing them to the appropriate binary RF.

    :param main_rf_dir:      Results directory of the main multiclass RF.
    :param coto_tabr_rf_dir: Results directory of the Coto/Tabr binary RF.
    :param lano_tabr_rf_dir: Results directory of the Lano/Tabr binary RF.
    :param input_path:       Chirp measures file to predict on.
    :param out_dir:          Directory for output artifacts.
    """

    def __init__(
        self,
        main_rf_dir:      str | Path,
        coto_tabr_rf_dir: str | Path,
        lano_tabr_rf_dir: str | Path,
        input_path:       str | Path,
        out_dir:          str | Path,
    ) -> None:
        self.input_path = Path(input_path)
        self.out_dir    = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.main_rf      = RFBundle(main_rf_dir,      name='main')
        self.coto_tabr_rf = RFBundle(coto_tabr_rf_dir, name='coto_tabr')
        self.lano_tabr_rf = RFBundle(lano_tabr_rf_dir, name='lano_tabr')

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """
        Execute the full hierarchical prediction pipeline and write
        output artifacts.

        :return: Predictions DataFrame (also written to
                 ``out_dir/predictions.csv``).
        """
        t0 = time.perf_counter()

        df = self._load_input()
        feature_cols = self._feature_cols(df)
        log.info(f'Feature columns: {len(feature_cols)}')

        X = df[feature_cols].values

        # ---- Stage 1 ------------------------------------------------- #
        log.info('Stage 1: main RF ...')
        s1_pred, s1_agree = self.main_rf.predict(X)

        # ---- Stage 2 routing ----------------------------------------- #
        log.info('Stage 2: routing ...')
        final_pred    = s1_pred.copy()
        final_agree   = s1_agree.copy()
        routing       = np.full(len(df), 'main', dtype=object)

        coto_tabr_mask = np.isin(s1_pred, list(_COTO_TABR_SPECIES))
        lano_tabr_mask = np.isin(s1_pred, list(_LANO_TABR_SPECIES))

        # Chirps routed to coto_tabr only (Coto predictions, not Tabr)
        coto_only_mask = coto_tabr_mask & ~lano_tabr_mask
        if coto_only_mask.any():
            p, a = self.coto_tabr_rf.predict(X[coto_only_mask])
            final_pred[coto_only_mask]  = p
            final_agree[coto_only_mask] = a
            routing[coto_only_mask]     = 'coto_tabr'
            log.info(f'  coto_tabr RF: {coto_only_mask.sum():,} chirps routed')

        # Chirps routed to lano_tabr only (Lano predictions, not Tabr)
        lano_only_mask = lano_tabr_mask & ~coto_tabr_mask
        if lano_only_mask.any():
            p, a = self.lano_tabr_rf.predict(X[lano_only_mask])
            final_pred[lano_only_mask]  = p
            final_agree[lano_only_mask] = a
            routing[lano_only_mask]     = 'lano_tabr'
            log.info(f'  lano_tabr RF: {lano_only_mask.sum():,} chirps routed')

        # Chirps routed to both (Tabr predictions) — take higher agreement
        both_mask = coto_tabr_mask & lano_tabr_mask
        if both_mask.any():
            p_ct, a_ct = self.coto_tabr_rf.predict(X[both_mask])
            p_lt, a_lt = self.lano_tabr_rf.predict(X[both_mask])
            # Pick whichever binary RF is more confident
            ct_wins = a_ct >= a_lt
            p_both  = np.where(ct_wins, p_ct, p_lt)
            a_both  = np.where(ct_wins, a_ct, a_lt)
            r_both  = np.where(ct_wins, 'coto_tabr', 'lano_tabr')
            final_pred[both_mask]  = p_both
            final_agree[both_mask] = a_both
            routing[both_mask]     = r_both
            log.info(
                f'  both RFs: {both_mask.sum():,} Tabr chirps; '
                f'coto_tabr won {ct_wins.sum():,}, '
                f'lano_tabr won {(~ct_wins).sum():,}'
            )

        # ---- Assemble output DataFrame ------------------------------- #
        out = pd.DataFrame({
            'file_id'             : df['file_id'].values
                                    if 'file_id'   in df.columns else np.nan,
            'chirp_idx'           : df['chirp_idx'].values
                                    if 'chirp_idx' in df.columns else np.nan,
            'rec_site'            : df['rec_site'].values
                                    if 'rec_site'  in df.columns else np.nan,
            'stage1_pred'         : s1_pred,
            'stage1_rf_agreement' : s1_agree.round(4),
            'final_pred'          : final_pred,
            'final_rf_agreement'  : final_agree.round(4),
            'routing'             : routing,
        })

        out.to_csv(self.out_dir / 'predictions.csv', index=False)
        log.info(f'Saved predictions.csv  ({len(out):,} rows)')

        # ---- Optional evaluation when ground truth is present -------- #
        if 'species' in df.columns:
            self._evaluate(df['species'].values, out)

        # ---- Routing summary ----------------------------------------- #
        routing_counts = pd.Series(routing).value_counts()
        log.info('Routing summary:')
        for stage, count in routing_counts.items():
            log.info(f'  {stage:12s}: {count:>10,} chirps '
                     f'({100 * count / len(out):.1f}%)')

        elapsed = time.perf_counter() - t0
        mins, secs = divmod(elapsed, 60)
        log.info(f'Done.  Elapsed: {int(mins)}m {secs:.1f}s')

        return out

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_input(self) -> pd.DataFrame:
        """
        Load the input chirp measures file via Utils.read_df_file().

        :return: Raw DataFrame with all columns intact.
        """
        log.info(f'Loading {self.input_path} ...')
        try:
            df = Utils.read_df_file(self.input_path)
        except Exception as exc:
            log.err(f'Cannot read input file: {exc}')
            sys.exit(1)
        log.info(f'Loaded {len(df):,} chirp rows, {len(df.columns)} columns')
        return df

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Identify feature columns: numeric columns not in ``_NON_FEATURE_COLS``.

        Uses the same logic as the training script so the feature matrix
        is identical at inference time.

        :param df: Input DataFrame.
        :return:   Ordered list of feature column names.
        """
        cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cols:
            log.err('No numeric feature columns found in input.')
            sys.exit(1)
        return cols

    def _evaluate(
        self,
        true_species: np.ndarray,
        out:          pd.DataFrame,
    ) -> None:
        """
        Compute and save a classification report and confusion matrix
        comparing ``final_pred`` against ground-truth species labels.

        Called only when the input contains a ``species`` column.

        Evaluation is restricted to rows whose true label is one of the
        species the main RF was trained on (``self.main_rf.le.classes_``).
        Rows with out-of-vocabulary true labels — species excluded via
        ``--exclude-species`` or ``--min-species-count`` during training —
        are tallied and logged separately so the omission is visible, but
        they are not included in the confusion matrix or classification
        report.  Including them would be misleading: the system was never
        designed to predict those species, so their forced misclassification
        into the nearest trained class would artificially depress all metrics.

        :param true_species: Array of ground-truth species code strings,
                             aligned to ``out``.
        :param out:          Predictions DataFrame containing ``final_pred``
                             and ``stage1_pred``.
        """
        log.info('Ground-truth labels found — computing evaluation metrics ...')

        known_species = set(self.main_rf.le.classes_)

        true_ser = pd.Series(true_species)

        # Mask 1: NaN or composite labels — always excluded
        nan_or_composite = (
            true_ser.isna() |
            true_ser.astype(str).str.contains('/', na=False)
        )

        # Mask 2: out-of-vocabulary species (not in main RF training set)
        oov_mask = (
            ~nan_or_composite &
            ~true_ser.isin(known_species)
        )

        # Log OOV summary before dropping them
        if oov_mask.any():
            oov_counts = true_ser[oov_mask].value_counts()
            log.info(
                f'{oov_mask.sum():,} chirp rows have out-of-vocabulary true '
                f'labels (species not in main RF training set) — excluded '
                f'from evaluation:'
            )
            for spp, cnt in oov_counts.items():
                log.info(f'  {spp}: {cnt:,}')

        valid = ~nan_or_composite & ~oov_mask
        y_true = true_ser[valid].values
        y_pred = out['final_pred'].values[valid]
        log.info(
            f'Evaluating on {valid.sum():,} labeled, non-composite, '
            f'in-vocabulary rows'
        )

        classes = sorted(known_species & set(y_true))

        if len(classes) == 0 or len(y_true) == 0:
            log.info('No labeled in-vocabulary rows — skipping classification report.')
            return
        report = classification_report(y_true, y_pred,
                                       labels=classes,
                                       target_names=classes,
                                       digits=4)
        (self.out_dir / 'classification_report.txt').write_text(report)
        log.info(f'\n{report}')

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            self.out_dir / 'confusion_matrix.csv'
        )
        n = len(classes)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n)))
        ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=classes).plot(
            ax=ax, colorbar=True, xticks_rotation=45, values_format='.2f'
        )
        ax.set_title('Confusion Matrix (normalised) — hierarchical RF')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
        log.info('Saved classification_report.txt, confusion_matrix.csv/png')

        # Per-stage accuracy summary
        s1_acc = (out['stage1_pred'].values[valid] == y_true).mean()
        fi_acc = (y_pred == y_true).mean()
        log.info(f'Stage 1 accuracy : {s1_acc:.4f}')
        log.info(f'Final   accuracy : {fi_acc:.4f}  '
                 f'(Δ = {fi_acc - s1_acc:+.4f})')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments for :class:`HierarchicalPredictor`.

    :return: ``argparse.Namespace`` with validated attributes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='hierarchical_rf_predict',
        description=(
            'Two-stage hierarchical Random Forest inference for bat species\n'
            'identification.\n\n'
            'Stage 1: main multiclass RF predicts species for every chirp.\n'
            'Stage 2: binary RFs resolve Coto/Tabr and Lano/Tabr confusions.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='PATH',
        help=(
            'Chirp-level measures file (.parquet/.pq preferred;\n'
            '.feather and .csv also accepted).\n'
            'A "species" column is optional; if present, evaluation\n'
            'metrics are computed and written to --out-dir.'
        ),
    )
    parser.add_argument(
        '--main-rf',
        required=True,
        metavar='DIR',
        help='Results directory of the main multiclass RF\n'
             '(must contain rf_model.joblib and label_encoder.joblib).',
    )
    parser.add_argument(
        '--coto-tabr-rf',
        required=True,
        metavar='DIR',
        help='Results directory of the Coto/Tabr binary RF.',
    )
    parser.add_argument(
        '--lano-tabr-rf',
        required=True,
        metavar='DIR',
        help='Results directory of the Lano/Tabr binary RF.',
    )
    parser.add_argument(
        '-o', '--out-dir',
        required=True,
        metavar='DIR',
        help='Directory for output artifacts (created if absent).',
    )

    args = parser.parse_args()

    for attr, label in [
        ('input',        '--input'),
        ('main_rf',      '--main-rf'),
        ('coto_tabr_rf', '--coto-tabr-rf'),
        ('lano_tabr_rf', '--lano-tabr-rf'),
    ]:
        p = Path(getattr(args, attr))
        if not p.exists():
            parser.error(f'{label} path not found: {p}')

    return args


def main() -> None:
    """
    CLI entry point for :class:`HierarchicalPredictor`.
    """
    args = _parse_args()

    predictor = HierarchicalPredictor(
        main_rf_dir      = args.main_rf,
        coto_tabr_rf_dir = args.coto_tabr_rf,
        lano_tabr_rf_dir = args.lano_tabr_rf,
        input_path       = args.input,
        out_dir          = args.out_dir,
    )
    predictor.run()


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()
