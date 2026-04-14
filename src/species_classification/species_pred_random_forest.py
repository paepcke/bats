#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-13 15:10:24
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/species_classification/species_pred_random_forest.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-14 13:36:24
#
# **********************************************************

"""
Train and evaluate a Random Forest classifier for bat species identification
from SonoBat acoustic measures.

Input
-----
A chirp-level Parquet file produced by ``sb_measures_postprocessing.py``
(``bats_<timestamp>.parquet``), containing one row per detected chirp with:

* All SonoBat acoustic measure columns (the feature set, normalised floats)
* ``species``    : SonoBat species label (4-char code, or composite
                   ``Myca/Myyu``-style string, or NaN for unlabelled rows)
* ``confidence`` : SonoBat confidence for that label (float)
* ``file_id``    : integer fragment key (stable across pipeline runs)
* ``chirp_idx``  : 0-based chirp position within the fragment
* ``rec_site``   : recording-site Categorical (e.g. ``barn``, ``lake2``)
* ``TimeInFile`` : chirp onset time within the 2-second fragment (seconds)

The file is read via ``Utils.read_df_file()``, which handles ``.parquet``
/ ``.pq``, ``.feather``, and ``.csv`` and works around the PyArrow
thrift-buffer limits that arise with large BatsData parquet files.
CSV and Feather inputs from the legacy ``sono_batch_processing.py`` pipeline
are still accepted for backward compatibility; the column differences are
handled transparently.

The non-feature columns are excluded from model training automatically.

Train / Val / Test Split
------------------------
Splitting is performed at the **file_id level** to prevent data leakage —
all chirps from a given 2-second fragment land in exactly one partition.
The split is stratified by the modal species of each file_id so that each
partition receives a proportional share of every species.

Class Imbalance
---------------
``class_weight='balanced'`` is passed to the Random Forest, causing each
tree to weight each sample inversely proportional to its class frequency.
This handles the extreme imbalance (Myca ~66%, Myyu ~22%) without
discarding data.

Species with fewer than ``--min-species-count`` fragments are excluded from
training and evaluation but are logged so the threshold choice is auditable.

Output Artifacts
----------------
All outputs are written under ``--out-dir``:

``rf_model.joblib``
    Serialised trained RandomForestClassifier.

``label_encoder.joblib``
    Serialised LabelEncoder mapping integer class indices to species codes.

``confusion_matrix.csv`` / ``confusion_matrix.png``
    Normalised confusion matrix on the test set.

``feature_importances.csv`` / ``feature_importances.png``
    Mean Gini impurity decrease per feature, sorted descending.

``classification_report.txt``
    Per-class precision, recall, F1, and support on the test set.

``test_predictions.csv``
    Test-set chirp rows with columns ``file_id``, ``chirp_idx``,
    ``rec_site``, ``species_true``, ``species_pred``, ``confidence``
    (max class probability from the RF).

``run_config.csv``
    All CLI parameters and derived statistics for reproducibility.

Typical Usage
-------------
::

    python species_pred_random_forest.py \\
        --input  /qnap/bats/all_data/bats_2026-04-08T23_40_36.627058.parquet \\
        --out-dir /qnap/bats/rf_results \\
        --min-species-count 500 \\
        --n-estimators 500 \\
        --test-frac 0.15 \\
        --val-frac  0.15
"""

import sys
import time
from pathlib import Path
from typing import Optional, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')       # non-interactive backend — safe on sextus / headless
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)

from logging_service import LoggingService
from sonobat_utils.utils import Utils

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are metadata / labels, never features.
# The primary set reflects sb_measures_postprocessing.py parquet output.
# Legacy names from sono_batch_processing.py are retained for backward
# compatibility with old feather/csv inputs.
_NON_FEATURE_COLS: frozenset[str] = frozenset([
    # Identifier columns (new pipeline)
    'file_id', 'chirp_idx', 'rec_site',
    # Label columns (new pipeline)
    'species', 'confidence',
    # TimeInFile is a per-fragment chirp offset (0-2 s); exclude it as a
    # feature to avoid fitting on recording-internal timing rather than
    # acoustics.  It is numeric so it must be listed explicitly here.
    'TimeInFile',
    # Legacy identifier/label columns (sono_batch_processing.py)
    'Filename', 'species_prob', 'species_2nd',
    # Other legacy pipeline columns
    'cntxt_sz', 'split', 'index',
    # SonoBat path/config columns (may survive if drop was skipped)
    'Path', 'ParentDir', 'NextDirUp', 'Version', 'Filter',
    'Preemphasis', 'MaxSegLnght',
])

# Default RF hyperparameters — sensible starting point for this data scale.
_DEFAULT_N_ESTIMATORS: int = 300
_DEFAULT_MAX_FEATURES: str = 'sqrt'     # standard for classification
_DEFAULT_MIN_SAMPLES_LEAF: int = 5      # smooths out noisy sparse classes
_DEFAULT_N_JOBS: int = -1               # use all cores (sextus has 48)

# Minimum fragments per species to include in training.
_DEFAULT_MIN_SPECIES_COUNT: int = 500

# Training mode choices.
_MODE_MULTICLASS: str = 'multiclass'
_MODE_BINARY:     str = 'binary'


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """
    Summary returned by :meth:`RFTrainer.run`.

    :param out_dir:          Directory where all artifacts were written.
    :param n_chirps_total:   Total chirp rows in the input file.
    :param n_chirps_used:    Chirp rows after label filtering and species
                             count threshold.
    :param species_kept:     Species codes included in training.
    :param species_excluded: Species codes excluded (below count threshold
                             or unlabeled).
    :param n_train:          Chirp rows in the training partition.
    :param n_val:            Chirp rows in the validation partition.
    :param n_test:           Chirp rows in the test partition.
    :param val_accuracy:     Overall accuracy on the validation set.
    :param test_accuracy:    Overall accuracy on the test set.
    :param elapsed_secs:     Wall-clock seconds for the full run.
    """
    out_dir:          Path
    n_chirps_total:   int
    n_chirps_used:    int
    species_kept:     list[str]
    species_excluded: list[str]
    n_train:          int
    n_val:            int
    n_test:           int
    val_accuracy:     float
    test_accuracy:    float
    elapsed_secs:     float
    # Binary-mode extras (None in multiclass mode).
    test_auc:         Optional[float] = None
    species_pair:     Optional[tuple[str, str]] = None

    def summary(self) -> str:
        """
        Return a human-readable multi-line run summary.

        :return: Formatted string with training statistics.
        """
        mins, secs = divmod(self.elapsed_secs, 60)
        elapsed_str = f'{int(mins)}m {secs:.1f}s' if mins else f'{secs:.1f}s'
        lines = [
            f"RF training complete:",
            f"  * {self.n_chirps_total:,} chirps in input",
            f"  * {self.n_chirps_used:,} chirps used for training",
            f"  * {len(self.species_kept)} species kept: "
            f"{', '.join(sorted(self.species_kept))}",
            f"  * {len(self.species_excluded)} species excluded: "
            f"{', '.join(sorted(self.species_excluded))}",
            f"  * Split — train: {self.n_train:,}  "
            f"val: {self.n_val:,}  test: {self.n_test:,}",
            f"  * Val  accuracy: {self.val_accuracy:.4f}",
            f"  * Test accuracy: {self.test_accuracy:.4f}",
        ]
        if self.test_auc is not None:
            lines.append(f"  * Test AUC:      {self.test_auc:.4f}")
        lines += [f"  * Elapsed: {elapsed_str}", f"  * Output:  {self.out_dir}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RFTrainer:
    """
    Train a Random Forest species classifier on SonoBat acoustic measures.

    :param input_path:        Path to the chirp-level measures file.
                              Read via ``Utils.read_df_file()``, which accepts
                              ``.parquet`` / ``.pq`` (primary), ``.feather``,
                              and ``.csv``.
    :param out_dir:           Directory for all output artifacts.
    :param min_species_count: Minimum number of labeled fragments for a
                              species to be included in training.
    :param val_frac:          Fraction of file_ids reserved for validation.
    :param test_frac:         Fraction of file_ids reserved for testing.
    :param n_estimators:      Number of trees in the Random Forest.
    :param max_features:      ``max_features`` parameter passed to
                              :class:`~sklearn.ensemble.RandomForestClassifier`.
    :param min_samples_leaf:  ``min_samples_leaf`` parameter.
    :param n_jobs:            Number of parallel jobs for RF training
                              (``-1`` = all available cores).
    :param random_state:      Random seed for reproducibility.
    """

    def __init__(
        self,
        input_path:        str | Path,
        out_dir:           str | Path,
        min_species_count: int   = _DEFAULT_MIN_SPECIES_COUNT,
        val_frac:          float = 0.15,
        test_frac:         float = 0.15,
        n_estimators:      int   = _DEFAULT_N_ESTIMATORS,
        max_features:      str   = _DEFAULT_MAX_FEATURES,
        min_samples_leaf:  int   = _DEFAULT_MIN_SAMPLES_LEAF,
        n_jobs:            int   = _DEFAULT_N_JOBS,
        random_state:      int   = 42,
        mode:              str   = 'multiclass',
        species_pair:      Optional[tuple[str, str]] = None,
    ) -> None:
        self.input_path        = Path(input_path)
        self.out_dir           = Path(out_dir)
        self.min_species_count = min_species_count
        self.val_frac          = val_frac
        self.test_frac         = test_frac
        self.n_estimators      = n_estimators
        self.max_features      = max_features
        self.min_samples_leaf  = min_samples_leaf
        self.n_jobs            = n_jobs
        self.random_state      = random_state
        self.mode              = mode
        self.species_pair      = species_pair

    # ------------------------------------------------------------------ #
    #  Data loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_data(self) -> pd.DataFrame:
        """
        Load the chirp-level measures file using
        :meth:`~sonobat_utils.utils.Utils.read_df_file`, which handles
        ``.parquet`` / ``.pq``, ``.feather``, and ``.csv`` and works around
        the PyArrow thrift-buffer limits that arise with large BatsData
        parquet files.

        :return: Raw DataFrame with all columns intact.
        :raises: ``SystemExit`` if the file cannot be read.
        """
        log.info(f'Loading {self.input_path} ...')
        try:
            df = Utils.read_df_file(self.input_path)
        except Exception as exc:
            log.warn(f'Cannot read input file {self.input_path}: {exc}')
            sys.exit(1)
        log.info(f'Loaded {len(df):,} chirp rows, {len(df.columns)} columns')
        self._log_column_inventory(df)
        return df

    def _log_column_inventory(self, df: pd.DataFrame) -> None:
        """
        Log which expected metadata columns are present or absent, so that
        pipeline schema mismatches are immediately visible.

        :param df: Freshly loaded DataFrame.
        """
        new_pipeline_cols = {'file_id', 'chirp_idx', 'rec_site',
                             'species', 'confidence', 'TimeInFile'}
        legacy_cols       = {'Filename', 'species_prob', 'species_2nd'}
        present_new    = new_pipeline_cols & set(df.columns)
        present_legacy = legacy_cols & set(df.columns)
        if present_new:
            log.info(f'New-pipeline columns present: {sorted(present_new)}')
        if present_legacy:
            log.info(
                f'Legacy pipeline columns present (backward-compat): '
                f'{sorted(present_legacy)}'
            )

    # ------------------------------------------------------------------ #
    #  Feature / label preparation                                        #
    # ------------------------------------------------------------------ #

    def _prepare(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
        """
        Filter to labeled rows, apply species count threshold, identify
        feature columns, and encode labels.

        :param df: Raw chirp DataFrame.
        :return:   Tuple of:

                   * ``features_df``    — DataFrame of numeric feature columns
                     only, aligned to ``labels``.
                   * ``labels``         — integer-encoded species Series.
                   * ``species_kept``   — species codes included.
                   * ``species_excluded`` — species codes excluded.
        """
        # Keep only rows with a clean species label.
        labeled = df[df['species'].notna()].copy()
        log.info(
            f'{len(labeled):,} rows have a species label '
            f'({len(df) - len(labeled):,} unlabeled dropped)'
        )

        # Apply per-species fragment count threshold.
        # Count by file_id first (fragment level), not chirp level,
        # so that species with many chirps per fragment aren't
        # artificially inflated past the threshold.
        frag_counts = (
            labeled.groupby('species')['file_id']
            .nunique()
            .rename('n_fragments')
        )
        kept_species    = frag_counts[
            frag_counts >= self.min_species_count
        ].index.tolist()
        excluded_species = frag_counts[
            frag_counts < self.min_species_count
        ].index.tolist()

        if excluded_species:
            log.info(
                f'Excluding {len(excluded_species)} species below '
                f'{self.min_species_count:,} fragment threshold: '
                f'{", ".join(sorted(excluded_species))}'
            )
        log.info(
            f'Training on {len(kept_species)} species: '
            f'{", ".join(sorted(kept_species))}'
        )

        labeled = labeled[labeled['species'].isin(kept_species)].copy()
        log.info(f'{len(labeled):,} chirp rows remain after species filtering')

        # Identify feature columns: numeric, not in the exclusion set.
        feature_cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        log.info(f'Feature columns: {len(feature_cols)}')

        # Drop rows with any NaN in features (rare but possible).
        before = len(labeled)
        labeled = labeled.dropna(subset=feature_cols)
        if len(labeled) < before:
            log.info(f'Dropped {before - len(labeled):,} rows with NaN features')

        features_df = labeled[feature_cols].reset_index(drop=True)
        species_ser = labeled['species'].reset_index(drop=True)

        return features_df, species_ser, sorted(kept_species), sorted(excluded_species)

    # ------------------------------------------------------------------ #
    #  Binary-mode: balanced subsampling at file_id level                #
    # ------------------------------------------------------------------ #

    def _balance_by_file_id(
        self,
        labeled:  pd.DataFrame,
        sp_a:     str,
        sp_b:     str,
    ) -> pd.DataFrame:
        """
        Subsample file_ids so that both species contribute the same number
        of fragments, then return all chirps from those file_ids.

        The smaller species determines the target count.  File_ids are
        sampled with a fixed seed for reproducibility.

        :param labeled: Labeled chirp DataFrame filtered to ``sp_a`` and
                        ``sp_b`` only.
        :param sp_a:    First species code.
        :param sp_b:    Second species code.
        :return:        Balanced DataFrame.
        """
        rng    = np.random.default_rng(self.random_state)
        fids_a = labeled[labeled['species'] == sp_a]['file_id'].unique()
        fids_b = labeled[labeled['species'] == sp_b]['file_id'].unique()
        n_keep = min(len(fids_a), len(fids_b))
        log.info(
            f'Binary balance: {sp_a} {len(fids_a):,} file_ids, '
            f'{sp_b} {len(fids_b):,} — subsampling both to {n_keep:,}'
        )
        sel_a    = rng.choice(fids_a, n_keep, replace=False)
        sel_b    = rng.choice(fids_b, n_keep, replace=False)
        keep     = set(sel_a.tolist()) | set(sel_b.tolist())
        balanced = labeled[labeled['file_id'].isin(keep)].copy()
        counts   = balanced['species'].value_counts()
        log.info(
            f'After balancing: {counts.get(sp_a, 0):,} {sp_a} chirps, '
            f'{counts.get(sp_b, 0):,} {sp_b} chirps'
        )
        return balanced

    # ------------------------------------------------------------------ #
    #  Train / val / test split at file_id level                         #
    # ------------------------------------------------------------------ #

    def _split_by_file_id(
        self,
        df:         pd.DataFrame,
        species_ser: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign each chirp row to train, val, or test by splitting at the
        ``file_id`` level, stratified by each file_id's modal species.

        All chirps from a given ``file_id`` land in exactly one partition,
        preventing any data leakage between splits.

        :param df:          Features DataFrame (must have ``file_id`` in the
                            original labeled DataFrame — passed via index).
        :param species_ser: Species Series aligned to ``df``.
        :return:            Tuple of boolean index arrays
                            ``(train_mask, val_mask, test_mask)``.
        """
        # Reconstruct file_id aligned to the filtered/reset index.
        # We stored species_ser from labeled; rebuild file_id the same way.
        # The caller passes the filtered 'labeled' df — access via _split.
        raise NotImplementedError  # replaced by _split_indices below

    def _split_indices(
        self,
        labeled: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.Index, pd.Index, pd.Index]:
        """
        Return row-index arrays for train, val, and test partitions,
        splitting at the ``file_id`` level stratified by modal species.

        :param labeled:      Filtered chirp DataFrame containing ``file_id``
                             and ``species`` columns.
        :param feature_cols: Feature column names (used only to confirm
                             alignment; not modified here).
        :return:             Tuple of pandas Index objects
                             ``(train_idx, val_idx, test_idx)`` into
                             ``labeled``.
        """
        rng = np.random.default_rng(self.random_state)

        # Modal species per file_id.
        modal = (
            labeled.groupby('file_id')['species']
            .agg(lambda s: s.mode().iloc[0])
            .rename('modal_species')
            .reset_index()
        )

        train_fids, val_fids, test_fids = [], [], []

        for spp, grp in modal.groupby('modal_species'):
            fids = grp['file_id'].values.copy()
            rng.shuffle(fids)
            n      = len(fids)
            n_test = max(1, round(n * self.test_frac))
            n_val  = max(1, round(n * self.val_frac))
            test_fids.extend(fids[:n_test].tolist())
            val_fids.extend(fids[n_test: n_test + n_val].tolist())
            train_fids.extend(fids[n_test + n_val:].tolist())

        train_set = set(train_fids)
        val_set   = set(val_fids)
        test_set  = set(test_fids)

        train_idx = labeled.index[labeled['file_id'].isin(train_set)]
        val_idx   = labeled.index[labeled['file_id'].isin(val_set)]
        test_idx  = labeled.index[labeled['file_id'].isin(test_set)]

        log.info(
            f'Split (by file_id): '
            f'train {len(train_fids):,} fids / {len(train_idx):,} chirps | '
            f'val {len(val_fids):,} fids / {len(val_idx):,} chirps | '
            f'test {len(test_fids):,} fids / {len(test_idx):,} chirps'
        )
        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------ #
    #  Output helpers                                                     #
    # ------------------------------------------------------------------ #

    def _save_confusion_matrix(
        self,
        y_true:        np.ndarray,
        y_pred:        np.ndarray,
        class_names:   list[str],
    ) -> None:
        """
        Save a normalised confusion matrix as both CSV and PNG.

        :param y_true:      True integer class labels.
        :param y_pred:      Predicted integer class labels.
        :param class_names: Ordered list of species code strings.
        """
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            self.out_dir / 'confusion_matrix.csv'
        )

        fig, ax = plt.subplots(figsize=(max(8, len(class_names)),
                                        max(6, len(class_names))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        disp.plot(ax=ax, colorbar=True, xticks_rotation=45,
                  values_format='.2f')
        ax.set_title('Confusion Matrix (normalised) — test set')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
        log.info('Saved confusion_matrix.csv and confusion_matrix.png')

    def _save_feature_importances(
        self,
        importances:  np.ndarray,
        feature_cols: list[str],
    ) -> None:
        """
        Save feature importances as CSV and a horizontal bar-chart PNG.

        :param importances:  Array of mean Gini impurity decrease per feature.
        :param feature_cols: Corresponding feature names.
        """
        fi = (
            pd.DataFrame({'feature': feature_cols, 'importance': importances})
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )
        fi.to_csv(self.out_dir / 'feature_importances.csv', index=False)

        top_n = min(30, len(fi))
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        ax.barh(fi['feature'][:top_n][::-1],
                fi['importance'][:top_n][::-1])
        ax.set_xlabel('Mean Gini impurity decrease')
        ax.set_title(f'Top {top_n} feature importances')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'feature_importances.png', dpi=150)
        plt.close(fig)
        log.info('Saved feature_importances.csv and feature_importances.png')

    def _save_roc_curve(
        self,
        rf:          RandomForestClassifier,
        X_test:      np.ndarray,
        y_test:      np.ndarray,
        pos_label:   int,
        class_name:  str,
    ) -> float:
        """
        Save an ROC curve plot and return the AUC score.

        :param rf:         Trained binary RandomForestClassifier.
        :param X_test:     Test feature array.
        :param y_test:     True integer labels.
        :param pos_label:  Integer label treated as the positive class.
        :param class_name: Display name for the positive class.
        :return:           AUC score.
        """
        proba   = rf.predict_proba(X_test)[:, pos_label]
        auc     = roc_auc_score(y_test, proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(
            y_test, proba,
            pos_label = pos_label,
            name      = f'{class_name}  (AUC={auc:.4f})',
            ax        = ax,
        )
        sp_a, sp_b = self.species_pair
        ax.set_title(f'ROC curve — binary RF  ({sp_a} vs {sp_b})')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'roc_curve.png', dpi=150)
        plt.close(fig)
        log.info(f'Saved roc_curve.png  (AUC={auc:.4f})')
        return auc

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> TrainingResult:
        """
        Execute the full training pipeline:

        1. Load data.
        2. Filter labels and apply species count threshold.
        3. Split by file_id (stratified by modal species).
        4. Train Random Forest with ``class_weight='balanced'``.
        5. Evaluate on val and test sets.
        6. Write all output artifacts.

        :return: :class:`TrainingResult` with summary statistics.
        """
        _t0 = time.perf_counter()
        if self.mode == _MODE_BINARY:
            return self._run_binary(_t0=_t0)
        return self._run_multiclass(_t0=_t0)

    def _run_multiclass(self, _t0: float) -> TrainingResult:
        """
        Full multiclass training pipeline.

        :param _t0: ``time.perf_counter()`` value captured at ``run()`` entry.
        :return:    :class:`TrainingResult`.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load ---------------------------------------------------- #
        raw_df = self._load_data()
        n_chirps_total = len(raw_df)

        # ---- Prepare ------------------------------------------------- #
        log.info('Preparing features and labels ...')
        feature_cols = [
            c for c in raw_df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(raw_df[c])
        ]

        labeled = raw_df[raw_df['species'].notna()].copy()
        log.info(
            f'{len(labeled):,} labeled rows '
            f'({n_chirps_total - len(labeled):,} unlabeled dropped)'
        )

        # Species count threshold (at fragment level).
        frag_counts = (
            labeled.groupby('species')['file_id']
            .nunique()
            .rename('n_fragments')
        )
        kept_species     = sorted(
            frag_counts[frag_counts >= self.min_species_count].index.tolist()
        )
        excluded_species = sorted(
            frag_counts[frag_counts < self.min_species_count].index.tolist()
        )
        if excluded_species:
            log.info(
                f'Excluding species below {self.min_species_count:,}-fragment '
                f'threshold: {", ".join(excluded_species)}'
            )
        log.info(f'Training on {len(kept_species)} species: {", ".join(kept_species)}')

        labeled = labeled[labeled['species'].isin(kept_species)].copy()
        labeled = labeled.dropna(subset=feature_cols).reset_index(drop=True)
        log.info(f'{len(labeled):,} chirp rows after all filtering')

        # ---- Split --------------------------------------------------- #
        log.info('Splitting by file_id (stratified by modal species) ...')
        train_idx, val_idx, test_idx = self._split_indices(labeled, feature_cols)

        X_train = labeled.loc[train_idx, feature_cols].values
        y_train_raw = labeled.loc[train_idx, 'species'].values
        X_val   = labeled.loc[val_idx,   feature_cols].values
        y_val_raw   = labeled.loc[val_idx,   'species'].values
        X_test  = labeled.loc[test_idx,  feature_cols].values
        y_test_raw  = labeled.loc[test_idx,  'species'].values

        le = LabelEncoder().fit(kept_species)
        y_train = le.transform(y_train_raw)
        y_val   = le.transform(y_val_raw)
        y_test  = le.transform(y_test_raw)

        # ---- Train --------------------------------------------------- #
        log.info(
            f'Training RandomForest: {self.n_estimators} trees, '
            f'max_features={self.max_features}, '
            f'min_samples_leaf={self.min_samples_leaf}, '
            f'n_jobs={self.n_jobs} ...'
        )
        rf = RandomForestClassifier(
            n_estimators      = self.n_estimators,
            max_features      = self.max_features,
            min_samples_leaf  = self.min_samples_leaf,
            class_weight      = 'balanced',
            n_jobs            = self.n_jobs,
            random_state      = self.random_state,
            verbose           = 1,
        )
        rf.fit(X_train, y_train)
        log.info('Training complete')

        # ---- Validate ------------------------------------------------ #
        val_accuracy  = rf.score(X_val, y_val)
        test_accuracy = rf.score(X_test, y_test)
        log.info(f'Val  accuracy: {val_accuracy:.4f}')
        log.info(f'Test accuracy: {test_accuracy:.4f}')

        # ---- Save artifacts ------------------------------------------ #
        log.info('Saving artifacts ...')

        joblib.dump(rf, self.out_dir / 'rf_model.joblib')
        log.info('Saved rf_model.joblib')

        joblib.dump(le, self.out_dir / 'label_encoder.joblib')
        log.info('Saved label_encoder.joblib')

        y_test_pred = rf.predict(X_test)
        report = classification_report(
            y_test, y_test_pred,
            target_names=le.classes_,
            digits=4,
        )
        (self.out_dir / 'classification_report.txt').write_text(report)
        log.info('Saved classification_report.txt')
        log.info(f'\n{report}')

        self._save_confusion_matrix(y_test, y_test_pred, list(le.classes_))
        self._save_feature_importances(rf.feature_importances_, feature_cols)

        # Test predictions with confidence.
        test_proba   = rf.predict_proba(X_test)
        confidence   = test_proba.max(axis=1)
        test_pred_df = pd.DataFrame({
            'file_id'      : labeled.loc[test_idx, 'file_id'].values,
            'chirp_idx'    : labeled.loc[test_idx, 'chirp_idx'].values
                             if 'chirp_idx' in labeled.columns else np.nan,
            'rec_site'     : labeled.loc[test_idx, 'rec_site'].values
                             if 'rec_site' in labeled.columns else np.nan,
            'species_true' : y_test_raw,
            'species_pred' : le.inverse_transform(y_test_pred),
            'confidence'   : confidence.round(4),
        })
        test_pred_df.to_csv(self.out_dir / 'test_predictions.csv', index=False)
        log.info('Saved test_predictions.csv')

        # Run config.
        elapsed = time.perf_counter() - _t0
        pd.DataFrame([
            {'parameter': 'input_path',         'value': str(self.input_path)},
            {'parameter': 'min_species_count',  'value': self.min_species_count},
            {'parameter': 'val_frac',           'value': self.val_frac},
            {'parameter': 'test_frac',          'value': self.test_frac},
            {'parameter': 'n_estimators',       'value': self.n_estimators},
            {'parameter': 'max_features',       'value': self.max_features},
            {'parameter': 'min_samples_leaf',   'value': self.min_samples_leaf},
            {'parameter': 'random_state',       'value': self.random_state},
            {'parameter': 'n_chirps_total',     'value': n_chirps_total},
            {'parameter': 'n_chirps_used',      'value': len(labeled)},
            {'parameter': 'species_kept',       'value': ','.join(kept_species)},
            {'parameter': 'species_excluded',   'value': ','.join(excluded_species)},
            {'parameter': 'n_train',            'value': len(train_idx)},
            {'parameter': 'n_val',              'value': len(val_idx)},
            {'parameter': 'n_test',             'value': len(test_idx)},
            {'parameter': 'val_accuracy',       'value': round(val_accuracy, 4)},
            {'parameter': 'test_accuracy',      'value': round(test_accuracy, 4)},
            {'parameter': 'elapsed_secs',       'value': round(elapsed, 1)},
        ]).to_csv(self.out_dir / 'run_config.csv', index=False)
        log.info(f'Saved run_config.csv  (elapsed: {elapsed:.1f}s)')

        return TrainingResult(
            out_dir          = self.out_dir.resolve(),
            n_chirps_total   = n_chirps_total,
            n_chirps_used    = len(labeled),
            species_kept     = kept_species,
            species_excluded = excluded_species,
            n_train          = len(train_idx),
            n_val            = len(val_idx),
            n_test           = len(test_idx),
            val_accuracy     = val_accuracy,
            test_accuracy    = test_accuracy,
            elapsed_secs     = elapsed,
        )


    def _run_binary(self, _t0: float) -> TrainingResult:
        """
        Binary classification pipeline for a single species pair.

        File_ids are balanced to the smaller class before splitting so that
        the RF sees equal recording-level representation of both species.
        AUC is reported as the primary metric alongside accuracy.

        :param _t0: ``time.perf_counter()`` value captured at ``run()`` entry.
        :return:    :class:`TrainingResult` with ``test_auc`` populated.
        """
        sp_a, sp_b = self.species_pair
        log.info(f'Binary mode: {sp_a} vs {sp_b}')
        self.out_dir.mkdir(parents=True, exist_ok=True)

        raw_df         = self._load_data()
        n_chirps_total = len(raw_df)
        feature_cols   = [
            c for c in raw_df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(raw_df[c])
        ]

        labeled = raw_df[raw_df['species'].isin([sp_a, sp_b])].copy()
        labeled = labeled.dropna(subset=feature_cols).reset_index(drop=True)
        log.info(f'After filtering to {sp_a}/{sp_b}: {len(labeled):,} chirp rows')

        missing = [s for s in (sp_a, sp_b) if s not in labeled['species'].unique()]
        if missing:
            log.warn(f'Species not found in data: {", ".join(missing)}')
            sys.exit(1)

        labeled     = self._balance_by_file_id(labeled, sp_a, sp_b)
        train_idx, val_idx, test_idx = self._split_indices(labeled, feature_cols)

        X_train     = labeled.loc[train_idx, feature_cols].values
        y_train_raw = labeled.loc[train_idx, 'species'].values
        X_val       = labeled.loc[val_idx,   feature_cols].values
        y_val_raw   = labeled.loc[val_idx,   'species'].values
        X_test      = labeled.loc[test_idx,  feature_cols].values
        y_test_raw  = labeled.loc[test_idx,  'species'].values

        le      = LabelEncoder().fit([sp_a, sp_b])
        y_train = le.transform(y_train_raw)
        y_val   = le.transform(y_val_raw)
        y_test  = le.transform(y_test_raw)

        log.info(f'Training binary RF: {self.n_estimators} trees, n_jobs={self.n_jobs} ...')
        rf = RandomForestClassifier(
            n_estimators     = self.n_estimators,
            max_features     = self.max_features,
            min_samples_leaf = self.min_samples_leaf,
            class_weight     = 'balanced',
            n_jobs           = self.n_jobs,
            random_state     = self.random_state,
            verbose          = 1,
        )
        rf.fit(X_train, y_train)
        log.info('Training complete')

        val_accuracy  = rf.score(X_val,  y_val)
        test_accuracy = rf.score(X_test, y_test)
        log.info(f'Val  accuracy: {val_accuracy:.4f}')
        log.info(f'Test accuracy: {test_accuracy:.4f}')

        pos_label = int(le.transform([sp_a])[0])
        test_auc  = self._save_roc_curve(
            rf, X_test, y_test, pos_label=pos_label, class_name=sp_a
        )
        log.info(f'Test AUC ({sp_a} as positive): {test_auc:.4f}')

        log.info('Saving artifacts ...')
        joblib.dump(rf, self.out_dir / 'rf_model.joblib')
        joblib.dump(le, self.out_dir / 'label_encoder.joblib')

        y_test_pred = rf.predict(X_test)
        report = classification_report(y_test, y_test_pred,
                                       target_names=le.classes_, digits=4)
        (self.out_dir / 'classification_report.txt').write_text(report)
        log.info(f'\n{report}')
        self._save_confusion_matrix(y_test, y_test_pred, list(le.classes_))
        self._save_feature_importances(rf.feature_importances_, feature_cols)

        pd.DataFrame({
            'file_id'      : labeled.loc[test_idx, 'file_id'].values,
            'chirp_idx'    : labeled.loc[test_idx, 'chirp_idx'].values
                             if 'chirp_idx' in labeled.columns else np.nan,
            'rec_site'     : labeled.loc[test_idx, 'rec_site'].values
                             if 'rec_site' in labeled.columns else np.nan,
            'species_true' : y_test_raw,
            'species_pred' : le.inverse_transform(y_test_pred),
            'confidence'   : rf.predict_proba(X_test).max(axis=1).round(4),
        }).to_csv(self.out_dir / 'test_predictions.csv', index=False)
        log.info('Saved test_predictions.csv')

        elapsed = time.perf_counter() - _t0
        pd.DataFrame([
            {'parameter': 'mode',            'value': _MODE_BINARY},
            {'parameter': 'species_a',        'value': sp_a},
            {'parameter': 'species_b',        'value': sp_b},
            {'parameter': 'n_estimators',     'value': self.n_estimators},
            {'parameter': 'max_features',     'value': self.max_features},
            {'parameter': 'min_samples_leaf', 'value': self.min_samples_leaf},
            {'parameter': 'random_state',     'value': self.random_state},
            {'parameter': 'n_chirps_total',   'value': n_chirps_total},
            {'parameter': 'n_chirps_used',    'value': len(labeled)},
            {'parameter': 'n_train',          'value': len(train_idx)},
            {'parameter': 'n_val',            'value': len(val_idx)},
            {'parameter': 'n_test',           'value': len(test_idx)},
            {'parameter': 'val_accuracy',     'value': round(val_accuracy, 4)},
            {'parameter': 'test_accuracy',    'value': round(test_accuracy, 4)},
            {'parameter': 'test_auc',         'value': round(test_auc, 4)},
            {'parameter': 'elapsed_secs',     'value': round(elapsed, 1)},
        ]).to_csv(self.out_dir / 'run_config.csv', index=False)
        log.info(f'Saved run_config.csv  (elapsed: {elapsed:.1f}s)')

        return TrainingResult(
            out_dir          = self.out_dir.resolve(),
            n_chirps_total   = n_chirps_total,
            n_chirps_used    = len(labeled),
            species_kept     = [sp_a, sp_b],
            species_excluded = [],
            n_train          = len(train_idx),
            n_val            = len(val_idx),
            n_test           = len(test_idx),
            val_accuracy     = val_accuracy,
            test_accuracy    = test_accuracy,
            elapsed_secs     = elapsed,
            test_auc         = test_auc,
            species_pair     = (sp_a, sp_b),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments for :class:`RFTrainer`.

    :return: ``argparse.Namespace`` with validated attributes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='species_pred_random_forest',
        description=(
            'Train a Random Forest bat species classifier on SonoBat\n'
            'acoustic measures produced by sb_measures_postprocessing.py.\n\n'
            'The train/val/test split is performed at the file_id level\n'
            '(stratified by modal species) to prevent data leakage.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='PATH',
        help=(
            'Chirp-level measures file produced by sb_measures_postprocessing.py.\n'
            'Read via Utils.read_df_file(); accepts .parquet/.pq (primary),\n'
            '.feather, and .csv (legacy sono_batch_processing.py inputs).'
        ),
    )
    parser.add_argument(
        '-o', '--out-dir',
        required=True,
        metavar='DIR',
        help='Directory for all output artifacts (created if absent).',
    )
    parser.add_argument(
        '--min-species-count',
        type=int,
        default=_DEFAULT_MIN_SPECIES_COUNT,
        metavar='N',
        help=(
            f'Minimum number of labeled fragments for a species to be\n'
            f'included in training (default: {_DEFAULT_MIN_SPECIES_COUNT}).'
        ),
    )
    parser.add_argument(
        '--val-frac',
        type=float,
        default=0.15,
        metavar='F',
        help='Fraction of file_ids for validation (default: 0.15).',
    )
    parser.add_argument(
        '--test-frac',
        type=float,
        default=0.15,
        metavar='F',
        help='Fraction of file_ids for test (default: 0.15).',
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=_DEFAULT_N_ESTIMATORS,
        metavar='N',
        help=f'Number of RF trees (default: {_DEFAULT_N_ESTIMATORS}).',
    )
    parser.add_argument(
        '--max-features',
        default=_DEFAULT_MAX_FEATURES,
        metavar='STR',
        help=(
            f'max_features for each split (default: {_DEFAULT_MAX_FEATURES}).\n'
            f'Accepts "sqrt", "log2", or a float fraction.'
        ),
    )
    parser.add_argument(
        '--min-samples-leaf',
        type=int,
        default=_DEFAULT_MIN_SAMPLES_LEAF,
        metavar='N',
        help=f'min_samples_leaf for RF (default: {_DEFAULT_MIN_SAMPLES_LEAF}).',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=_DEFAULT_N_JOBS,
        metavar='N',
        help='Parallel jobs for RF training (-1 = all cores, default: -1).',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        metavar='N',
        help='Random seed for reproducibility (default: 42).',
    )

    parser.add_argument(
        '--mode',
        choices=[_MODE_MULTICLASS, _MODE_BINARY],
        default=_MODE_MULTICLASS,
        help=(
            f'Training mode (default: {_MODE_MULTICLASS}).\n'
            f'  {_MODE_MULTICLASS}  trains all species above --min-species-count.\n'
            f'  {_MODE_BINARY}      trains two species only; requires\n'
            f'               --species-pair; file_ids balanced; AUC reported.'
        ),
    )
    parser.add_argument(
        '--species-pair',
        nargs=2,
        metavar='SPECIES',
        default=None,
        help='Two species codes for binary mode, e.g. --species-pair Lano Tabr',
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        parser.error(f'Input file not found: {args.input}')
    if args.val_frac + args.test_frac >= 1.0:
        parser.error('val-frac + test-frac must be < 1.0')
    if args.mode == _MODE_BINARY and not args.species_pair:
        parser.error('--species-pair is required for --mode binary')
    if args.mode == _MODE_MULTICLASS and args.species_pair:
        parser.error('--species-pair is only used with --mode binary')

    args.input   = Path(args.input)
    args.out_dir = Path(args.out_dir)
    return args


def main() -> None:
    """
    CLI entry point for :class:`RFTrainer`.
    """
    args = _parse_args()

    trainer = RFTrainer(
        input_path        = args.input,
        out_dir           = args.out_dir,
        min_species_count = args.min_species_count,
        val_frac          = args.val_frac,
        test_frac         = args.test_frac,
        n_estimators      = args.n_estimators,
        max_features      = args.max_features,
        min_samples_leaf  = args.min_samples_leaf,
        n_jobs            = args.n_jobs,
        random_state      = args.random_state,
        mode              = args.mode,
        species_pair      = tuple(args.species_pair) if args.species_pair else None,
    )

    result = trainer.run()
    log.info(result.summary())
    sys.exit(0 if result.test_accuracy > 0 else 1)


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()
