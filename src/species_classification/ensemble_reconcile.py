#!/usr/bin/env python
# *********************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-24 17:36:28
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-24 17:37:14
# *********************************************

"""
ensemble_reconcile.py
======================
Reconcile per-chirp predictions from the Random Forest and CNN models
into a single unified predictions CSV consumed by
:class:`~species_classification.rf_confidence_join.RFConfidenceJoiner`.

**This is a stub.**  The interface contract — input columns, output
columns, CLI arguments, and combination logic — is fully specified so
that :mod:`rf_confidence_join` and the rest of the downstream pipeline
can be written against it now.  The internals of
:meth:`EnsembleReconciler._combine` are marked ``# TODO`` and will be
filled in once ``train_cnn.py`` and its inference counterpart are
available.

Workflow position
-----------------
::

    measures.csv
        │
        ├──→ hierarchical_rf_predict.py   →  rf_predictions/predictions.csv
        │
        └──→ cnn_predict.py  (upcoming)   →  cnn_predictions/predictions.csv
                                                   │
                                        ensemble_reconcile.py    ← HERE
                                                   │
                                        ensemble_predictions.csv
                                                   │
                                        rf_confidence_join.py
                                                   │
                                        measures_classified.csv
                                                   │
                                        from_scratch_postprocessing.py
                                                   │
                                        bats_<ts>.parquet

Input contract
--------------
Both prediction CSVs must contain:

``file_id``             Integer chop identifier (from PathEncoder).
``chirp_idx``           0-based chirp rank within the chop.
``final_pred``          Winning species label (string, e.g. ``'Myca'``).
``final_prob``          Probability of the winning class in [0, 1].
                        For the RF this is ``final_rf_agreement``
                        (renamed before passing here); for the CNN it
                        is the softmax probability of the argmax class.

Both CSVs must also contain one column per species in the label set
with the name ``prob_<species>`` (e.g. ``prob_Myca``, ``prob_Tabr``),
carrying the per-class probability for that chirp.  These columns are
used for the weighted combination before taking the argmax.

The RF predictions CSV additionally carries:

``routing``             Which sub-classifier was used (omnibus /
                        coto_tabr / lano_tabr).  Passed through
                        unchanged to the output for diagnostics.

Output contract
---------------
The output CSV written to ``--out-csv`` contains:

``file_id``             Unchanged.
``chirp_idx``           Unchanged.
``final_pred``          Reconciled winning species label.
``final_prob``          Combined probability of the winning class.
``prob_<species>``      Combined per-class probabilities (one column
                        per species, same names as inputs).
``source``              Which model(s) contributed:
                        ``'rf_only'`` | ``'cnn_only'`` | ``'ensemble'``.
                        ``'rf_only'`` is used whenever the CNN
                        predictions CSV is absent or does not contain
                        this (file_id, chirp_idx) — providing graceful
                        degradation during the transition period before
                        the CNN is fully deployed.
``routing``             Passed through from RF predictions.

The ``final_prob`` column in the output is what
:class:`~species_classification.rf_confidence_join.RFConfidenceJoiner`
uses as ``Prob`` in the SonoBat-analog confidence formula, in place of
the RF-only ``final_rf_agreement`` column it currently reads.

Combination method
------------------
The intended combination is a **weighted geometric mean** of per-class
probabilities, re-normalised to sum to 1 before taking the argmax::

    combined[c] = rf_prob[c] ** w_rf  ×  cnn_prob[c] ** w_cnn
    combined    = combined / combined.sum()          # re-normalise
    final_pred  = argmax(combined)
    final_prob  = combined[final_pred]

Starting weights: ``w_rf = w_cnn = 0.5``.  Override via
``--weight-rf`` / ``--weight-cnn``.

The geometric mean is preferred over arithmetic mean because it
penalises cases where one model is highly confident and the other is
near-uniform (i.e. uncertain) more aggressively than arithmetic
averaging would, producing a conservative combined probability that
better matches the semantics of SonoBat's ``Prob`` term.

Graceful degradation
--------------------
If ``--cnn-predictions-csv`` is omitted, or if a ``(file_id, chirp_idx)``
pair is present in the RF predictions but absent from the CNN predictions
(e.g. because the CNN was run on a subset), the RF prediction is passed
through unchanged with ``source = 'rf_only'``.  This allows the pipeline
to run end-to-end before the CNN is available, and to handle partial CNN
coverage during rollout.

CLI usage
---------
Full ensemble (once CNN is available)::

    python ensemble_reconcile.py \\
        --rf-predictions-csv   /qnap/src/marsh_stanford_processed/rf_predictions/predictions.csv \\
        --cnn-predictions-csv  /qnap/src/marsh_stanford_processed/cnn_predictions/predictions.csv \\
        --out-csv              /qnap/src/marsh_stanford_processed/ensemble_predictions.csv \\
        --weight-rf   0.5 \\
        --weight-cnn  0.5

RF-only interim mode (before CNN is available)::

    python ensemble_reconcile.py \\
        --rf-predictions-csv  /qnap/src/marsh_stanford_processed/rf_predictions/predictions.csv \\
        --out-csv             /qnap/src/marsh_stanford_processed/ensemble_predictions.csv

In both cases the output path fed into ``rf_confidence_join.py`` is
``ensemble_predictions.csv``.  The downstream invocation does not change
between interim and full-ensemble modes.
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

from logging_service import LoggingService

# ---------------------------------------------------------------------------
# Default ensemble weights
# ---------------------------------------------------------------------------

_WEIGHT_RF_DEFAULT:  float = 0.5
_WEIGHT_CNN_DEFAULT: float = 0.5

# Column prefix for per-class probability columns in both input CSVs.
_PROB_COL_PREFIX: str = 'prob_'


class EnsembleReconciler:
    """
    Reconcile RF and CNN per-chirp predictions into a single unified
    predictions CSV.

    :param rf_predictions_csv:  Path to ``predictions.csv`` from
                                ``hierarchical_rf_predict.py``.
                                Required.
    :param cnn_predictions_csv: Path to ``predictions.csv`` from
                                ``cnn_predict.py``.  Optional — if
                                omitted or if a chirp has no CNN
                                prediction, the RF prediction is passed
                                through with ``source='rf_only'``.
    :param out_csv:             Destination path for the unified
                                predictions CSV.
    :param weight_rf:           Weight for the RF in the geometric mean
                                combination (default 0.5).
    :param weight_cnn:          Weight for the CNN in the geometric mean
                                combination (default 0.5).
    """

    def __init__(
        self,
        rf_predictions_csv:  str | Path,
        out_csv:             str | Path,
        cnn_predictions_csv: str | Path | None = None,
        weight_rf:           float = _WEIGHT_RF_DEFAULT,
        weight_cnn:          float = _WEIGHT_CNN_DEFAULT,
    ) -> None:
        self.log                 = LoggingService()
        self.rf_predictions_csv  = Path(rf_predictions_csv)
        self.cnn_predictions_csv = (Path(cnn_predictions_csv)
                                    if cnn_predictions_csv is not None
                                    else None)
        self.out_csv             = Path(out_csv)
        self.weight_rf           = weight_rf
        self.weight_cnn          = weight_cnn

        if abs(weight_rf + weight_cnn - 1.0) > 1e-6:
            self.log.err(
                f"--weight-rf ({weight_rf}) + --weight-cnn ({weight_cnn}) "
                f"must sum to 1.0"
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full reconciliation workflow.

        Steps:

        1. Load and validate RF predictions (required).
        2. Load CNN predictions if provided.
        3. Combine per-class probabilities or pass RF through.
        4. Write unified predictions CSV.

        :return: None
        :raises SystemExit: On missing files or schema violations.
        """
        rf_preds  = self._load_rf_predictions()
        cnn_preds = self._load_cnn_predictions()   # None if not provided
        unified   = self._combine(rf_preds, cnn_preds)
        self._write(unified)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_rf_predictions(self) -> pd.DataFrame:
        """
        Load and validate the RF predictions CSV.

        Required columns: ``file_id``, ``chirp_idx``, ``final_pred``,
        ``final_rf_agreement``, and at least one ``prob_<species>``
        column.  ``final_rf_agreement`` is renamed to ``final_prob``
        here so that downstream code is model-agnostic.

        :return: RF predictions DataFrame.
        :raises SystemExit: If the file is missing or required columns
                            are absent.
        """
        if not self.rf_predictions_csv.exists():
            self.log.err(
                f"RF predictions file not found: {self.rf_predictions_csv}"
            )
            sys.exit(1)

        self.log.info(f"Loading RF predictions: {self.rf_predictions_csv}")
        df = pd.read_csv(self.rf_predictions_csv, low_memory=False)
        self.log.info(f"  {len(df):,} rows, {len(df.columns)} columns")

        required = {'file_id', 'chirp_idx', 'final_pred',
                    'final_rf_agreement'}
        missing  = required - set(df.columns)
        if missing:
            self.log.err(
                f"RF predictions CSV missing required columns: {missing}. "
                f"Was it produced by hierarchical_rf_predict.py?"
            )
            sys.exit(1)

        prob_cols = [c for c in df.columns
                     if c.startswith(_PROB_COL_PREFIX)]
        if not prob_cols:
            self.log.err(
                f"RF predictions CSV contains no '{_PROB_COL_PREFIX}*' "
                f"per-class probability columns.  "
                f"hierarchical_rf_predict.py must write one column per "
                f"species in the label set."
            )
            sys.exit(1)

        # Rename to the model-agnostic name used by the rest of this module.
        df = df.rename(columns={'final_rf_agreement': 'final_prob'})
        self.log.info(
            f"  RF species: "
            f"{sorted(c[len(_PROB_COL_PREFIX):] for c in prob_cols)}"
        )
        return df

    def _load_cnn_predictions(self) -> pd.DataFrame | None:
        """
        Load and validate the CNN predictions CSV, if provided.

        Required columns: ``file_id``, ``chirp_idx``, ``final_pred``,
        ``final_prob``, and the same ``prob_<species>`` columns present
        in the RF predictions.

        :return: CNN predictions DataFrame, or ``None`` if
                 ``--cnn-predictions-csv`` was not supplied.
        :raises SystemExit: If the file is specified but missing or has
                            schema violations.
        """
        if self.cnn_predictions_csv is None:
            self.log.info(
                "No CNN predictions CSV supplied — "
                "running in RF-only interim mode. "
                "All rows will have source='rf_only'."
            )
            return None

        if not self.cnn_predictions_csv.exists():
            self.log.err(
                f"CNN predictions file not found: "
                f"{self.cnn_predictions_csv}"
            )
            sys.exit(1)

        self.log.info(f"Loading CNN predictions: {self.cnn_predictions_csv}")
        df = pd.read_csv(self.cnn_predictions_csv, low_memory=False)
        self.log.info(f"  {len(df):,} rows, {len(df.columns)} columns")

        required = {'file_id', 'chirp_idx', 'final_pred', 'final_prob'}
        missing  = required - set(df.columns)
        if missing:
            self.log.err(
                f"CNN predictions CSV missing required columns: {missing}. "
                f"Expected columns: {sorted(required)}."
            )
            sys.exit(1)

        prob_cols = [c for c in df.columns
                     if c.startswith(_PROB_COL_PREFIX)]
        if not prob_cols:
            self.log.err(
                f"CNN predictions CSV contains no '{_PROB_COL_PREFIX}*' "
                f"per-class probability columns."
            )
            sys.exit(1)

        self.log.info(
            f"  CNN species: "
            f"{sorted(c[len(_PROB_COL_PREFIX):] for c in prob_cols)}"
        )
        return df

    # ------------------------------------------------------------------
    # Combination  (TODO: implement once CNN is available)
    # ------------------------------------------------------------------

    def _combine(
        self,
        rf_preds:  pd.DataFrame,
        cnn_preds: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Combine RF and CNN per-class probabilities into a unified
        predictions DataFrame.

        **Current behaviour (interim / RF-only mode):**
        If ``cnn_preds`` is ``None``, the RF predictions are passed
        through unchanged with a ``source='rf_only'`` column added.
        This is the correct behaviour until ``cnn_predict.py`` exists.

        **Intended behaviour (full ensemble):**

        For each chirp row present in both CSVs::

            combined[c] = rf_prob[c] ** w_rf  ×  cnn_prob[c] ** w_cnn
            combined    = combined / combined.sum()      # re-normalise
            final_pred  = argmax(combined)
            final_prob  = combined[final_pred]
            source      = 'ensemble'

        For chirps present only in the RF CSV (no CNN coverage)::

            pass through RF prediction unchanged
            source = 'rf_only'

        For chirps present only in the CNN CSV (no RF prediction —
        should not occur in normal operation but handled defensively)::

            pass through CNN prediction unchanged
            source = 'cnn_only'

        Species label set mismatch between the two CSVs (e.g. CNN was
        trained on a different species list) is a hard error.

        # TODO: implement the geometric mean combination block once
        #       cnn_predict.py and its output schema are finalised.
        #       The RF-only pass-through below is intentional interim
        #       behaviour, not a placeholder to be deleted — it must
        #       remain as the fallback for chirps with no CNN coverage.

        :param rf_preds:  RF predictions DataFrame (required).
        :param cnn_preds: CNN predictions DataFrame, or ``None``.
        :return:          Unified predictions DataFrame matching the
                          output contract in the module docstring.
        """
        if cnn_preds is None:
            # ── Interim RF-only mode ───────────────────────────────────
            out = rf_preds.copy()
            out['source'] = 'rf_only'
            self.log.info(
                f"RF-only mode: passing {len(out):,} rows through "
                f"unchanged."
            )
            return out

        # ── TODO: full ensemble combination ───────────────────────────
        # Validate that both CSVs share the same species label set.
        rf_prob_cols  = sorted(c for c in rf_preds.columns
                               if c.startswith(_PROB_COL_PREFIX))
        cnn_prob_cols = sorted(c for c in cnn_preds.columns
                               if c.startswith(_PROB_COL_PREFIX))
        if rf_prob_cols != cnn_prob_cols:
            self.log.err(
                f"RF and CNN predictions use different species sets:\n"
                f"  RF:  {rf_prob_cols}\n"
                f"  CNN: {cnn_prob_cols}\n"
                f"Both models must be trained on the same species label "
                f"set before ensemble reconciliation is possible."
            )
            sys.exit(1)

        # TODO: implement weighted geometric mean combination here.
        #
        # Sketch:
        #   merged = rf_preds.merge(cnn_preds, on=['file_id','chirp_idx'],
        #                           how='outer', suffixes=('_rf','_cnn'),
        #                           indicator=True)
        #
        #   For rows in both ('both'):
        #     for col in prob_cols:
        #       combined[col] = (merged[col+'_rf'] ** self.weight_rf
        #                        * merged[col+'_cnn'] ** self.weight_cnn)
        #     combined = combined.div(combined[prob_cols].sum(axis=1),axis=0)
        #     final_pred = combined[prob_cols].idxmax(axis=1).str[len(prefix):]
        #     final_prob = combined[prob_cols].max(axis=1)
        #     source     = 'ensemble'
        #
        #   For rows in RF only:
        #     pass through rf values, source = 'rf_only'
        #
        #   For rows in CNN only:
        #     pass through cnn values, source = 'cnn_only'

        raise NotImplementedError(
            "Full ensemble combination is not yet implemented. "
            "Omit --cnn-predictions-csv to run in RF-only interim mode."
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _write(self, df: pd.DataFrame) -> None:
        """
        Write the unified predictions CSV.

        :param df: Unified predictions DataFrame.
        :return: None
        """
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_csv, index=False)
        if 'source' in df.columns:
            counts = df['source'].value_counts().to_dict()
            self.log.info(f"Source breakdown: {counts}")
        self.log.info(
            f"Wrote {len(df):,} rows → {self.out_csv}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for :class:`EnsembleReconciler`.

    :return: Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog='ensemble_reconcile.py',
        description=(
            'Reconcile RF and CNN per-chirp predictions into a single '
            'unified predictions CSV for rf_confidence_join.py.\n\n'
            'Run without --cnn-predictions-csv for RF-only interim mode '
            '(before the CNN is available).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Full ensemble (once CNN is available):
              python ensemble_reconcile.py \\
                  --rf-predictions-csv  .../rf_predictions/predictions.csv \\
                  --cnn-predictions-csv .../cnn_predictions/predictions.csv \\
                  --out-csv             .../ensemble_predictions.csv \\
                  --weight-rf  0.5 \\
                  --weight-cnn 0.5

            RF-only interim mode (before CNN is available):
              python ensemble_reconcile.py \\
                  --rf-predictions-csv  .../rf_predictions/predictions.csv \\
                  --out-csv             .../ensemble_predictions.csv

            In both cases the output path is the same; rf_confidence_join.py
            does not change between interim and full-ensemble modes.
        """),
    )
    parser.add_argument(
        '--rf-predictions-csv',
        required=True, metavar='CSV', type=Path,
        help='predictions.csv from hierarchical_rf_predict.py.',
    )
    parser.add_argument(
        '--cnn-predictions-csv',
        default=None, metavar='CSV', type=Path,
        help=(
            'predictions.csv from cnn_predict.py.  '
            'Omit to run in RF-only interim mode.'
        ),
    )
    parser.add_argument(
        '--out-csv',
        required=True, metavar='CSV', type=Path,
        help='Destination path for the unified predictions CSV.',
    )
    parser.add_argument(
        '--weight-rf',
        type=float, default=_WEIGHT_RF_DEFAULT, metavar='FLOAT',
        help=(
            f'RF weight for geometric mean combination '
            f'(default: {_WEIGHT_RF_DEFAULT}). '
            f'Must sum to 1.0 with --weight-cnn.'
        ),
    )
    parser.add_argument(
        '--weight-cnn',
        type=float, default=_WEIGHT_CNN_DEFAULT, metavar='FLOAT',
        help=(
            f'CNN weight for geometric mean combination '
            f'(default: {_WEIGHT_CNN_DEFAULT}). '
            f'Must sum to 1.0 with --weight-rf.'
        ),
    )

    args = parser.parse_args()

    if not args.rf_predictions_csv.exists():
        parser.error(
            f'--rf-predictions-csv not found: {args.rf_predictions_csv}'
        )
    if (args.cnn_predictions_csv is not None
            and not args.cnn_predictions_csv.exists()):
        parser.error(
            f'--cnn-predictions-csv not found: {args.cnn_predictions_csv}'
        )
    if abs(args.weight_rf + args.weight_cnn - 1.0) > 1e-6:
        parser.error(
            f'--weight-rf ({args.weight_rf}) + '
            f'--weight-cnn ({args.weight_cnn}) must sum to 1.0'
        )
    return args


def main() -> None:
    """
    CLI entry point for :class:`EnsembleReconciler`.

    :return: None
    """
    args = _parse_args()
    EnsembleReconciler(
        rf_predictions_csv  = args.rf_predictions_csv,
        cnn_predictions_csv = args.cnn_predictions_csv,
        out_csv             = args.out_csv,
        weight_rf           = args.weight_rf,
        weight_cnn          = args.weight_cnn,
    ).run()


if __name__ == '__main__':
    main()