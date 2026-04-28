#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-24 17:19:27
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-28 09:40:16
# **********************************************************
"""
rf_confidence_join.py
======================
Join the per-chirp predictions produced by
:class:`~species_classification.hierarchical_rf_predict.HierarchicalPredictor`
back onto the raw measures CSV from
:class:`~chirp_detection.chirp_measures_extraction.MeasureExtractor`, then
compute a confidence score that faithfully replicates SonoBat's composite
formula using only quantities available in our from-scratch pipeline.

Workflow position
-----------------
::

    chirp_measures_extraction.py  →  measures.csv
                                          |
    hierarchical_rf_predict.py    →  predictions.csv
                                          |
    rf_confidence_join.py         →  measures_classified.csv   ← HERE
                                          |
    from_scratch_postprocessing.py →  bats_<ts>.parquet

The SonoBat formula and our exact replication
----------------------------------------------
SonoBat's composite confidence (from CumulativeSonoBatch) is::

    confidence = Prob × (0.7 × (#Maj / #Accp) + 0.3 × log1p(#Accp) / log1p(30))

The three SB quantities are all computed within a single 2-second chop:

``Prob``
    SonoBat's discriminant probability for the winning species call.

``#Accp``
    Count of calls in this chop that passed SonoBat's quality gate.
    The gate is a fixed global threshold on SonoBat's internal Quality
    score (``MinAccpQuality``), capped at ``Max#CallsConsidered``.
    From the CumulativeSonoBatch headers on the barn and lake2 corpora::

        MinAccpQuality      = 0.60
        Max#CallsConsidered = 48

    So: ``#Accp = min(count of chirps with Quality >= 0.60, 48)``

``#Maj``
    Count of accepted calls (Quality >= 0.60) whose discriminant
    probability for any species exceeded SonoBat's DP threshold
    (typically 0.90) AND whose winning species matches the chop-level
    species assignment.

Our pipeline has direct equivalents for all three, drawn from
``measures.csv`` (Quality per chirp) and ``predictions.csv``
(``final_pred``, ``final_rf_agreement`` per chirp):

``Prob``   →  ``final_rf_agreement``:  fraction of RF trees that voted
              for the winning class for this chirp.

``#Accp``  →  ``n_accp``:  count of chirps in this ``file_id`` with
              ``Quality >= ACCP_QUALITY_THRESH (0.60)``, capped at
              ``MAX_CALLS_CONSIDERED (48)``.

``#Maj``   →  ``n_maj``:  count of Quality-passing chirps in this
              ``file_id`` whose ``final_rf_agreement >=
              MAJ_PROB_THRESH (0.90)`` AND whose ``final_pred`` matches
              the plurality species of this chop.

The formula then becomes, per chirp::

    consensus  = n_maj / n_accp
    evidence   = log1p(n_accp) / log1p(ACCP_LOG_CEIL)   # ACCP_LOG_CEIL = 30
    confidence = final_rf_agreement
                 × (0.7 × consensus + 0.3 × evidence)

All output values are in [0, 1] by the same construction as the SB formula.

Residual approximation
----------------------
SonoBat's ``#Maj`` requires SB's own DP >= 0.90 per call, which is not
identical to ``final_rf_agreement >= 0.90`` — it is a different classifier
with a different probability scale.  This is an unavoidable approximation;
no closer proxy exists without SonoBat's internal model.

CNN integration (future)
-------------------------
When ``train_cnn.py`` is available its per-chirp softmax probability can be
combined with ``final_rf_agreement`` before applying the formula::

    combined_prob = rf_prob ** w_rf × cnn_prob ** w_cnn   (w_rf + w_cnn = 1)

Add a ``--cnn-predictions`` argument and replace ``rf_prob`` with
``combined_prob``; the rest of the formula is unchanged.

CLI usage
---------
::

    python rf_confidence_join.py \\
        --measures-csv    /qnap/src/marsh_stanford_processed/measures.csv \\
        --predictions-csv /qnap/src/marsh_stanford_processed/predictions.csv \\
        --out-csv         /qnap/src/marsh_stanford_processed/measures_classified.csv

Override SonoBat constants if your corpus used different SonoBat settings::

    python rf_confidence_join.py ... \\
        --accp-quality-thresh  0.80 \\
        --max-calls-considered 48 \\
        --maj-prob-thresh      0.90
"""

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from logging_service import LoggingService

# ---------------------------------------------------------------------------
# SonoBat constants — from CumulativeSonoBatch headers on barn + lake2 data.
# Override via CLI if your corpus used different SonoBat settings.
# ---------------------------------------------------------------------------

# MinAccpQuality in CumulativeSonoBatch: minimum Quality for a chirp to count
# toward #Accp.  Confirmed 0.60 from barn and lake2 headers and empirically
# validated against 20.8M chirps in quality_threshold_calibration.py.
_ACCP_QUALITY_THRESH_DEFAULT: float = 0.60

# Max#CallsConsidered in CumulativeSonoBatch: hard cap on #Accp per chop.
_MAX_CALLS_CONSIDERED_DEFAULT: int = 48

# Analog of SonoBat's DP threshold: minimum final_rf_agreement for a
# quality-passing chirp to count toward n_maj.
_MAJ_PROB_THRESH_DEFAULT: float = 0.90

# Fixed log-normalisation ceiling matching SonoBat's formula.
_ACCP_LOG_CEIL: int = 30

# Confidence formula weights — must match SonoBatPostProcessor constants.
_WEIGHT_CONSENSUS: float = 0.7
_WEIGHT_EVIDENCE:  float = 0.3


class RFConfidenceJoiner:
    """
    Join RF predictions onto the measures CSV, compute the SonoBat-analog
    confidence score, and write ``measures_classified.csv`` ready for
    :class:`~sonobat_utils.from_scratch_postprocessing.FromScratchPostProcessor`.

    :param measures_csv:         Path to raw measures CSV from
                                 ``chirp_measures_extraction.py``.
    :param predictions_csv:      Path to ``predictions.csv`` from
                                 ``hierarchical_rf_predict.py``.
    :param out_csv:              Destination path for the classified
                                 measures CSV.
    :param accp_quality_thresh:  Minimum Quality for a chirp to count
                                 toward n_accp (default 0.60, matching
                                 SonoBat MinAccpQuality on barn/lake2).
    :param max_calls_considered: Hard cap on n_accp per chop (default 48,
                                 matching SonoBat Max#CallsConsidered).
    :param maj_prob_thresh:      Minimum final_rf_agreement for a
                                 quality-passing chirp to count toward
                                 n_maj (default 0.90, analog of SonoBat
                                 DP threshold).
    """

    def __init__(
        self,
        measures_csv:         str | Path,
        predictions_csv:      str | Path,
        out_csv:              str | Path,
        accp_quality_thresh:  float = _ACCP_QUALITY_THRESH_DEFAULT,
        max_calls_considered: int   = _MAX_CALLS_CONSIDERED_DEFAULT,
        maj_prob_thresh:      float = _MAJ_PROB_THRESH_DEFAULT,
        chop_report:          str | Path | None = None,
        manifest:             str | Path | None = None,
        chop_duration:        float = 2.0,
    ) -> None:
        self.log                  = LoggingService()
        self.measures_csv         = Path(measures_csv)
        self.predictions_csv      = Path(predictions_csv)
        self.out_csv              = Path(out_csv)
        self.accp_quality_thresh  = accp_quality_thresh
        self.max_calls_considered = max_calls_considered
        self.maj_prob_thresh      = maj_prob_thresh
        self.chop_report          = Path(chop_report) if chop_report else None
        self.manifest             = Path(manifest)    if manifest    else None
        self.chop_duration        = chop_duration

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full join and confidence computation.

        Steps:

        1. Load and validate both CSVs.
        2. Join predictions onto measures on ``(file_id, chirp_idx)``.
        3. Error on any unmatched measures rows.
        4. Compute per-chirp confidence using the SonoBat-analog formula.
        5. Write ``measures_classified.csv``.

        :return: None
        :raises SystemExit: On missing columns, unmatched rows, or I/O
                            errors.
        """
        measures, predictions = self._load_and_validate()
        merged                = self._join(measures, predictions)
        merged                = self._attach_source_duration(merged)
        merged                = self._compute_confidence(merged)
        self._write(merged)

    # ------------------------------------------------------------------
    # Load and validate
    # ------------------------------------------------------------------

    def _attach_source_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach a ``source_duration_s`` column to the merged DataFrame.

        The column is not used in the confidence formula itself (which
        retains the fixed ``log1p(30)`` ceiling for cross-site
        comparability) but is carried through to ``measures_classified.csv``
        so downstream analysis can stratify confidence values by recording
        length.  This matters because marsh recordings are 5-second files
        while barn/lake2 recordings are 2-second chops — confidence values
        are structurally lower for shorter recordings with fewer calls.

        Three resolution paths, tried in order:

        **Path 1 — ``--chop-report``** (from-scratch pipeline):
            ``chop_report.csv`` from ``wav_chopper.py`` has
            ``source_duration_s`` per ``file_id``.  Joined directly.

        **Path 2 — ``--manifest``** (legacy SonoBat pipeline):
            ``manifest.csv`` has ``time_in_file_ms`` per chirp.
            ``source_duration_s ≈ max(time_in_file_ms)/1000 + 0.005``
            per ``file_id`` (the +5ms accounts for the tail of the last
            detected call).  For SonoBat 2-second chops this consistently
            returns ~1.995s, confirming the approach.

        **Path 3 — ``--chop-duration``** (fallback constant):
            Broadcasts the scalar value to all rows.  Default 2.0s.
            Used when neither of the above files is available.

        :param df: Merged measures + predictions DataFrame with
                   ``file_id`` column.
        :return:   DataFrame with ``source_duration_s`` column added.
        """
        if self.chop_report is not None:
            # ── Path 1: chop_report.csv ───────────────────────────────
            self.log.info(
                f"Resolving source_duration_s from chop report: "
                f"{self.chop_report}"
            )
            cr = pd.read_csv(self.chop_report,
                             usecols=['file_id', 'source_duration_s'],
                             low_memory=False)
            # One row per chunk; take the first (all share the same source).
            dur_map = (
                cr.drop_duplicates('file_id')
                  .set_index('file_id')['source_duration_s']
            )
            df = df.copy()
            df['source_duration_s'] = df['file_id'].map(dur_map)
            n_missing = df['source_duration_s'].isna().sum()
            if n_missing:
                self.log.warn(
                    f"{n_missing:,} chirp rows have no matching file_id "
                    f"in chop_report — filling with chop_duration="
                    f"{self.chop_duration}s."
                )
                df['source_duration_s'] = df['source_duration_s'].fillna(
                    self.chop_duration
                )

        elif self.manifest is not None:
            # ── Path 2: manifest.csv (legacy SonoBat) ─────────────────
            self.log.info(
                f"Resolving source_duration_s from manifest: "
                f"{self.manifest}"
            )
            mf = pd.read_csv(self.manifest,
                             usecols=['file_id', 'time_in_file_ms'],
                             low_memory=False)
            mf['time_in_file_ms'] = pd.to_numeric(
                mf['time_in_file_ms'], errors='coerce'
            )
            dur_map = (
                mf.groupby('file_id')['time_in_file_ms']
                  .max()
                  .div(1000.0)          # ms → s
                  .add(0.005)           # +5ms for call tail
            )
            df = df.copy()
            df['source_duration_s'] = df['file_id'].map(dur_map)
            n_missing = df['source_duration_s'].isna().sum()
            if n_missing:
                self.log.warn(
                    f"{n_missing:,} chirp rows have no matching file_id "
                    f"in manifest — filling with chop_duration="
                    f"{self.chop_duration}s."
                )
                df['source_duration_s'] = df['source_duration_s'].fillna(
                    self.chop_duration
                )
            self.log.info(
                f"  source_duration_s: "
                f"median={df['source_duration_s'].median():.3f}s  "
                f"min={df['source_duration_s'].min():.3f}s  "
                f"max={df['source_duration_s'].max():.3f}s"
            )

        else:
            # ── Path 3: constant fallback ──────────────────────────────
            self.log.info(
                f"source_duration_s: using constant "
                f"{self.chop_duration}s for all rows "
                f"(no --chop-report or --manifest supplied)."
            )
            df = df.copy()
            df['source_duration_s'] = self.chop_duration

        return df

    def _load_and_validate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both CSVs and verify required columns are present.

        :return: ``(measures_df, predictions_df)`` tuple.
        :raises SystemExit: If a file is missing or required columns are
                            absent.
        """
        for path in (self.measures_csv, self.predictions_csv):
            if not path.exists():
                self.log.err(f"File not found: {path}")
                sys.exit(1)

        self.log.info(f"Loading measures:    {self.measures_csv}")
        measures = pd.read_csv(self.measures_csv, low_memory=False)
        self.log.info(
            f"  {len(measures):,} rows, {len(measures.columns)} columns"
        )

        self.log.info(f"Loading predictions: {self.predictions_csv}")
        predictions = pd.read_csv(self.predictions_csv, low_memory=False)
        self.log.info(
            f"  {len(predictions):,} rows, {len(predictions.columns)} columns"
        )

        # Quality is required because n_accp and n_maj both depend on it.
        required_measures = {'file_id', 'chirp_idx', 'Quality'}
        required_preds    = {'file_id', 'chirp_idx',
                             'final_pred', 'final_rf_agreement'}

        missing_m = required_measures - set(measures.columns)
        missing_p = required_preds    - set(predictions.columns)

        if missing_m:
            self.log.err(
                f"measures CSV missing required columns: {missing_m}. "
                f"Was it produced by chirp_measures_extraction.py?"
            )
            sys.exit(1)
        if missing_p:
            self.log.err(
                f"predictions CSV missing required columns: {missing_p}. "
                f"Was it produced by hierarchical_rf_predict.py?"
            )
            sys.exit(1)

        return measures, predictions

    # ------------------------------------------------------------------
    # Join
    # ------------------------------------------------------------------

    def _join(
        self,
        measures:    pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inner-join predictions onto measures on ``(file_id, chirp_idx)``.

        Every measures row must have a matching prediction — if any are
        unmatched the pipeline is in an inconsistent state and the run is
        aborted.  Orphan prediction rows (predictions with no matching
        measures row, e.g. chirps dropped upstream) are logged as a
        warning only.

        :param measures:    Measures DataFrame.
        :param predictions: Predictions DataFrame.
        :return:            Merged DataFrame.
        :raises SystemExit: If any measures row is unmatched.
        """
        pred_cols = [c for c in
                     ('file_id', 'chirp_idx',
                      'final_pred', 'final_rf_agreement', 'routing')
                     if c in predictions.columns]

        merged = measures.merge(
            predictions[pred_cols],
            on=['file_id', 'chirp_idx'],
            how='left',
            indicator=True,
        )

        unmatched = (merged['_merge'] == 'left_only').sum()
        if unmatched:
            sample = merged.loc[
                merged['_merge'] == 'left_only',
                ['file_id', 'chirp_idx']
            ].head(10)
            self.log.err(
                f"{unmatched:,} measures rows have no matching prediction "
                f"row — did you run hierarchical_rf_predict.py on this "
                f"measures CSV?\n"
                f"Sample unmatched rows:\n{sample.to_string(index=False)}"
            )
            sys.exit(1)

        measures_keys = set(
            zip(measures['file_id'].values, measures['chirp_idx'].values)
        )
        pred_keys = set(
            zip(predictions['file_id'].values,
                predictions['chirp_idx'].values)
        )
        orphans = pred_keys - measures_keys
        if orphans:
            self.log.warn(
                f"{len(orphans):,} prediction rows have no matching "
                f"measures row (chirps dropped upstream?) — ignored."
            )

        merged.drop(columns=['_merge'], inplace=True)
        self.log.info(f"Join complete: {len(merged):,} matched rows")
        return merged

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the SonoBat-analog confidence score for each chirp row.

        **n_accp  (≈ #Accp)**
            Count of chirps per ``file_id`` with
            ``Quality >= accp_quality_thresh``, capped at
            ``max_calls_considered``.  Mirrors SonoBat's
            ``MinAccpQuality=0.60`` / ``Max#CallsConsidered=48``
            confirmed from barn/lake2 CumulativeSonoBatch headers.

        **n_maj  (≈ #Maj)**
            Count of Quality-passing chirps per ``file_id`` whose
            ``final_rf_agreement >= maj_prob_thresh`` AND whose
            ``final_pred`` matches the plurality species for this chop.
            The plurality species is determined solely from the set of
            majority-candidate chirps (Quality-passing AND
            high-agreement), so the n_maj numerator and the denominator
            used to derive the chop winner are drawn from the same pool.
            Mirrors SonoBat's DP-threshold gate on #Maj.

        **rf_prob  (≈ Prob)**
            ``final_rf_agreement``: fraction of RF trees that voted for
            the winning class for this individual chirp.

        :param df: Merged DataFrame with ``Quality``, ``final_pred``,
                   and ``final_rf_agreement`` columns present.
        :return:   DataFrame with ``species`` and ``confidence`` columns
                   added.  Internal working columns (``_quality_pass``,
                   ``_maj_cand``, ``chop_species``) are dropped before
                   returning.
        """
        df = df.copy()
        rf_prob = df['final_rf_agreement'].astype(float)

        # ── Quality gate (replicates MinAccpQuality) ──────────────────
        quality_pass = df['Quality'].astype(float) >= self.accp_quality_thresh
        df['_quality_pass'] = quality_pass

        # ── n_accp: quality-gated count per chop, capped ──────────────
        n_accp = (
            quality_pass.astype(int)
            .groupby(df['file_id'])
            .transform('sum')
            .clip(upper=self.max_calls_considered)
            .astype(float)
        )
        self.log.info(
            f"n_accp (Quality >= {self.accp_quality_thresh}, "
            f"cap {self.max_calls_considered}): "
            f"median={n_accp.median():.1f}  max={n_accp.max():.0f}"
        )

        # ── Majority candidates: quality-passing AND high RF agreement ─
        maj_cand = quality_pass & (rf_prob >= self.maj_prob_thresh)
        df['_maj_cand'] = maj_cand

        # ── Plurality species per chop (from majority candidates only) ─
        # Build vote counts per (file_id, final_pred) among maj_cands,
        # take the argmax per file_id.
        cand_rows = df.loc[maj_cand, ['file_id', 'final_pred']]

        if len(cand_rows) == 0:
            self.log.warn(
                "No chirps passed both Quality >= "
                f"{self.accp_quality_thresh} and RF agreement >= "
                f"{self.maj_prob_thresh} in any chop.  All confidence "
                "values will be 0.  Check --accp-quality-thresh and "
                "--maj-prob-thresh."
            )
            df['species']    = df['final_pred']
            df['confidence'] = 0.0
            return df.drop(
                columns=[c for c in ('_quality_pass', '_maj_cand')
                         if c in df.columns]
            )

        vote_counts = (
            cand_rows
            .groupby(['file_id', 'final_pred'])
            .size()
            .rename('votes')
            .reset_index()
        )
        plurality = (
            vote_counts
            .sort_values('votes', ascending=False)
            .drop_duplicates('file_id', keep='first')
            .rename(columns={'final_pred': 'chop_species'})
            [['file_id', 'chop_species']]
        )
        df = df.merge(plurality, on='file_id', how='left')

        # file_ids with no majority candidate fall back to this chirp's
        # own final_pred (confidence will be 0 anyway via n_accp=0).
        df['chop_species'] = df['chop_species'].fillna(df['final_pred'])

        # ── n_maj: maj_cand chirps that match the chop's plurality ─────
        n_maj = (
            (df['_maj_cand'] & (df['final_pred'] == df['chop_species']))
            .astype(int)
            .groupby(df['file_id'])
            .transform('sum')
            .astype(float)
        )

        # ── Consensus and evidence ─────────────────────────────────────
        safe_n_accp = n_accp.replace(0, np.nan)
        consensus   = (n_maj / safe_n_accp).fillna(0.0).clip(0.0, 1.0)
        evidence    = (
            np.log1p(n_accp) / np.log1p(_ACCP_LOG_CEIL)
        ).clip(0.0, 1.0)

        # ── SonoBat formula ────────────────────────────────────────────
        confidence = (
            rf_prob * (
                _WEIGHT_CONSENSUS * consensus
                + _WEIGHT_EVIDENCE  * evidence
            )
        ).clip(0.0, 1.0)

        df['species']    = df['chop_species']
        df['confidence'] = confidence.round(6)

        # ── Diagnostics ───────────────────────────────────────────────
        n_zero_accp = (n_accp == 0).sum()
        if n_zero_accp:
            self.log.warn(
                f"{n_zero_accp:,} chirp rows belong to chops where no "
                f"chirp passed Quality >= {self.accp_quality_thresh}; "
                f"their confidence is 0."
            )
        self.log.info(
            f"Confidence: "
            f"min={confidence.min():.4f}  "
            f"median={confidence.median():.4f}  "
            f"mean={confidence.mean():.4f}  "
            f"max={confidence.max():.4f}"
        )
        self.log.info(
            f"n_maj median={n_maj.median():.1f}  "
            f"consensus median={consensus.median():.3f}  "
            f"evidence median={evidence.median():.3f}"
        )

        return df.drop(
            columns=[c for c in ('_quality_pass', '_maj_cand')
                     if c in df.columns]
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _write(self, df: pd.DataFrame) -> None:
        """
        Write the classified measures CSV, dropping RF-internal working
        columns not needed by
        :class:`~sonobat_utils.from_scratch_postprocessing.FromScratchPostProcessor`.

        Retained: all original measures columns plus ``species`` and
        ``confidence``.  Dropped: ``final_pred``, ``final_rf_agreement``,
        ``routing``, ``chop_species``.

        :param df: Fully classified and merged DataFrame.
        :return: None
        """
        drop_cols = [c for c in
                     ('final_pred', 'final_rf_agreement',
                      'routing', 'chop_species')
                     if c in df.columns]
        df = df.drop(columns=drop_cols)

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_csv, index=False)
        self.log.info(f"Wrote {len(df):,} rows → {self.out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for :class:`RFConfidenceJoiner`.

    :return: Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog='rf_confidence_join.py',
        description=(
            'Join hierarchical RF predictions onto the measures CSV and '
            'compute a SonoBat-analog confidence score.\n\n'
            'Produces measures_classified.csv ready for '
            'from_scratch_postprocessing.py.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Default SonoBat constants (confirmed from barn/lake2 headers):
              --accp-quality-thresh   0.60   (MinAccpQuality)
              --max-calls-considered  48     (Max#CallsConsidered)
              --maj-prob-thresh       0.90   (DP threshold analog)

            Example:
              python rf_confidence_join.py \\
                  --measures-csv    /qnap/src/marsh_stanford_processed/measures.csv \\
                  --predictions-csv /qnap/src/marsh_stanford_processed/predictions.csv \\
                  --out-csv         /qnap/src/marsh_stanford_processed/measures_classified.csv
        """),
    )
    parser.add_argument(
        '--measures-csv',
        required=True, metavar='CSV', type=Path,
        help='Raw measures CSV from chirp_measures_extraction.py.',
    )
    parser.add_argument(
        '--predictions-csv',
        required=True, metavar='CSV', type=Path,
        help='predictions.csv from hierarchical_rf_predict.py.',
    )
    parser.add_argument(
        '--out-csv',
        required=True, metavar='CSV', type=Path,
        help='Destination path for the classified measures CSV.',
    )
    parser.add_argument(
        '--accp-quality-thresh',
        type=float, default=_ACCP_QUALITY_THRESH_DEFAULT, metavar='FLOAT',
        help=(
            f'Minimum Quality for a chirp to count toward n_accp. '
            f'Default: {_ACCP_QUALITY_THRESH_DEFAULT} '
            f'(SonoBat MinAccpQuality confirmed from barn/lake2).'
        ),
    )
    parser.add_argument(
        '--max-calls-considered',
        type=int, default=_MAX_CALLS_CONSIDERED_DEFAULT, metavar='N',
        help=(
            f'Hard cap on n_accp per chop. '
            f'Default: {_MAX_CALLS_CONSIDERED_DEFAULT} '
            f'(SonoBat Max#CallsConsidered confirmed from barn/lake2).'
        ),
    )
    parser.add_argument(
        '--maj-prob-thresh',
        type=float, default=_MAJ_PROB_THRESH_DEFAULT, metavar='FLOAT',
        help=(
            f'Minimum final_rf_agreement for a quality-passing chirp to '
            f'count toward n_maj. '
            f'Default: {_MAJ_PROB_THRESH_DEFAULT} '
            f'(analog of SonoBat DP threshold).'
        ),
    )

    dur_group = parser.add_mutually_exclusive_group()
    dur_group.add_argument(
        '--chop-report',
        default=None, metavar='CSV', type=Path, dest='chop_report',
        help=(
            'chop_report.csv from wav_chopper.py (from-scratch pipeline). '
            'Provides source_duration_s per file_id by joining on file_id. '
            'Takes priority over --manifest and --chop-duration.'
        ),
    )
    dur_group.add_argument(
        '--manifest',
        default=None, metavar='CSV', type=Path,
        help=(
            'manifest.csv from the legacy SonoBat pipeline. '
            'source_duration_s is reconstructed as '
            'max(time_in_file_ms)/1000 + 0.005 per file_id. '
            'Used when --chop-report is not available.'
        ),
    )
    dur_group.add_argument(
        '--chop-duration',
        type=float, default=2.0, metavar='SECS', dest='chop_duration',
        help=(
            'Fallback constant source duration in seconds broadcast to '
            'all rows when neither --chop-report nor --manifest is '
            'supplied.  Default: 2.0 (SonoBat standard chop length).'
        ),
    )

    args = parser.parse_args()
    for attr, flag in [('measures_csv',    '--measures-csv'),
                       ('predictions_csv', '--predictions-csv')]:
        if not getattr(args, attr).exists():
            parser.error(f'{flag} path not found: {getattr(args, attr)}')
    for attr, flag in [('chop_report', '--chop-report'),
                       ('manifest',    '--manifest')]:
        val = getattr(args, attr)
        if val is not None and not val.exists():
            parser.error(f'{flag} path not found: {val}')
    return args


def main() -> None:
    """
    CLI entry point for :class:`RFConfidenceJoiner`.

    :return: None
    """
    args = _parse_args()
    RFConfidenceJoiner(
        measures_csv         = args.measures_csv,
        predictions_csv      = args.predictions_csv,
        out_csv              = args.out_csv,
        accp_quality_thresh  = args.accp_quality_thresh,
        max_calls_considered = args.max_calls_considered,
        maj_prob_thresh      = args.maj_prob_thresh,
        chop_report          = args.chop_report,
        manifest             = args.manifest,
        chop_duration        = args.chop_duration,
    ).run()


if __name__ == '__main__':
    main()