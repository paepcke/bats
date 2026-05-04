#!/usr/bin/env python
# *******************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-03 20:13:23
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-03 20:26:34
# *******************************************

"""
rf_sb_validation_optional.py

Compare SonoBat species predictions against hierarchical RF predictions
for the marsh recording site.

SonoBat predictions come from a postprocessed parquet file covering a
random sample of marsh recordings.  RF predictions come from
hierarchical_rf_predict.py run over all marsh recordings.  The two
datasets are joined on (filename_stem, chirp_idx); only chirps present
in both are evaluated.

Filters applied before comparison
----------------------------------
* Composite SonoBat species (slash-separated, e.g. ``Lano/Tabr``) are
  excluded from the confusion matrix but counted and logged.
* Rows where SonoBat's ``species`` is NaN are excluded.

Confidence stratification
--------------------------
Results are reported for three SonoBat confidence tiers:

  * High    : confidence >= 0.90
  * Medium  : 0.65 <= confidence < 0.90
  * Low     : confidence < 0.65

Daytime breakdown
------------------
If ``was_daytime`` is populated in the parquet (non-null), a separate
per-tier summary is produced for daytime vs. night-time chirps.

Outputs (all written to --out-dir)
------------------------------------
``comparison_summary.txt``
    Human-readable summary: counts, per-tier agreement rates, daytime
    breakdown.

``confusion_matrix_<tier>.csv`` / ``confusion_matrix_<tier>.png``
    Normalised confusion matrix (SonoBat label as rows / true axis,
    RF final_pred as columns / predicted axis) for each confidence tier
    and for all tiers combined.

``per_species_agreement.csv``
    Per-species agreement rate, count, and confidence-tier breakdown.

``joined_chirps.csv``
    Full joined table with both SonoBat and RF columns, for ad-hoc
    downstream analysis.

Typical usage
-------------
::

    python sb_rf_comparison.py \\
        --parquet   /data/all_data_marsh/bats_2026-05-03T18_22_16.766575.parquet \\
        --rf-preds  /data2/marsh_stanford_processed/rf_predictions/predictions.csv \\
        --out-dir   /data2/marsh_stanford_processed/sb_random_validation
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from logging_service import LoggingService
from sonobat_utils.sb_measures_postprocessing import BatsData

log = LoggingService()

# ---------------------------------------------------------------------------
# Confidence tier boundaries
# ---------------------------------------------------------------------------

_TIER_HIGH_MIN   = 0.90
_TIER_MEDIUM_MIN = 0.65   # same threshold used at postprocessing time

_TIERS: list[tuple[str, float, float]] = [
    ('high',   _TIER_HIGH_MIN,   1.01),
    ('medium', _TIER_MEDIUM_MIN, _TIER_HIGH_MIN),
    ('low',    0.0,              _TIER_MEDIUM_MIN),
    ('all',    0.0,              1.01),
]


# ---------------------------------------------------------------------------
# Class Comparator
# ---------------------------------------------------------------------------

class Comparator:
    """
    Load, join, and compare SonoBat and RF predictions for marsh recordings.

    :param parquet_path: Path to the postprocessed marsh parquet produced
                         by ``sb_measures_postprocessing.py``.
    :param rf_preds_path: Path to ``predictions.csv`` from
                          ``hierarchical_rf_predict.py``.
    :param out_dir: Directory where all output artifacts are written.
    """

    def __init__(
        self,
        parquet_path:  str | Path,
        rf_preds_path: str | Path,
        out_dir:       str | Path,
    ) -> None:
        self.parquet_path  = Path(parquet_path)
        self.rf_preds_path = Path(rf_preds_path)
        self.out_dir       = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full comparison pipeline and write all output artifacts.

        :return: None
        """
        # ---- Load -------------------------------------------------------
        log.info(f"Loading parquet: {self.parquet_path}")
        bats      = BatsData.read_parquet(self.parquet_path)
        parquet_df = bats.df.copy()
        file_map   = bats.file_map          # int -> full path string
        log.info(f"  {len(parquet_df):,} chirp rows in parquet")

        log.info(f"Loading RF predictions: {self.rf_preds_path}")
        rf_df = pd.read_csv(self.rf_preds_path, dtype={'file_id': str,
                                                        'chirp_idx': 'Int64'})
        log.info(f"  {len(rf_df):,} chirp rows in RF predictions")

        # ---- Build stem column for parquet ------------------------------
        # file_map values are full path strings; stem = basename without ext.
        int_to_stem: dict[int, str] = {
            fid: Path(path).stem
            for fid, path in file_map.items()
        }
        parquet_df['file_id_stem'] = parquet_df['file_id'].map(int_to_stem)
        n_unmapped = parquet_df['file_id_stem'].isna().sum()
        if n_unmapped:
            log.warn(f"  {n_unmapped:,} parquet rows had no file_map entry "
                     f"— will be lost in join")

        # RF predictions already use stem as file_id
        rf_df = rf_df.rename(columns={'file_id': 'file_id_stem'})

        # ---- Inner join on (file_id_stem, chirp_idx) --------------------
        joined = parquet_df.merge(
            rf_df,
            on=['file_id_stem', 'chirp_idx'],
            how='inner',
            suffixes=('_sb', '_rf'),
        )
        log.info(f"  {len(joined):,} chirps matched after inner join")

        n_parquet_only = len(parquet_df) - len(joined)
        n_rf_only      = len(rf_df) - len(joined)
        log.info(f"  {n_parquet_only:,} parquet chirps with no RF match "
                 f"(not in RF input)")
        log.info(f"  {n_rf_only:,} RF chirps with no SonoBat label "
                 f"(outside SonoBat sample)")

        # ---- Filter: drop NaN SonoBat species ---------------------------
        n_before = len(joined)
        joined = joined[joined['species'].notna()].copy()
        log.info(f"  {n_before - len(joined):,} rows dropped: NaN SonoBat species")

        # ---- Separate composite species ---------------------------------
        composite_mask = joined['species'].str.contains('/', na=False)
        n_composite    = composite_mask.sum()
        if n_composite:
            log.info(f"  {n_composite:,} rows with composite SonoBat species "
                     f"set aside (not included in confusion matrices)")
        joined_eval     = joined[~composite_mask].copy()
        joined_composite = joined[composite_mask].copy()

        log.info(f"  {len(joined_eval):,} rows available for evaluation")

        # ---- Daytime availability ---------------------------------------
        has_daytime = joined_eval['was_daytime'].notna().any()
        if has_daytime:
            n_day   = (joined_eval['was_daytime'] == True).sum()
            n_night = (joined_eval['was_daytime'] == False).sum()
            n_unk   = joined_eval['was_daytime'].isna().sum()
            log.info(f"  was_daytime populated: "
                     f"{n_day:,} day / {n_night:,} night / {n_unk:,} unknown")
        else:
            log.info("  was_daytime: all None — daytime breakdown skipped")

        # ---- Save joined table ------------------------------------------
        joined.to_csv(self.out_dir / 'joined_chirps.csv', index=False)
        log.info(f"Saved joined_chirps.csv ({len(joined):,} rows)")

        # ---- Per-tier confusion matrices and agreement ------------------
        summary_lines: list[str] = []
        summary_lines += self._header_block(
            len(parquet_df), len(rf_df), len(joined),
            n_parquet_only, n_rf_only,
            n_composite, len(joined_eval),
        )

        all_species = sorted(
            set(joined_eval['species'].unique()) |
            set(joined_eval['final_pred'].unique())
        )

        per_species_rows: list[dict] = []

        for tier_name, lo, hi in _TIERS:
            mask = (joined_eval['confidence'] >= lo) & \
                   (joined_eval['confidence'] <  hi)
            tier_df = joined_eval[mask]
            if tier_df.empty:
                summary_lines.append(
                    f"\n[{tier_name.upper()} confidence tier: no rows]\n")
                continue

            agree      = (tier_df['species'] == tier_df['final_pred']).mean()
            n_tier     = len(tier_df)
            summary_lines.append(
                f"\n--- Confidence tier: {tier_name.upper()} "
                f"(n={n_tier:,}, agreement={agree:.3f}) ---"
            )

            # Confusion matrix
            y_true = tier_df['species'].values
            y_pred = tier_df['final_pred'].values
            tier_species = sorted(set(y_true) | set(y_pred))

            self._save_confusion_matrix(
                y_true, y_pred, tier_species, tier_name
            )

            # Per-species agreement for this tier
            for spp in tier_species:
                spp_mask   = tier_df['species'] == spp
                spp_df     = tier_df[spp_mask]
                spp_agree  = (spp_df['final_pred'] == spp).mean()
                per_species_rows.append({
                    'tier':      tier_name,
                    'species':   spp,
                    'n_chirps':  len(spp_df),
                    'agreement': round(spp_agree, 4),
                })

            # Per-species summary lines for this tier
            for spp in tier_species:
                spp_rows = [r for r in per_species_rows
                            if r['tier'] == tier_name and r['species'] == spp]
                if spp_rows:
                    r = spp_rows[0]
                    summary_lines.append(
                        f"  {spp:6s}  n={r['n_chirps']:>7,}  "
                        f"agreement={r['agreement']:.3f}"
                    )

            # Daytime breakdown within this tier
            if has_daytime and tier_name != 'all':
                for dt_val, dt_label in [(True, 'daytime'), (False, 'nighttime')]:
                    dt_mask = tier_df['was_daytime'] == dt_val
                    dt_df   = tier_df[dt_mask]
                    if dt_df.empty:
                        continue
                    dt_agree = (dt_df['species'] == dt_df['final_pred']).mean()
                    summary_lines.append(
                        f"    {dt_label:10s}  n={len(dt_df):>7,}  "
                        f"agreement={dt_agree:.3f}"
                    )

        # ---- Per-species agreement CSV ----------------------------------
        ps_df = pd.DataFrame(per_species_rows)
        ps_df.to_csv(self.out_dir / 'per_species_agreement.csv', index=False)
        log.info("Saved per_species_agreement.csv")

        # ---- Composite species summary ----------------------------------
        if n_composite:
            summary_lines.append(
                f"\n--- Composite SonoBat species (excluded from matrices) ---"
            )
            for spp, cnt in (joined_composite['species']
                             .value_counts().items()):
                rf_agree = (joined_composite[
                    joined_composite['species'] == spp
                ]['final_pred'].value_counts(normalize=True)
                    .rename_axis('rf_pred')
                    .reset_index())
                top = rf_agree.iloc[0]
                summary_lines.append(
                    f"  {spp:15s}  n={cnt:>6,}  "
                    f"RF most-common: {top['rf_pred']} "
                    f"({top['proportion']:.1%})"
                )

        # ---- Write summary ----------------------------------------------
        summary_text = '\n'.join(summary_lines) + '\n'
        summary_path = self.out_dir / 'comparison_summary.txt'
        summary_path.write_text(summary_text)
        log.info(f"Saved comparison_summary.txt")
        log.info(f"\n{summary_text}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _header_block(
        self,
        n_parquet:      int,
        n_rf:           int,
        n_joined:       int,
        n_parquet_only: int,
        n_rf_only:      int,
        n_composite:    int,
        n_eval:         int,
    ) -> list[str]:
        """
        Build the header section of the summary text.

        :param n_parquet:      Rows in parquet before join.
        :param n_rf:           Rows in RF predictions before join.
        :param n_joined:       Rows after inner join.
        :param n_parquet_only: Parquet rows with no RF match.
        :param n_rf_only:      RF rows with no SonoBat label.
        :param n_composite:    Composite SonoBat species rows.
        :param n_eval:         Rows used for evaluation.
        :return: List of text lines.
        """
        return [
            "SonoBat vs RF comparison — marsh site",
            "=" * 50,
            f"Parquet chirps (SonoBat sample)  : {n_parquet:>10,}",
            f"RF prediction chirps (all marsh) : {n_rf:>10,}",
            f"Matched after inner join         : {n_joined:>10,}",
            f"  Parquet-only (no RF match)     : {n_parquet_only:>10,}",
            f"  RF-only (no SonoBat label)     : {n_rf_only:>10,}",
            f"Composite SonoBat species        : {n_composite:>10,}",
            f"Rows used for evaluation         : {n_eval:>10,}",
        ]

    def _save_confusion_matrix(
        self,
        y_true:      np.ndarray,
        y_pred:      np.ndarray,
        class_names: list[str],
        tier_name:   str,
    ) -> None:
        """
        Save a normalised confusion matrix as CSV and PNG for one tier.

        Rows = SonoBat (ground truth), columns = RF prediction.

        :param y_true:      SonoBat species labels.
        :param y_pred:      RF final_pred labels.
        :param class_names: Sorted union of all species in this tier.
        :param tier_name:   Label used in filenames and title.
        :return: None
        """
        cm = confusion_matrix(y_true, y_pred,
                              labels=class_names, normalize='true')
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            self.out_dir / f'confusion_matrix_{tier_name}.csv'
        )

        n   = len(class_names)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n)))
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=class_names
        ).plot(ax=ax, colorbar=True, xticks_rotation=45, values_format='.2f')
        ax.set_ylabel('SonoBat (reference)')
        ax.set_xlabel('RF final_pred')
        ax.set_title(
            f'SonoBat vs RF — marsh  '
            f'[confidence: {tier_name}]  (normalised)'
        )
        fig.tight_layout()
        fig.savefig(self.out_dir / f'confusion_matrix_{tier_name}.png', dpi=150)
        plt.close(fig)
        log.info(f"Saved confusion_matrix_{tier_name}.csv/png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Populated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog='sb_rf_comparison',
        description=textwrap.dedent("""\
            Compare SonoBat species predictions against hierarchical RF
            predictions for marsh recordings.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--parquet',
        required=False,
        default='/data/all_data_marsh/bats_2026-05-03T18_22_16.766575.parquet',
        metavar='PATH',
        help='Postprocessed marsh parquet from sb_measures_postprocessing.py.',
    )
    parser.add_argument(
        '--rf-preds',
        required=False,
        default='/data2/marsh_stanford_processed/rf_predictions/predictions.csv',
        metavar='PATH',
        help='predictions.csv from hierarchical_rf_predict.py.',
    )
    parser.add_argument(
        '--out-dir',
        required=False,
        default='/data2/marsh_stanford_processed/sb_random_validation',
        metavar='DIR',
        help='Directory for output artifacts (created if absent).',
    )
    args = parser.parse_args()

    for attr, label in [('parquet', '--parquet'), ('rf_preds', '--rf-preds')]:
        p = Path(getattr(args, attr))
        if not p.exists():
            parser.error(f'{label} not found: {p}')

    return args


def main() -> None:
    """
    CLI entry point for :class:`Comparator`.

    :return: None
    """
    args = _parse_args()
    Comparator(
        parquet_path  = args.parquet,
        rf_preds_path = args.rf_preds,
        out_dir       = args.out_dir,
    ).run()


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()