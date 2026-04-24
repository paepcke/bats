#!/usr/bin/env python
# **********************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-23 17:04:55
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-23 17:20:34
# **********************************************
"""
quality_threshold_calibration.py
==================================
Calibrate the ``--min-chirp-quality`` threshold for
``chirp_measures_extraction.py`` by comparing SonoBat's per-chirp
``Quality`` scores against the ``#Accp`` acceptance counts in the
matching ``CumulativeSonoBatch`` files.

Background
----------
SonoBat's Long File Parser runs its own per-call quality gate before
tallying ``#Accp`` and ``#Maj`` in the CumulativeSonoBatch output.
The controlling parameter is ``AccpQuality`` (0.60 in the barn/lake2
data).  Within each 2-sec chop SB ranks all detected chirps by their
``Quality`` score and accepts the top ``#Accp`` of them.

``chirp_measures_extraction.py`` has no equivalent gate: it writes every
detected chirp regardless of quality, so our ``n_accp`` proxy in
``rf_confidence_join.py`` currently overestimates ``#Accp``.

This script computes the empirical Quality threshold that reproduces
SB's acceptance rate across the barn and lake2 corpora, giving us a
data-driven default.

Method
------
For each 2-sec chop that has at least one row in CumulativeParameters
and a matching row in CumulativeSonoBatch:

1. Retrieve ``#Accp`` from CumulativeSonoBatch (number of quality-passed
   calls SB accepted).
2. Sort the chirp rows for that chop by ``Quality`` descending.
3. Label the top ``#Accp`` rows as "accepted", the rest as "rejected".
4. Collect Quality values for both groups across all chops.

From the two distributions we compute:

* The Quality value at which the accepted/rejected distributions cross
  (equal density) — the natural decision boundary.
* The Quality percentile of the accepted group's minimum — the
  ``keep_top_fraction`` that reproduces SB's gate.
* A precision-recall-style curve: for each candidate threshold, report
  what fraction of SB-accepted chirps we retain and what fraction of
  SB-rejected chirps we incorrectly retain.

Outputs
-------
All written to ``--out-dir``:

``quality_distributions.png``
    Overlaid histograms of Quality for accepted vs rejected chirps,
    per site and combined.

``threshold_curve.png``
    Retention rate of accepted chirps and false-retention rate of
    rejected chirps as a function of Quality threshold.

``calibration_report.txt``
    Summary statistics and recommended default threshold.

``chirp_quality_detail.csv``
    One row per chirp: filename, site, Quality, accepted (bool).
    Useful for further analysis.

Usage
-----
::

    python quality_threshold_calibration.py \\
        --root-dirs /qnap/bats/barn /qnap/bats/lake2 \\
        --rec-sites barn lake2 \\
        --out-dir   /qnap/bats/calibration
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logging_service import LoggingService


class QualityThresholdCalibrator:
    """
    Calibrate the chirp quality threshold from paired CumulativeParameters
    and CumulativeSonoBatch files.

    :param root_dirs: Root directories to search, one per recording site.
    :param rec_sites: Site label for each root directory.
    :param out_dir:   Directory for all output files.
    """

    _PARAMS_PAT  = re.compile(r'.*_CumulativeParameters_v[0-9.]+\.txt$')
    _SPECIES_PAT = re.compile(r'.*_CumulativeSonoBatch_v[0-9.]+\.txt$')

    def __init__(
        self,
        root_dirs: list[Path],
        rec_sites: list[str],
        out_dir:   Path,
    ) -> None:
        self.log       = LoggingService()
        self.root_dirs = root_dirs
        self.rec_sites = rec_sites
        self.out_dir   = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full calibration workflow and write all outputs.

        :return: None
        """
        detail_df = self._build_detail()
        if detail_df.empty:
            self.log.err("No matched chirp data found. Check --root-dirs.")
            sys.exit(1)

        self._save_detail(detail_df)
        threshold, stats = self._analyze(detail_df)
        self._plot_distributions(detail_df)
        self._plot_threshold_curve(detail_df)
        self._write_report(threshold, stats, detail_df)

        self.log.info(f"Recommended default threshold: {threshold:.3f}")
        self.log.info(f"All outputs written to {self.out_dir}")

    # ------------------------------------------------------------------
    # Data assembly
    # ------------------------------------------------------------------

    def _build_detail(self) -> pd.DataFrame:
        """
        Load CumulativeParameters and CumulativeSonoBatch files for all
        sites, join them on filename stem, and label each chirp row as
        accepted or rejected based on SB's #Accp count.

        SB accepts the #Accp highest-Quality chirps within each chop.
        When #Accp equals the number of detected chirps, all are accepted.
        When #Accp is zero (no detection row in SonoBatch), all are rejected.

        :return: DataFrame with columns:
                 ``filename``, ``site``, ``Quality``, ``accepted``,
                 ``n_detected``, ``n_accp``.
        """
        records = []

        for root_dir, site in zip(self.root_dirs, self.rec_sites):
            self.log.info(f"Loading {site} from {root_dir} ...")

            params_files  = sorted(
                p for p in root_dir.rglob('*') if self._PARAMS_PAT.match(str(p))
            )
            species_files = sorted(
                p for p in root_dir.rglob('*') if self._SPECIES_PAT.match(str(p))
            )

            if not params_files:
                self.log.warn(f"No CumulativeParameters files under {root_dir}")
                continue
            if not species_files:
                self.log.warn(f"No CumulativeSonoBatch files under {root_dir}")
                continue

            # Load and normalize paths to bare stems in both files.
            params_dfs  = []
            for pf in params_files:
                df = pd.read_csv(pf, sep='\t',
                                 usecols=['Path', 'TimeInFile', 'Quality'],
                                 low_memory=False)
                df['stem'] = df['Path'].astype(str)\
                    .str.replace('\\', '/', regex=False)\
                    .apply(lambda p: Path(p).stem)
                params_dfs.append(df[['stem', 'TimeInFile', 'Quality']])

            species_dfs = []
            for sf in species_files:
                df = pd.read_csv(sf, sep='\t',
                                 usecols=['Path', '#Accp'],
                                 low_memory=False)
                df['stem'] = df['Path'].astype(str)\
                    .str.replace('\\', '/', regex=False)\
                    .apply(lambda p: Path(p).stem)
                # Fill missing #Accp (no-detection rows) with 0.
                df['#Accp'] = pd.to_numeric(df['#Accp'], errors='coerce').fillna(0).astype(int)
                species_dfs.append(df[['stem', '#Accp']])

            params_df  = pd.concat(params_dfs,  ignore_index=True)
            species_df = pd.concat(species_dfs, ignore_index=True)

            # ── Deduplication ─────────────────────────────────────────
            # The same Cumulative files can appear in multiple root_dirs
            # due to human error (e.g. two barn directories that overlap).
            # A chirp is uniquely identified by (stem, TimeInFile): same
            # recording, same onset time within the chop.  For the species
            # side the key is just stem (one row per chop per Cumulative
            # file); keep the row with the highest #Accp so that genuine
            # detections are not discarded in favour of a no-detection row
            # from an earlier run.

            n_params_before  = len(params_df)
            n_species_before = len(species_df)

            params_df = (
                params_df
                .sort_values('Quality', ascending=False)   # keep higher Quality on tie
                .drop_duplicates(subset=['stem', 'TimeInFile'], keep='first')
                .reset_index(drop=True)
            )
            species_df = (
                species_df
                .sort_values('#Accp', ascending=False)     # keep higher #Accp on tie
                .drop_duplicates(subset='stem', keep='first')
                .reset_index(drop=True)
            )

            n_params_dups  = n_params_before  - len(params_df)
            n_species_dups = n_species_before - len(species_df)

            if n_params_dups or n_species_dups:
                self.log.warn(
                    f"  {site}: removed duplicates — "
                    f"{n_params_dups:,} chirp rows (Parameters), "
                    f"{n_species_dups:,} chop rows (SonoBatch). "
                    f"This indicates overlapping Cumulative files across "
                    f"root_dirs for this site."
                )
            else:
                self.log.info(f"  {site}: no duplicate rows detected.")

            self.log.info(
                f"  {site}: {params_df['stem'].nunique():,} chops in Parameters, "
                f"{len(species_df):,} rows in SonoBatch"
            )

            # Join #Accp onto chirp rows.
            merged = params_df.merge(species_df, on='stem', how='left')
            merged['#Accp'] = merged['#Accp'].fillna(0).astype(int)
            merged['Quality'] = pd.to_numeric(merged['Quality'],
                                              errors='coerce')
            merged = merged.dropna(subset=['Quality'])

            # Label accepted chirps: within each chop, the top #Accp
            # by Quality are accepted.  Ties are broken arbitrarily
            # (rank method='first') — same as SB's behaviour.
            def _label(grp: pd.DataFrame) -> pd.Series:
                n_accp = int(grp['#Accp'].iloc[0])
                if n_accp == 0:
                    return pd.Series(False, index=grp.index)
                # rank descending: rank 1 = highest Quality
                ranks = grp['Quality'].rank(method='first', ascending=False)
                return ranks <= n_accp

            merged['accepted'] = merged.groupby('stem', group_keys=False)\
                .apply(_label)

            # Derive n_detected per chop for reporting.
            n_det = merged.groupby('stem')['Quality'].transform('count')
            merged['n_detected'] = n_det
            merged['n_accp']     = merged['#Accp']
            merged['site']       = site
            merged.rename(columns={'stem': 'filename'}, inplace=True)

            records.append(
                merged[['filename', 'site', 'Quality',
                         'accepted', 'n_detected', 'n_accp']]
            )
            self.log.info(
                f"  {site}: {merged['accepted'].sum():,} accepted chirps, "
                f"{(~merged['accepted']).sum():,} rejected chirps"
            )

        if not records:
            return pd.DataFrame()
        return pd.concat(records, ignore_index=True)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(
        self,
        df: pd.DataFrame,
    ) -> tuple[float, dict]:
        """
        Compute the recommended threshold and summary statistics.

        The threshold is chosen as the Quality value that maximises the
        F1-like score:

            F1 = 2 × precision × recall / (precision + recall)

            recall    = fraction of SB-accepted chirps retained above threshold
            precision = fraction of chirps above threshold that are SB-accepted

        This balances retaining true accepted chirps against admitting
        rejected ones.  The crossover Quality (equal-density point of the
        two distributions) is also reported for reference.

        :param df: Detail DataFrame from :meth:`_build_detail`.
        :return:   Tuple of (recommended_threshold, stats_dict).
        """
        q_accepted = df.loc[df['accepted'],  'Quality'].values
        q_rejected = df.loc[~df['accepted'], 'Quality'].values

        # Candidate thresholds: every distinct Quality value in the data.
        candidates = np.sort(df['Quality'].unique())

        best_thresh = candidates[0]
        best_f1     = 0.0
        f1_curve    = []

        for t in candidates:
            tp = (q_accepted >= t).sum()    # accepted chirps above threshold
            fp = (q_rejected >= t).sum()    # rejected chirps above threshold
            fn = (q_accepted <  t).sum()    # accepted chirps below threshold

            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            f1_curve.append((t, recall, precision, f1))

            if f1 > best_f1:
                best_f1     = f1
                best_thresh = t

        # Also compute the crossover (equal-density) point using KDE.
        from scipy.stats import gaussian_kde
        q_range = np.linspace(df['Quality'].min(), df['Quality'].max(), 500)
        try:
            kde_acc = gaussian_kde(q_accepted)(q_range)
            kde_rej = gaussian_kde(q_rejected)(q_range)
            diff    = kde_acc - kde_rej
            # Crossover: where diff changes sign from negative to positive
            sign_changes = np.where(np.diff(np.sign(diff)) > 0)[0]
            crossover = float(q_range[sign_changes[0]]) if len(sign_changes) else np.nan
        except Exception:
            crossover = np.nan

        acceptance_rate = df['accepted'].mean()
        median_accp     = df.groupby('filename')['n_accp'].first().median()
        median_detected = df.groupby('filename')['n_detected'].first().median()

        stats = {
            'n_chirps_total'   : len(df),
            'n_accepted'       : int(df['accepted'].sum()),
            'n_rejected'       : int((~df['accepted']).sum()),
            'acceptance_rate'  : acceptance_rate,
            'median_n_accp'    : median_accp,
            'median_n_detected': median_detected,
            'q_accepted_mean'  : float(q_accepted.mean()),
            'q_accepted_p10'   : float(np.percentile(q_accepted, 10)),
            'q_accepted_min'   : float(q_accepted.min()),
            'q_rejected_mean'  : float(q_rejected.mean()),
            'q_rejected_p90'   : float(np.percentile(q_rejected, 90)),
            'crossover_quality': crossover,
            'best_f1_threshold': best_thresh,
            'best_f1'          : best_f1,
            'f1_curve'         : f1_curve,
        }

        return best_thresh, stats

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_distributions(self, df: pd.DataFrame) -> None:
        """
        Overlaid Quality histograms: accepted vs rejected, per site and
        combined.

        :param df: Detail DataFrame.
        :return:   None
        """
        sites    = sorted(df['site'].unique())
        n_panels = len(sites) + 1          # one per site + combined
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(5 * n_panels, 4),
                                 sharey=False)
        if n_panels == 1:
            axes = [axes]

        panels = [(site, df[df['site'] == site]) for site in sites]
        panels.append(('combined', df))

        for ax, (label, sub) in zip(axes, panels):
            q_acc = sub.loc[sub['accepted'],  'Quality']
            q_rej = sub.loc[~sub['accepted'], 'Quality']
            bins = np.linspace(sub['Quality'].min(), sub['Quality'].max(), 50)
            ax.hist(q_acc, bins=bins, alpha=0.6, label='accepted',
                    density=True, color='steelblue')
            ax.hist(q_rej, bins=bins, alpha=0.6, label='rejected',
                    density=True, color='salmon')
            ax.set_title(label)
            ax.set_xlabel('Quality')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

        fig.suptitle('Chirp Quality: SB-accepted vs SB-rejected', y=1.02)
        fig.tight_layout()
        fig.savefig(self.out_dir / 'quality_distributions.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.log.info("Saved quality_distributions.png")

    def _plot_threshold_curve(self, df: pd.DataFrame) -> None:
        """
        Plot recall, precision, and F1 as a function of Quality threshold.

        :param df: Detail DataFrame.
        :return:   None
        """
        q_accepted = df.loc[df['accepted'],  'Quality'].values
        q_rejected = df.loc[~df['accepted'], 'Quality'].values
        candidates = np.sort(df['Quality'].unique())

        recalls, precisions, f1s = [], [], []
        for t in candidates:
            tp = (q_accepted >= t).sum()
            fp = (q_rejected >= t).sum()
            fn = (q_accepted <  t).sum()
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        best_idx = int(np.argmax(f1s))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(candidates, recalls,    label='recall (accepted retained)', color='steelblue')
        ax.plot(candidates, precisions, label='precision',                  color='darkorange')
        ax.plot(candidates, f1s,        label='F1',                         color='green', lw=2)
        ax.axvline(candidates[best_idx], color='green', linestyle='--', alpha=0.7,
                   label=f'best F1 threshold = {candidates[best_idx]:.3f}')
        ax.set_xlabel('Quality threshold')
        ax.set_ylabel('Rate')
        ax.set_title('Acceptance precision / recall vs Quality threshold')
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(self.out_dir / 'threshold_curve.png', dpi=150)
        plt.close(fig)
        self.log.info("Saved threshold_curve.png")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _write_report(
        self,
        threshold: float,
        stats:     dict,
        df:        pd.DataFrame,
    ) -> None:
        """
        Write a human-readable calibration report.

        :param threshold: Recommended threshold.
        :param stats:     Statistics dict from :meth:`_analyze`.
        :param df:        Detail DataFrame (for per-site breakdown).
        :return:          None
        """
        lines = [
            "=" * 65,
            "chirp_measures_extraction.py  --min-chirp-quality calibration",
            "=" * 65,
            "",
            f"Sites analysed : {', '.join(self.rec_sites)}",
            f"Total chirps   : {stats['n_chirps_total']:,}",
            f"  SB-accepted  : {stats['n_accepted']:,}  "
            f"({100*stats['acceptance_rate']:.1f}%)",
            f"  SB-rejected  : {stats['n_rejected']:,}  "
            f"({100*(1-stats['acceptance_rate']):.1f}%)",
            "",
            "Median chop stats (across all file_ids):",
            f"  chirps detected per chop  : {stats['median_n_detected']:.1f}",
            f"  chirps accepted (#Accp)   : {stats['median_n_accp']:.1f}",
            "",
            "Quality distribution — SB-accepted chirps:",
            f"  mean  : {stats['q_accepted_mean']:.4f}",
            f"  p10   : {stats['q_accepted_p10']:.4f}   "
            f"(10% of accepted chirps fall below this)",
            f"  min   : {stats['q_accepted_min']:.4f}",
            "",
            "Quality distribution — SB-rejected chirps:",
            f"  mean  : {stats['q_rejected_mean']:.4f}",
            f"  p90   : {stats['q_rejected_p90']:.4f}   "
            f"(90% of rejected chirps fall below this)",
            "",
        ]

        if not np.isnan(stats['crossover_quality']):
            lines += [
                f"Distribution crossover (equal density): "
                f"{stats['crossover_quality']:.4f}",
                "  (Reference only — not used for the recommendation)",
                "",
            ]

        lines += [
            f"Best-F1 threshold  : {stats['best_f1_threshold']:.4f}  "
            f"(F1 = {stats['best_f1']:.4f})",
            "",
            "=" * 65,
            f"RECOMMENDED DEFAULT  --min-chirp-quality = "
            f"{stats['best_f1_threshold']:.3f}",
            "=" * 65,
            "",
            "At this threshold:",
        ]

        # Compute per-site stats at the recommended threshold.
        t = threshold
        for site in sorted(df['site'].unique()):
            sub      = df[df['site'] == site]
            q_acc    = sub.loc[sub['accepted'],  'Quality'].values
            q_rej    = sub.loc[~sub['accepted'], 'Quality'].values
            tp       = (q_acc >= t).sum()
            fp       = (q_rej >= t).sum()
            fn       = (q_acc <  t).sum()
            recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fp_rate  = fp / len(q_rej) if len(q_rej) > 0 else 0.0
            lines.append(
                f"  {site:8s}: retain {recall*100:.1f}% of SB-accepted chirps, "
                f"admit {fp_rate*100:.1f}% of SB-rejected chirps"
            )

        lines += [
            "",
            "Notes",
            "-----",
            "* 'Accepted' means SB included the chirp in #Accp for that chop.",
            "* The threshold is applied to the Quality column produced by",
            "  chirp_measures_extraction.py, which mirrors the Quality column",
            "  in CumulativeParameters.",
            "* Adjust the default conservatively upward if the marsh site",
            "  has noisier recordings than barn/lake2.",
        ]

        report_text = "\n".join(lines)
        (self.out_dir / 'calibration_report.txt').write_text(report_text)
        self.log.info("Saved calibration_report.txt")
        print("\n" + report_text)

    # ------------------------------------------------------------------
    # Save detail CSV
    # ------------------------------------------------------------------

    def _save_detail(self, df: pd.DataFrame) -> None:
        """
        Write the per-chirp detail CSV.

        :param df: Detail DataFrame.
        :return:   None
        """
        df.to_csv(self.out_dir / 'chirp_quality_detail.csv', index=False)
        self.log.info(
            f"Saved chirp_quality_detail.csv  ({len(df):,} rows)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.

    :return: None
    """
    parser = argparse.ArgumentParser(
        prog='quality_threshold_calibration.py',
        description=textwrap.dedent("""\
            Calibrate the --min-chirp-quality threshold for
            chirp_measures_extraction.py from paired CumulativeParameters
            and CumulativeSonoBatch files.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--root-dirs', required=True, nargs='+', metavar='DIR', type=Path,
        help='Root directories to search, one per recording site.',
    )
    parser.add_argument(
        '--rec-sites', required=True, nargs='+', metavar='SITE',
        help='Site label for each root directory (same order).',
    )
    parser.add_argument(
        '--out-dir', required=True, metavar='DIR', type=Path,
        help='Directory for all output files.',
    )

    args = parser.parse_args()

    if len(args.root_dirs) != len(args.rec_sites):
        parser.error(
            f"--root-dirs ({len(args.root_dirs)}) and "
            f"--rec-sites ({len(args.rec_sites)}) must have the same count."
        )
    for d in args.root_dirs:
        if not d.exists():
            parser.error(f"--root-dirs path not found: {d}")

    QualityThresholdCalibrator(
        root_dirs = args.root_dirs,
        rec_sites = args.rec_sites,
        out_dir   = args.out_dir,
    ).run()


if __name__ == '__main__':
    main()