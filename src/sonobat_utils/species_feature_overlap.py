#!/usr/bin/env python
# *********************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-17 18:40:59
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-17 18:47:06
# *********************************************
#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-04-17
# @File:   src/species_classification/species_feature_overlap.py
#
# **********************************************************

"""
Analyse how well individual SonoBat acoustic measures separate two bat
species, using three complementary metrics per feature:

  overlap_coeff  Area under min(KDE_sp1, KDE_sp2) — the fraction of the
                 combined distribution shared by both species.  Range [0,1];
                 lower means better separation.

  cohens_d       Normalised mean difference: (μ₁ − μ₂) / pooled_std.
                 |d| ≥ 0.8 is conventionally "large".  Sign indicates
                 which species has the higher mean.

  f_stat         One-way ANOVA F-statistic across the two groups.
  p_value        Corresponding p-value.  Small p means the group means
                 are unlikely to be equal given within-group variance.

The four metrics are complementary:

  Low OVL + high F   → well separated; RF should exploit this feature.
  High OVL + high F  → means differ significantly but distributions are
                       wide and overlapping; noisy signal.
  High OVL + low F   → distributions genuinely overlap; no usable signal.
  Low OVL + low F    → small-sample artefact; treat with caution.

Output
------
``overlap_<SP1>_<SP2>.csv``
    Full results table, one row per feature, sorted by overlap_coeff
    ascending (most discriminating features first).

``overlap_<SP1>_<SP2>.png``
    Horizontal bar chart of overlap_coeff for the top-N features
    (default 15), with Cohen's d values annotated.

Typical Usage
-------------
::

    python species_feature_overlap.py \\
        --input   /data/bats_2026-04-14T23_44_31.660585.parquet \\
        --species Lano Tabr \\
        --top-n   15 \\
        --out-dir /data/random_forest/overlap_analysis
"""

import sys
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, f_oneway

from logging_service import LoggingService
from sonobat_utils.utils import Utils

log = LoggingService()

# Columns that are never acoustic features — same set as the RF scripts.
_NON_FEATURE_COLS: frozenset[str] = frozenset([
    'file_id', 'chirp_idx', 'rec_site',
    'species', 'confidence',
    'TimeInFile',
    'Filename', 'species_prob', 'species_2nd',
    'cntxt_sz', 'split', 'index',
    'Path', 'ParentDir', 'NextDirUp', 'Version', 'Filter',
    'Preemphasis', 'MaxSegLnght',
])


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for multiprocessing pickle
# ---------------------------------------------------------------------------

def _compute_one_feature(
    args: tuple,
) -> dict | None:
    """
    Compute all metrics for a single feature column.

    Defined at module level (not as a method) so that
    :mod:`multiprocessing` can pickle it for worker processes.

    :param args: Tuple of ``(col, x, y, sp1, sp2, n_grid)`` where
                 ``x`` and ``y`` are 1-D float arrays of feature values
                 for species 1 and species 2 respectively.
    :return:     Dict of metrics, or ``None`` if either array is too
                 small to compute KDE.
    """
    col, x, y, sp1, sp2, n_grid = args

    if len(x) < 2 or len(y) < 2:
        return None

    # ANOVA
    f_stat, p_value = f_oneway(x, y)

    # Cohen's d
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * x.var(ddof=1) +
                  (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    pooled_std = float(np.sqrt(pooled_var))
    cohens_d   = float((x.mean() - y.mean()) / pooled_std) if pooled_std else 0.0

    # Overlap coefficient
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if lo == hi:
        ovl = 1.0
    else:
        grid  = np.linspace(lo, hi, n_grid)
        kde_x = gaussian_kde(x)(grid)
        kde_y = gaussian_kde(y)(grid)
        kde_x /= kde_x.sum()
        kde_y /= kde_y.sum()
        ovl = float(np.minimum(kde_x, kde_y).sum())

    return {
        'feature'        : col,
        f'{sp1}_mean'    : round(float(x.mean()), 4),
        f'{sp2}_mean'    : round(float(y.mean()), 4),
        f'{sp1}_std'     : round(float(x.std()),  4),
        f'{sp2}_std'     : round(float(y.std()),  4),
        'cohens_d'       : round(cohens_d,         4),
        'f_stat'         : round(float(f_stat),    2),
        'p_value'        : round(float(p_value),   6),
        'overlap_coeff'  : round(ovl,              4),
    }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class SpeciesFeatureOverlap:
    """
    Compute per-feature separation metrics between two bat species.

    :param parquet_path: Chirp measures file (.parquet/.pq/.feather/.csv).
    :param species_pair: Tuple of two species codes, e.g. ``('Lano', 'Tabr')``.
    :param n_grid:       Number of grid points for KDE integration when
                         computing the overlap coefficient (default 1000).
    :param n_jobs:       Number of worker processes for parallel feature
                         computation.  ``-1`` uses all available cores
                         (default ``-1``).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        species_pair: tuple[str, str],
        n_grid:       int = 1000,
        n_jobs:       int = -1,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.sp1, self.sp2 = species_pair
        self.n_grid  = n_grid
        self.n_jobs  = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self._results: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute(self) -> pd.DataFrame:
        """
        Load data, compute all metrics for every numeric feature, and
        return a DataFrame sorted by ``overlap_coeff`` ascending.

        The result is also cached in ``self._results`` for use by
        :meth:`plot`.

        :return: DataFrame with columns ``feature``, ``sp1_mean``,
                 ``sp2_mean``, ``sp1_std``, ``sp2_std``, ``cohens_d``,
                 ``f_stat``, ``p_value``, ``overlap_coeff``.
        """
        df = self._load()
        feature_cols = self._feature_cols(df)

        sp1_df = df[df['species'] == self.sp1][feature_cols]
        sp2_df = df[df['species'] == self.sp2][feature_cols]

        log.info(
            f'{self.sp1}: {len(sp1_df):,} chirps   '
            f'{self.sp2}: {len(sp2_df):,} chirps   '
            f'features: {len(feature_cols)}   '
            f'workers: {self.n_jobs}'
        )

        # Build job args — each worker gets its feature slice as numpy arrays
        # so the DataFrame isn't pickled repeatedly.
        job_args = [
            (
                col,
                sp1_df[col].dropna().values,
                sp2_df[col].dropna().values,
                self.sp1,
                self.sp2,
                self.n_grid,
            )
            for col in feature_cols
        ]

        with Pool(processes=self.n_jobs) as pool:
            raw_rows = pool.map(_compute_one_feature, job_args)

        rows = [r for r in raw_rows if r is not None]
        skipped = len(job_args) - len(rows)
        if skipped:
            log.info(f'Skipped {skipped} feature(s) — insufficient data.')

        results = (
            pd.DataFrame(rows)
            .sort_values('overlap_coeff', ascending=True)
            .reset_index(drop=True)
        )
        self._results = results
        log.info(f'Computed metrics for {len(results)} features.')
        return results

    def plot(
        self,
        top_n:    int = 15,
        out_path: Path | None = None,
    ) -> Path:
        """
        Save a horizontal bar chart of ``overlap_coeff`` for the
        ``top_n`` most discriminating features, with Cohen's d
        annotated on each bar.

        :param top_n:    Number of features to show (default 15).
        :param out_path: Output PNG path.  If None, a default name is
                         constructed from the species pair.
        :return:         Path to the saved PNG.
        """
        if self._results is None:
            raise RuntimeError('Call compute() before plot().')

        top = self._results.head(top_n).iloc[::-1]  # reverse for bottom-up bars

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))

        bars = ax.barh(top['feature'], top['overlap_coeff'],
                       color='steelblue', alpha=0.85)

        # Annotate each bar with Cohen's d
        for bar, (_, row) in zip(bars, top.iterrows()):
            d_str = f"d={row['cohens_d']:+.2f}"
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                d_str, va='center', ha='left', fontsize=8, color='dimgray'
            )

        ax.set_xlabel('Overlap coefficient  (lower = better separation)')
        ax.set_title(
            f'Top {top_n} most discriminating features\n'
            f'{self.sp1} vs {self.sp2}  —  overlap coefficient'
        )
        ax.set_xlim(0, min(1.15, top['overlap_coeff'].max() + 0.15))
        ax.axvline(0.5, color='tomato', linestyle='--', linewidth=0.8,
                   label='OVL = 0.5')
        ax.legend(fontsize=8)
        fig.tight_layout()

        if out_path is None:
            out_path = Path(f'overlap_{self.sp1}_{self.sp2}.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f'Saved plot: {out_path}')
        return out_path

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load(self) -> pd.DataFrame:
        """
        Load the measures file and filter to the two species of interest.

        :return: DataFrame containing only rows for sp1 and sp2.
        :raises SystemExit: If the file cannot be read or either species
                            is absent.
        """
        log.info(f'Loading {self.parquet_path} ...')
        try:
            df = Utils.read_df_file(self.parquet_path)
        except Exception as exc:
            log.err(f'Cannot read input file: {exc}')
            sys.exit(1)
        log.info(f'Loaded {len(df):,} rows.')

        missing = [s for s in (self.sp1, self.sp2)
                   if s not in df['species'].values]
        if missing:
            log.err(
                f'Species not found in data: {", ".join(missing)}\n'
                f'Available: {sorted(df["species"].dropna().unique())}'
            )
            sys.exit(1)

        df = df[df['species'].isin([self.sp1, self.sp2])].copy()
        # Drop composite rows that might have slipped through
        df = df[~df['species'].str.contains('/', na=False)]
        log.info(
            f'After filtering to {self.sp1}/{self.sp2}: {len(df):,} rows.'
        )
        return df

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Return numeric feature column names, excluding metadata columns.

        :param df: Input DataFrame.
        :return:   List of feature column names.
        """
        return [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Validated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog='species_feature_overlap',
        description=(
            'Compute per-feature separation metrics (overlap coefficient,\n'
            "Cohen's d, ANOVA F-statistic) between two bat species.\n\n"
            'Output: overlap_<SP1>_<SP2>.csv (all features) and\n'
            '        overlap_<SP1>_<SP2>.png (top-N chart).'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='PATH',
        help='Chirp measures file (.parquet/.pq preferred;\n'
             '.feather and .csv also accepted).',
    )
    parser.add_argument(
        '--species',
        nargs=2,
        required=True,
        metavar='SPECIES',
        help='Two species codes to compare, e.g. --species Lano Tabr',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=15,
        metavar='N',
        help='Number of features to show in the bar chart (default: 15).\n'
             'CSV always contains all features.',
    )
    parser.add_argument(
        '--n-grid',
        type=int,
        default=1000,
        metavar='N',
        help='Grid resolution for KDE overlap integration (default: 1000).',
    )
    parser.add_argument(
        '-o', '--out-dir',
        default='.',
        metavar='DIR',
        help='Output directory (default: current directory).',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plot generation; write CSV only.',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        metavar='N',
        help='Worker processes for parallel computation\n'
             '(-1 = all cores, default: -1).',
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        parser.error(f'Input file not found: {args.input}')

    return args


def main() -> None:
    """
    CLI entry point for :class:`SpeciesFeatureOverlap`.
    """
    args    = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp1, sp2 = args.species
    analyser = SpeciesFeatureOverlap(
        parquet_path = args.input,
        species_pair = (sp1, sp2),
        n_grid       = args.n_grid,
        n_jobs       = args.n_jobs,
    )

    results = analyser.compute()

    csv_path = out_dir / f'overlap_{sp1}_{sp2}.csv'
    results.to_csv(csv_path, index=False)
    log.info(f'Saved {csv_path}')

    if not args.no_plot:
        png_path = out_dir / f'overlap_{sp1}_{sp2}.png'
        analyser.plot(top_n=args.top_n, out_path=png_path)

    # Print top-10 summary to stdout
    log.info(f'\nTop 10 most discriminating features ({sp1} vs {sp2}):')
    log.info(
        results.head(10)
        [['feature', 'overlap_coeff', 'cohens_d', 'f_stat', 'p_value']]
        .to_string(index=False)
    )


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()
    