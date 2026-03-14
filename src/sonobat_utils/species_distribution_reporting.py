#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-03-13 10:08:55
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/species_distribution_reporting.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-14 10:49:26
# **********************************************************


"""
Analyse and report species distributions in a chirp-level measures DataFrame.

Expected input: a ``.csv`` or ``.feather`` file produced by
``sono_batch_processing.py``, containing columns including
``species``, ``species_prob``, ``species_2nd``, and ``file_id``.

Actions
-------
``list-species``
    Print raw counts and percentages for high-confidence species labels
    (``species_prob >= 0.98`` by default).

``purity-analysis``
    For each ``file_id``, compute the fraction of chirps assigned to the
    majority species.  Outputs a purity-distribution histogram, a purity-vs-
    probability scatter, and a CSV of impure file_ids.

``univariate-overlap``
    For each confusable species pair (selected from a confusion matrix CSV),
    compute the per-feature Bhattacharyya coefficient — 1.0 means identical
    distributions, 0.0 means fully separated.  Results are ranked by overlap
    (most confused features first) and saved as CSV + bar chart.

``multivariate-overlap``
    Fit a multivariate Gaussian to each species' feature cloud and compute the
    Bhattacharyya distance between the two clouds.  A distance near 0 means
    the species are essentially the same cloud in feature space; larger values
    indicate increasing separability.

``umap``
    Project the two species' chirps into 2D (optionally 3D) with UMAP and
    save a scatter plot.  Visually shows whether the two clouds are separable.

Typical Usage
-------------
::

    # List species distribution
    python species_distribution_reporting.py data.feather \\
        --action list-species

    # Purity analysis
    python species_distribution_reporting.py data.feather \\
        --action purity-analysis --out-dir ./reports

    # All three overlap analyses for the top-3 confused pairs
    python species_distribution_reporting.py data.feather \\
        --action univariate-overlap \\
        --confusion-matrix-csv confusion_matrix.csv \\
        --top-n-pairs 3 \\
        --out-dir ./overlap_reports

    python species_distribution_reporting.py data.feather \\
        --action umap --umap-3d \\
        --confusion-matrix-csv confusion_matrix.csv \\
        --top-n-pairs 3 \\
        --out-dir ./overlap_reports

    Poorest classifications from the confusion matrix plus
    comparison between Myca/Myyu and Myth/Myev:

    python species_distribution_reporting.py data.feather \
       --action univariate-overlap multivariate-overlap \
       --confusion-matrix-csv confusion_matrix.csv \
       --top-n-pairs 3 \
       --species-pairs Myca Myyu Myth Myev \
       --out-dir ./overlap_reports        
   """

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from logging_service import LoggingService
from sonobat_utils.utils import Utils

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are never acoustic features.
_NON_FEATURE_COLS: frozenset[str] = frozenset([
    'Filename', 'file_id',
    'species', 'species_prob', 'species_2nd',
    'chirp_idx', 'cntxt_sz', 'split', 'index',
    'Path', 'ParentDir', 'NextDirUp', 'Version', 'Filter',
    'Preemphasis', 'MaxSegLnght', 'MinAccpQuality',
    'Max#CallsConsidered', 'TimeInFile',
])

# Minimum probability to count a chirp as confidently identified.
_DEFAULT_CONF_THRESHOLD: float = 0.98


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the list of numeric feature columns, excluding all metadata and
    label columns defined in :data:`_NON_FEATURE_COLS`.

    :param df: Chirp-level measures DataFrame.
    :return:   Sorted list of feature column names.
    """
    return sorted([
        c for c in df.columns
        if c not in _NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ])


def _worst_pairs(
    cm_csv:   Path,
    top_n:    int,
) -> list[tuple[str, str, float]]:
    """
    Load a normalised confusion matrix CSV and return the top-N species pairs
    ranked by off-diagonal confusion rate (symmetric: max of both directions).

    The confusion matrix CSV is expected to have species codes as both the
    index column and the header row, as produced by
    :meth:`RFTrainer._save_confusion_matrix`.

    :param cm_csv: Path to the normalised confusion matrix CSV.
    :param top_n:  Number of worst pairs to return.
    :return:       List of ``(species_a, species_b, confusion_rate)`` tuples,
                   sorted descending by confusion rate.
    """
    cm = pd.read_csv(cm_csv, index_col=0)
    species = cm.index.tolist()
    pairs: list[tuple[str, str, float]] = []

    for i, sp_a in enumerate(species):
        for sp_b in species[i + 1:]:
            # Symmetric confusion: average of both off-diagonal cells.
            rate = (cm.loc[sp_a, sp_b] + cm.loc[sp_b, sp_a]) / 2.0
            pairs.append((sp_a, sp_b, rate))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_n]


# ---------------------------------------------------------------------------
# Class SpeciesDistribReporter
# ---------------------------------------------------------------------------

class SpeciesDistribReporter:
    """
    Print raw counts and percentage distribution of high-confidence species
    labels.

    :param df:                  Chirp-level measures DataFrame.
    :param conf_threshold:      Minimum ``species_prob`` to count a chirp as
                                confidently identified.
    """

    def __init__(
        self,
        df:              pd.DataFrame,
        conf_threshold:  float = _DEFAULT_CONF_THRESHOLD,
    ) -> None:
        self.distrib = self.compute_distrib(df, conf_threshold)

    #------------------------------------
    # compute_distrib
    #-------------------

    def compute_distrib(
        self,
        df:             pd.DataFrame,
        conf_threshold: float,
    ) -> pd.DataFrame:
        """
        Compute and print species counts and percentages for rows where
        ``species_prob >= conf_threshold``.

        :param df:             Chirp-level measures DataFrame.
        :param conf_threshold: Minimum ``species_prob`` threshold.
        :return:               DataFrame with columns ``count`` and
                               ``percentage``, indexed by species.
        """
        confident = df[df['species_prob'] >= conf_threshold]
        n = len(confident)
        counts = confident.groupby('species').size().rename('count')
        pct    = (100.0 * counts / n).rename('percentage').round(4)
        self.distrib = pd.concat([counts, pct], axis=1)
        print(f"\nSpecies distribution "
              f"(species_prob >= {conf_threshold}, n={n:,}):\n")
        print(self.distrib.to_string())
        print()
        return self.distrib


# ---------------------------------------------------------------------------
# Class ChirpSeqSpeciesPurityReporter
# ---------------------------------------------------------------------------

class ChirpSeqSpeciesPurityReporter:
    """
    Examine how often all chirps within a ``file_id`` fragment share the same
    species label.

    A purity of 1.0 for a ``file_id`` means every chirp in that fragment was
    assigned the same species — perfect internal consistency.

    :param df:      Chirp-level measures DataFrame.
    :param out_dir: Directory for output files.
    """

    def __init__(self, df: pd.DataFrame, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.df      = df
        self.metrics: Optional[pd.DataFrame] = None

    #------------------------------------
    # compute_purity
    #-------------------

    def compute_purity(self) -> None:
        """
        Compute per-``file_id`` purity: fraction of chirps assigned to the
        majority species.  Stores results in :attr:`metrics`.
        """
        log.info('Computing purity metrics per file_id ...')

        counts = (
            self.df.groupby(['file_id', 'species'])
            .size()
            .reset_index(name='count')
        )
        totals              = counts.groupby('file_id')['count'].transform('sum')
        counts['proportion'] = counts['count'] / totals

        self.metrics = (
            counts
            .sort_values('count', ascending=False)
            .drop_duplicates('file_id')
            .rename(columns={'species': 'primary_species',
                             'proportion': 'purity'})
        )
        self.metrics['is_pure'] = self.metrics['purity'] == 1.0

        purity_rate = self.metrics['is_pure'].mean() * 100
        log.info(
            f'Purity analysis complete: '
            f'{purity_rate:.2f}% of file_ids are homogeneous'
        )

    #------------------------------------
    # save_reports
    #-------------------

    def save_reports(self) -> None:
        """
        Save a CSV listing all impure file_ids (purity < 1.0), sorted by
        ascending purity.
        """
        imperfect_path = self.out_dir / 'imperfect_files.csv'
        imperfect      = (
            self.metrics[self.metrics['purity'] < 1.0]
            .sort_values('purity')
        )
        imperfect.to_csv(imperfect_path, index=False)
        log.info(f'Saved {len(imperfect):,} impure file_ids to {imperfect_path}')

    #------------------------------------
    # plot_visuals
    #-------------------

    def plot_visuals(self) -> None:
        """
        Save two diagnostic plots:

        * Purity distribution histogram (log y-scale).
        * Scatter of purity vs mean ``species_prob`` per file_id.
        """
        log.info('Generating purity charts ...')

        # Chart 1: purity distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.metrics['purity'], bins=20, kde=False,
                     color='skyblue', ax=ax)
        ax.set_yscale('log')
        ax.set_title('Distribution of file_id purity  (ideal = 1.0)')
        ax.set_xlabel('Purity  (majority species count / total chirp count)')
        ax.set_ylabel('Number of file_ids')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'purity_distribution.png', dpi=150)
        plt.close(fig)

        # Chart 2: purity vs mean probability
        avg_prob = self.df.groupby('file_id')['species_prob'].mean()
        plot_df  = self.metrics.set_index('file_id').join(avg_prob)
        fig, ax  = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=plot_df, x='purity', y='species_prob',
                        alpha=0.1, ax=ax)
        ax.set_title('Purity vs. mean species_prob per file_id')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'purity_vs_prob.png', dpi=150)
        plt.close(fig)

        log.info(f'Purity charts saved to {self.out_dir}')

    #------------------------------------
    # run
    #-------------------

    def run(self) -> None:
        """
        Run the full purity analysis: compute, save CSV, save plots.
        """
        self.compute_purity()
        self.save_reports()
        self.plot_visuals()


# ---------------------------------------------------------------------------
# Class SpeciesOverlapAnalyzer
# ---------------------------------------------------------------------------

class SpeciesOverlapAnalyzer:
    """
    Quantify and visualise the overlap between two species in acoustic feature
    space, supporting three complementary analyses:

    * **Univariate Bhattacharyya** — per-feature overlap coefficient (0=fully
      separated, 1=identical distributions), ranked descending.
    * **Multivariate Bhattacharyya** — distance between two multivariate
      Gaussian fits in the full feature space (0=same cloud, larger=more
      separable).
    * **UMAP projection** — 2-D (or optional 3-D) scatter plot of the two
      species' chirps in reduced feature space.

    All outputs for a species pair are written to
    ``<out_dir>/<species_a>_<species_b>/``.

    :param df:       Chirp-level measures DataFrame containing ``species`` and
                     all acoustic feature columns.
    :param out_dir:  Root output directory; one subdirectory is created per
                     pair.
    :param n_sample: Maximum number of chirps per species to use for UMAP and
                     multivariate analysis (random sample; ``None`` = use all).
    """

    def __init__(
        self,
        df:       pd.DataFrame,
        out_dir:  str | Path,
        n_sample: Optional[int] = 20_000,
    ) -> None:
        self.df       = df
        self.out_dir  = Path(out_dir)
        self.n_sample = n_sample
        self._fcols   = _feature_cols(df)
        log.info(f'SpeciesOverlapAnalyzer: {len(self._fcols)} feature columns')

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _pair_arrays(
        self,
        sp_a: str,
        sp_b: str,
        sample: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract and optionally subsample the feature arrays for two species.

        :param sp_a:   First species code.
        :param sp_b:   Second species code.
        :param sample: If ``True`` and :attr:`n_sample` is set, randomly
                       sample up to ``n_sample`` rows per species.
        :return:       Tuple ``(X_a, X_b)`` of float64 arrays with shape
                       ``(n_chirps, n_features)``.
        """
        rng = np.random.default_rng(42)

        def _extract(sp: str) -> np.ndarray:
            rows = (
                self.df[self.df['species'] == sp][self._fcols]
                .dropna()
                .values.astype(np.float64)
            )
            if sample and self.n_sample and len(rows) > self.n_sample:
                idx  = rng.choice(len(rows), self.n_sample, replace=False)
                rows = rows[idx]
            return rows

        X_a = _extract(sp_a)
        X_b = _extract(sp_b)
        log.info(
            f'  {sp_a}: {len(X_a):,} chirps   {sp_b}: {len(X_b):,} chirps'
        )
        return X_a, X_b

    @staticmethod
    def _pair_dir(out_dir: Path, sp_a: str, sp_b: str) -> Path:
        """
        Create and return the output subdirectory for a species pair.

        :param out_dir: Root output directory.
        :param sp_a:    First species code.
        :param sp_b:    Second species code.
        :return:        ``out_dir / '<sp_a>_<sp_b>'``, created if absent.
        """
        d = out_dir / f'{sp_a}_{sp_b}'
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ #
    #  Univariate Bhattacharyya                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bhattacharyya_coeff_1d(
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 100,
    ) -> float:
        """
        Estimate the Bhattacharyya coefficient between two 1-D samples using
        equal-width histogram bins spanning the combined range.

        The coefficient is 1.0 when the distributions are identical and 0.0
        when they share no support.

        :param x:      Sample from the first distribution.
        :param y:      Sample from the second distribution.
        :param n_bins: Number of histogram bins.
        :return:       Bhattacharyya coefficient in ``[0, 1]``.
        """
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if lo == hi:
            return 1.0
        bins  = np.linspace(lo, hi, n_bins + 1)
        px, _ = np.histogram(x, bins=bins, density=True)
        py, _ = np.histogram(y, bins=bins, density=True)
        # Normalise to probability mass per bin
        dx    = bins[1] - bins[0]
        px    = px * dx
        py    = py * dx
        return float(np.sum(np.sqrt(px * py)))

    def univariate_overlap(
        self,
        sp_a: str,
        sp_b: str,
    ) -> pd.DataFrame:
        """
        Compute the Bhattacharyya coefficient for every feature independently,
        rank features from most overlapping (hardest to discriminate) to least,
        and save results as a CSV and horizontal bar chart.

        :param sp_a: First species code.
        :param sp_b: Second species code.
        :return:     DataFrame with columns ``feature`` and
                     ``bhattacharyya_coeff``, sorted descending.
        """
        log.info(f'Univariate overlap: {sp_a} vs {sp_b} ...')
        X_a, X_b = self._pair_arrays(sp_a, sp_b, sample=False)
        pair_dir = self._pair_dir(self.out_dir, sp_a, sp_b)

        coeffs = [
            self._bhattacharyya_coeff_1d(X_a[:, i], X_b[:, i])
            for i in range(len(self._fcols))
        ]
        result = (
            pd.DataFrame({'feature': self._fcols,
                          'bhattacharyya_coeff': coeffs})
            .sort_values('bhattacharyya_coeff', ascending=False)
            .reset_index(drop=True)
        )
        csv_path = pair_dir / 'univariate_overlap.csv'
        result.to_csv(csv_path, index=False)
        log.info(f'  Saved {csv_path}')

        # Bar chart — most overlapping features at top
        n_show = min(40, len(result))
        fig, ax = plt.subplots(figsize=(9, max(5, n_show * 0.28)))
        ax.barh(
            result['feature'][:n_show][::-1],
            result['bhattacharyya_coeff'][:n_show][::-1],
            color='steelblue',
        )
        ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8,
                   label='0.5 overlap')
        ax.set_xlabel('Bhattacharyya coefficient  (1 = identical, 0 = separated)')
        ax.set_title(f'Per-feature overlap: {sp_a} vs {sp_b}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        png_path = pair_dir / 'univariate_overlap.png'
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        log.info(f'  Saved {png_path}')

        return result

    # ------------------------------------------------------------------ #
    #  Multivariate Bhattacharyya                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bhattacharyya_distance_mv(
        X_a: np.ndarray,
        X_b: np.ndarray,
    ) -> float:
        """
        Compute the Bhattacharyya distance between two multivariate Gaussian
        distributions fitted to ``X_a`` and ``X_b``.

        The distance is 0 when the two clouds are identical and increases
        without bound as they become more separated.

        Uses the closed-form expression for multivariate Gaussians::

            D_B = (1/8)(mu_a - mu_b)^T Sigma^{-1} (mu_a - mu_b)
                + (1/2) ln( det(Sigma) / sqrt(det(Sigma_a) det(Sigma_b)) )

        where ``Sigma = (Sigma_a + Sigma_b) / 2``.

        :param X_a: Array of shape ``(n_a, d)`` for species A.
        :param X_b: Array of shape ``(n_b, d)`` for species B.
        :return:    Bhattacharyya distance (float >= 0).
        """
        mu_a   = X_a.mean(axis=0)
        mu_b   = X_b.mean(axis=0)
        cov_a  = np.cov(X_a, rowvar=False)
        cov_b  = np.cov(X_b, rowvar=False)
        cov_m  = (cov_a + cov_b) / 2.0

        # Regularise to avoid singular matrices (rare but possible with
        # constant columns or near-zero-variance features).
        eps    = 1e-6 * np.eye(cov_m.shape[0])
        cov_m  += eps
        cov_a  += eps
        cov_b  += eps

        diff   = mu_a - mu_b
        try:
            inv_m  = np.linalg.inv(cov_m)
            sign_m, logdet_m = np.linalg.slogdet(cov_m)
            sign_a, logdet_a = np.linalg.slogdet(cov_a)
            sign_b, logdet_b = np.linalg.slogdet(cov_b)
            mahal  = 0.125 * float(diff @ inv_m @ diff)
            logdet = 0.5 * (logdet_m - 0.5 * (logdet_a + logdet_b))
            return mahal + logdet
        except np.linalg.LinAlgError as exc:
            log.warn(f'  Multivariate Bhattacharyya failed: {exc}')
            return float('nan')

    def multivariate_overlap(
        self,
        sp_a: str,
        sp_b: str,
    ) -> float:
        """
        Compute and save the multivariate Bhattacharyya distance between two
        species in the full acoustic feature space.

        A distance near 0 means the two species occupy essentially the same
        region of feature space; larger values indicate increasing
        separability.

        :param sp_a: First species code.
        :param sp_b: Second species code.
        :return:     Bhattacharyya distance.
        """
        log.info(f'Multivariate overlap: {sp_a} vs {sp_b} ...')
        X_a, X_b = self._pair_arrays(sp_a, sp_b, sample=True)
        pair_dir = self._pair_dir(self.out_dir, sp_a, sp_b)

        dist = self._bhattacharyya_distance_mv(X_a, X_b)
        log.info(f'  Bhattacharyya distance: {dist:.4f}')

        result = pd.DataFrame([{
            'species_a':              sp_a,
            'species_b':              sp_b,
            'bhattacharyya_distance': round(dist, 6),
            'n_chirps_a':             len(X_a),
            'n_chirps_b':             len(X_b),
            'n_features':             len(self._fcols),
        }])
        csv_path = pair_dir / 'multivariate_overlap.csv'
        result.to_csv(csv_path, index=False)
        log.info(f'  Saved {csv_path}')
        return dist

    # ------------------------------------------------------------------ #
    #  UMAP projection                                                    #
    # ------------------------------------------------------------------ #

    def umap_projection(
        self,
        sp_a:     str,
        sp_b:     str,
        n_components: int = 2,
    ) -> None:
        """
        Project the two species' chirps into 2-D (or 3-D) with UMAP and save
        a scatter plot.

        Requires the ``umap-learn`` package (``pip install umap-learn``).

        :param sp_a:         First species code.
        :param sp_b:         Second species code.
        :param n_components: Number of UMAP dimensions (2 or 3).
        """
        try:
            import umap
        except ImportError:
            log.warn(
                'umap-learn is not installed.  '
                'Run: pip install umap-learn'
            )
            return

        log.info(f'UMAP projection ({n_components}D): {sp_a} vs {sp_b} ...')
        X_a, X_b = self._pair_arrays(sp_a, sp_b, sample=True)
        pair_dir = self._pair_dir(self.out_dir, sp_a, sp_b)

        X      = np.vstack([X_a, X_b])
        labels = np.array([sp_a] * len(X_a) + [sp_b] * len(X_b))

        log.info(f'  Fitting UMAP on {len(X):,} points x {X.shape[1]} features ...')
        reducer   = umap.UMAP(n_components=n_components, random_state=42)
        # This commented version runs multi-threaded, but is
        # not reproducible:
        #reducer   = umap.UMAP(n_components=n_components, random_state=42,
        #                      n_jobs=-1)
        embedding = reducer.fit_transform(X)
        log.info('  UMAP fit complete')

        colours = {sp_a: 'steelblue', sp_b: 'tomato'}

        if n_components == 2:
            fig, ax = plt.subplots(figsize=(9, 7))
            for sp, colour in colours.items():
                mask = labels == sp
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=colour, label=sp, alpha=0.3, s=4, linewidths=0,
                )
            ax.set_xlabel('UMAP-1')
            ax.set_ylabel('UMAP-2')
            ax.set_title(f'UMAP 2-D projection: {sp_a} vs {sp_b}')
            ax.legend(markerscale=3)
            fig.tight_layout()
            png_path = pair_dir / 'umap_2d.png'
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            log.info(f'  Saved {png_path}')

        else:  # 3-D
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection='3d')
            for sp, colour in colours.items():
                mask = labels == sp
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                    c=colour, label=sp, alpha=0.3, s=3, linewidths=0,
                )
            ax.set_xlabel('UMAP-1')
            ax.set_ylabel('UMAP-2')
            ax.set_zlabel('UMAP-3')
            ax.set_title(f'UMAP 3-D projection: {sp_a} vs {sp_b}')
            ax.legend(markerscale=3)
            fig.tight_layout()
            png_path = pair_dir / 'umap_3d.png'
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            log.info(f'  Saved {png_path}')

    # ------------------------------------------------------------------ #
    #  Batch runner                                                       #
    # ------------------------------------------------------------------ #

    def run_pairs(
        self,
        pairs:        list[tuple[str, str, float]],
        actions:      list[str],
        n_components: int = 2,
    ) -> None:
        """
        Run one or more overlap analyses for each species pair.

        :param pairs:        List of ``(sp_a, sp_b, confusion_rate)`` tuples
                             as returned by :func:`_worst_pairs`.
        :param actions:      List of action strings — any subset of
                             ``'univariate-overlap'``,
                             ``'multivariate-overlap'``, ``'umap'``.
        :param n_components: UMAP dimensionality (2 or 3); ignored for
                             non-umap actions.
        """
        for sp_a, sp_b, rate in pairs:
            log.info(
                f'--- Pair: {sp_a} / {sp_b}  '
                f'(confusion rate: {rate:.3f}) ---'
            )
            for action in actions:
                if action == 'univariate-overlap':
                    self.univariate_overlap(sp_a, sp_b)
                elif action == 'multivariate-overlap':
                    self.multivariate_overlap(sp_a, sp_b)
                elif action == 'umap':
                    self.umap_projection(sp_a, sp_b, n_components=n_components)
                else:
                    log.warn(f'Unknown action: {action}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ACTIONS = (
    'list-species',
    'purity-analysis',
    'univariate-overlap',
    'multivariate-overlap',
    'umap',
)

_OVERLAP_ACTIONS = frozenset([
    'univariate-overlap',
    'multivariate-overlap',
    'umap',
])


def _parse_args():
    """
    Parse command-line arguments.

    :return: Validated ``argparse.Namespace``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawTextHelpFormatter,
        description='Report and analyse species distributions in a chirp measures file.',
    )
    parser.add_argument(
        'input',
        help='Path to a .csv or .feather chirp measures file.',
    )
    parser.add_argument(
        '--action',
        required=True,
        nargs='+',
        choices=_ACTIONS,
        metavar='ACTION',
        help=(
            'One or more analyses to run (space-separated).  Choices:\n'
            '  list-species          — print species counts and percentages\n'
            '  purity-analysis       — per-file_id species purity\n'
            '  univariate-overlap    — per-feature Bhattacharyya coefficients\n'
            '  multivariate-overlap  — multivariate Bhattacharyya distance\n'
            '  umap                  — UMAP 2-D/3-D projection scatter plot\n'
            '\n'
            'Example: --action univariate-overlap multivariate-overlap umap'
        ),
    )
    parser.add_argument(
        '-o', '--out-dir',
        dest='out_dir',
        default='./reports',
        help='Root output directory (default: ./reports).',
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=_DEFAULT_CONF_THRESHOLD,
        metavar='F',
        help=(
            f'Minimum species_prob to count a chirp as confidently labeled\n'
            f'(used by list-species; default: {_DEFAULT_CONF_THRESHOLD}).'
        ),
    )
    parser.add_argument(
        '--confusion-matrix-csv',
        default=None,
        metavar='CSV',
        help=(
            'Normalised confusion matrix CSV produced by\n'
            'species_pred_random_forest.py.  Required for overlap actions\n'
            'unless --species-pairs is also provided.'
        ),
    )
    parser.add_argument(
        '--top-n-pairs',
        type=int,
        default=5,
        metavar='N',
        help='Number of worst-confused species pairs to analyse (default: 5).',
    )
    parser.add_argument(
        '--species-pairs',
        nargs='+',
        default=[],
        metavar='SPECIES',
        help=(
            'Explicit species pairs to analyse, given as a flat even-length\n'
            'space-separated list of species codes:\n'
            '  --species-pairs Myca Myyu Lano Tabr\n'
            'analyses Myca/Myyu and Lano/Tabr.  May be combined with\n'
            '--confusion-matrix-csv; the resulting pairs are unioned.\n'
            'Satisfies the overlap-action source requirement on its own.'
        ),
    )
    parser.add_argument(
        '--n-sample',
        type=int,
        default=20_000,
        metavar='N',
        help=(
            'Maximum chirps per species for UMAP and multivariate analysis\n'
            '(random subsample; 0 = use all; default: 20000).'
        ),
    )
    parser.add_argument(
        '--umap-3d',
        action='store_true',
        help='Project to 3-D instead of 2-D for the umap action.',
    )

    args = parser.parse_args()

    # Post-parse validation.
    overlap_requested = [a for a in args.action if a in _OVERLAP_ACTIONS]

    if overlap_requested:
        has_cm_source     = bool(args.confusion_matrix_csv)
        has_pair_source   = bool(args.species_pairs)
        if not has_cm_source and not has_pair_source:
            parser.error(
                f'Overlap actions ({", ".join(overlap_requested)}) require '
                f'at least one of --confusion-matrix-csv or --species-pairs.'
            )
        if args.species_pairs and len(args.species_pairs) % 2 != 0:
            parser.error(
                f'--species-pairs requires an even number of species codes; '
                f'got {len(args.species_pairs)}.'
            )

    if args.confusion_matrix_csv and not Path(args.confusion_matrix_csv).exists():
        parser.error(
            f'Confusion matrix file not found: {args.confusion_matrix_csv}'
        )

    args.input    = Path(args.input)
    args.out_dir  = Path(args.out_dir)
    args.n_sample = args.n_sample if args.n_sample > 0 else None
    return args


def main() -> None:
    """
    CLI entry point for species distribution reporting and overlap analysis.
    """
    args = _parse_args()

    # ---- Load data --------------------------------------------------- #
    log.info(f'Reading {args.input} ...')
    try:
        df = Utils.read_df_file(str(args.input))
        log.info(
            f'Loaded {len(df):,} rows, '
            f'{df["file_id"].nunique():,} unique file_ids'
        )
    except Exception as exc:
        log.warn(f'Cannot read {args.input}: {exc}')
        sys.exit(1)

    # ---- Dispatch ---------------------------------------------------- #
    actions          = args.action          # now a list
    overlap_actions  = [a for a in actions if a in _OVERLAP_ACTIONS]
    simple_actions   = [a for a in actions if a not in _OVERLAP_ACTIONS]

    # Simple actions first — no confusion matrix needed.
    for action in simple_actions:
        if action == 'list-species':
            SpeciesDistribReporter(df, conf_threshold=args.conf_threshold)
        elif action == 'purity-analysis':
            reporter = ChirpSeqSpeciesPurityReporter(df, args.out_dir)
            reporter.run()
        else:
            log.warn(f'Unhandled action: {action}')

    # Overlap actions — assemble pairs from both sources, then run.
    if overlap_actions:
        pairs: list[tuple[str, str, float]] = []

        # Pairs from confusion matrix (ranked by confusion rate).
        if args.confusion_matrix_csv:
            cm_pairs = _worst_pairs(
                Path(args.confusion_matrix_csv),
                top_n=args.top_n_pairs,
            )
            log.info(
                f'Top {len(cm_pairs)} confused pairs from confusion matrix: '
                + '  '.join(f'{a}/{b} ({r:.3f})' for a, b, r in cm_pairs)
            )
            pairs.extend(cm_pairs)

        # Explicit pairs from --species-pairs (confusion rate reported as NaN).
        if args.species_pairs:
            sp = args.species_pairs
            explicit = [
                (sp[i], sp[i + 1], float('nan'))
                for i in range(0, len(sp), 2)
            ]
            # Union: skip any pair already present from the confusion matrix.
            existing = {(a, b) for a, b, _ in pairs} | {(b, a) for a, b, _ in pairs}
            new_explicit = [(a, b, r) for a, b, r in explicit
                            if (a, b) not in existing]
            if new_explicit:
                log.info(
                    f'Adding {len(new_explicit)} explicit pair(s): '
                    + '  '.join(f'{a}/{b}' for a, b, _ in new_explicit)
                )
            pairs.extend(new_explicit)

        analyzer = SpeciesOverlapAnalyzer(
            df       = df,
            out_dir  = args.out_dir,
            n_sample = args.n_sample,
        )
        n_components = 3 if args.umap_3d else 2
        analyzer.run_pairs(pairs, actions=overlap_actions,
                           n_components=n_components)


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()
