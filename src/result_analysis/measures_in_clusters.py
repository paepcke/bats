#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-19 18:33:23
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-21 08:13:11
"""
Given each measure's normality in each cluster in file bats_measures_normality_all.csv,
and the all-measures cluster assignments, examine whether any values are particularly important
for cluster membership.

Then determine which measures are important for which clusters.
Example usage:

    # After running your importance analysis:
    results = analyze_measure_cluster_importance(
        df=chirp_data,
        measure_cols=chirp_data.columns[:-1],  # All except cluster column
        cluster_col='cluster',
        importance_tier='High',
        summary_df=analysis.summary(),
        alpha=0.05,
        correction_method='fdr_bh'  # FDR correction recommended
    )

    # Print summary
    print_analysis_summary(results)

    # Save results
    results['pairwise_tests'].to_csv('pairwise_cluster_tests.csv', index=False)
    results['measure_summary'].to_csv('measure_discrimination_summary.csv', index=False)
    results['cluster_profiles'].to_csv('cluster_profiles.csv', index=False)

    # Access specific results
    high_importance_pairs = results['pairwise_tests'][
        results['pairwise_tests']['significant']
    ]
"""

import argparse
from enum import StrEnum
import os
from pathlib import Path
import sys
from typing import Literal

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scikit_posthocs import posthoc_dunn

from sonobat_utils.utils import Utils
from logging_service import LoggingService

# Significance levels for importance of bat measures
# to cluster membership:
class ImportanceTier(StrEnum):
    NEGLIGIBLE = 'negligible'
    LOW = 'low'
    MEDIUM = 'medium'
    LARGE = 'large'
    ANYRANKED = 'all'

class DunnCorrectionMethods(StrEnum):
    BONFERRONI =     'bonferroni'
    SIDAK =          'sidak'
    HOLM_SIDAK =     'holm-sidak'
    HOLM =           'holm'
    SIMES_HOCHBERG = 'simes-hochberg'
    HOMMEL =         'hommel'
    FDR_BH =         'fdr_bh'
    FDR_BY =         'fdr_by'
    FDR_TSBH =       'fdr_tsbh'
    FDR_TSBKY =      'fdr_tsbky'

RELEVANT_COLS = [
    'TimeInFile', 'PrecedingIntrvl', 'HiFreq', 'Bndwdth', 'FreqMaxPwr',
    'PrcntMaxAmpDur', 'FreqKnee', 'PrcntKneeDur', 'StartF', 'UpprKnFreq',
    'HiFtoUpprKnAmp', 'HiFtoKnAmp', 'HiFtoFcAmp', 'UpprKnToKnAmp',
    'KnToFcAmp', 'LdgToFcAmp', 'FreqCtr', 'FFwd32dB', 'FFwd20dB',
    'FFwd15dB', 'FBak5dB', 'FFwd5dB', 'Bndw32dB', 'Amp1stQrtl',
    'Amp2ndQrtl', 'Amp3rdQrtl', 'Amp4thQrtl', '1st10kHzSlp',
    '1st5to15kHzSlp', '1st10kHzExp', '1st5to15kHzExp', 'AmpK@start',
    'chirp_idx','idiom_start', 'idiom_end','in_idiom', 'is_first', 'is_last'
    ]

# =====================================================================
# Class KruskalWallisTester
# =====================================================================

class KruskalWallisTester:
    """
    Performs Kruskal-Wallis tests (continuous features) or chi-squared tests
    (boolean / binary features) and computes comparable effect sizes across
    cluster assignments.

    We assume that the data df contains a column 'cluster' with a cluster ID.

    :param infile: Path to a .csv or .feather file containing the measures
                   and a cluster ID column.
    :param col_names: Optional list of column names to include. If None, all
                      columns except cluster_id_col are used.
    :param cluster_id_col: Name of the column containing cluster assignments.
    """

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(
        self,
        infile: str | Path,
        col_names: list[str] | None = None,
        cluster_id_col: str = 'cluster'
    ):
        
        self.log = LoggingService()

        df_raw = Utils.read_df_file(infile)
        if col_names is None:
            self.col_names = list(self.df.columns.drop(cluster_id_col))
        else:
            self.col_names = col_names

        self.df, missing_cols = Utils.extract_cols_safely(df_raw, self.col_names)
        if len(missing_cols) > 0:
            self.log.warn(f"KuskalWallisTester: Requested columns {missing_cols} not in given df")
            # Remove the missing cols from self.col_names:
            self.col_names = [col for col in self.col_names if col not in missing_cols]

        self.cluster_labels = df_raw[cluster_id_col]
        self.n = len(self.df)
        self.k = len(np.unique(self.cluster_labels))

        if len(self.cluster_labels) != self.n:
            raise ValueError(
                f"cluster_labels length ({len(self.cluster_labels)}) "
                f"must match DataFrame length ({self.n})."
            )

        self.results_: pd.DataFrame | None = None

    #------------------------------------
    # _is_binary
    #-------------------

    def _is_binary(self, col: str) -> bool:
        """
        Return True if column contains only two distinct non-NaN values,
        i.e. is boolean or binary integer.

        :param col: Column name.
        :return: True if binary/boolean.
        """
        return len(self.df[col].dropna().unique()) <= 2

    #------------------------------------
    # _eta_squared_kw
    #-------------------

    def _eta_squared_kw(self, H: float) -> float:
        """
        Compute eta-squared effect size from the KW H statistic.

        :param H: Kruskal-Wallis H statistic.
        :return: Eta-squared value in [0, 1].
        """
        return max(0.0, (H - self.k + 1) / (self.n - self.k))

    #------------------------------------
    # _cramers_v
    #-------------------

    def _cramers_v(self, chi2: float) -> float:
        """
        Compute Cramér's V effect size from a chi-squared statistic on a 2 x k table.

        :param chi2: Chi-squared statistic.
        :return: Cramér's V in [0, 1].
        """
        # For a 2-row table, min_dim - 1 = 1, so V simplifies to sqrt(chi2 / n)
        return min(1.0, np.sqrt(chi2 / (self.n * (min(2, self.k) - 1))))

    #------------------------------------
    # _run_kw
    #-------------------

    def _run_kw(self, col: str) -> dict:
        """
        Run Kruskal-Wallis test for a single continuous feature.

        :param col: Column name.
        :return: Result dict with H_statistic, p_value, effect_size,
                 effect_size_metric, and test.
        """
        self.log.info(f"Running Kruskal-Wallis on {len(self.df)} chirps, col {col}...")
        groups = [
            self.df.loc[self.cluster_labels == label, col].dropna().values
            for label in np.unique(self.cluster_labels)
        ]
        groups = [g for g in groups if len(g) > 0]
        H, p = stats.kruskal(*groups)
        return {
            "H_statistic": H,
            "p_value": p,
            "effect_size": self._eta_squared_kw(H),
            "effect_size_metric": "eta_squared",
            "test": "kruskal-wallis",
        }

    #------------------------------------
    # _run_chi2
    #-------------------

    def _run_chi2(self, col: str) -> dict:
        """
        Run chi-squared test of independence for a binary/boolean feature.

        :param col: Column name.
        :return: Result dict with H_statistic (chi2), p_value, effect_size,
                 effect_size_metric, and test.
        """
        self.log.info(f"Running chi^2 on column {col}...")
        contingency = pd.crosstab(self.df[col], self.cluster_labels)
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        return {
            "H_statistic": chi2,    # unified column name for downstream compatibility
            "p_value": p,
            "effect_size": self._cramers_v(chi2),
            "effect_size_metric": "cramers_v",
            "test": "chi-squared",
        }

    #------------------------------------
    # run
    #-------------------

    def run(self) -> pd.DataFrame:
        """
        Run the appropriate test for each selected feature. Binary/boolean
        columns are routed to chi-squared + Cramér's V; all others use
        Kruskal-Wallis + eta-squared.

        :return: DataFrame indexed by feature name with columns:
                 H_statistic, p_value, effect_size, effect_size_metric,
                 test, effect_size_rank. Sorted by effect_size descending.
        """
        records = []

        for col in self.col_names:
            result = {"feature": col}
            if self._is_binary(col):
                result.update(self._run_chi2(col))
            else:
                result.update(self._run_kw(col))
            records.append(result)

        self.results_ = (
            pd.DataFrame(records)
            .set_index("feature")
            .sort_values("effect_size", ascending=False)
        )
        self.results_["effect_size_rank"] = np.arange(1, len(self.results_) + 1)

        self.log.info("Done running Kruskal-Wallis tests")
        return self.results_

    #------------------------------------
    # summary
    #-------------------

    def summary(self, top_n: int | None = None) -> pd.DataFrame:
        """
        Return a formatted summary of results, optionally limited to top N features.

        :param top_n: Number of top features to return. If None, returns all.
        :return: DataFrame of results, sorted by effect_size descending,
                 with an added effect_size_label column.
        """
        if self.results_ is None:
            raise RuntimeError("Call run() before summary().")

        def _label(row):
            e = row["effect_size"]
            if row["effect_size_metric"] == "eta_squared":
                if e >= 0.14: return "large"
                if e >= 0.06: return "medium"
                if e >= 0.01: return "small"
                return "negligible"
            else:  # cramers_v
                if e >= 0.50: return "large"
                if e >= 0.30: return "medium"
                if e >= 0.10: return "small"
                return "negligible"

        out = self.results_.copy()
        if top_n is not None:
            out = out.head(top_n)
        out["effect_size_label"] = out.apply(_label, axis=1)
        return out


# =====================================================================
# Class RandomForestTester
# =====================================================================

class RandomForestTester:
    """
    Fits a RandomForestClassifier to predict cluster membership from
    the feature matrix, then extracts both Gini-based and permutation
    importances as a measure of each feature's discriminative power.

    :param infile: Path to a .csv or .feather file containing the measures
                   and a cluster ID column.
    :param col_names: Optional list of column names to include. If None, all
                      columns except cluster_id_col are used.
    :param cluster_id_col: Name of the column containing cluster assignments.
    :param n_estimators: Number of trees in the forest.
    :param random_state: Random seed for reproducibility.
    :param n_jobs: Number of parallel jobs for fitting and permutation importance.
    """

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(
        self,
        infile: str | Path,
        col_names: list[str] | None = None,
        cluster_id_col: str = 'cluster',
        n_estimators: int = 300,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.log = LoggingService()
        
        df_raw = Utils.read_df_file(infile)
        if col_names is None:
            self.col_names = list(self.df.columns.drop(cluster_id_col))
        else:
            self.col_names = col_names

        self.X, missing_cols = Utils.extract_cols_safely(df_raw, self.col_names)
        if len(missing_cols) > 0:
            self.log.warn(f"RandomForest Tester: Requested columns {missing_cols} not in given df")
            # Remove the missing cols from self.col_names:
            self.col_names = [col for col in self.col_names if col not in missing_cols]

        self.y = df_raw[cluster_id_col].values
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.rf_: RandomForestClassifier | None = None
        self.results_: pd.DataFrame | None = None

    #------------------------------------
    # run
    #-------------------

    def run(self, perm_n_repeats: int = 10) -> pd.DataFrame:
        """
        Fit the random forest and compute both Gini and permutation importances.

        Permutation importance is computed on the training data, which is
        appropriate here since the goal is descriptive (characterizing the
        clusters) rather than predictive generalization.

        :param perm_n_repeats: Number of permutation rounds per feature.
                               Higher values give more stable estimates at
                               greater compute cost.
        :return: DataFrame indexed by feature name with columns:
                 gini_importance, gini_rank, perm_importance, perm_importance_std,
                 perm_rank. Sorted by gini_importance descending.
        """
        self.log.info(f"Fitting Random Forest to {len(self.X)} chirps using {self.n_jobs} processes...")

        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.rf_.fit(self.X, self.y)

        # Gini-based importances
        gini_imp = self.rf_.feature_importances_

        # Permutation importances
        perm_result = permutation_importance(
            self.rf_, self.X, self.y,
            n_repeats=perm_n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.results_ = pd.DataFrame({
            "gini_importance": gini_imp,
            "perm_importance": perm_result.importances_mean,
            "perm_importance_std": perm_result.importances_std,
        }, index=self.col_names)

        self.results_.index.name = "feature"
        self.results_.sort_values("gini_importance", ascending=False, inplace=True)
        self.results_["gini_rank"] = np.arange(1, len(self.results_) + 1)

        # Rank by permutation importance separately
        perm_sorted_idx = self.results_["perm_importance"].rank(
            ascending=False, method="min"
        ).astype(int)
        self.results_["perm_rank"] = perm_sorted_idx

        self.log.info("Done fitting Random Forest")
        
        return self.results_

    #------------------------------------
    # summary
    #-------------------

    def summary(self, top_n: int | None = None) -> pd.DataFrame:
        """
        Return a formatted summary of RF importances, sorted by Gini importance.

        :param top_n: Number of top features to return. If None, returns all.
        :return: DataFrame of results.
        """
        if self.results_ is None:
            raise RuntimeError("Call run() before summary().")

        out = self.results_.copy()
        if top_n is not None:
            out = out.head(top_n)
        return out


# =====================================================================
# Class CombinedAnalysis
# =====================================================================

class CombinedAnalysis:
    """
    Runs both KruskalWallisTester and RandomForestTester on the same
    dataset and merges their results into a single summary DataFrame,
    ranked by mean rank across all three importance signals
    (KW effect size, Gini importance, permutation importance).

    :param infile: Path to a .csv or .feather file.
    :param col_names: Optional list of feature columns. If None, all columns
                      except cluster_id_col are used.
    :param cluster_id_col: Name of the cluster assignment column.
    :param rf_n_estimators: Number of trees for the random forest.
    :param rf_random_state: Random seed for the random forest.
    :param n_jobs: Parallel jobs for the random forest.
    """

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(
        self,
        infile: str | Path,
        col_names: list[str] | None = None,
        cluster_id_col: str = 'cluster',
        rf_n_estimators: int = 300,
        rf_random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.log = LoggingService()
        
        self.infile = infile
        self.col_names = col_names
        self.cluster_id_col = cluster_id_col
        self.rf_n_estimators = rf_n_estimators
        self.rf_random_state = rf_random_state
        self.n_jobs = n_jobs

        self.kw_tester_: KruskalWallisTester | None = None
        self.rf_tester_: RandomForestTester | None = None
        self.results_: pd.DataFrame | None = None

    #------------------------------------
    # run
    #-------------------

    def run(self, perm_n_repeats: int = 10) -> pd.DataFrame:
        """
        Execute both analyses and merge results.

        :param perm_n_repeats: Permutation rounds passed to RandomForestTester.
        :return: Merged DataFrame indexed by feature, sorted by mean_rank ascending.
                 Columns: kw_effect_size, kw_effect_size_metric, kw_test, kw_rank,
                 gini_importance, gini_rank, perm_importance, perm_importance_std,
                 perm_rank, mean_rank.
        """
        # --- KW ---
        self.kw_tester_ = KruskalWallisTester(
            self.infile,
            col_names=self.col_names,
            cluster_id_col=self.cluster_id_col,
        )
        # Save the data df that the Kruskal-Wallis test uses:
        self.df = self.kw_tester_.df
        kw_results = self.kw_tester_.run()

        # --- RF ---
        self.rf_tester_ = RandomForestTester(
            self.infile,
            col_names=self.col_names,
            cluster_id_col=self.cluster_id_col,
            n_estimators=self.rf_n_estimators,
            random_state=self.rf_random_state,
            n_jobs=self.n_jobs,
        )
        rf_results = self.rf_tester_.run(perm_n_repeats=perm_n_repeats)

        # --- Merge ---
        kw_cols = kw_results[[
            "effect_size", "effect_size_metric", "test", "effect_size_rank"
        ]].rename(columns={
            "effect_size":        "kw_effect_size",
            "effect_size_metric": "kw_effect_size_metric",
            "test":               "kw_test",
            "effect_size_rank":   "kw_rank",
        })

        rf_cols = rf_results[[
            "gini_importance", "gini_rank",
            "perm_importance", "perm_importance_std", "perm_rank",
        ]]

        merged = kw_cols.join(rf_cols, how="inner")

        # Mean rank across the three ranking signals
        merged["mean_rank"] = (
            merged[["kw_rank", "gini_rank", "perm_rank"]].mean(axis=1)
        )
        merged.sort_values("mean_rank", inplace=True)

        self.results_ = merged
        return self.results_

    #------------------------------------
    # summary
    #-------------------

    def summary(self, top_n: int | None = None) -> pd.DataFrame:
        """
        Return the merged results with a human-readable effect size label
        and composite importance tier.
        
        :param top_n: Number of top features to return. If None, returns all.
        :return: Summary DataFrame sorted by mean_rank ascending.
        """
        if self.results_ is None:
            raise RuntimeError("Call run() before summary().")
        
        def _label(row):
            """Apply effect size labels based on metric type."""
            e = row["kw_effect_size"]
            if row["kw_effect_size_metric"] == "eta_squared":
                if e >= 0.14: return "large"
                if e >= 0.06: return "medium"
                if e >= 0.01: return "small"
                return "negligible"
            else:  # cramers_v
                if e >= 0.50: return "large"
                if e >= 0.30: return "medium"
                if e >= 0.10: return "small"
                return "negligible"
        
        def _composite_importance(row):
            """
            Assign composite importance tier based on:
            - High: large effect size AND top 10 mean rank
            - Medium-High: large effect size OR top 10 mean rank
            - Low: everything else
            """
            kw_label = row['kw_effect_size_label']
            mean_rank = row['mean_rank']
            
            if kw_label == 'large' and mean_rank <= 10:
                return 'High'
            elif kw_label == 'large' or mean_rank <= 10:
                return 'Medium-High'
            else:
                return 'Low'
        
        # Create output dataframe
        out = self.results_.copy()
        
        # The index of self.results_ should contain the measure names from self.df
        # Add them as an explicit column
        out['measure_name'] = out.index
        
        # Add effect size label
        out["kw_effect_size_label"] = out.apply(_label, axis=1)
        
        # Add composite importance
        out['composite_importance'] = out.apply(_composite_importance, axis=1)
        
        # Convert composite_importance to ordered categorical
        importance_order = ['High', 'Medium-High', 'Low']
        out['composite_importance'] = pd.Categorical(
            out['composite_importance'],
            categories=importance_order,
            ordered=True
        )
        
        # Add priority rank within each importance tier
        out = out.sort_values(['composite_importance', 'mean_rank'], ascending=[True, True])
        out['priority_rank'] = out.groupby('composite_importance', observed=True).cumcount() + 1
        
        # Reorder columns to put key info first
        first_cols = [
            'priority_rank',
            'measure_name',
            'composite_importance',
            'kw_effect_size',
            'kw_effect_size_label',
            'kw_rank',
            'mean_rank'
        ]
        other_cols = [col for col in out.columns if col not in first_cols]
        out = out[first_cols + other_cols]
        
        # Apply top_n filter if requested
        if top_n is not None:
            out = out.head(top_n)
        
        return out

    #------------------------------------
    # save
    #-------------------

    def save(self, 
             summary: pd.DataFrame, 
             outdir: str | Path,
             force: bool = False) -> None:
        """
        Save the combined summary to a .csv file.

        :param summary: a df with test results and 
            qualitative cluster-wide importance labels for each
            column.
        :param outdir: destination directory.
        :param force: whether to overwrite the dst file, or ask user
        """
        outdir = Path(outdir)
        if self.results_ is None:
            raise RuntimeError("Call run() before save().")
        out_fname = 'meas_importance_all_clusters.csv'
        outfile = outdir / out_fname
        self.log.info(f"Writing measures importances to {outfile}...")
        Utils.write_outfile(self.summary(), outfile, force=True)

class PostHocTests:

    '''
    Runs analyzes importance of given bat measures for
    membership in particular clusters. Use this after running
    the CombinedAnalysis, which generates a summary table of
    which measures are large, medium, small, or negligible.

    This class's run() method takes that summary table, and
    the original data, and creates three more dataframes 
    from posthoc tests. They describe which clusters rely
    most on which measures for pairwisely discriminating
    against other clusters.
    '''
    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self,
                 df_info: str | Path | pd.DataFrame,
                 measure_cols: list[str],
                 cluster_col: str = 'cluster',
                 importance_tier: ImportanceTier = ImportanceTier.LARGE,
                 summary_df_info: str | Path | pd.DataFrame = None,
                 alpha: float = 0.05,
                 correction_method: DunnCorrectionMethods = DunnCorrectionMethods.FDR_BH
                 ):
        self.log = LoggingService()

        if isinstance(df_info, pd.DataFrame):
            df_raw = df_info
        else:
            df_raw = Utils.read_df_file(df_info)
        if summary_df_info is not None:
            if isinstance(summary_df_info, pd.DataFrame):
                self.summary_df = summary_df_info
            else:
                self.summary_df = Utils.read_df_file(summary_df_info)

        # Final data df, and measure columns to include:
        self.df, missing_cols = Utils.extract_cols_safely(df_raw, measure_cols)
        if len(missing_cols) > 0:
            self.log.warn(f"PosthocTests: Requested columns {missing_cols} not in given df")
            # Remove the missing cols from measure_cols:
            self.measure_cols = [col for col in measure_cols if col not in missing_cols]
        else:
            self.measure_cols = measure_cols

        # Cluster grouping var:
        if cluster_col not in df_raw:
            raise ValueError(f"Cluster column {cluster_col} not found in df_info")
        self.cluster_col = cluster_col

        # Check tier spec:
        if importance_tier not in ImportanceTier:
            raise ValueError(f"Importance tier must be one of {list(ImportanceTier)}, not {importance_tier}")
        self.importance_tier = importance_tier

        # Correction methods:
        if correction_method not in DunnCorrectionMethods:
            raise ValueError(f"Posthoc test must be one of {list(DunnCorrectionMethods)}, not {correction_method}")
        self.correction_method = correction_method

        self.result_dfs: dict[str: pd.DataFrame] | None = None

    #------------------------------------
    # run
    #-------------------    

    def run(self) -> dict[str: pd.DataFrame]:
        """
        Analyze which measures discriminate which cluster pairs, with proper
        multiple testing correction.

        This creates a measure-cluster importance analysis that:
        1. Runs Dunn's post-hoc tests for high-importance measures
        2. Applies FDR correction to control false discovery rate
        3. Summarizes which clusters each measure best discriminates
        """
        # Filter measures by importance if summary provided
        if self.summary_df is not None and self.importance_tier != ImportanceTier.ANYRANKED:
            high_measures = self.summary_df[
                self.summary_df['composite_importance'] == str(self.importance_tier)
            ]['measure_name'].tolist()
            measure_cols = [m for m in self.measure_cols if m in high_measures]
            self.log.info(f"Analyzing {len(measure_cols)} {self.importance_tier} importance measures")
        else:
            # All measures examined, no matter what their importance
            measure_cols = self.measure_cols
        
        n_clusters = self.df[self.cluster_col].nunique()
        n_pairs = n_clusters * (n_clusters - 1) // 2
        
        info = (f"\nTotal pairwise comparisons per measure: {n_pairs}"
                "   Total tests across {len(measure_cols)} measures: {n_pairs * len(measure_cols)}"
                f"  Multiple testing correction: {self.correction_method}")
        self.log.info(info)
        
        all_results = []
        
        # Run Dunn's test for each measure
        for measure in measure_cols:
            self.log.info(f"\nProcessing {measure}...")
            
            # Prepare data for this measure
            measure_data = self.df[[self.cluster_col, measure]].dropna()
            
            # Run Dunn's post-hoc test
            dunn_result = posthoc_dunn(
                measure_data,
                val_col=measure,
                group_col=self.cluster_col,
                p_adjust=self.correction_method
            )
            
            # Extract pairwise results
            clusters = sorted(measure_data[self.cluster_col].unique())
            for i, c1 in enumerate(clusters):
                for c2 in clusters[i+1:]:
                    p_val = dunn_result.loc[c1, c2]
                    
                    # Calculate effect size (rank-biserial correlation)
                    group1 = measure_data[measure_data[self.cluster_col] == c1][measure]
                    group2 = measure_data[measure_data[self.cluster_col] == c2][measure]
                    
                    # Mann-Whitney U test for effect size
                    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    n1, n2 = len(group1), len(group2)
                    # Rank-biserial correlation
                    r = 1 - (2*u_stat) / (n1 * n2)
                    
                    # Mean values for interpretation
                    mean1 = group1.mean()
                    mean2 = group2.mean()
                    
                    all_results.append({
                        'measure': measure,
                        'cluster_1': c1,
                        'cluster_2': c2,
                        'cluster_pair': f"{c1} vs {c2}",
                        'p_value': p_val,
                        'significant': p_val < self.alpha,
                        'effect_size_r': r,
                        'mean_cluster_1': mean1,
                        'mean_cluster_2': mean2,
                        'mean_diff': mean1 - mean2,
                        'direction': 'higher' if mean1 > mean2 else 'lower',
                        'n_cluster_1': n1,
                        'n_cluster_2': n2
                    })
        
        # Create results dataframe
        pairwise_df = pd.DataFrame(all_results)
        
        # Add significance symbols
        def sig_symbol(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return 'ns'
        
        pairwise_df['significance'] = pairwise_df['p_value'].apply(sig_symbol)
        
        # Summarize per measure
        measure_summary = pairwise_df.groupby('measure').agg({
            'significant': 'sum',
            'p_value': lambda x: (x < self.alpha).sum(),
            'effect_size_r': lambda x: x.abs().mean()
        }).reset_index()
        
        measure_summary.columns = ['measure', 'n_significant_pairs', 
                                    'n_significant_pairs_check', 'mean_abs_effect_size']
        measure_summary['total_pairs'] = n_pairs
        measure_summary['pct_significant'] = 100 * measure_summary['n_significant_pairs'] / n_pairs
        measure_summary = measure_summary.drop('n_significant_pairs_check', axis=1)
        
        # Sort by discrimination power
        measure_summary = measure_summary.sort_values('n_significant_pairs', ascending=False)
        
        # Create cluster profiles: which measures best characterize each cluster
        cluster_profiles = []
        for cluster in sorted(self.df[self.cluster_col].unique()):
            cluster_data = pairwise_df[
                (pairwise_df['cluster_1'] == cluster) | (pairwise_df['cluster_2'] == cluster)
            ].copy()
            
            # For this cluster, find measures that significantly discriminate it from others
            # and calculate average effect size
            for measure in measure_cols:
                measure_data = cluster_data[cluster_data['measure'] == measure]
                n_sig = measure_data['significant'].sum()
                avg_effect = measure_data['effect_size_r'].abs().mean()
                
                # Determine if this cluster tends to be higher or lower for this measure
                # Look at comparisons where this cluster is involved
                this_cluster_higher = 0
                this_cluster_lower = 0
                
                for _, row in measure_data.iterrows():
                    if row['cluster_1'] == cluster:
                        if row['mean_cluster_1'] > row['mean_cluster_2']:
                            this_cluster_higher += 1
                        else:
                            this_cluster_lower += 1
                    else:  # cluster_2 == cluster
                        if row['mean_cluster_2'] > row['mean_cluster_1']:
                            this_cluster_higher += 1
                        else:
                            this_cluster_lower += 1
                
                tendency = 'high' if this_cluster_higher > this_cluster_lower else 'low'
                
                cluster_profiles.append({
                    'cluster': cluster,
                    'measure': measure,
                    'n_significant_discriminations': n_sig,
                    'avg_effect_size': avg_effect,
                    'tendency': tendency,
                    'importance_score': n_sig * avg_effect  # Combined metric
                })
        
        cluster_profiles_df = pd.DataFrame(cluster_profiles)
        
        # For each cluster, identify top discriminating measures
        cluster_profiles_df = cluster_profiles_df.sort_values(
            ['cluster', 'importance_score'], 
            ascending=[True, False]
        )

        self.result_dfs = {
            'pairwise_tests': pairwise_df,
            'measure_summary': measure_summary,
            'cluster_profiles': cluster_profiles_df
        }
        return self.result_dfs

    #------------------------------------
    # save
    #-------------------    

    def save(self, 
             result_dfs: dict[str: pd.DataFrame], 
             outdir: str | Path,
             force: bool = False
             ):
        outdir = Path(outdir)
        out1_path = outdir / 'meas_towards_clusters_pairwise.csv'
        out2_path = outdir / 'meas_towards_clusters_measures_summary.csv'
        out3_path = outdir / 'meas_towards_clusters_cluster_profiles.csv'
        self.log.info(f"Writing pairwise tests to {out1_path}")
        Utils.write_outfile(result_dfs['pairwise_tests'], out1_path, force)
        self.log.info(f"Writing measures summary to {out2_path}")
        Utils.write_outfile(result_dfs['measure_summary'], out2_path, force)
        self.log.info(f"Writing cluster profiles to {out3_path}")
        Utils.write_outfile(result_dfs['cluster_profiles'], out3_path, force)

    #------------------------------------
    # print_analysis_summary
    #-------------------

    def print_analysis_summary(results: dict, top_n: int = 5):
        """Print a human-readable summary of the analysis."""
        
        print("\n" + "="*80)
        print("MEASURE DISCRIMINATION POWER")
        print("="*80)
        print("\nMeasures ranked by number of cluster pairs they significantly discriminate:\n")
        print(results['measure_summary'].to_string(index=False))
        
        print("\n" + "="*80)
        print("CLUSTER CHARACTERIZATION")
        print("="*80)
        print(f"\nTop {top_n} discriminating measures per cluster:\n")
        
        for cluster in sorted(results['cluster_profiles']['cluster'].unique()):
            cluster_data = results['cluster_profiles'][
                results['cluster_profiles']['cluster'] == cluster
            ].head(top_n)
            
            print(f"\nCluster {cluster}:")
            for _, row in cluster_data.iterrows():
                print(f"  {row['measure']:20s} - {row['n_significant_discriminations']:2.0f} sig pairs, "
                    f"effect={row['avg_effect_size']:.2f}, tends {row['tendency']}")
        
        print("\n" + "="*80)
        print("HIGHLY DISCRIMINATING PAIRS")
        print("="*80)
        print("\nCluster pairs with strong differences (|effect size| > 0.5, p < 0.05):\n")
        
        strong_effects = results['pairwise_tests'][
            (results['pairwise_tests']['significant']) &
            (results['pairwise_tests']['effect_size_r'].abs() > 0.5)
        ].sort_values('effect_size_r', key=abs, ascending=False)
        
        if len(strong_effects) > 0:
            print(strong_effects[['measure', 'cluster_pair', 'effect_size_r', 
                                'p_value', 'direction']].head(20).to_string(index=False))
        else:
            print("No pairs with |effect size| > 0.5")



# --------------------- Main -------------

if __name__ == "__main__":
    desc = "Computes importance of bat measures for cluster membership."
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawTextHelpFormatter,
        description=desc,
    )
    parser.add_argument('infile',
                        help='path to .csv or .feather file')
    
    parser.add_argument('-a', '--autocols',
                        action='store_true',
                        default=False,
                        help='include all suitable columns in the analysis; if yes, --cols is ignored')

    parser.add_argument('-c', '--cols',
                        type=str,
                        nargs='+',
                        help='Repeatable: columns to include; default: all',
                        default=None)
    parser.add_argument('-o', '--outdir',
                        type=str,
                        help='output directory for all results',
                        default=None)

    parser.add_argument('-p', '--print',
                        action='store_true',
                        default=False,
                        help='print summary results to console')

    parser.add_argument('-f', '--force',
                        action='store_true',
                        default=False,
                        help='overwrite existing summary if it exists; default: ask permission in console')

    parser.add_argument('--top-n',
                        type=int,
                        help='Print only top N features in summary',
                        default=None)

    parser.add_argument('-j', '--numjobs',
                        type=int,
                        help='number of jobs for Random Forest; default: all reasonably possible resources',
                        default=-2)

    args = parser.parse_args()

    analysis = CombinedAnalysis(
        args.infile, 
        col_names=args.cols if not args.autocols else RELEVANT_COLS,
        n_jobs=args.numjobs
        )
    analysis.run()

    summary = analysis.summary(top_n=args.top_n)
    posthocs = PostHocTests(
                df_info = args.infile,
                measure_cols = RELEVANT_COLS,
                cluster_col = 'cluster',
                importance_tier = ImportanceTier.LARGE,
                summary_df_info = summary,
                alpha = 0.05,
                correction_method = DunnCorrectionMethods.FDR_BH
    )
    posthoc_res = posthocs.run()

    if args.print:
        print('=================================================')
        print('    Summary of Overall Cluster Relevance Tests')
        print('=================================================')
        print(summary.to_string())

        print('=================================================')
        print('    Summary of Measure Cluster Relevance')
        print('=================================================')
        posthocs.print_analysis_summary(posthoc_res)

    if args.outfile:
        analysis.save(summary, args.outdir, args.force)
        posthocs.save(posthoc_res, args.outdir, args.force)