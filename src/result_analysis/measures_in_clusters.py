# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-19 18:33:23
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-19 19:50:40
"""
Given each measure's normality in each cluster in file bats_measures_normality_all.csv,
and the all-measures cluster assignments, examine whether any values are particularly important
for cluster membership.
"""

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from scipy import stats


class KruskalWallisTester:

    #------------------------------------
    # Constructor
    #-------------------

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

    def __init__(
        self,
        infile: str | Path,
        col_names: list[str] | None = None,
        cluster_id_col: str = 'cluster'
    ):
        df = self.read_df_file(infile)

        if col_names is not None:
            self.col_names = col_names
        else:
            self.col_names = list(df.columns.drop(cluster_id_col))

        self.df = df[self.col_names].copy()
        self.cluster_labels = df[cluster_id_col]
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
        vals = self.df[col].dropna().unique()
        return len(vals) <= 2

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
        groups = [
            self.df.loc[self.cluster_labels == label, col].dropna().values
            for label in np.unique(self.cluster_labels)
        ]
        # Drop any empty groups (e.g. NaN-heavy clusters)
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
        contingency = pd.crosstab(self.df[col], self.cluster_labels)
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        return {
            "H_statistic": chi2,   # unified column name for downstream compatibility
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

    #------------------------------------
    # read_df_file
    #-------------------

    def read_df_file(self, fpath: str | Path):
        """
        Read a DataFrame from a .csv or .feather file.

        :param fpath: Path to the file.
        :return: DataFrame.
        """
        fpath = Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"File {fpath} not found")
        if fpath.suffix == '.feather':
            df = pd.read_feather(fpath)
        elif fpath.suffix == '.csv':
            df = pd.read_csv(fpath)
        return df


# --------------------- Main -------------

if __name__ == "__main__":
    desc = '''Computes importance of bat measures for cluster membership '''
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=desc
                                     )
    parser.add_argument('infile',
                        help='path to .csv or .feather file')

    parser.add_argument('-c', '--cols',
                        type=str,
                        nargs='+',
                        help='Repeatable: columns to include; default: all',
                        default=None
                        )
    parser.add_argument('-o', '--outfile',
                        type=str,
                        help='outfile for result',
                        default=None
                        )

    args = parser.parse_args()
    kw_test = KruskalWallisTester(args.infile, args.cols)
    kw_test.run()
    print(kw_test.summary(top_n=10))