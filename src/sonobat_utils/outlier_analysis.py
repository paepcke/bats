# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-23 12:22:14
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-13 10:08:22

# Detect outliers in measurements.

import sys

import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from sonobat_utils.utils import Utils

@dataclass
class OutlierAnalysis:
    """
    Container for outlier analysis results.
    
    :param column: Column name
    :param outlier_values: Array of outlier values
    :param outlier_indices: Indices of outliers in original dataframe
    :param mean: Column mean
    :param std: Column standard deviation
    :param outlier_std: Standard deviation of outlier values
    :param outlier_cv: Coefficient of variation of outliers (std/mean)
    :param n_outliers: Number of outliers found
    :param is_artifact_suspect: Whether outliers show artifact-like behavior
    """
    column: str
    outlier_values: np.ndarray
    outlier_indices: np.ndarray
    mean: float
    std: float
    outlier_std: float
    outlier_cv: float
    n_outliers: int
    is_artifact_suspect: bool


class OutlierDetector:
    """
    Detect potential equipment artifacts in bat chirp measurements.
    
    Identifies outliers that show suspiciously low variation, which may
    indicate recording equipment artifacts rather than biological variation.
    """
    
    def __init__(
        self,
        sd_threshold: float = 2.0,
        cv_threshold: float = 0.05,
        min_outliers: int = 3
    ):
        """
        Initialize the outlier detector.
        
        :param sd_threshold: Number of standard deviations to define outliers
        :param cv_threshold: Coefficient of variation threshold below which outliers
                             are considered suspiciously uniform (default 0.05 = 5%)
        :param min_outliers: Minimum number of outliers required to assess variation
        """
        self.sd_threshold = sd_threshold
        self.cv_threshold = cv_threshold
        self.min_outliers = min_outliers
        self.results: Optional[Dict[str, OutlierAnalysis]] = None
    
    def detect(self, df: pd.DataFrame) -> Dict[str, OutlierAnalysis]:
        """
        Detect potential equipment artifacts in the dataframe.
        
        :param df: DataFrame with bat chirp measurements
        :return: Dictionary mapping column names to OutlierAnalysis objects
        """
        results = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            # Skip if all NaN
            if df[column].isna().all():
                continue
                
            # Calculate statistics
            col_data = df[column].dropna()
            mean = col_data.mean()
            std = col_data.std()
            
            if std == 0:
                continue
            
            # Identify outliers (more than sd_threshold SDs from mean)
            z_scores = np.abs((col_data - mean) / std)
            outlier_mask = z_scores > self.sd_threshold
            
            if outlier_mask.sum() < self.min_outliers:
                continue
                
            outlier_values = col_data[outlier_mask].values
            outlier_indices = col_data[outlier_mask].index.values
            
            # Calculate variation within outliers
            outlier_std = np.std(outlier_values)
            outlier_mean = np.mean(outlier_values)
            
            # Coefficient of variation (CV) - normalized measure of variation
            # Use absolute mean to handle negative values
            if outlier_mean != 0:
                outlier_cv = outlier_std / np.abs(outlier_mean)
            else:
                outlier_cv = np.inf if outlier_std > 0 else 0
            
            # Flag as artifact suspect if variation is too low
            is_artifact_suspect = (
                outlier_cv < self.cv_threshold and 
                np.isfinite(outlier_cv)
            )
            
            results[column] = OutlierAnalysis(
                column=column,
                outlier_values=outlier_values,
                outlier_indices=outlier_indices,
                mean=mean,
                std=std,
                outlier_std=outlier_std,
                outlier_cv=outlier_cv,
                n_outliers=len(outlier_values),
                is_artifact_suspect=is_artifact_suspect
            )
        
        self.results = results
        return results
    
    def print_report(self, show_all: bool = False) -> None:
        """
        Print a summary report of potential artifacts.
        
        :param show_all: If True, show all columns with outliers; if False, only show suspects
        :return: None
        """
        if self.results is None:
            print("No analysis has been run yet. Call detect() first.")
            return
        
        suspects = {k: v for k, v in self.results.items() if v.is_artifact_suspect}
        
        if suspects:
            print("=" * 80)
            print("POTENTIAL EQUIPMENT ARTIFACTS DETECTED")
            print("=" * 80)
            
            for column, analysis in suspects.items():
                print(f"\n{column}:")
                print(f"  Column mean: {analysis.mean:.4f}, std: {analysis.std:.4f}")
                print(f"  Outliers found: {analysis.n_outliers}")
                print(f"  Outlier range: [{analysis.outlier_values.min():.4f}, "
                      f"{analysis.outlier_values.max():.4f}]")
                print(f"  Outlier std: {analysis.outlier_std:.4f}")
                print(f"  Outlier CV: {analysis.outlier_cv:.4f} ⚠️  (suspiciously low)")
                print(f"  Sample values: {analysis.outlier_values[:5]}")
        else:
            print("No potential equipment artifacts detected.")
        
        if show_all:
            normal = {k: v for k, v in self.results.items() if not v.is_artifact_suspect}
            if normal:
                print("\n" + "=" * 80)
                print("OUTLIERS WITH NORMAL BIOLOGICAL VARIATION")
                print("=" * 80)
                
                for column, analysis in normal.items():
                    print(f"\n{column}:")
                    print(f"  Outliers found: {analysis.n_outliers}")
                    print(f"  Outlier CV: {analysis.outlier_cv:.4f} ✓ (normal variation)")
    
    def get_suspect_columns(self) -> List[str]:
        """
        Get list of columns with suspected artifacts.
        
        :return: List of column names with suspected artifacts
        """
        if self.results is None:
            return []
        return [col for col, analysis in self.results.items() if analysis.is_artifact_suspect]
    
    def get_suspect_indices(self, column: str) -> Optional[np.ndarray]:
        """
        Get indices of suspected artifacts for a specific column.
        
        :param column: Column name
        :return: Array of indices or None if column not found or not suspect
        """
        if self.results is None or column not in self.results:
            return None
        
        analysis = self.results[column]
        if analysis.is_artifact_suspect:
            return analysis.outlier_indices
        return None

def main():
    """
    CLI entry point for outlier detection.
    
    :return: None
    """
    parser = argparse.ArgumentParser(
        description="Detect potential equipment artifacts in bat chirp measurements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'infile',
        type=str,
        help='Path to input data file (CSV, Feather, Parquet, Excel, or Pickle)'
    )
    
    parser.add_argument(
        '--sd-threshold',
        type=float,
        default=2.0,
        help='Number of standard deviations to define outliers'
    )
    
    parser.add_argument(
        '--cv-threshold',
        type=float,
        default=0.05,
        help='Coefficient of variation threshold for artifact detection (e.g., 0.05 = 5%%)'
    )
    
    parser.add_argument(
        '--min-outliers',
        type=int,
        default=3,
        help='Minimum number of outliers required to assess variation'
    )

    parser.add_argument(
        '-i', '--ignore',
        type=str,
        nargs='+',
        help='repeatable: columns to ignore'
    )

    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all columns with outliers, not just suspected artifacts'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.infile}")
    df = Utils.read_df_file(args.infile)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

    ignored_cols = args.ignore
    if len(ignored_cols) > 0:
        # Ensure that all cols are present in the df:
        for col in ignored_cols:
            if col not in df.columns:
                print(f"Column {col} not in the df, but listed in cols to ignore")
                sys.exit(1)

        df = df.drop(columns=ignored_cols)
    
    # Create detector and run analysis
    detector = OutlierDetector(
        sd_threshold=args.sd_threshold,
        cv_threshold=args.cv_threshold,
        min_outliers=args.min_outliers
    )
    
    print(f"Running outlier detection with parameters:")
    print(f"  SD threshold: {args.sd_threshold}")
    print(f"  CV threshold: {args.cv_threshold}")
    print(f"  Min outliers: {args.min_outliers}\n")
    
    detector.detect(df)
    detector.print_report(show_all=args.show_all)
    
    # Print summary
    suspect_cols = detector.get_suspect_columns()
    if suspect_cols:
        print(f"\n\nSummary: {len(suspect_cols)} column(s) with suspected artifacts:")
        for col in suspect_cols:
            indices = detector.get_suspect_indices(col)
            print(f"  - {col}: {len(indices)} suspect data points")

if __name__ == "__main__":
    main()
    