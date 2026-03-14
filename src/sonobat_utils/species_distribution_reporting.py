#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-13 10:08:55
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/species_distribution_reporting.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-13 16:42:35
#
# **********************************************************

# Given a .csv/.feather file with measures that have the per sample
# species column available, determine, print, and display the 
# distribution of species.
# Expected columns:
#   species,species_prob,species_2nd

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

from sonobat_utils.utils import Utils
from logging_service import LoggingService

log = LoggingService()

from sonobat_utils.utils import Utils
#from result_analysis.charting import Charter

class SpeciesDistribReporter:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, df: pd.DataFrame):
        self.compute_distrib(df)

    #------------------------------------
    # compute_distrib
    #-------------------

    def compute_distrib(self, df: pd.DataFrame) -> pd.Series:
        confident_idents = df[df['species_prob'] >= 0.98]
        n = len(confident_idents)
        grp = confident_idents.groupby('species')
        print(f"Raw counts:\n{grp.size()}")
        print(f"Percentages: \n{100 * grp.size() / n}")

# -------------------- Class ChirpSeqSpeciesPurityReporter --------------
class ChirpSeqSpeciesPurityReporter:
    """
    Examines how often all of the chirps inside a 
    measures df were determined to be the same species.
    A 'Purity' value of 1.0 for one file_id (sequence) is perfect: 
    all chirps in that sequence were deemed to be the same
    species
    """
    def __init__(self, df: pd.DataFrame, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.df = df
        self.metrics = None

    def compute_purity(self):
        log.info("Computing purity metrics per file_id...")
        
        # Group by file_id and species to get counts
        counts = self.df.groupby(['file_id', 'species']).size().reset_index(name='count')
        
        # Calculate total rows per file_id
        totals = counts.groupby('file_id')['count'].transform('sum')
        counts['proportion'] = counts['count'] / totals
        
        # Identify the primary (majority) species and its purity
        self.metrics = counts.sort_values('count', ascending=False).drop_duplicates('file_id')
        self.metrics = self.metrics.rename(columns={'species': 'primary_species', 'proportion': 'purity'})
        
        # Add a flag for 'Imperfect' groups
        self.metrics['is_pure'] = self.metrics['purity'] == 1.0
        
        purity_rate = self.metrics['is_pure'].mean() * 100
        log.info(f"Analysis complete. Dataset Purity: {purity_rate:.2f}% of files are homogenous.")

    def save_reports(self):
        # Save CSV of problematic files (purity < 1.0)
        imperfect_path = self.out_dir / "imperfect_files.csv"
        imperfect_files = self.metrics[self.metrics['purity'] < 1.0].sort_values('purity')
        imperfect_files.to_csv(imperfect_path, index=False)
        log.info(f"Saved list of imperfect groups to {imperfect_path}")

    def plot_visuals(self):
        log.info("Generating diagnostic charts...")
        
        # Chart 1: Purity Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.metrics['purity'], bins=20, kde=False, color='skyblue')
        plt.title('Distribution of File Purity (Ideal = 1.0)')
        plt.xlabel('Purity (Majority Species Count / Total Group Count)')
        plt.ylabel('Number of File Groups')
        plt.yscale('log')  # Log scale useful for 471k groups
        plt.savefig(self.out_dir / "purity_distribution.png")
        
        # Chart 2: Purity vs Probability (Are errors high-confidence?)
        # Merging average probability back to see if "messy" files have lower confidence
        avg_prob = self.df.groupby('file_id')['species_prob'].mean()
        plot_df = self.metrics.set_index('file_id').join(avg_prob)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=plot_df, x='purity', y='species_prob', alpha=0.1)
        plt.title('Purity vs. Average Species Probability')
        plt.savefig(self.out_dir / "purity_vs_prob.png")
        
        plt.close('all')
        log.info(f"Charts saved to output directory {self.out_dir}")

    def run(self):
        self.compute_purity()
        self.save_reports()
        self.plot_visuals()


# ------------------------- Main ------------------------
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Report distribution of species in measures df"
                                     )

    parser.add_argument('input',
                        help='path to .csv or .feather measures file',
                        default=None)
    parser.add_argument("--o", "--out-dir", dest="out_dir", default="./reports", help="Output directory")

    args = parser.parse_args()
    try:
        print(f"Reading df from {args.input}...")
        df = Utils.read_df_file(args.input)
        log.info(f"Loaded {len(df)} rows with {df['file_id'].nunique()} unique file_ids.")
    except Exception as e:
        print(f"Cannot read df file from {args.input}: {e}")

    # Simple list of species percentages:
    SpeciesDistribReporter(df)
    
    try:
        reporter = ChirpSeqSpeciesPurityReporter(df, args.out_dir)
        reporter.run()
    except Exception as e:
        log.err(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
