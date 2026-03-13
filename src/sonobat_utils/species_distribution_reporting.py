#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-13 10:08:55
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/species_distribution_reporting.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-13 12:26:46
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

from sonobat_utils.utils import Utils
from result_analysis.charting import Charter

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

# ------------------------- Main ------------------------
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Report distribution of species in measures df"
                                     )

    parser.add_argument('input',
                        help='path to .csv or .feather measures file',
                        default=None)

    args = parser.parse_args()
    try:
        print(f"Reading df from {args.input}...")
        df = Utils.read_df_file(args.input)
    except Exception as e:
        print(f"Cannot read df file from {args.input}: {e}")
    
    SpeciesDistribReporter(df)

if __name__ == "__main__":
    main()
