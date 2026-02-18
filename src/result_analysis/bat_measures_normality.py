# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-17 19:03:14
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-17 19:38:23

import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class NormalityChecker:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, df_info: pd.DataFrame | str, cols: str | list[str]):
        '''
        Given a dataframe or file to .feather or .csv that can load
        as a df. Also given one or more column names. Determine the
        degree of normality in each column.

        :param df: dataframe, or path to it
        :param cols: column(s) to analyze
        '''

        if type(df_info) == str:
            df = self.read_df(df_info)
        elif isinstance(df_info, pd.DataFrame):
            df = df_info
        else:
            raise TypeError(f"Df info must be a df or a file path str, not {df_info}")
        
        if type(cols) != list:
            cols = [cols]

        # Adjust font sizes:
        # Set global sizes
        plt.rcParams.update({
            'font.size': 18,          # Base font size
            'axes.titlesize': 24,     # Title size
            'axes.labelsize': 20,     # X and Y label size
            'xtick.labelsize': 16,    # X tick label size
            'ytick.labelsize': 16,    # Y tick label size
            'legend.fontsize': 16,    # Legend size
            'lines.linewidth': 3,     # Thicker lines for visibility
            'lines.markersize': 10    # Larger markers
        })        

        passed = []
        failed = []
        for col_name, col_vals in df.items():
            if col_name not in cols:
                continue
            col_passed = self.analyze_normality(col_vals)
            if col_passed:
                passed.append(col_name)
            else:
                failed.append(col_name)

        print(f"Passed: \n{passed}\nFailed:\n{failed}")
                
    #------------------------------------
    # analyze_normality
    #-------------------

    def analyze_normality(self, data: pd.Series):
        
        col_name = data.name
        # 'kde=True' draws the smooth density line over the bars
        sns.histplot(data, kde=True, color='skyblue', bins=30)

        plt.title(f"Distribution of {col_name} with Kernel Density Estimate")
        plt.show(block=True)

        is_normal = input(f"Normal distribution (y/n)? ")
        return is_normal == 'y'
            

    #------------------------------------
    # read_df
    #-------------------        

    def read_df(self, fpath: Path | str) -> pd.DataFrame:
        '''
        Given a path to a supposed dataframe on disk,
        load and return the df. Reads .feather or .csv 

        :param fpath: path to data
        :raises FileNotFoundError: if file not found
        :raises TypeError: if file not .feather or .csv
        :return: the loaded df
        '''
        ppath = Path(fpath)
        if not ppath.exists():
            raise FileNotFoundError(f"File {fpath} not found")
        if ppath.suffix == '.feather':
            df = pd.read_feather(ppath)
        elif ppath.suffix == '.csv':
            df = pd.read_csv(ppath)
        else:
            raise TypeError(f"Input file must be .feather or .csv, not {fpath}")
        return df
        
#----------- main --------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Check normality of given dataframe columns."
                                     )
    parser.add_argument('infile',
                        help='path to data',
                        )
    
    parser.add_argument('-c', '--cols',
                        type=str,
                        nargs='+',
                        help='Repeatable: columnns to check; all if omitted')

    args = parser.parse_args()

    NormalityChecker(args.infile, 
                     args.cols
                     )
