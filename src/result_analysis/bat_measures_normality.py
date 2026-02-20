#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-17 19:03:14
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-19 09:01:51

import argparse
import os
from pathlib import Path
import sys
from matplotlib.widgets import Button
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from logging_service.logging_service import LoggingService

PHYSICAL_MEASURES = [
    'TimeInFile',
    'PrecedingIntrvl',
    'HiFreq',
    'Bndwdth',
    'FreqMaxPwr',
    'PrcntMaxAmpDur',
    'FreqKnee',
    'PrcntKneeDur',
    'StartF',
    'UpprKnFreq',
    'HiFtoUpprKnAmp',
    'HiFtoKnAmp',
    'HiFtoFcAmp',
    'UpprKnToKnAmp',
    'KnToFcAmp',
    'LdgToFcAmp',
    'FreqCtr',
    'FFwd32dB',
    'FFwd20dB',
    'FFwd15dB',
    'FBak5dB',
    'FFwd5dB',
    'Bndw32dB',
    '1st10kHzSlp',
    '1st5to15kHzSlp',
    '1st10kHzExp',
    '1st5to15kHzExp'
    ]   


# ------------------------ Class NormalityChecker -------------
class NormalityChecker:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, 
                 df_info: pd.DataFrame | str, 
                 tests: list[str], 
                 by_cluster: bool,
                 all_numerics: bool,
                 cols: str | list[str],
                 outfile: str | None
                 ):
        '''
        Given a dataframe or file to .feather or .csv that can load
        as a df. Also given one or more column names. Determine the
        degree of normality in each column.

        Test options are:
           o 'vizcurve': binned histogram of values plus normal curve overlaid
           o 'vizqq'   : Quantile-Quantile (QQ) plot
           o 'shapiro' : Shapiro-Wilk test

        The QQ plot: the closer points to diagonal, the more normal.

        The result, apart from the interim charts that one can save
        when they appear is a multi-index df like:
                               test     normal
             measure cluster                  
             HiFreq  0        vizcurve    True
             Bndwdth 0        vizcurve    True
             HiFreq  1        vizcurve   False        

        :param df: dataframe, or path to it
        :param tests: the type of tests to perform
        :param by_cluster: whether to test each measure across all clusters,
            or separately for each cluster.
        :param all_numerics: whether to run tests on all the PHYSICAL_MEASURES
        :param cols: column(s) to analyze
        '''

        self.log = LoggingService()

        if type(df_info) == str:
            df = self.read_df(df_info)
        elif isinstance(df_info, pd.DataFrame):
            df = df_info
        else:
            raise TypeError(f"Df info must be a df or a file path str, not {df_info}")
        
        if all_numerics:
            cols = PHYSICAL_MEASURES
        elif cols is not None:
            if type(cols) == str:
                cols = [cols]
            # Detect early if a col doesn't exist in df
            all_cols = df.columns
            for col in cols:
                if col not in all_cols:
                    raise ValueError(f"Column {col} is not in df")
        else:
            # Run over all columns
            cols = df.columns                

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

        if by_cluster:
            cluster_grp = df.groupby('cluster')
            cluster_result_dfs = []
            for cluster_id, df_slice in cluster_grp:
                cluster_result_dfs.append(
                    self.run_tests(df_slice,tests, cols, f"cluster {cluster_id}"))
            self.res_df = pd.concat(cluster_result_dfs)
        else:
            self.res_df = self.run_tests(df, tests, cols, 'all_clusters')

        # The result has a multi-level index
        self.res_df.index.names = ['measure', 'cluster']

        if outfile is not None:
            self.write_outfile(self.res_df, outfile)
            
        print(self.res_df)

    #------------------------------------
    # run_tests
    #-------------------

    def run_tests(self, 
                  df_slice: pd.DataFrame, 
                  tests: list[str],
                  cols: list[str],
                  cluster_id: int | str) -> pd.DataFrame:

        tst_res = {}
        for test in tests:
            if test == 'vizcurve':
                self.log.info("Starting normality histogram viz...")

                # Show each histogram, and ask user for judgement: yes/no:
                for col_name, col_vals in df_slice[cols].items():

                    self.log.info(f"Histogram for column {col_name}")
                    
                    col_passed = self.normality_histogram(col_vals, cluster_id=cluster_id)
                    if col_passed:
                        tst_res[(col_name, cluster_id)] = {'test' : test, 'normal': True, 'n' : len(col_vals)}
                    else:
                        tst_res[(col_name, cluster_id)] = {'test' : test, 'normal': False, 'n' : len(col_vals)}

                res_df = pd.DataFrame.from_dict(tst_res, orient='index')

            elif test == 'shapiro':
                res_df = self.shapiro_wilk(df_slice, cols, cluster_id=cluster_id)

        return res_df


    #------------------------------------
    # normality_histogram
    #-------------------

    def normality_histogram(self, data: pd.Series, cluster_id: str = None):
        
        is_normal: bool

        col_name = data.name
        # 'kde=True' draws the smooth density line over the bars
        sns.histplot(data, kde=True, color='skyblue', bins=30)

        if cluster_id is None:
            title = f"Is this {col_name} distribution normal?" 
        else:
            title = f"Is this {col_name} distribution in {cluster_id} normal?" 
        plt.title(title)
        
        # Room for "Is this normal" buttons
        # Move the plot up and shrink it slightly to make room for big labels
        plt.subplots_adjust(bottom=0.35, top=0.85)        

        # Place Yes/No buttons relative to that new bottom
        # [left, bottom, width, height]
        ax_yes = plt.axes([0.3, 0.08, 0.15, 0.1]) 
        ax_no  = plt.axes([0.5, 0.08, 0.15, 0.1])
        
        self.btn_yes = Button(ax_yes, 'Yes', color='lightgreen', hovercolor='green')
        self.btn_no = Button(ax_no, 'No', color='tomato', hovercolor='red')
        
        # Action when clicking the Yes button:
        def set_yes(event):
            nonlocal is_normal
            is_normal = True
            plt.close() # Close the window once decided

        # Action when clicking the No button:
        def set_no(event):
            nonlocal is_normal
            is_normal = False
            plt.close()

        # 4. Define what happens on click
        self.btn_yes.on_clicked(set_yes)
        self.btn_no.on_clicked(set_no)        

        plt.show(block=True)
        return is_normal
            
    #------------------------------------
    # shapiro_wilk
    #-------------------     

    def shapiro_wilk(self, 
                     df: pd.DataFrame, 
                     measure_cols: list[str],
                     cluster_id: str = None
                     ) -> pd.DataFrame:
        '''
        Selects each measure_cols from df in turn, and
        applies a shapiro-wilk test to the selected pd.Series. 
        Returns results as a df with one row for each column.
        Information in the return multi-index df: 

             (measure_name, cluster_id), W_statistic, p_value, is_normal

        The is_normal bool is True if p_value >= 0.05: the hypothesis
        that the measure is NOT normal is rejected in that case.

        :param df: the data
        :param measure_cols: the columns to test for normality
        :param cluster_id: not used for grouping; just to 
            identify the cluster in the returned df
        :return: dataframe with the result columns
        '''
        result_dict = {}
        measure: pd.Series
        for measure_nm, measure in df[measure_cols].items():
            # Examine measure overall, not per cluster
            stat, pval = shapiro(measure)
            result_dict[(measure_nm, cluster_id)] = {
                'W_statistic': float(stat),
                'p_value': float(pval),
                'n': len(df)
            }
            # Determine normality: depends on the sample size:
            if stat >= 0.95:
                # No matter the sample size: normal
                result_dict[(measure_nm, cluster_id)]['normal'] = True 
                continue
            # With lower Shapiro-Wilk W: sample size comes in:
            n = len(df)
            # Small n:
            if n < 50 and pval > 0.05 and stat > 0.90:
                result_dict[(measure_nm, cluster_id)]['normal'] = True 
                continue
            if n in range(50,500):
                # p means probability that data came from normal distribution:
                if pval < 0.05:
                    result_dict[(measure_nm, cluster_id)]['normal'] = False
                else:
                    result_dict[(measure_nm, cluster_id)]['normal'] = True
                continue
            if n >= 500:
                # pval is hyper sensitive:
                if stat < 0.90 and pval < 0.001:
                    result_dict[(measure_nm, cluster_id)]['normal'] = False
                else:
                    result_dict[(measure_nm, cluster_id)]['normal'] = True
                continue

        results_df = pd.DataFrame.from_dict(result_dict, orient='index')
        return results_df

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
        
    #------------------------------------
    # write_outfile
    #-------------------

    def write_outfile(self, df: pd.DataFrame, outfile: str):

        while True:
            if not os.path.exists(outfile):
                # Outfile does not already exist; all good
                break

            resp = input(f"File '{outfile}' exists; overwrite/new path/cancel (o/p/c): ").lower()

            if resp == 'o':
                print(f"Overwriting {outfile}...")
                break  # Exit loop and use current outfile
            
            elif resp == 'p':
                new_path = input("Enter new file path: ")
                outfile = new_path
                # Loop restarts to check if the NEW path also exists
            
            elif resp == 'c':
                print("Not saving result.")
                return
            
            else:
                print("Invalid input. Please enter 'o', 'p', or 'c'.")            

        outpath = Path(outfile)
        if outpath.suffix == '.csv':
            df.to_csv(outfile)
        elif outpath.suffix == '.feather':
            df.to_feather(outfile)
        elif outpath.suffix == '':
            outpath_default = outpath.with_suffix('.csv')
            print(f"Writing to {outpath_default}")
            df.to_csv(outpath_default)


#----------- main --------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Check normality of given dataframe columns."
                                     )
    parser.add_argument('infile',
                        help='path to data',
                        )

    parser.add_argument('tests',
                        choices=['vizcurve', 'vizqq', 'shapiro'],
                        nargs='+',
                        help='Repeatable: normal-curve, QQ-chart, shapiro-wilk numeric test')

    parser.add_argument('-l', '--clustered',
                        action='store_true',
                        default=False,
                        help='whether to run tests separately for each cluster')

    parser.add_argument('-n', '--numerics',
                        action='store_true',
                        default=False,
                        help='analyze for all numeric columns; --cols ignored if this option is set')

    parser.add_argument('-c', '--cols',
                        type=str,
                        nargs='+',
                        default=None,
                        help='Repeatable: columnns to check; if omitted, all-numerics or all, if not all-numerics')

    parser.add_argument('-o', '--outfile',
                        default=None,
                        help='optional outfile; options are .csv and .feather',
                        )

    args = parser.parse_args()

    NormalityChecker(args.infile,
                     args.tests,
                     args.clustered,
                     args.numerics,
                     args.cols,
                     args.outfile
                     )

