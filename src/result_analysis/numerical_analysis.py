# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2025-10-29 09:53:54
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/result_analysis/numerical_analysis.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2025-11-20 09:09:50
#
# **********************************************************

'''
Given chirp test data that was fed into a BatChat model, and
the model's prediction outputs, compute accuracy values. Facilities
for measure-by-measure accuracy, overall accuracy, and multi-model
accuracy are provided.
'''

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import List
import pyarrow.feather as feather
import pyarrow as pa
import pandas as pd



class PredictionAnalyzer:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, input_chirp_file, predictions_file=None):
        self.in_chirps = self.read_file(input_chirp_file)
        if predictions_file is not None:
            self.pred_chirps = self.read_file(predictions_file)
        else:
            self.pred_chirps = None


    #------------------------------------
    # verify_residual_err0
    #-------------------

    def verify_residual_err0(self, df):
        df_pred = self.make_perfect_prediction(df)
        res_err = self.residual_error(df, df_pred)

        numeric_cols = res_err.select_dtypes(include='number').columns

        assert(res_err.iloc[:-1][numeric_cols] ==0 ).all().all()

        return res_err

    #------------------------------------
    # residual_error
    #-------------------    

    def residual_error(self, df_orig, df_pred):

        # Collect the names of cols whose values are all numeric:
        # Check if there are any non-numeric values in "numeric" columns
        numeric_cols = df_orig.select_dtypes(include='number').columns        

        # Subtract the original from the predictions for
        # numeric values:
        # Must reset index of second df so that it matches the
        # df from which we are subtracting: 0,1,2
        res_err = df_pred[numeric_cols] - df_orig.iloc[1:][numeric_cols].reset_index(drop=True)

        # Add back the SHIFTED non-numeric columns:
        non_numerics = set(df_orig.columns) - set(numeric_cols)
        for col in non_numerics:
            res_err[col] = df_pred[col].values

        return res_err

    #------------------------------------
    # make_perfect_prediction
    #-------------------

    def make_perfect_prediction(self, df):
        # The shift op does make a copy, but df
        # is passed by reference, so for other ops
        # this would be needed:
        df = df.copy()
        predicted_df = df.shift(-1)
        return predicted_df

    #------------------------------------
    # read_file
    #-------------------

    def read_file(self, path):
        path_obj = Path(path)
        if path_obj.suffix == '.feather':
            df = feather.read_feather(path)
        elif path_obj.suffix == '.csv':
            df = pd.read_csv(path)
        
        return df

# ---------------------- main() ----------------
def main():

    desc = ("Takes a set of input vectors, a set of output predictions \n"
            "for row n+1, where n is the nth input."
    )
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Here is what this package does."
                                     )

    parser.add_argument('-l', '--errLogFile',
                        help=('fully qualified log file name to which info and error \n'
                              'messages are directed. Default: stdout.'),
                        dest='errLogFile',
                        default=None)
    parser.add_argument('-d', '--dryRun',
                        help=('show what script would do if run normally; \n'
                              'no actual downloads or other changes are performed.'),
                        action='store_true')
    parser.add_argument('my_integers',
                        type=int,
                        nargs='+',
                        help='Repeatable: integers. Will show as list in my_integers')

    args = parser.parse_args()


    cur_dir = Path(__file__).parent
    in_data = cur_dir.joinpath('data/scaled_chirps_2024-06-25T12_55_03.feather')

    analyzer = PredictionAnalyzer(in_data)
    res_err  = analyzer.verify_residual_err0(analyzer.in_chirps)

# ---------------------- Main ----------------

if __name__ == "__main__":
    main()
