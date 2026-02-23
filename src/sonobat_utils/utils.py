# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-20 08:53:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-22 19:05:39

# =====================================================================
# Class Utilities
# =====================================================================

from pathlib import Path
import pickle

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
import pandas as pd


class Utils:
    """
    Shared utility methods for data loading and result persistence.
    """

    #------------------------------------
    # read_df_file
    #-------------------

    @staticmethod
    def read_df_file(fpath: str | Path) -> pd.DataFrame:
        """
        Read a DataFrame from a .csv or .feather file.

        :param fpath: Path to the file.
        :return: DataFrame.
        """
        if not isinstance(fpath, Path) and not isinstance(fpath, str):
            raise TypeError(f"Path must be str or Path, not {fpath}")
        
        fpath = Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"File {fpath} not found")
        if fpath.suffix == '.feather':
            df = pd.read_feather(fpath)
        elif fpath.suffix == '.csv':
            df = pd.read_csv(fpath)
            # If df was saved as df.to_csv() without
            # an addtional index=False arg, we'll have
            # a first column named 'Unnamed: 0'. Remove
            # that if it exists:
            if df.columns[0] == 'Unnamed: 0':
                df.drop(columns=['Unnamed: 0'], inplace=True)
        else:
            raise ValueError(f"Unsupported file type: {fpath.suffix}. Use .csv or .feather.")

        return df

    #------------------------------------
    # extract_cols_safely
    #-------------------

    @staticmethod
    def extract_cols_safely(df: pd.DataFrame, cols: str | list[str]) -> tuple[pd.DataFrame, list[str]]:
        '''
        Copies the given columns from the given df.
        If any of the given columns do not exist in the
        df, still returns the df with the cols that are
        available. 

        Returns a tuple with the df copy and a list of
        columns that are not included.

        :param df: dataframe from which to extract
        :param cols: column(s) to extract
        :return: tuple with df extract and cols that were not present
        '''
        if type(cols) == str:
            cols = [cols]
        available_cols = set(df.columns)
        missing_cols   = set(cols) - available_cols
        
        cols_to_copy   = [col for col in cols if col in available_cols]

        res_df = df[cols_to_copy].copy()
        return (res_df, missing_cols)


    #------------------------------------
    # write_outfile
    #-------------------

    @staticmethod
    def write_df_outfile(df: pd.DataFrame, 
                         outfile: str | Path,
                         force: bool = False
                         ) -> None:
        '''
        Writes dataframe to file in either .csv or .feather format.
        If force is False (default), asks user for confirmation and
        alternative outfile if outfile already exists. 

        Suffix ouf outfile determines output format. If no
        suffix, .csv is assumed. Only .csv and .feather are supported.

        :param df: dataframe to save
        :param outfile: where to save
        :param force: whether to overwrite, defaults to False
        '''

        outfile = Path(outfile)
        if outfile.suffix == '':
            outfile = outfile.with_suffix('.csv')

        if outfile.suffix not in ['.csv', '.feather']:
            raise NotImplementedError(f"Only .csv and .feather are supported for writing, not {outfile}")

        while True:
            if not outfile.exists() or force:
                # Outfile does not already exist, or
                # we may overwrite; all good
                break

            resp = input(f"File '{outfile}' exists; overwrite/new path/cancel (o/p/c): ").lower()

            if resp == 'o':
                print(f"Overwriting {outfile}...")
                break  # Exit loop and use current outfile
            
            elif resp == 'p':
                new_path = input("Enter new file path: ")
                outfile = Path(new_path)
                # Loop restarts to check if the NEW path also exists
            
            elif resp == 'c':
                print("Not saving result.")
                return
            
            else:
                print("Invalid input. Please enter 'o', 'p', or 'c'.")            

        if outfile.suffix == '.csv':
            df.to_csv(outfile, index=False)
        elif outfile.suffix == '.feather':
            df.to_feather(outfile)

    #------------------------------------
    # write_scaler_outfile
    #-------------------

    @staticmethod
    def write_scaler_outfile(scaler, outfile: str | Path, derive_fname: bool = False):
        '''
        Write an sklearn scaler to a file. If derive_fname, 
        then the file will be called:

            <outfile-stem>_scaler.pks

        else the the file will be pickled to the 
        given file.

        :param scaler: the scaler object to write
        :param outfile: outfile 
        :param derive_fname: whether or not to create
           the output file from the given outfile.
        '''
        if derive_fname:
            scaler_method = scaler.__class__.__name__
            scaler_name = f"{outfile.stem}_scaler_{scaler_method}.pks"
            scaler_path = outfile.with_name(scaler_name)
        else:
            scaler_path = outfile
        with open(scaler_path, 'wb') as fd:
            pickle.dump(scaler, fd)


    #------------------------------------
    # right_size_fontsizes
    #-------------------

    @staticmethod
    def right_size_fontsizes():
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

    #------------------------------------
    # add_is_first_last
    #-------------------              

    @staticmethod
    def add_is_first_last(df_raw:pd.DataFrame) -> pd.DataFrame:
        '''
        Checks whether the columns is_first and is_last
        are present. Adds them if not. The df is assumed
        to be a full bats measures df that includes the
        chirp sequence grouping column file_id, and the 
        index of each chirp within its sequence: chirp_idx.

        The new columns are bools indicating whether the
        respective chirp is the first or last in its sequence.

        :param df_raw: df to be augmented
        :return: augmented df
        '''
        if 'is_first' not in df_raw.columns:
            df_raw['is_first'] = df_raw['chirp_idx'] == 0

        if 'is_last' not in df_raw.columns:
            # 1. Group by the file_id
            # 2. Transform the chirp_idx to find the max within each group
            # 3. Compare that max to the original chirp_idx
            df_raw['is_last'] = df_raw.groupby('file_id')['chirp_idx'].transform('max') == df_raw['chirp_idx']

        return df_raw