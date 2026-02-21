# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-20 08:53:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-21 12:41:40

# =====================================================================
# Class Utilities
# =====================================================================

from pathlib import Path

from matplotlib import pyplot as plt
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
    def write_outfile(df: pd.DataFrame, 
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
        