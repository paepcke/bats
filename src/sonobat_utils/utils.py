# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-20 08:53:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-08 18:57:10

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

    PARQUET_FILES_THRIFT_LIMIT = 1_000_000_000

    #------------------------------------
    # read_df_file
    #-------------------

    @staticmethod
    def read_df_file(fpath: str | Path, **kwargs) -> pd.DataFrame:
        """
        Read a DataFrame from a .csv, .feather, or .parquet file.
        The kwargs are 

        :param fpath: Path to the file.
        :return: DataFrame.
        """
        if not isinstance(fpath, Path) and not isinstance(fpath, str):
            raise TypeError(f"Path must be str or Path, not {fpath}")
        
        fpath = Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"File {fpath} not found")
        if fpath.suffix == '.feather':
            df = pd.read_feather(fpath, **kwargs)
        elif fpath.suffix in ['.csv', '.tsv']:
            fld_sep = '\t' if fpath.suffix == '.tsv' else ','
            df = pd.read_csv(fpath, sep=fld_sep, **kwargs)
            # If df was saved as df.to_csv() without
            # an addtional index=False arg, we'll have
            # a first column named 'Unnamed: 0'. Remove
            # that if it exists:
            if df.columns[0] == 'Unnamed: 0':
                df.drop(columns=['Unnamed: 0'], inplace=True)
        elif fpath.suffix in ['.parquet', '.pq']:
            # If caller provided a thrift_string_size_limit or
            # thrift_container_size_limit, it will be respected
            # by the kwargs.setdefault(...) statements. If not
            # provided in the call, allow a large amount.
            kwargs.setdefault('thrift_string_size_limit', Utils.PARQUET_FILES_THRIFT_LIMIT)
            kwargs.setdefault('thrift_container_size_limit', Utils.PARQUET_FILES_THRIFT_LIMIT)
            df = pd.read_parquet(fpath, **kwargs)
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
                         ) -> Path | bool:
        '''
        Writes dataframe to file in either .csv or .feather format.
        If force is False (default), asks user for confirmation and
        alternative outfile if outfile already exists. 

        Suffix of outfile determines output format. If no
        suffix, .csv is assumed. Only .csv and .feather are supported.

        Returns the outfile to which df was written,
        or False if outfile existed, and human user asked to abort.

        :param df: dataframe to save
        :param outfile: where to save
        :param force: whether to overwrite, defaults to False
        :returns the actual outfile path to which df was written, 
            or False if user canceled.
        '''

        outfile = Path(outfile)
        if outfile.suffix == '':
            outfile = outfile.with_suffix('.csv')

        if outfile.suffix not in ['.csv', '.feather']:
            raise NotImplementedError(f"Only .csv and .feather are supported for writing, not {outfile}")

        outfile = Utils.solicit_overwrite_permission(outfile, force)

        if not outfile:
            # Caller does not wish to overwrite, nor
            # did they provide an alternative path
            return False

        if outfile.suffix == '.csv':
            df.to_csv(outfile, index=False)
        elif outfile.suffix == '.feather':
            df.to_feather(outfile)

        return outfile

    #------------------------------------
    # write_scaler_outfile
    #-------------------

    @staticmethod
    def write_scaler_outfile(
        scaler: BaseEstimator, 
        outfile: str | Path, 
        derive_fname: bool = False,
        force: bool = False) -> Path | bool:
        '''
        Write an sklearn scaler to a file. If derive_fname, then
        assume the file name convention:
           scaled data in the form 
             all_chirp_measures_scaled_min_max.csv:
           i.e.    
                 <root>_scaled_<scale-method>.{csv|feather}

           The scaler metadata in the form:
             all_chirp_measures_scaler_metadata_min_max.pks
           i.e.    
                 <root>_scaler_metadata_<scale-method>.pks

        When derive_fname is True, then outfile is assumed to be
        that name of the scaled data file in the proper form as
        per above. Error if deviation from the template is detected.
        The scaler metadata name will be generated, and the scaler
        info will be written to it.

        For derive_fname is False, the scaler metadata file 
        will be pickled to the given outfile.

        If file exists, user is asked to provide an alternative
        name, allow overwrite, or to abort. If force is True,
        overwrite of already existing file with same name is
        automatic; user won't be consulted.

        Returns the outfile to which df was written,
        or False if outfile existed, and human user asked to abort.

        :param scaler: the scaler object to write
        :param outfile: outfile verbatim if derive_fname is False,
            else path to the associated scaled-data file
        :param derive_fname: whether or not to create
           the output file from the given outfile.
        :returns the path to which scaler data was written,
            or False if user aborted
        '''
        if derive_fname:
            scaler_dst = Utils._derive_scaler_metadata_nm(outfile)
        else:
            scaler_dst = Path(outfile)

        outfile = Utils.solicit_overwrite_permission(scaler_dst, force)

        if not outfile:
            return False
        
        with open(outfile, 'wb') as fd:
            pickle.dump(scaler, fd)
        return outfile

    #------------------------------------
    # solicit_overwrite_permission
    #-------------------

    @staticmethod
    def solicit_overwrite_permission(
        outfile: str | Path, 
        force: bool = False) -> Path | bool:
        outfile = Path(outfile)
        while True:
            if not outfile.exists() or force:
                # Outfile does not already exist, or
                # we may overwrite; all good
                return outfile

            resp = input(f"File '{outfile}' exists; overwrite/new path/cancel (o/p/c): ").lower()

            if resp == 'o':
                return outfile
            
            elif resp == 'p':
                new_path = input("Enter new file path: ")
                outfile = Path(new_path)
                # Loop restarts to check if the NEW path also exists
            
            elif resp == 'c':
                return False
            
            else:
                print("Invalid input. Please enter 'o', 'p', or 'c'.")            

    #------------------------------------
    # _derive_scaler_metadata_nm
    #-------------------

    @staticmethod
    def _derive_scaler_metadata_nm(scaled_data_outfile: str | Path):
        '''
        Create the path to a file destination for scaler metadate. The
        path will follow the convention:

            <root>_scaler_metadata_<scale-method>.pks

        The destination path will be based on the scaled_data_outfile. 
        That path is assumed to be of the form:

            <root>_scaled_<scale-method>.{csv|feather}

        such as: all_chirp_measures_scaled_min_max.csv

        :param scaled_data_outfile: full path to scaled data for
            which scaler metadata is being saved        
        :return: path to where scaler metadata is to be saved 
        '''
        ref_fpath = Path(scaled_data_outfile)

        # Sanity check for reference fname. 
        # Get like:
        #    ['all', 'chirp', 'measures', 'scaled', 'robust']
        # or like:
        #    ['mydata', 'scaled', 'min', 'max]
        elements = ref_fpath.stem.split('_')

        # We expect everything before 'scaled' to be
        # the dataset name, which varies, and all after
        # 'scaled' to be the scaling method.

        if len(elements) < 2:
            msg = (f"Scaler destination file only derivable from names conforming to " 
                   f"<root>_scaled_<scale_method>.<suffix>, not {scaled_data_outfile}")
            raise ValueError(msg)

        # Find the index of the last 'scaled'
        # We reverse the list to find the first occurrence from the end
        target = 'scaled'
        
        if target not in elements:
            msg = (f"File name {scaled_data_outfile} does not conform " 
                   "to <root>_scaled_<scale-method>.<suffix>")
            raise ValueError(msg)
            
        # Calculate the index from the end
        last_index = len(elements) - 1 - elements[::-1].index(target)
        
        # Slice the list
        root = '_'.join(elements[:last_index])
        scaling_method = '_'.join(elements[last_index + 1:])
        
        dst_nm = f"{root}_scaler_metadata_{scaling_method}.pks"
        dst_path = ref_fpath.parent / dst_nm
        return dst_path

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