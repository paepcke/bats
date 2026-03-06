#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-22 08:26:51
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-25 09:46:59

'''
Facilities around scaling dataframes, including 
checking a current dataframe against a scikit scaler.pkl
files
'''
import argparse
from enum import StrEnum
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import joblib

from sonobat_utils.utils import Utils
from logging_service import LoggingService

class ScaleMethod(StrEnum):
    Z_SCORE = 'z-score'
    MIN_MAX = 'min-max'
    ROBUST = 'robust'
    DESCRIBE = 'describe' # Just describe an existing .pks file content
    

# --------------------- Class Scaling ---------------

class Scaling:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, 
                 df_info: str | Path | pd.DataFrame,
                 force: bool, 
                 actions: ScaleMethod | list[ScaleMethod],
                 outfiles: list[str] | list[Path],
                 cols: str | list[str] | None = None
                 ):

        self.log = LoggingService()

        if isinstance(actions, ScaleMethod):
            actions = [actions]
        elif len(actions) == 0:
            raise ValueError("Must provide at least one ScaleMethod action")

        # If action ScaleMethod.DESCRIBE, that must be the only
        # action, because then df_info must be a .pks file:
        
        if ScaleMethod.DESCRIBE in actions and len(actions) > 1:
            raise ValueError("For describing an existing scaler file, the DESCRIBE action must be the only action")
        
        if actions[0] == ScaleMethod.DESCRIBE:
            # df_info must be a .pks scaler file
            self.describe_scaler(df_info)
            # All done; and that was the only action
            return

        # Real scaling:

        # Must have an outfile for every action:
        if len(outfiles) != len(actions):
            raise ValueError(f"Must have as many outfiles as actions (actions: {len(actions)}, outfiles: {len(outfiles)})")

        df = Utils.read_df_file(df_info)
        
        for action, outfile in zip(actions, outfiles):
            outfile = Path(outfile)
            if action == ScaleMethod.Z_SCORE:
                new_df, scaler = self.z_score_scale(df, cols)

            elif action == ScaleMethod.MIN_MAX:
                new_df, scaler = self.min_max_scale(df, cols)

            elif action == ScaleMethod.ROBUST:
                new_df, scaler = self.robust_scale(df, cols)
                
            # Write scaled df to file:
            true_df_outfile = Utils.write_df_outfile(new_df, outfile, force)
            if true_df_outfile:
                self.log.info(f"Wrote scaled df to {true_df_outfile}")
                # Also write scaler metadata to file:
                true_scaler_dst_path = Utils.write_scaler_outfile(
                    scaler, 
                    outfile, 
                    derive_fname=True,
                    force=force)
                if true_scaler_dst_path:
                    self.log.info(f"Wrote scaling metadata to {true_scaler_dst_path}")
                else:
                    self.log.warn("Did NOT write scaler metadata file; user aborted")
            else:
                msg = "Did NOT write scaled df or scaler metadata to file; user aborted"
                self.log.warn(msg)


    #------------------------------------
    # z_score_scale
    #-------------------

    def z_score_scale(
            self, 
            df: str | Path | pd.DataFrame,
            cols: list[str] | None = None) -> tuple[str | BaseEstimator]:
        '''
        Returns a new df with z-score scaled values,
        and a scaler object.

        :param df: dataframe to (partially) transform
        :param cols: columns to transform; all if None, defaults to None
        :return: tuple of tranformed df and the used scaler
        '''
        
        if cols is not None:
            df_to_scale = df[cols].copy()
        else:
            df_to_scale = df.copy()

        scaler = StandardScaler()

        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_to_scale),
            columns=df_to_scale.columns,
            index=df_to_scale.index
        )

        # If only a subset of the original df 
        # was scaled, add the other cols back in:
        if cols is not None:
            cols_to_add = [col for col in df if col not in df_to_scale.columns]
            new_df = pd.concat([df_scaled, df[cols_to_add]], axis='columns')
        else:
            new_df = df_scaled

        return new_df, scaler

    #------------------------------------
    # robust_scale
    #-------------------

    def robust_scale(
            self, 
            df: str | Path | pd.DataFrame,
            cols: list[str] | None = None) -> tuple[str | BaseEstimator]:
        '''
        Returns a new df with scaled values via the sklearn,
        RobustScaler, and a scaler object.

        :param df: dataframe to (partially) transform
        :param cols: columns to transform; all if None, defaults to None
        :return: tuple of tranformed df and the used scaler
        '''
        
        if cols is not None:
            df_to_scale = df[cols].copy()
        else:
            df_to_scale = df.copy()

        # Center around the Median, and scale the
        # values to the inter-quartile range between
        # 25.0 and 75.0
        scaler = RobustScaler(with_centering=True, with_scaling=True)

        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_to_scale),
            columns=df_to_scale.columns,
            index=df_to_scale.index
        )

        # If only a subset of the original df 
        # was scaled, add the other cols back in:
        if cols is not None:
            cols_to_add = [col for col in df if col not in df_to_scale.columns]
            new_df = pd.concat([df_scaled, df[cols_to_add]], axis='columns')
        else:
            new_df = df_scaled

        return new_df, scaler

    #------------------------------------
    # min_max_scale
    #-------------------

    def min_max_scale(
            self, 
            df: str | Path | pd.DataFrame,
            cols: list[str] | None = None) -> tuple[str | BaseEstimator]:
        '''
        Returns a new df with min-max scaled values (all [0,1]),
        and a scaler object.

        :param df: dataframe to (partially) transform
        :param cols: columns to transform; all if None, defaults to None
        :return: tuple of tranformed df and the used scaler
        '''
        
        if cols is not None:
            df_to_scale = df[cols].copy()
        else:
            df_to_scale = df.copy()

        scaler = MinMaxScaler()

        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_to_scale),
            columns=df_to_scale.columns,
            index=df_to_scale.index
        )

        # If only a subset of the original df 
        # was scaled, add the other cols back in:
        if cols is not None:
            cols_to_add = [col for col in df if col not in df_to_scale.columns]
            new_df = pd.concat([df_scaled, df[cols_to_add]], axis='columns')
        else:
            new_df = df_scaled

        return new_df, scaler


    #------------------------------------
    # describe_scaler
    #-------------------

    def describe_scaler(self, 
                        scaler_info: str | Path | BaseEstimator,
                        verbose: bool = False
                        ):

        scaler = self.load_scaler(scaler_info)

        print(f"--- Scaler Identification ---")
        print(f"Class: {scaler.__class__.__name__}")

        if hasattr(scaler, 'output_distribution'):
            print(f"Target Distribution: {scaler.output_distribution}")
            print(f"Number of Quantiles: {scaler.n_quantiles}")

        # Check for feature names (if using scikit-learn 1.0+)
        if hasattr(scaler, 'feature_names_in_'):
            print(f"Original Feature Names: {scaler.feature_names_in_}")

        if verbose:
            # List all 'learned' parameters (those ending in _)
            learned_params = {k: v for k, v in vars(scaler).items() if k.endswith('_')}

            for param, value in learned_params.items():
                print(f"{param}: {value}")

    #------------------------------------
    # load_scaler
    #-------------------    

    def load_scaler(self, scaler_info: str | Path | BaseEstimator) -> BaseEstimator:

        if isinstance(scaler_info, BaseEstimator):
            # Arg is already a scaler:
            return scaler_info
        
        fpath = Path(scaler_info)
        if not fpath.exists():
            raise FileNotFoundError(f"Scaler file {scaler_info} not found")
        
        scaler = joblib.load(fpath)
        return scaler

# -------------------- Main -------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Dataframe scaling utitlities"
                                     )

    parser.add_argument('infile',
                        help='file to dataframe (.csv or .feather), or scaler file (.pks)',
                        )

    parser.add_argument('action',
                        type=ScaleMethod,
                        nargs='+',
                        help='Repeatable: scale action(s) to apply')
    
    parser.add_argument('-f', '--force',
                        action='store_true',
                        help='whether or not to overwrite existing files without consulting human user')
    
    parser.add_argument('-c', '--column',
                        nargs='+',
                        help='repeatable: column name(s) to include in scaling')

    parser.add_argument('-o', '--outfile',
                        nargs='+',
                        help='one file path for each df; corresponding scaler saved to <fname>_scaler.pks')
    
    args = parser.parse_args()

    actions = args.action
    # If ScaleMethod.DESCRIBE, that must be the only
    # action, because then infile must be a .pks file:
    if ScaleMethod.DESCRIBE in actions and len(actions) > 1:
        print("For describing an existing scaler file, the DESCRIBE action must be the only action")
        sys.exit(1)
    # We need as many outfiles as scaling actions:
    outfiles = args.outfile
    if ScaleMethod.DESCRIBE not in actions and len(actions) != len(outfiles):
        print("Must provide as many outfiles as scaling methods")
        sys.exit(1)

    scaling = Scaling(args.infile, args.force, actions, outfiles, args.column)

    # proj_root = Path(__file__).parent.parent.parent
    # scaler_path = proj_root / 'src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/split_scaler.pkl'
    # scaling.describe_scaler(scaler_path, verbose=False)

