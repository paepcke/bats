#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-25 08:35:59
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-25 08:48:13

'''
Adds is_start, is_end, and seq_duration to 
given full chirps df. 
'''

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

from sonobat_utils.utils import Utils
from logging_service import LoggingService


class MeasuresAugmenter:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, 
                 infile: str | Path, 
                 outfile: str | Path | None = None,
                 force: bool = False
                 ):
        self.log = LoggingService()
        
        df = Utils.read_df_file(infile)
        self.augmented_df = self.add_seq_info(df)
        if outfile is not None:
            true_outfile = Utils.write_df_outfile(self.augmented_df, outfile, force)
            self.log.info(f"Wrote augmented df to {true_outfile}")

    #------------------------------------
    # add_seq_info
    #-------------------

    def add_seq_info(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Add three columns: 'seq_duration', 'is_first', and 'is_last'
        that indicate for each chirp whether it is the first or last
        in its sequence, and the length of the sequence in msecs.
        We use that info later in node files.

        :param df: all chirp data
        :return: augmented copy of df
        '''
        # Create the grouping object once to save processing time
        grouped = df.groupby('file_id')

        # 1. Identify the first and last chirps
        df['is_first'] = df['chirp_idx'] == grouped['chirp_idx'].transform('min')
        df['is_last']  = df['chirp_idx'] == grouped['chirp_idx'].transform('max')

        # 2. Get the end time (duration) for the entire sequence
        df['seq_duration'] = grouped['TimeInFile'].transform('max')

        return df


def parse_args():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Add sequence start/end/duration cols to chirps df"
                                     )
    parser.add_argument('infile',
                        help='file to df to be imported (.csv or .feather)',
                        )


    parser.add_argument('-o', '--outfile',
                        help='destination file for augmented df',
                        default=None)

    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    infile = Path(args.infile)
    if not infile.exists():
        print(f"Input file {infile} not found")
        sys.exit(1)
    MeasuresAugmenter(infile, args.outfile)

if __name__ == "__main__":
    main()
