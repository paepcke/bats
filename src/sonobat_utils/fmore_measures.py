#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-14 11:45:56
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/fmore_measures.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-14 14:20:25
#
# **********************************************************

"""
A simple CLI level 'more' utility to show successive rows of
a measures df on the console. CLI args allow showing only 
rows with particular values in cols, and/or only the values
of particular columns.

Examples:
    fmore_measures.py --filter "species == 'Tabr'"  my_measures.csv

"""

import argparse
import os
import sys
import termios
import tty
from typing import Iterable
import pandas as pd

from sonobat_utils.utils import Utils

class MeasuresPager:

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 df: pd.DataFrame, 
                 filter: str | None = None, 
                 columns: list[str] | None = None):
        pages = []
        cursor = -1
        for page in self.pager(df, filter, columns):
            pages.append(page)
            cursor += 1
            self._show_and_ask(cursor, pages)

    #------------------------------------
    # pager
    #-------------------

    def pager(self, 
              df: pd.DataFrame, 
              filter: str | None = None,
              columns: str | list[str] | None = None
              ) -> Iterable[str]:
        '''
        Generator for dataframe pages, right-sized to terminal window
        and filtered if filter is provided.

        :param df: _description_
        :type df: _type_
        :param filter: _description_
        :type filter: _type_
        :yield: _description_
        :rtype: _type_
        '''
        term_height = os.get_terminal_size().lines - 4
        num_rows = len(df)
        if columns is None:
            columns = list(df.columns)
        filtered_df = df.query(filter)
        ln_counter = 0
        while ln_counter < num_rows:
            slice_bnd = min(ln_counter + term_height, num_rows)
            yield filtered_df.iloc[ln_counter:slice_bnd][columns]
            ln_counter += term_height

    #------------------------------------
    # _show_and_ask
    #-------------------            

    def _show_and_ask(self, cursor: int, pages: list[str]):
        local_cursor = cursor
        while True:
            print(pages[local_cursor])
            print(':', end='')
            nxt = self.getch()
            if nxt in ('q', 'Q', 'quit'):
                print()
                sys.exit(0)
            if nxt in [' ', '\n']:
                return
            if nxt == 'b':
                local_cursor = max(local_cursor -1, 0)
                continue
            
    #------------------------------------
    # getch
    #-------------------
    
    def getch(self) -> str:
        """Reads a single character from stdin on Unix-like systems."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# ----------------------- Main ---------------------

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Page through df, optionally filtered."
                                     )

    parser.add_argument('input',
                        help='path to dataframe (.csv or .feather)',
    )
    parser.add_argument('-f', '--filter',
                        default=None,
                        help="optional filter string for df.query(); e.g. species == 'Tabr'"
    )
    parser.add_argument('-c', '--columns',
                        nargs="+",
                        default=None,
                        help="list of columns to display; default: all"
    )


    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"Cannot read df at {args.input}")
    df = Utils.read_df_file(args.input)
    # Sanity check:
    if args.columns is not None:
        bad_cols = [col 
                    for col 
                    in args.columns
                    if col not in df.columns]
        if len(bad_cols) > 0:
            print(f"Columns not all in df: {bad_cols}")
            sys.exit(1)
    MeasuresPager(df, args.filter, args.columns)

if __name__ == "__main__":
    main()
