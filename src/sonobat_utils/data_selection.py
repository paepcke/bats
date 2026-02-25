#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-24 18:32:08
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-25 08:18:58

import argparse
from enum import StrEnum
import os
from pathlib import Path
import sys

import pandas as pd

from sonobat_utils.utils import Utils
from logging_service import LoggingService

class PopulationType(StrEnum):
    IDIOM_INTERNAL   = 'idiom-internal'
    IDIOM_STARTS     = 'idiom-starts'
    IDIOM_ENDS       = 'idiom-ends'
    IDIOM_ANY        = 'idiom-any'
    NON_IDIOM_RANDOM = 'non-idiom-random'

class RandSelector(StrEnum):
    N_SAMPLES         = 'n-samples'
    FRAC              = 'frac'
    MATCH_IN_IDIOM    = 'match-in-idiom'  # Match population of in_idiom chirps
    MATCH_IDIOM_START = 'match-idiom-start'
    MATCH_IDIOM_END   = 'match-idiom-end'
    ALL               = 'all'
    

class DataSelector:

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self):
        self.log = LoggingService()

    #------------------------------------
    # select
    #-------------------

    def select(self, 
               in_info: str | Path | pd.DataFrame,
               pop_type: PopulationType,
               rand_selector: RandSelector = RandSelector.N_SAMPLES,
               quantity: int | float = -1,
               seed: int = 42,
               outfile: str | Path = None
               ) -> pd.DataFrame:
        
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be int, not {seed}")

        df_all = Utils.read_df_file(in_info)
        
        # WHAT to get:
        if pop_type == PopulationType.IDIOM_ANY:
            # In or on edge of idiom
            df_pop = df_all[df_all['in_idiom']]
        elif pop_type == PopulationType.IDIOM_STARTS:
            # Only idiom starts
            df_pop = df_all[df_all['idiom_start']]
        elif pop_type == PopulationType.IDIOM_ENDS:
            # Only idiom ends
            df_pop = df_all[df_all['idiom_end']]
        elif pop_type == PopulationType.IDIOM_INTERNAL:
            # Only inside idioms 
            df_pop = df_all[
                (df_all['in_idiom'] == True) & 
                (~df_all['idiom_start']) & 
                (~df_all['idiom_end'])
            ]
        elif pop_type == PopulationType.NON_IDIOM_RANDOM:
            # Anything not inside or on the edge of an idiom:
            df_pop = df_all[(~df_all['in_idiom'] == True)]

            #****df_pop = df_all[~df_all['in_idiom']]
        
        # HOW MUCH to get:
        if rand_selector == RandSelector.N_SAMPLES \
            or rand_selector == RandSelector.ALL:
            # A given number of rows:
            if quantity == -1 or rand_selector == RandSelector.ALL:
                # No sampling; get all of what qualifies
                df_final = df_pop
            else:
                # Random n samples:
                df_final = df_pop.sample(n=quantity, replace=False, random_state=seed)
        elif rand_selector == RandSelector.FRAC:
            # A given fraction of rows
            df_final = df_pop.sample(frac=quantity, replace=False, random_state=seed)

        # Match population size of in_idiom or idiom_start/end:
        elif rand_selector == RandSelector.MATCH_IN_IDIOM:
            target_pop_num = df_all['in_idiom'].sum()
            df_final = df_pop.sample(n=target_pop_num, replace=False, random_state=seed)
        elif rand_selector in [RandSelector.MATCH_IDIOM_START, RandSelector.MATCH_IDIOM_END]:
            target_pop_num = df_all['idiom_start'].sum()
            df_final = df_pop.sample(n=target_pop_num, replace=False, random_state=seed)

        if outfile is not None:
            self.log.info(f"Writing selected rows to {outfile}")
            Utils.write_df_outfile(df_final, outfile)

        return df_final

# ------------- Main --------------

#------------------------------------
# parse_args
#-------------------
def parse_args():

    desc = 'Creates bats subpopulation files'
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=desc
                                     )
    parser.add_argument('infile',
                        help='path to full data')

    parser.add_argument('--population_type',
                        required=True,
                        type=PopulationType,
                        help='which type of subpopulation to extract',
                        )

    parser.add_argument('--rand_selector',
                        required=True,                        
                        type=RandSelector,
                        help='select all/n-samples/fraction/matching-other-pops of rows',
                        )

    parser.add_argument('--quantity',
                        help=('how many rows number for n-samples/fraction; ' 
                              'ignored for other RandSelector members')
                        )
    
    parser.add_argument('-o', '--outfile',
                        help='where to write the resulting df',
                        default=None)

    parser.add_argument('-s', '--seed',
                        type=int,
                        default=42,
                        help='random state for repeatability',
                        )

    args = parser.parse_args()
    return args

#------------------------------------
# main
#-------------------

def main():
    args = parse_args()

    # Check the quantity input:
    quantity = args.quantity
    population_type = args.population_type
    rand_selector = args.rand_selector
    if rand_selector in [RandSelector.FRAC, RandSelector.N_SAMPLES]:
        if type(quantity) not in (int, float):
            print(f"For rand selectors FRAC and N_SAMPLES, quantity must be a number, not {quantity}")
            sys.exit(1)

    if population_type == PopulationType.NON_IDIOM_RANDOM:
        if rand_selector not in [RandSelector.ALL,
                                 RandSelector.MATCH_IN_IDIOM,
                                 RandSelector.MATCH_IDIOM_START,
                                 RandSelector.MATCH_IDIOM_END
                                 ]:
            msg = ("If selecting from non-idiom chirps, then rand_selector must be "
                   "'all', 'match-idiom-start', or 'match-idiom-end. "
                   f"not {rand_selector}"
                   )
            print(msg)
            sys.exit(1)

    data_selector = DataSelector()
    data_selector.select(
        args.infile,
        args.population_type,
        args.rand_selector,
        args.quantity,
        args.seed,
        args.outfile
    )

# ------------- Main Section --------------

if __name__ == "__main__":
    main()
