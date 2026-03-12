#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-11 15:59:39
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sono_batch_processing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-11 18:42:17
#
# **********************************************************

"""
When SonoBat extracts measures from chirp files, two files are produced:
   xxx_Parameters_xxx.txt
   xxx_SonoBatch_....txt

Each SonoBatch file is a 34-column summary of a single 
chirp file. The main content is the bat species with 
confidence measures. 

Our workflow follows the SonoBat recommendation of chopping 
recordings into 2-sec fragments before running their analysis. 
So each SonoBatch file is actually a SonoBatch result *fragment*
of the chirp sequence in a recording.

This class reads a series of the SonoBatch fragments, and 
   1. combines them into a dataframe. It then 
   2. composites a second dataframe that combines the information
      for one entire chirp sequence into each row.

This second bats_id dataframe can be joined with measures files
to fill in a 'species' column for each row.
"""

import sys
from pathlib import Path
import pandas as pd

from logging_service import LoggingService

log = LoggingService()

class SonoBatchCombinator:

    SONOBATCH_COLS = [
        'Path',
        'Filename',
        'HiF',
        'LoF',
        'SppAccp',
        'Prob',
        '#Maj',
        '#Accp',
        '~Spp',
        '~Prob',
        'Fc mean',
        'Fc StdDev',
        'Dur mean',
        'Dur StdDev',
        'calls/sec',
        'mean HiFreq',
        'mean LoFreq',
        'mean UpprSlp',
        'mean LwrSlp',
        'mean TotalSlp',
        'mean PrecedingIntvl',
        '1st',
        '2nd',
        '3rd',
        '4th',
        '<--All spp in sqnc classified with a ANN>0.40 in order of prevalence',
        'ParentDir',
        'NextDirUp',
        'FileLength(sec)',
        'Version',
        'Filter',
        'AccpQuality',
        'AccpQualForTally',
        'Max#CallsConsidered'
    ]

    NEEDED_COLS = [
        'Filename',    # Key for later merging into measures df
        'SppAccp',     # Accepted species
        '#Maj',        # Count of pulses matching most frequent species
        '#Accp',       # Count of pulses meeting the criteria for the final ID
        '1st',
        '2nd',
        '3rd',
        'AccpQuality'  # A confidence metric based on call clarity and consistency.
    ]
    
    # Dict of columns wanted from each SonoBatch table.
    # Maps column name to its position in the header files.
    # Initialized in constructor:
    NEEDED_COL_IDXS: dict[str, int] = {}

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, paths: list[str | Path]):
        self.paths = paths
        # Init the column value extraction index dict:
        for col_nm in SonoBatchCombinator.NEEDED_COLS:
            SonoBatchCombinator.NEEDED_COL_IDXS[col_nm] = \
                SonoBatchCombinator.SONOBATCH_COLS.index(col_nm)

    #------------------------------------
    # load_sono_batch_reports
    #-------------------

    def run(self) -> pd.DataFrame:
        payloads: list[dict[str, str|int|float]] = []
        with open(self.paths[0], 'r') as fd:
            lines = fd.readlines()
            for line in lines[1:]:
                fields = line.split('\t')
                payload = {col_nm : fields[idx] if col_nm != 'Filename' else fields[idx][:-4]
                           for col_nm, idx
                           in SonoBatchCombinator.NEEDED_COL_IDXS.items()
                }
                payloads.append(payload)
        df = pd.DataFrame(payloads)
        return df

def parse_args():
    """
    Parse command-line arguments.

    :return: ``args`` namespace (``args.inputs`` is a list of raw Path objects).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='chirp_measures_extraction',
        description=(
            'Combine SonoBatch species ID summaries written by SonoBat 30.x.\n\n'
            'Inputs can be any mix of:\n'
            '  • individual xxx_SonoBatch_....txt files\n'
            '  • directories (searched for such files; use -r to recurse)\n'
            'Output: CSV with one row per chirp *series*'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'input',
        nargs='+',
        help=(
            'one or more xxxSonoBatchxxx.txt files or directories.\n'
            'Directories are searched at top level only; use -r to recurse.'
        ),
    )
    parser.add_argument(
        '-o', '--out-csv',
        help='destination .csv path for final result',
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='descend into subdirectories when a directory is given',
    )
    parser.add_argument(
        '--done-csv',
        nargs='+',
        default=[],
        metavar='CSV',
        help=(
            'one or more previously-written SonoBatch CSVs.\n'
            'Files whose stem (file_id) already appears in any of these\n'
            'CSVs are skipped, enabling incremental runs.'
        ),
    )
    args = parser.parse_args()

    # Validate inputs exist; warn on unrecognised extensions but still pass
    # them through so _iter_paths() can emit its own warning with full context.
    inputs: list[Path] = []
    for item in args.input:
        p = Path(item)
        suffix = p.suffix.lower()
        if not p.exists() and suffix not in ('.txt'):
            print(f"Warning: '{item}' does not exist and is not a recognised type — skipping",
                  file=sys.stderr)
            continue
        inputs.append(p)

    if not inputs:
        parser.error('No valid inputs found.')

    args.inputs = inputs
    out_csv = Path(args.out_csv)
    Path.mkdir(out_csv.parent, parents=True, exist_ok=True)
    args.out_csv = out_csv
    return args


def main():
    args = parse_args()
    combinator = SonoBatchCombinator(args.input)
    df: pd.DataFrame
    df = combinator.run()
    df.to_csv(args.out_csv)
    log.info(f"Wrote {len(df)} lines to {args.out_csv}")

# ------------------- Main Section --------------
if __name__ == "__main__":
    main()
