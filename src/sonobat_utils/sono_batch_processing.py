#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-11 15:59:39
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sono_batch_processing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-12 15:43:42
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

Output Columns
--------------
The output CSV contains one row per recording (file_id) with:
    - file_id: base recording name (without _2secs, _t000000ms, etc.)
    - species_accepted: primary species determination
    - species_prob: probability/confidence for accepted species
    - n_maj: count of pulses matching most frequent species (summed)
    - n_accp: count of pulses meeting criteria for final ID (summed)
    - species_1st: most prevalent species
    - species_2nd: second most prevalent species
    - species_3rd: third most prevalent species
    - accp_quality_mean: mean acceptance quality across fragments
    - n_fragments: number of 2-sec fragments for this recording

Typical Usage
-------------
::

    from sono_batch_processing import SonoBatchCombinator
    
    combinator = SonoBatchCombinator(
        inputs=['path/to/sonobatch_files/', 'file.txt'],
        out_csv='species_determinations.csv',
        recursive=True
    )
    result = combinator.run()
    print(f"Processed {result.n_sequences} sequences from {result.n_fragments} fragments")
"""

import sys
import csv
from pathlib import Path
from typing import Optional, Sequence
from dataclasses import dataclass

import pandas as pd

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CombinationResult:
    """Result summary from SonoBatchCombinator.run()"""
    out_csv: Path
    n_fragments: int
    n_sequences: int
    n_skipped: int
    
    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"SonoBatch combination complete:\n"
            f"  • {self.n_fragments:,} fragments processed\n"
            f"  • {self.n_sequences:,} sequences identified\n"
            f"  • {self.n_skipped:,} fragments skipped (already done)\n"
            f"  • Output: {self.out_csv}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SonoBatchCombinator:
    """
    Discover, parse, and coalesce SonoBat 30.x batch output files.
    
    :param inputs:     One or more paths — can be individual SonoBatch .txt 
                       files or directories (use ``recursive=True`` to descend).
    :param out_csv:    Destination CSV path for sequence-level species IDs.
    :param recursive:  If True, descend into subdirectories when a directory 
                       is given.
    :param done_stems: Set of file_id stems already processed (from prior runs).
                       Build with :meth:`load_done_stems`.
    """

    # Column names in SonoBatch files (34 columns)
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

    # Subset of columns we actually need
    NEEDED_COLS = [
        'Filename',    # Key for extracting file_id
        'SppAccp',     # Accepted species
        'Prob',        # Probability for accepted species
        '#Maj',        # Count of pulses matching most frequent species
        '#Accp',       # Count of pulses meeting the criteria for the final ID
        '1st',         # Most prevalent species
        '2nd',         # Second most prevalent
        '3rd',         # Third most prevalent
        'AccpQuality'  # Confidence metric based on call clarity and consistency
    ]
    
    # Dict mapping column name → column index in SONOBATCH_COLS
    # Initialized once in __init__
    NEEDED_COL_IDXS: dict[str, int] = {}

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(
        self,
        inputs:       Sequence[str | Path],
        out_csv:      str | Path,
        recursive:    bool          = False,
        done_stems:   Optional[set] = None,
    ):
        """
        :param inputs:     One or more SonoBatch .txt files or directories.
        :param out_csv:    Output CSV path for sequence-level species IDs.
        :param recursive:  Descend into subdirectories when a directory is given.
        :param done_stems: Set of file_id stems already processed. Build with
                           :meth:`load_done_stems`.
        """
        self.inputs      = [Path(p) for p in inputs]
        self.out_csv     = Path(out_csv)
        self.recursive   = recursive
        self.done_stems  = done_stems or set()
        
        # Initialize column index mapping (only once)
        if not SonoBatchCombinator.NEEDED_COL_IDXS:
            for col_nm in SonoBatchCombinator.NEEDED_COLS:
                SonoBatchCombinator.NEEDED_COL_IDXS[col_nm] = \
                    SonoBatchCombinator.SONOBATCH_COLS.index(col_nm)

    # ------------------------------------------------------------------ #
    #  Done-stems helper                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_done_stems(cls, csv_paths: Sequence[str | Path]) -> set:
        """
        Read one or more previously-written sequence CSV files and return the 
        set of file_id values they contain.
        
        These stems are used to skip files that have already been processed, 
        enabling incremental runs over large datasets.
        
        :param csv_paths: Paths to existing sequence CSV files.
        :return:          Set of file_id strings.
        """
        stems: set = set()
        for p in csv_paths:
            p = Path(p)
            if not p.exists():
                log.warn(f'--done-csv file not found, skipping: {p}')
                continue
            try:
                df = pd.read_csv(p, usecols=['file_id'])
                stems.update(df['file_id'].dropna().unique().tolist())
                log.info(f'Loaded {len(df):,} done rows from {p}')
            except Exception as exc:
                log.warn(f'Could not read done-csv {p}: {exc}')
        log.info(f'Total already-done file_ids: {len(stems):,}')
        return stems

    # ------------------------------------------------------------------ #
    #  File discovery                                                    #
    # ------------------------------------------------------------------ #

    def _iter_sonobatch_files(self):
        """
        Yield SonoBatch .txt file paths from all inputs.
        
        Rules:
        * If input is a .txt file with '_SonoBatch_' in the name → yield it
        * If input is a directory → glob for *SonoBatch*.txt files
        * Skip files whose base file_id is in done_stems
        
        :yields: Path objects for SonoBatch files.
        """
        seen: set[Path] = set()
        
        for inp in self.inputs:
            if inp.is_file():
                # Single file case
                if '_SonoBatch_' in inp.name and inp.suffix.lower() == '.txt':
                    rp = inp.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        # Extract file_id to check done_stems
                        file_id = self._extract_file_id(rp)
                        if file_id not in self.done_stems:
                            yield rp
                        else:
                            log.info(f'Skipping already-done file_id: {file_id}')
                else:
                    log.warn(f'File does not appear to be a SonoBatch file: {inp}')
                    
            elif inp.is_dir():
                # Directory case
                pattern = '**/*SonoBatch*.txt' if self.recursive else '*SonoBatch*.txt'
                for p in inp.glob(pattern):
                    rp = p.resolve()
                    if rp in seen:
                        continue
                    seen.add(rp)
                    file_id = self._extract_file_id(rp)
                    if file_id not in self.done_stems:
                        yield rp
                    else:
                        log.info(f'Skipping already-done file_id: {file_id}')
            else:
                log.warn(f'Input does not exist or is not a file/directory: {inp}')

    @staticmethod
    def _extract_file_id(sonobatch_path: Path) -> str:
        """
        Extract the base file_id from a SonoBatch filename.
        
        Examples:
            20220907_2secs_SonoBatch_v30.2.20250912.txt → 20220907
            batch1_SonoBatch_v30.txt → batch1
        
        :param sonobatch_path: Path to SonoBatch file
        :return:               Base file_id (everything before _SonoBatch_)
        """
        name = sonobatch_path.stem
        # Split on _SonoBatch_ and take everything before it
        parts = name.split('_SonoBatch_')
        if len(parts) >= 1:
            base = parts[0]
            # Remove common suffixes like _2secs
            base = base.replace('_2secs', '')
            return base
        return name

    # ------------------------------------------------------------------ #
    #  Parsing                                                           #
    # ------------------------------------------------------------------ #

    def _parse_sonobatch_file(self, path: Path) -> pd.DataFrame:
        """
        Parse a single SonoBatch .txt file into a DataFrame with needed columns.
        
        :param path: Path to SonoBatch file
        :return:     DataFrame with columns from NEEDED_COLS plus 'file_id'
        """
        payloads: list[dict] = []
        
        try:
            with open(path, 'r', encoding='utf-8') as fd:
                lines = fd.readlines()
                
            # Skip header line (line 0)
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                fields = line.split('\t')
                
                # Skip lines that don't have enough columns
                if len(fields) < len(SonoBatchCombinator.SONOBATCH_COLS):
                    continue
                
                # Extract needed columns
                payload = {}
                for col_nm, idx in SonoBatchCombinator.NEEDED_COL_IDXS.items():
                    if col_nm == 'Filename':
                        # Strip .wav extension from filename
                        val = fields[idx]
                        if val.endswith('.wav'):
                            val = val[:-4]
                        payload[col_nm] = val
                    else:
                        # Store raw value
                        payload[col_nm] = fields[idx]
                
                payloads.append(payload)
                
        except Exception as exc:
            log.warn(f'Error parsing {path}: {exc}')
            return pd.DataFrame()
        
        if not payloads:
            log.warn(f'No valid rows found in {path}')
            return pd.DataFrame()
            
        df = pd.DataFrame(payloads)
        
        # Add file_id column (extract from Filename column)
        # Filename format: bats-20220907_000040_2secs
        # file_id should be: bats-20220907
        df['file_id'] = df['Filename'].apply(self._extract_file_id_from_fragment)
        
        # Filter out already-done file_ids
        if self.done_stems:
            before_count = len(df)
            df = df[~df['file_id'].isin(self.done_stems)]
            after_count = len(df)
            if before_count > after_count:
                log.info(f'Filtered out {before_count - after_count} already-done fragments')
        
        return df

    @staticmethod
    def _extract_file_id_from_fragment(fragment_name: str) -> str:
        """
        Extract the base file_id from a fragment filename.
        
        Examples:
            bats-20220907_000040_2secs → bats-20220907
            recording_123456_2secs → recording
        
        :param fragment_name: Fragment filename (without .wav)
        :return:              Base file_id
        """
        # Remove _2secs suffix if present
        if fragment_name.endswith('_2secs'):
            fragment_name = fragment_name[:-6]
        
        # Split on underscore and take first two parts for date-based naming
        # e.g., bats-20220907_000040 → bats-20220907
        parts = fragment_name.split('_')
        if len(parts) >= 2:
            # Check if second part looks like a timestamp (6 digits)
            if len(parts[1]) == 6 and parts[1].isdigit():
                return parts[0]
        
        # Otherwise just return the first part
        return parts[0] if parts else fragment_name

    # ------------------------------------------------------------------ #
    #  Coalescing                                                        #
    # ------------------------------------------------------------------ #

    def _coalesce_sequences(self, fragments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Coalesce fragment-level data into sequence-level species determinations.
        
        For each file_id (recording), aggregate across all 2-sec fragments to 
        produce a single species determination.
        
        Strategy:
        - Sum #Maj and #Accp counts across fragments
        - Use most common SppAccp as the sequence-level species
        - Average Prob and AccpQuality
        - Aggregate 1st/2nd/3rd species across fragments
        
        :param fragments_df: DataFrame with one row per fragment
        :return:             DataFrame with one row per sequence (file_id)
        """
        if fragments_df.empty:
            return pd.DataFrame()
        
        # Convert numeric columns to proper types
        fragments_df['#Maj'] = pd.to_numeric(fragments_df['#Maj'], errors='coerce').fillna(0).astype(int)
        fragments_df['#Accp'] = pd.to_numeric(fragments_df['#Accp'], errors='coerce').fillna(0).astype(int)
        fragments_df['Prob'] = pd.to_numeric(fragments_df['Prob'], errors='coerce').fillna(0.0)
        fragments_df['AccpQuality'] = pd.to_numeric(fragments_df['AccpQuality'], errors='coerce').fillna(0.0)
        
        sequences = []
        
        for file_id, group in fragments_df.groupby('file_id'):
            # Find most common accepted species
            species_counts = group['SppAccp'].value_counts()
            # Remove empty strings
            species_counts = species_counts[species_counts.index != '']
            
            if species_counts.empty:
                species_accepted = ''
                species_prob = 0.0
            else:
                species_accepted = species_counts.index[0]
                # Average probability for this species across fragments
                species_mask = group['SppAccp'] == species_accepted
                species_prob = group.loc[species_mask, 'Prob'].mean()
            
            # Sum majority and accepted counts
            n_maj = group['#Maj'].sum()
            n_accp = group['#Accp'].sum()
            
            # Aggregate 1st/2nd/3rd species across fragments
            # Filter out empty strings before counting
            all_1st = group['1st'][group['1st'] != ''].value_counts()
            all_2nd = group['2nd'][group['2nd'] != ''].value_counts()
            all_3rd = group['3rd'][group['3rd'] != ''].value_counts()
            
            species_1st = all_1st.index[0] if len(all_1st) > 0 else ''
            species_2nd = all_2nd.index[0] if len(all_2nd) > 0 else ''
            species_3rd = all_3rd.index[0] if len(all_3rd) > 0 else ''
            
            # Average acceptance quality
            accp_quality_mean = group['AccpQuality'].mean()
            
            sequences.append({
                'file_id': file_id,
                'species_accepted': species_accepted,
                'species_prob': round(species_prob, 4),
                'n_maj': n_maj,
                'n_accp': n_accp,
                'species_1st': species_1st,
                'species_2nd': species_2nd,
                'species_3rd': species_3rd,
                'accp_quality_mean': round(accp_quality_mean, 2),
                'n_fragments': len(group)
            })
        
        return pd.DataFrame(sequences)

    # ------------------------------------------------------------------ #
    #  Main run method                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> CombinationResult:
        """
        Discover, parse, and coalesce SonoBatch files into sequence-level 
        species determinations.
        
        :return: CombinationResult with summary statistics
        """
        log.info(f'SonoBatchCombinator: processing {len(self.inputs)} input(s)')
        
        # Discover all SonoBatch files
        sonobatch_files = list(self._iter_sonobatch_files())
        log.info(f'Found {len(sonobatch_files)} SonoBatch file(s) to process')
        
        if not sonobatch_files:
            log.warn('No SonoBatch files found to process')
            # Create empty output file with correct columns
            empty_df = pd.DataFrame(columns=[
                'file_id', 'species_accepted', 'species_prob', 'n_maj', 'n_accp',
                'species_1st', 'species_2nd', 'species_3rd', 'accp_quality_mean', 
                'n_fragments'
            ])
            empty_df.to_csv(self.out_csv, index=False)
            return CombinationResult(
                out_csv=self.out_csv.resolve(),
                n_fragments=0,
                n_sequences=0,
                n_skipped=len(self.done_stems)
            )
        
        # Parse all files into one big DataFrame of fragments
        all_fragments = []
        for path in sonobatch_files:
            log.info(f'Parsing {path.name}...')
            df = self._parse_sonobatch_file(path)
            if not df.empty:
                all_fragments.append(df)
        
        if not all_fragments:
            log.warn('No valid data extracted from SonoBatch files')
            empty_df = pd.DataFrame(columns=[
                'file_id', 'species_accepted', 'species_prob', 'n_maj', 'n_accp',
                'species_1st', 'species_2nd', 'species_3rd', 'accp_quality_mean',
                'n_fragments'
            ])
            empty_df.to_csv(self.out_csv, index=False)
            return CombinationResult(
                out_csv=self.out_csv.resolve(),
                n_fragments=0,
                n_sequences=0,
                n_skipped=len(self.done_stems)
            )
        
        fragments_df = pd.concat(all_fragments, ignore_index=True)
        log.info(f'Parsed {len(fragments_df)} fragment rows')
        
        # Coalesce into sequence-level determinations
        log.info('Coalescing fragments into sequences...')
        sequences_df = self._coalesce_sequences(fragments_df)
        log.info(f'Created {len(sequences_df)} sequence-level determinations')
        
        # Write output
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        sequences_df.to_csv(self.out_csv, index=False)
        log.info(f'Wrote {len(sequences_df)} sequences to {self.out_csv}')
        
        return CombinationResult(
            out_csv=self.out_csv.resolve(),
            n_fragments=len(fragments_df),
            n_sequences=len(sequences_df),
            n_skipped=len(self.done_stems)
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    :return: ``args`` namespace (``args.inputs`` is a list of raw Path objects).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='sono_batch_processing',
        description=(
            'Combine SonoBat species ID summaries written by SonoBat 30.x.\n\n'
            'Inputs can be any mix of:\n'
            '  • individual xxx_SonoBatch_....txt files\n'
            '  • directories (searched for such files; use -r to recurse)\n'
            'Output: CSV with one row per chirp *sequence* (recording)'
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
        required=True,
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
            'Files whose file_id already appears in any of these\n'
            'CSVs are skipped, enabling incremental runs.'
        ),
    )
    args = parser.parse_args()

    # Validate inputs exist; warn on unrecognised extensions but still pass
    # them through so _iter_sonobatch_files() can emit its own warning with full context.
    inputs: list[Path] = []
    for item in args.input:
        p = Path(item)
        if not p.exists():
            print(f"Warning: '{item}' does not exist — skipping",
                  file=sys.stderr)
            continue
        inputs.append(p)

    if not inputs:
        parser.error('No valid inputs found.')

    args.inputs = inputs
    out_csv = Path(args.out_csv)
    args.out_csv = out_csv
    return args


def main():
    args = parse_args()
    
    done_stems = SonoBatchCombinator.load_done_stems(args.done_csv) if args.done_csv else set()
    
    combinator = SonoBatchCombinator(
        inputs=args.inputs,
        out_csv=args.out_csv,
        recursive=args.recursive,
        done_stems=done_stems
    )
    
    result = combinator.run()
    log.info(result.summary())
    sys.exit(0 if result.n_sequences > 0 else 1)


# ------------------- Main Section --------------
if __name__ == "__main__":
    main()