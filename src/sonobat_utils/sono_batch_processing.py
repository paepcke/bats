#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-11 15:59:39
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sono_batch_processing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-12 16:40:48
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

This module reads a series of the SonoBatch fragments, and
   1. combines them into a dataframe. It then
   2. composites a second dataframe that combines the information
      for one entire chirp sequence into each row.

This second bats_id dataframe can be joined with measures files
to fill in a 'species' column for each row.

Fragment Filename Convention
----------------------------
The 2-second fragment filenames embedded in SonoBatch files take the form::

    <prefix>-<YYYYMMDD>_<HHMMSS>_2secs.wav

e.g.::

    lake2-20220123_064819_2secs.wav
    bats-20220706_000013_2secs.wav
    barn-20220411_000047_2secs.wav

Because the original 50-second recordings are no longer available,
recording boundaries are recovered by **timestamp clustering**: consecutive
2-sec fragments whose absolute timestamps are separated by less than
``gap_seconds`` (default 10 s) are assumed to belong to the same original
recording.  The synthesised ``file_id`` for each cluster is::

    <prefix>-<YYYYMMDD>_<HHMMSS>

where ``<HHMMSS>`` is the timestamp of the *first* fragment in the cluster.

Output Columns
--------------
The output CSV contains one row per recording with:

    - file_id            : integer join key (``pd.factorize`` of recording_name),
                           compatible with the integer ``file_id`` used in other
                           pipeline dataframes
    - recording_name     : human-readable synthesised key derived from the first
                           fragment's stem, e.g. ``lake2-20220123_064819``
    - species_accepted   : primary species determination
    - species_prob       : mean probability for accepted species
    - n_maj              : total pulses matching most frequent species
    - n_accp             : total pulses meeting criteria for final ID
    - species_1st        : most prevalent species across fragments
    - species_2nd        : second most prevalent species
    - species_3rd        : third most prevalent species
    - accp_quality_mean  : mean acceptance quality across fragments
    - n_fragments        : number of 2-sec fragments for this recording

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
    print(result.summary())
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from dataclasses import dataclass

import pandas as pd

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex that matches the date+time embedded in a fragment filename.
# Captures: group(1) = YYYYMMDD, group(2) = HHMMSS
_TS_RE = re.compile(r'(\d{8})_(\d{6})')

# Gap (seconds) between fragment timestamps that signals a new recording.
# Original recordings were ~50 sec; 2-sec fragments span at most that window.
# A gap larger than this implies a new, distinct recording event.
_DEFAULT_GAP_SECONDS: int = 10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CombinationResult:
    """Result summary returned by :meth:`SonoBatchCombinator.run`."""

    out_csv:     Path
    n_fragments: int
    n_sequences: int
    n_skipped:   int

    def summary(self) -> str:
        """
        Return a human-readable result summary.

        :return: Multi-line string with fragment/sequence counts and output path.
        """
        return (
            f"SonoBatch combination complete:\n"
            f"  • {self.n_fragments:,} fragments processed\n"
            f"  • {self.n_sequences:,} sequences identified\n"
            f"  • {self.n_skipped:,} file_ids skipped (already done)\n"
            f"  • Output: {self.out_csv}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SonoBatchCombinator:
    """
    Discover, parse, and coalesce SonoBat 30.x batch output files.

    Because the original 50-second recordings are no longer available, recording
    boundaries are recovered from the absolute timestamps encoded in fragment
    filenames.  Fragments whose timestamps are within ``gap_seconds`` of each
    other are grouped into the same recording; the synthesised ``file_id`` is
    ``<prefix>-<YYYYMMDD>_<HHMMSS>`` using the *first* fragment's timestamp.

    :param inputs:      One or more paths — individual SonoBatch ``.txt`` files
                        or directories.  Use ``recursive=True`` to descend.
    :param out_csv:     Destination CSV path for sequence-level species IDs.
    :param recursive:   If ``True``, descend into subdirectories when a
                        directory is given.
    :param done_stems:  Set of integer ``file_id`` values already present in a
                        prior run's output CSV.  Build with
                        :meth:`load_done_stems`.
    :param gap_seconds: Maximum inter-fragment timestamp gap (seconds) that
                        still counts as the same original recording.
    """

    # ------------------------------------------------------------------ #
    #  SonoBatch file schema                                             #
    # ------------------------------------------------------------------ #

    # All 34 column names, in order, as they appear in the SonoBatch file.
    SONOBATCH_COLS: list[str] = [
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
        'Max#CallsConsidered',
    ]

    # Subset of columns actually needed for species aggregation.
    NEEDED_COLS: list[str] = [
        'Filename',
        'SppAccp',
        'Prob',
        '#Maj',
        '#Accp',
        '1st',
        '2nd',
        '3rd',
        'AccpQuality',
    ]

    # Column-name → 0-based index within SONOBATCH_COLS.
    # Populated once on first instantiation.
    _NEEDED_COL_IDXS: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  Constructor                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        inputs:      Sequence[str | Path],
        out_csv:     str | Path,
        recursive:   bool           = False,
        done_stems:  Optional[set]  = None,
        gap_seconds: int            = _DEFAULT_GAP_SECONDS,
    ) -> None:
        self.inputs      = [Path(p) for p in inputs]
        self.out_csv     = Path(out_csv)
        self.recursive   = recursive
        self.done_stems  = done_stems or set()
        self.gap_seconds = gap_seconds

        # Build column-index lookup once for the lifetime of the class.
        if not SonoBatchCombinator._NEEDED_COL_IDXS:
            for col in SonoBatchCombinator.NEEDED_COLS:
                SonoBatchCombinator._NEEDED_COL_IDXS[col] = (
                    SonoBatchCombinator.SONOBATCH_COLS.index(col)
                )

    # ------------------------------------------------------------------ #
    #  Public helper: load done stems                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_done_stems(cls, csv_paths: Sequence[str | Path]) -> set:
        """
        Read one or more previously-written sequence CSV files and return the
        set of integer ``file_id`` values they contain.

        These ids are used to skip recordings already processed, enabling
        incremental runs over large datasets.

        :param csv_paths: Paths to existing sequence CSV files.
        :return:          Set of integer ``file_id`` values.
        """
        stems: set = set()
        for p in csv_paths:
            p = Path(p)
            if not p.exists():
                log.warn(f'--done-csv not found, skipping: {p}')
                continue
            try:
                df = pd.read_csv(p, usecols=['file_id'])
                new = set(df['file_id'].dropna().astype(int).unique().tolist())
                stems.update(new)
                log.info(f'Loaded {len(new):,} done file_ids from {p}')
            except Exception as exc:
                log.warn(f'Could not read done-csv {p}: {exc}')
        log.info(f'Total already-done file_ids: {len(stems):,}')
        return stems

    # ------------------------------------------------------------------ #
    #  Fragment timestamp utilities                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_fragment_ts(fragment_name: str) -> Optional[datetime]:
        """
        Extract the absolute timestamp from a 2-sec fragment filename.

        Expected format: ``<prefix>-<YYYYMMDD>_<HHMMSS>_2secs``

        :param fragment_name: Fragment stem (no ``.wav`` extension).
        :return:              Parsed :class:`datetime`, or ``None`` if unparseable.
        """
        m = _TS_RE.search(fragment_name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        except ValueError:
            return None

    @staticmethod
    def _location_date_prefix(fragment_name: str) -> str:
        """
        Return the ``<location>-<YYYYMMDD>`` prefix of a fragment filename.

        This prefix groups fragments that belong to the same detector and night
        before timestamp-clustering is applied within that group.

        Examples::

            lake2-20220123_064819_2secs  →  lake2-20220123
            bats-20220706_000013_2secs   →  bats-20220706
            barn-20220411_000047_2secs   →  barn-20220411

        :param fragment_name: Fragment stem (no ``.wav`` extension).
        :return:              ``<location>-<YYYYMMDD>`` string, or the full
                              name if the pattern is not found.
        """
        m = _TS_RE.search(fragment_name)
        if not m:
            return fragment_name
        # Everything up to (and including) the date portion
        date_end = m.start() + 8          # end of YYYYMMDD within the match
        return fragment_name[: m.start() + 8]

    # ------------------------------------------------------------------ #
    #  Recording-boundary recovery via timestamp clustering              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assign_recording_ids(
        group_df:    pd.DataFrame,
        gap_seconds: int,
    ) -> pd.Series:
        """
        Assign a synthesised ``file_id`` to each fragment row by clustering
        consecutive timestamps.

        Fragments are sorted by their absolute timestamp.  A new cluster (i.e.
        a new original recording) is started whenever the gap to the previous
        fragment exceeds ``gap_seconds``.  The synthesised ``file_id`` for each
        cluster is ``<prefix>-<YYYYMMDD>_<HHMMSS>`` of the cluster's *first*
        fragment.

        Rows whose ``Filename`` does not contain a parseable timestamp are
        assigned the sentinel ``file_id`` ``"unknown"``.

        :param group_df:    Sub-DataFrame for one ``<location>-<YYYYMMDD>``
                            prefix (i.e. one detector night).
        :param gap_seconds: Inter-fragment gap threshold (seconds).
        :return:            :class:`pandas.Series` of ``recording_name`` strings,
                            aligned to ``group_df.index``.
        """
        result = pd.Series('unknown', index=group_df.index, dtype=str)

        # Parse a timestamp for every row; keep NaT where parsing fails.
        timestamps: pd.Series = group_df['Filename'].map(
            SonoBatchCombinator._parse_fragment_ts
        )

        valid_mask = timestamps.notna()
        if not valid_mask.any():
            return result

        valid_idx  = timestamps[valid_mask].sort_values().index
        prev_ts:   Optional[datetime] = None
        cluster_id: str               = ''

        for idx in valid_idx:
            ts = timestamps[idx]
            if prev_ts is None or (ts - prev_ts).total_seconds() > gap_seconds:
                # Start a new cluster; synthesise a file_id from this timestamp.
                name = group_df.at[idx, 'Filename']
                # Strip _2secs suffix for a clean key
                clean = name[:-6] if name.endswith('_2secs') else name
                cluster_id = clean          # e.g. "lake2-20220123_064819"
            result[idx] = cluster_id
            prev_ts = ts

        return result

    # ------------------------------------------------------------------ #
    #  File discovery                                                    #
    # ------------------------------------------------------------------ #

    def _iter_sonobatch_files(self):
        """
        Yield resolved paths to SonoBatch ``.txt`` files from all inputs.

        Files are de-duplicated.  A file is only yielded when at least one
        fragment inside it (checked later during parsing) has a ``file_id``
        not already in ``done_stems``.  The file-level quick-skip used here
        checks whether the SonoBatch filename's date component suggests the
        file *might* contain new data; full filtering happens in
        :meth:`_parse_sonobatch_file`.

        :yields: :class:`pathlib.Path` objects for unprocessed SonoBatch files.
        """
        seen: set[Path] = set()

        for inp in self.inputs:
            if inp.is_file():
                if '_SonoBatch_' in inp.name and inp.suffix.lower() == '.txt':
                    rp = inp.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        yield rp
                else:
                    log.warn(f'File does not look like a SonoBatch file: {inp}')

            elif inp.is_dir():
                pattern = '**/*SonoBatch*.txt' if self.recursive else '*SonoBatch*.txt'
                for p in inp.glob(pattern):
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        yield rp
            else:
                log.warn(f'Input does not exist or is not a file/directory: {inp}')

    # ------------------------------------------------------------------ #
    #  Parsing                                                           #
    # ------------------------------------------------------------------ #

    def _parse_sonobatch_file(self, path: Path) -> pd.DataFrame:
        """
        Parse a single SonoBatch ``.txt`` file into a DataFrame.

        Only the columns listed in :attr:`NEEDED_COLS` are retained, plus a
        synthesised ``file_id`` column that identifies the original recording.
        Rows whose ``file_id`` already appears in :attr:`done_stems` are
        dropped so that incremental runs remain efficient.

        :param path: Path to a SonoBatch file.
        :return:     DataFrame with :attr:`NEEDED_COLS` plus ``file_id``,
                     or an empty DataFrame on any parse error.
        """
        payloads: list[dict] = []

        try:
            with open(path, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()
        except Exception as exc:
            log.warn(f'Cannot open {path}: {exc}')
            return pd.DataFrame()

        # Line 0 is the header; data starts at line 1.
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < len(SonoBatchCombinator.SONOBATCH_COLS):
                continue

            row: dict = {}
            for col, idx in SonoBatchCombinator._NEEDED_COL_IDXS.items():
                val = fields[idx]
                if col == 'Filename':
                    # Strip .wav extension; keep the rest intact for clustering.
                    if val.lower().endswith('.wav'):
                        val = val[:-4]
                row[col] = val
            payloads.append(row)

        if not payloads:
            log.warn(f'No valid rows found in {path}')
            return pd.DataFrame()

        df = pd.DataFrame(payloads)

        # ---- Assign file_ids via timestamp clustering ---- #
        # Group by <location>-<YYYYMMDD> prefix first so that clustering is
        # only applied within one detector/night combination.
        df['_prefix'] = df['Filename'].map(
            SonoBatchCombinator._location_date_prefix
        )

        file_ids = pd.Series('unknown', index=df.index, dtype=str)
        for prefix, grp in df.groupby('_prefix', sort=False):
            ids = SonoBatchCombinator._assign_recording_ids(grp, self.gap_seconds)
            file_ids.update(ids)

        df['recording_name'] = file_ids
        df.drop(columns=['_prefix'], inplace=True)

        # ---- Drop already-done recordings ---- #
        # done_stems holds integer file_ids from prior runs; map them back
        # via recording_name is not yet possible here (factorize runs later),
        # so we carry all fragments through and filter after coalescing.
        # (done_stems filtering on the coalesced frame happens in run().)

        return df

    # ------------------------------------------------------------------ #
    #  Coalescing (vectorised)                                          #
    # ------------------------------------------------------------------ #

    def _coalesce_sequences(self, fragments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Coalesce fragment-level rows into one row per original recording.

        Uses fully vectorised pandas ``groupby``/``agg`` operations for
        performance on large datasets (hundreds of thousands of fragments).

        Aggregation strategy per recording (``file_id``):

        * ``species_accepted``  — mode of non-empty ``SppAccp`` values
        * ``species_prob``      — mean ``Prob`` for the accepted species
        * ``n_maj``             — sum of ``#Maj``
        * ``n_accp``            — sum of ``#Accp``
        * ``species_1st/2nd/3rd`` — mode of non-empty rank-species columns
        * ``accp_quality_mean`` — mean ``AccpQuality``
        * ``n_fragments``       — row count

        :param fragments_df: DataFrame with one row per 2-sec fragment,
                             including a ``file_id`` column.
        :return:             DataFrame with one row per recording.
        """
        if fragments_df.empty:
            return pd.DataFrame()

        df = fragments_df.copy()

        # Coerce numeric columns; treat missing/invalid as 0.
        for col in ('#Maj', '#Accp'):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        for col in ('Prob', 'AccpQuality'):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Replace empty strings with NaN so mode/aggregations ignore them.
        for col in ('SppAccp', '1st', '2nd', '3rd'):
            df[col] = df[col].replace('', pd.NA)

        def _mode_or_empty(s: pd.Series) -> str:
            """Return the most frequent non-null value, or empty string."""
            clean = s.dropna()
            if clean.empty:
                return ''
            return clean.mode().iloc[0]

        # First pass: standard aggregations grouped by recording_name.
        agg = df.groupby('recording_name', sort=False).agg(
            n_maj             = ('#Maj',        'sum'),
            n_accp            = ('#Accp',       'sum'),
            accp_quality_mean = ('AccpQuality', 'mean'),
            n_fragments       = ('Filename',    'count'),
            species_accepted  = ('SppAccp',     _mode_or_empty),
            species_1st       = ('1st',         _mode_or_empty),
            species_2nd       = ('2nd',         _mode_or_empty),
            species_3rd       = ('3rd',         _mode_or_empty),
        ).reset_index()

        # Second pass: mean Prob only for the accepted species per recording.
        prob_rows = df[df['SppAccp'].notna()].copy()
        prob_rows = prob_rows.merge(
            agg[['recording_name', 'species_accepted']],
            on='recording_name', how='left'
        )
        prob_rows = prob_rows[prob_rows['SppAccp'] == prob_rows['species_accepted']]
        prob_mean = (
            prob_rows.groupby('recording_name')['Prob']
            .mean()
            .rename('species_prob')
            .reset_index()
        )

        agg = agg.merge(prob_mean, on='recording_name', how='left')
        agg['species_prob'] = agg['species_prob'].fillna(0.0)

        # Round for readability.
        agg['species_prob']      = agg['species_prob'].round(4)
        agg['accp_quality_mean'] = agg['accp_quality_mean'].round(2)

        # Assign integer file_id via factorize — same convention used across
        # the broader chirp-measures pipeline so joins work without remapping.
        agg['file_id'] = pd.factorize(agg['recording_name'])[0]

        # Enforce column order: integer file_id first, then readable name.
        return agg[[
            'file_id', 'recording_name',
            'species_accepted', 'species_prob',
            'n_maj', 'n_accp',
            'species_1st', 'species_2nd', 'species_3rd',
            'accp_quality_mean', 'n_fragments',
        ]]

    # ------------------------------------------------------------------ #
    #  Main entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> CombinationResult:
        """
        Discover, parse, and coalesce SonoBatch files into sequence-level
        species determinations, then write the result to :attr:`out_csv`.

        :return: :class:`CombinationResult` with summary statistics.
        """
        _EMPTY_COLS = [
            'file_id', 'recording_name',
            'species_accepted', 'species_prob',
            'n_maj', 'n_accp',
            'species_1st', 'species_2nd', 'species_3rd',
            'accp_quality_mean', 'n_fragments',
        ]

        def _write_empty() -> CombinationResult:
            pd.DataFrame(columns=_EMPTY_COLS).to_csv(self.out_csv, index=False)
            return CombinationResult(
                out_csv     = self.out_csv.resolve(),
                n_fragments = 0,
                n_sequences = 0,
                n_skipped   = len(self.done_stems),
            )

        log.info(f'SonoBatchCombinator: processing {len(self.inputs)} input(s)')
        sonobatch_files = list(self._iter_sonobatch_files())
        log.info(f'Found {len(sonobatch_files):,} SonoBatch file(s) to process')

        if not sonobatch_files:
            log.warn('No SonoBatch files found — writing empty output')
            return _write_empty()

        all_fragments: list[pd.DataFrame] = []
        for path in sonobatch_files:
            log.info(f'Parsing {path.name} …')
            frag_df = self._parse_sonobatch_file(path)
            if not frag_df.empty:
                all_fragments.append(frag_df)

        if not all_fragments:
            log.warn('No valid data extracted — writing empty output')
            return _write_empty()

        fragments_df = pd.concat(all_fragments, ignore_index=True)
        log.info(f'Total fragment rows: {len(fragments_df):,}')

        log.info('Coalescing fragments into sequences …')
        sequences_df = self._coalesce_sequences(fragments_df)

        # Drop recordings whose integer file_id was already written in a
        # prior incremental run.  This is the authoritative done-filter;
        # fragment-level filtering is not possible before factorize runs.
        if self.done_stems:
            before = len(sequences_df)
            sequences_df = sequences_df[
                ~sequences_df['file_id'].isin(self.done_stems)
            ]
            log.info(
                f'Skipped {before - len(sequences_df):,} already-done recordings'
            )

        log.info(f'Sequences identified: {len(sequences_df):,}')

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        sequences_df.to_csv(self.out_csv, index=False)
        log.info(f'Wrote {len(sequences_df):,} rows to {self.out_csv}')

        return CombinationResult(
            out_csv     = self.out_csv.resolve(),
            n_fragments = len(fragments_df),
            n_sequences = len(sequences_df),
            n_skipped   = len(self.done_stems),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments.

    :return: ``argparse.Namespace`` with validated ``inputs``, ``out_csv``,
             ``recursive``, and ``done_csv`` attributes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='sono_batch_processing',
        description=(
            'Combine SonoBat species ID summaries written by SonoBat 30.x.\n\n'
            'Inputs can be any mix of:\n'
            '  • individual xxx_SonoBatch_....txt files\n'
            '  • directories (searched for such files; use -r to recurse)\n\n'
            'Output: CSV with one row per original recording (chirp sequence).'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'input',
        nargs='+',
        help=(
            'One or more SonoBatch .txt files or directories.\n'
            'Directories are searched at the top level; use -r to recurse.'
        ),
    )
    parser.add_argument(
        '-o', '--out-csv',
        required=True,
        help='Destination .csv path for the coalesced result.',
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Descend into subdirectories when a directory is given.',
    )
    parser.add_argument(
        '--done-csv',
        nargs='+',
        default=[],
        metavar='CSV',
        help=(
            'One or more previously-written sequence CSVs.\n'
            'Recordings whose file_id already appears in any of these\n'
            'files are skipped, enabling incremental runs.'
        ),
    )
    parser.add_argument(
        '--gap-seconds',
        type=int,
        default=_DEFAULT_GAP_SECONDS,
        metavar='N',
        help=(
            f'Inter-fragment timestamp gap (seconds) that signals a new\n'
            f'original recording (default: {_DEFAULT_GAP_SECONDS}).'
        ),
    )

    args = parser.parse_args()

    inputs: list[Path] = []
    for item in args.input:
        p = Path(item)
        if not p.exists():
            print(f"Warning: '{item}' does not exist — skipping", file=sys.stderr)
            continue
        inputs.append(p)

    if not inputs:
        parser.error('No valid inputs found.')

    args.inputs  = inputs
    args.out_csv = Path(args.out_csv)
    return args


def main() -> None:
    """
    CLI entry point for :class:`SonoBatchCombinator`.
    """
    args = _parse_args()

    done_stems = (
        SonoBatchCombinator.load_done_stems(args.done_csv)
        if args.done_csv else set()
    )

    combinator = SonoBatchCombinator(
        inputs      = args.inputs,
        out_csv     = args.out_csv,
        recursive   = args.recursive,
        done_stems  = done_stems,
        gap_seconds = args.gap_seconds,
    )

    result = combinator.run()
    log.info(result.summary())
    sys.exit(0 if result.n_sequences > 0 else 1)


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()