#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-11 15:59:39
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sono_batch_processing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-13 11:12:57
#
# **********************************************************

"""
Extract fragment-level species labels from SonoBat 30.x batch output files
and join them with the corresponding acoustic-measures rows from the
``_Parameters_`` files.

Background
----------
SonoBat's Long File Parser chops each ~50-second recording into 2-second
``.wav`` fragments and writes two output files per batch:

``xxx_Parameters_xxx.txt``
    One row **per detected chirp** with ~100 acoustic measures.  Multiple
    rows may share the same ``Filename`` (fragment stem) when more than one
    chirp was detected in a 2-second window.

``xxx_SonoBatch_xxx.txt``
    One row **per fragment** summarising the species ID for that window:
    accepted species, probability, ranked alternatives, call-quality metrics.

This module:

1. Discovers all per-night ``_SonoBatch_`` files under one or more input
   directories (cumulative / nightly-summary files are silently skipped).
2. Parses each file and retains the three species columns needed downstream:
   ``species``, ``species_prob``, and ``species_2nd``.
3. Reads the corresponding ``_Parameters_`` files (same directory, same date
   prefix) to obtain the full chirp-level measures.
4. Joins measures to species on ``Filename`` — the fragment stem written by
   SonoBat into both file types and therefore identical by construction.
5. Assigns a stable integer ``file_id`` per unique ``Filename``.  On
   incremental runs the prior ``filename_to_id.csv`` is loaded so that
   existing integers are never remapped; new fragments receive ids that
   extend beyond the current maximum.
6. Writes three output files:

   ``<out_path>``
       One row per chirp with all measures plus ``species``,
       ``species_prob``, ``species_2nd``, and integer ``file_id``.
       Written as ``.feather`` (fast, compact) when ``use_feather=True``,
       otherwise as ``.csv``.

   ``<stem>_filename_to_id.csv``
       ``Filename`` to ``file_id`` lookup table for joins with other
       pipeline dataframes.

   ``<stem>_config.csv``
       Run-level statistics (file counts, chirp counts, species coverage,
       elapsed time).

Fragment Filename Convention
----------------------------
Fragment stems embedded in both file types take the form::

    <location>-<YYYYMMDD>_<HHMMSS>_2secs

e.g.::

    lake2-20220123_064819_2secs
    bats-20220706_000013_2secs
    barn-20220411_000047_2secs

This stem is the natural unique key across all sites and dates.  The integer
``file_id`` is a groupby/join convenience derived from it.

Output Columns (chirp-level CSV)
---------------------------------
All columns from the ``_Parameters_`` file are retained verbatim, plus:

    - ``species``       : SonoBat accepted species for the containing fragment
    - ``species_prob``  : SonoBat confidence (0-1) for that assignment
    - ``species_2nd``   : SonoBat second-ranked species (possible alternative)
    - ``file_id``       : stable integer key; groupby('file_id') yields all
                          chirps from one 2-second fragment

Downstream sequence-level species
----------------------------------
Recording-level species can be derived from this table without any
pre-aggregation::

    seq_species = (
        chirps_df
        .groupby('recording_name')['species']
        .agg(lambda s: s.mode().iloc[0] if s.notna().any() else pd.NA)
    )

where ``recording_name`` can be added via timestamp-clustering if needed.

Typical Usage
-------------
::

    from sono_batch_processing import SpeciesLabeler

    labeler = SpeciesLabeler(
        inputs=['/qnap/bats/barn_sonobat3_2_processed',
                '/qnap/bats/lake2_sonobat3_2_processed'],
        out_csv='/qnap/bats/chirps_with_species.csv',
        recursive=True,
    )
    result = labeler.run()
    print(result.summary())
"""

import sys
import time
from pathlib import Path
from typing import Optional, Sequence
from dataclasses import dataclass

import pandas as pd

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-night SonoBatch files to skip — they match the glob but use a different
# schema and would produce only spurious warnings if parsed.
_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    'Cumulative',
    'NightlySummary',
    'BatchSummary',
    'SonoBatStaging',
)

# All 34 column names in a per-night SonoBatch file, in order.
_SONOBATCH_COLS: list[str] = [
    'Path', 'Filename', 'HiF', 'LoF',
    'SppAccp', 'Prob',
    '#Maj', '#Accp', '~Spp', '~Prob',
    'Fc mean', 'Fc StdDev', 'Dur mean', 'Dur StdDev', 'calls/sec',
    'mean HiFreq', 'mean LoFreq', 'mean UpprSlp', 'mean LwrSlp',
    'mean TotalSlp', 'mean PrecedingIntvl',
    '1st', '2nd', '3rd', '4th',
    '<--All spp in sqnc classified with a ANN>0.40 in order of prevalence',
    'ParentDir', 'NextDirUp', 'FileLength(sec)',
    'Version', 'Filter', 'AccpQuality', 'AccpQualForTally',
    'Max#CallsConsidered',
]

# Column indices used during parsing (computed once at import time).
_IDX_FILENAME: int = _SONOBATCH_COLS.index('Filename')
_IDX_SPPACCP:  int = _SONOBATCH_COLS.index('SppAccp')
_IDX_PROB:     int = _SONOBATCH_COLS.index('Prob')
_IDX_2ND:      int = _SONOBATCH_COLS.index('2nd')

# Parameters-file columns that carry no information for a classifier:
#   Path / ParentDir / NextDirUp  — stale Windows paths, redundant with Filename
#   Version / Filter              — SonoBat run constants, identical across all rows
#   Preemphasis / MaxSegLnght     — additional run-configuration constants
_COLS_TO_DROP: frozenset[str] = frozenset([
    'Path', 'ParentDir', 'NextDirUp',
    'Version', 'Filter', 'Preemphasis', 'MaxSegLnght',
])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LabelingResult:
    """
    Summary returned by :meth:`SpeciesLabeler.run`.

    :param out_csv:            Path to the chirp-level output CSV.
    :param n_sonobatch_files:  Number of per-night SonoBatch files parsed.
    :param n_fragments:        Total fragment rows parsed from SonoBatch files.
    :param n_chirps_raw:       Chirp rows loaded from Parameters files before join.
    :param n_chirps_labeled:   Chirp rows that received a species label.
    :param n_chirps_unlabeled: Chirp rows with no matching SonoBatch entry.
    :param n_skipped:          Fragment Filenames skipped (already done).
    :param elapsed_secs:       Wall-clock seconds for the full run.
    """
    out_csv:            Path
    n_sonobatch_files:  int
    n_fragments:        int
    n_chirps_raw:       int
    n_chirps_labeled:   int
    n_chirps_unlabeled: int
    n_skipped:          int
    elapsed_secs:       float

    def summary(self) -> str:
        """
        Return a human-readable multi-line run summary.

        :return: Formatted string with all result statistics.
        """
        mins, secs = divmod(self.elapsed_secs, 60)
        elapsed_str = f'{int(mins)}m {secs:.1f}s' if mins else f'{secs:.1f}s'
        pct = (
            100.0 * self.n_chirps_labeled / self.n_chirps_raw
            if self.n_chirps_raw else 0.0
        )
        return (
            f"SpeciesLabeler complete:\n"
            f"  * {self.n_sonobatch_files:,} SonoBatch files parsed\n"
            f"  * {self.n_fragments:,} fragment rows extracted\n"
            f"  * {self.n_chirps_raw:,} chirp rows loaded from Parameters files\n"
            f"  * {self.n_chirps_labeled:,} chirps labeled ({pct:.1f}%)\n"
            f"  * {self.n_chirps_unlabeled:,} chirps without a species match\n"
            f"  * {self.n_skipped:,} fragments skipped (already done)\n"
            f"  * Elapsed: {elapsed_str}\n"
            f"  * Output:  {self.out_csv}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpeciesLabeler:
    """
    Parse SonoBat per-night output files and produce a chirp-level CSV
    with acoustic measures plus species labels, ready for CNN/RF training.

    For each pair of ``_SonoBatch_`` / ``_Parameters_`` files the labeler:

    * Reads species columns from the SonoBatch file (one row per fragment).
    * Reads all measure columns from the Parameters file (one row per chirp).
    * Joins on ``Filename`` — the fragment stem written identically by SonoBat
      into both files, and therefore an exact key by construction.
    * Assigns a stable integer ``file_id`` per unique ``Filename`` that is
      consistent across incremental runs (see :meth:`_assign_file_ids`).

    :param inputs:      One or more directories or individual SonoBatch
                        ``.txt`` files.  Use ``recursive=True`` to descend.
    :param out_csv:     Destination path for the chirp-level output CSV.
    :param recursive:   If ``True``, descend into subdirectories.
    :param done_csv:    Path to a previously-written output CSV whose
                        ``Filename`` values should be skipped.  New rows are
                        appended so the output grows incrementally.
    :param id_map_csv:  Path to a previously-written ``filename_to_id.csv``.
                        When provided, existing ``file_id`` integers are
                        preserved and new filenames are assigned ids that
                        extend beyond the prior maximum.
    :param use_feather: If ``True``, write the chirp output as a ``.feather``
                        file instead of ``.csv``.  Feather is typically 5-10x
                        faster to write and read and roughly half the size for
                        this kind of numeric-heavy DataFrame.  The
                        ``filename_to_id`` and ``config`` sidecar files are
                        always written as CSV regardless of this flag.
    """

    def __init__(
        self,
        inputs:     Sequence[str | Path],
        out_csv:    str | Path,
        recursive:  bool                 = False,
        done_csv:   Optional[str | Path] = None,
        id_map_csv: Optional[str | Path] = None,
        use_feather: bool                = False,
    ) -> None:
        self.inputs      = [Path(p) for p in inputs]
        self.out_csv     = Path(out_csv)
        self.recursive   = recursive
        self.done_csv    = Path(done_csv)   if done_csv   else None
        self.id_map_csv  = Path(id_map_csv) if id_map_csv else None
        self.use_feather = use_feather

    # ------------------------------------------------------------------ #
    #  File discovery                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_per_night_sonobatch(p: Path) -> bool:
        """
        Return ``True`` iff *p* is a per-night SonoBatch fragment file.

        Accepted:  any name containing ``_SonoBatch_`` but none of the
                   keywords in :data:`_EXCLUDE_KEYWORDS`.
        Rejected:  cumulative, nightly-summary, and staging files.

        :param p: Candidate path.
        :return:  ``True`` if the file should be processed.
        """
        name = p.name
        if '_SonoBatch_' not in name:
            return False
        return not any(kw in name for kw in _EXCLUDE_KEYWORDS)

    def _iter_sonobatch_files(self):
        """
        Yield resolved, de-duplicated paths to per-night SonoBatch files
        from all inputs.

        :yields: :class:`pathlib.Path` for each qualifying file.
        """
        seen: set[Path] = set()
        for inp in self.inputs:
            if inp.is_file():
                if self._is_per_night_sonobatch(inp):
                    rp = inp.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        yield rp
                else:
                    log.warn(f'Not a per-night SonoBatch file: {inp}')
            elif inp.is_dir():
                pattern = '**/*SonoBatch*.txt' if self.recursive else '*SonoBatch*.txt'
                for p in inp.glob(pattern):
                    if not self._is_per_night_sonobatch(p):
                        log.info(f'Skipping summary/staging file: {p.name}')
                        continue
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        yield rp
            else:
                log.warn(f'Input does not exist or is not a file/directory: {inp}')

    @staticmethod
    def _parameters_path_for(sonobatch_path: Path) -> Optional[Path]:
        """
        Derive the ``_Parameters_`` file path paired with a SonoBatch file.

        Both files live in the same directory and share the same date prefix;
        only the ``_SonoBatch_`` / ``_Parameters_`` infix differs.

        Example::

            20220706_2secs_SonoBatch_v30.2.20250912.txt
            ->  20220706_2secs_Parameters_v30.2.20250912.txt

        :param sonobatch_path: Path to a per-night SonoBatch file.
        :return:               Corresponding Parameters path, or ``None``
                               if the file does not exist.
        """
        candidate = Path(
            str(sonobatch_path).replace('_SonoBatch_', '_Parameters_')
        )
        if candidate.exists():
            return candidate
        log.warn(
            f'No matching Parameters file for {sonobatch_path.name} '
            f'(expected {candidate.name})'
        )
        return None

    # ------------------------------------------------------------------ #
    #  Parsing                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_sonobatch_file(path: Path) -> pd.DataFrame:
        """
        Parse a per-night SonoBatch file into a fragment-level species table.

        Retains only ``Filename``, ``species``, ``species_prob``, and
        ``species_2nd``.  The ``.wav`` extension is stripped from
        ``Filename`` to match the stem used in Parameters files.  Rows with
        fewer fields than expected are silently skipped.

        :param path: Path to a per-night SonoBatch ``.txt`` file.
        :return:     DataFrame with columns
                     ``['Filename', 'species', 'species_prob', 'species_2nd']``,
                     or an empty DataFrame on error.
        """
        rows: list[dict] = []
        try:
            with open(path, encoding='utf-8') as fh:
                lines = fh.readlines()
        except Exception as exc:
            log.warn(f'Cannot open {path}: {exc}')
            return pd.DataFrame()

        n_cols = len(_SONOBATCH_COLS)
        for line in lines[1:]:      # line 0 is the header
            line = line.rstrip('\n')
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < n_cols:
                continue
            fname = fields[_IDX_FILENAME]
            if fname.lower().endswith('.wav'):
                fname = fname[:-4]
            rows.append({
                'Filename'    : fname,
                'species'     : fields[_IDX_SPPACCP] or pd.NA,
                'species_prob': fields[_IDX_PROB]    or pd.NA,
                'species_2nd' : fields[_IDX_2ND]     or pd.NA,
            })

        if not rows:
            log.warn(f'No valid rows in {path.name}')
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['species_prob'] = pd.to_numeric(df['species_prob'], errors='coerce')
        return df

    @staticmethod
    def _parse_parameters_file(path: Path) -> pd.DataFrame:
        """
        Read a ``_Parameters_`` file into a chirp-level measures DataFrame.

        All columns are retained verbatim.  The ``Filename`` column (fragment
        stem) is the join key; any trailing ``.wav`` extension is stripped
        defensively.

        :param path: Path to a ``_Parameters_`` ``.txt`` file.
        :return:     DataFrame with one row per detected chirp, or an empty
                     DataFrame on error.
        """
        try:
            df = pd.read_csv(path, sep='\t', low_memory=False)
        except Exception as exc:
            log.warn(f'Cannot read Parameters file {path}: {exc}')
            return pd.DataFrame()

        if 'Filename' not in df.columns:
            log.warn(f'No Filename column in {path.name} — skipping')
            return pd.DataFrame()

        # Strip .wav if present (defensive; SonoBat usually omits it here).
        df['Filename'] = df['Filename'].str.replace(
            r'\.wav$', '', regex=True, case=False
        )

        # Drop columns that are redundant or carry no classifier signal.
        df.drop(
            columns=[c for c in _COLS_TO_DROP if c in df.columns],
            inplace=True,
        )
        return df

    # ------------------------------------------------------------------ #
    #  Done-set helper                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_done_filenames(done_csv: Path) -> set[str]:
        """
        Return the set of ``Filename`` stems already present in a prior
        output CSV, enabling incremental runs.

        :param done_csv: Path to an existing chirp-level output CSV.
        :return:         Set of ``Filename`` strings to skip.
        """
        if not done_csv.exists():
            log.warn(f'done-csv not found: {done_csv}')
            return set()
        try:
            df = pd.read_csv(done_csv, usecols=['Filename'])
            stems = set(df['Filename'].dropna().unique().tolist())
            log.info(f'Loaded {len(stems):,} done Filenames from {done_csv}')
            return stems
        except Exception as exc:
            log.warn(f'Could not read done-csv {done_csv}: {exc}')
            return set()

    # ------------------------------------------------------------------ #
    #  Stable integer file_id                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assign_file_ids(
        df:         pd.DataFrame,
        id_map_csv: Optional[Path],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assign a stable integer ``file_id`` to every chirp row via ``Filename``.

        On a fresh run integers are assigned by order of first appearance.
        On an incremental run the prior mapping is loaded first; known
        Filenames keep their existing integers and new Filenames are
        assigned integers extending beyond the prior maximum.  This ensures
        that no earlier row ever has its ``file_id`` remapped.

        :param df:         Chirp-level DataFrame containing a ``Filename``
                           column.
        :param id_map_csv: Path to a previously-written
                           ``filename_to_id.csv``, or ``None`` for a fresh
                           run.
        :return:           Tuple of (updated DataFrame with ``file_id``
                           column added, full Filename-to-id mapping
                           DataFrame with columns
                           ``['Filename', 'file_id']``).
        """
        prior = pd.DataFrame(columns=['Filename', 'file_id'])

        if id_map_csv and id_map_csv.exists():
            try:
                prior = pd.read_csv(id_map_csv)
                log.info(
                    f'Loaded {len(prior):,} prior Filename->id mappings '
                    f'from {id_map_csv}'
                )
            except Exception as exc:
                log.warn(f'Could not read id-map CSV {id_map_csv}: {exc}')

        known_names: set[str] = set(prior['Filename'].tolist()) if len(prior) else set()
        all_names              = df['Filename'].unique()
        new_names              = [n for n in all_names if n not in known_names]

        if new_names:
            next_id  = int(prior['file_id'].max()) + 1 if len(prior) else 0
            log.info(
                f'  Assigning ids {next_id:,} .. '
                f'{next_id + len(new_names) - 1:,} '
                f'to {len(new_names):,} new Filenames'
            )
            new_rows = pd.DataFrame({
                'Filename': new_names,
                'file_id' : range(next_id, next_id + len(new_names)),
            })
            full_map = pd.concat([prior, new_rows], ignore_index=True)
        else:
            log.info('  All Filenames already mapped — no new ids needed')
            full_map = prior.copy()

        log.info(f'  Merging file_id into {len(df):,} chirp rows ...')
        df = df.merge(full_map[['Filename', 'file_id']], on='Filename', how='left')
        log.info('  file_id merge complete')
        return df, full_map

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> LabelingResult:
        """
        Discover SonoBatch files, join with Parameters measures, attach
        species labels, assign stable integer ``file_id`` values, and
        write output.

        On an incremental run (``done_csv`` provided and present on disk):

        * Fragment ``Filename`` stems already in the done set are excluded
          before the join.
        * The chirp output CSV is appended to rather than overwritten.
        * The ``filename_to_id`` map is extended without remapping old ids.

        :return: :class:`LabelingResult` with full run statistics.
        """
        _t0 = time.perf_counter()

        # ---- Load done set ------------------------------------------- #
        done_filenames: set[str] = set()
        if self.done_csv and self.done_csv.exists():
            done_filenames = self.load_done_filenames(self.done_csv)
            log.info(f'{len(done_filenames):,} fragment Filenames already done')

        # ---- Discover SonoBatch files --------------------------------- #
        sonobatch_files = list(self._iter_sonobatch_files())
        log.info(f'Found {len(sonobatch_files):,} per-night SonoBatch files')

        if not sonobatch_files:
            log.warn('No SonoBatch files found — nothing to do')
            return LabelingResult(
                out_csv            = self.out_csv.resolve(),
                n_sonobatch_files  = 0,
                n_fragments        = 0,
                n_chirps_raw       = 0,
                n_chirps_labeled   = 0,
                n_chirps_unlabeled = 0,
                n_skipped          = len(done_filenames),
                elapsed_secs       = time.perf_counter() - _t0,
            )

        # ---- Parse, filter, join ------------------------------------- #
        all_chirps: list[pd.DataFrame] = []
        n_fragments_total = 0

        for sb_path in sonobatch_files:
            log.info(f'Processing {sb_path.name} ...')

            species_df = self._parse_sonobatch_file(sb_path)
            if species_df.empty:
                continue

            n_fragments_total += len(species_df)

            # Drop already-done fragments before touching the Parameters file.
            if done_filenames:
                before     = len(species_df)
                species_df = species_df[
                    ~species_df['Filename'].isin(done_filenames)
                ]
                skipped = before - len(species_df)
                if skipped:
                    log.info(f'  Skipped {skipped:,} already-done fragments')
            if species_df.empty:
                log.info('  All fragments already done — skipping Parameters file')
                continue

            params_path = self._parameters_path_for(sb_path)
            if params_path is None:
                continue
            measures_df = self._parse_parameters_file(params_path)
            if measures_df.empty:
                continue

            # Restrict measures to new fragments only, then join.
            new_filenames = set(species_df['Filename'].tolist())
            measures_df   = measures_df[
                measures_df['Filename'].isin(new_filenames)
            ]
            if measures_df.empty:
                log.warn(
                    f'  No measures rows matched species Filenames '
                    f'in {params_path.name}'
                )
                continue

            joined = measures_df.merge(species_df, on='Filename', how='left')
            all_chirps.append(joined)

        # ---- Bail out if nothing new --------------------------------- #
        if not all_chirps:
            log.warn('No new chirp data after filtering — nothing written')
            return LabelingResult(
                out_csv            = self.out_csv.resolve(),
                n_sonobatch_files  = len(sonobatch_files),
                n_fragments        = n_fragments_total,
                n_chirps_raw       = 0,
                n_chirps_labeled   = 0,
                n_chirps_unlabeled = 0,
                n_skipped          = len(done_filenames),
                elapsed_secs       = time.perf_counter() - _t0,
            )

        log.info(f'Concatenating {len(all_chirps):,} per-night DataFrames ...')
        chirps_df          = pd.concat(all_chirps, ignore_index=True)
        log.info(f'Concat complete: {len(chirps_df):,} chirp rows')
        n_chirps_raw       = len(chirps_df)
        n_chirps_labeled   = int(chirps_df['species'].notna().sum())
        n_chirps_unlabeled = n_chirps_raw - n_chirps_labeled

        if n_chirps_unlabeled:
            log.warn(
                f'{n_chirps_unlabeled:,} chirps have no species match '
                f'(fragment present in Parameters but absent from SonoBatch)'
            )

        # ---- Stable integer file_id ---------------------------------- #
        log.info('Assigning stable file_id integers ...')
        chirps_df, full_map = self._assign_file_ids(chirps_df, self.id_map_csv)

        # ---- Write output -------------------------------------------- #
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Derive stem regardless of whether the extension is .csv or .feather.
        out_str = str(self.out_csv)
        for ext in ('.feather', '.csv'):
            if out_str.endswith(ext):
                stem = out_str[: -len(ext)]
                break
        else:
            stem = out_str

        id_map_path = Path(stem + '_filename_to_id.csv')
        config_path = Path(stem + '_config.csv')

        # Resolve the actual output path — honour use_feather regardless of
        # what extension the caller put on out_csv.
        if self.use_feather:
            out_path = Path(stem + '.feather')
        else:
            out_path = Path(stem + '.csv')

        is_incremental = bool(
            self.done_csv and self.done_csv.exists() and out_path.exists()
        )

        log.info(
            f'{"Appending" if is_incremental else "Writing"} '
            f'{len(chirps_df):,} chirp rows to {out_path} ...'
        )
        if self.use_feather:
            if is_incremental:
                import pyarrow.feather as feather
                prior_df = feather.read_feather(out_path)
                combined = pd.concat([prior_df, chirps_df], ignore_index=True)
                feather.write_feather(combined, out_path)
            else:
                chirps_df.reset_index(drop=True).to_feather(out_path)
        else:
            write_mode   = 'a' if is_incremental else 'w'
            write_header = not is_incremental
            chirps_df.to_csv(out_path, mode=write_mode, header=write_header, index=False)
        log.info(
            f'{"Appended" if is_incremental else "Wrote"} '
            f'{len(chirps_df):,} chirp rows to {out_path}'
        )

        full_map.to_csv(id_map_path, index=False)
        log.info(f'Wrote {len(full_map):,} Filename->id mappings to {id_map_path}')

        elapsed = time.perf_counter() - _t0
        pd.DataFrame([
            {'parameter': 'n_sonobatch_files',  'value': len(sonobatch_files)},
            {'parameter': 'n_fragments',        'value': n_fragments_total},
            {'parameter': 'n_chirps_raw',       'value': n_chirps_raw},
            {'parameter': 'n_chirps_labeled',   'value': n_chirps_labeled},
            {'parameter': 'n_chirps_unlabeled', 'value': n_chirps_unlabeled},
            {'parameter': 'n_skipped',          'value': len(done_filenames)},
            {'parameter': 'elapsed_secs',       'value': round(elapsed, 1)},
        ]).to_csv(config_path, index=False)
        log.info(f'Wrote config to {config_path}')

        return LabelingResult(
            out_csv            = out_path.resolve(),
            n_sonobatch_files  = len(sonobatch_files),
            n_fragments        = n_fragments_total,
            n_chirps_raw       = n_chirps_raw,
            n_chirps_labeled   = n_chirps_labeled,
            n_chirps_unlabeled = n_chirps_unlabeled,
            n_skipped          = len(done_filenames),
            elapsed_secs       = elapsed,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments for :class:`SpeciesLabeler`.

    :return: ``argparse.Namespace`` with validated attributes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='sono_batch_processing',
        description=(
            'Join SonoBat species labels with acoustic measures to produce\n'
            'a chirp-level training CSV.\n\n'
            'Inputs can be any mix of:\n'
            '  * individual xxx_SonoBatch_....txt files\n'
            '  * directories containing such files (use -r to recurse)\n\n'
            'Each SonoBatch file is paired automatically with its\n'
            'co-located xxx_Parameters_....txt file.\n\n'
            'For incremental runs, pass --done-csv pointing at the prior\n'
            'output and --id-map-csv pointing at the prior\n'
            'filename_to_id.csv.  New rows are appended and existing\n'
            'file_id integers are never remapped.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'input',
        nargs='+',
        help='One or more SonoBatch .txt files or directories.',
    )
    parser.add_argument(
        '-o', '--out-csv',
        required=True,
        help='Destination .csv for the chirp-level output.',
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Descend into subdirectories.',
    )
    parser.add_argument(
        '--done-csv',
        default=None,
        metavar='CSV',
        help=(
            'Previously-written chirp CSV.  Fragment Filenames already\n'
            'present are skipped; new rows are appended.'
        ),
    )
    parser.add_argument(
        '--id-map-csv',
        default=None,
        metavar='CSV',
        help=(
            'Previously-written filename_to_id.csv.  Existing integer\n'
            'file_ids are preserved; new Filenames extend the sequence.'
        ),
    )
    parser.add_argument(
        '-f', '--use-feather',
        action='store_true',
        help=(
            'Write the chirp output as a .feather file instead of .csv.\n'
            'Feather is typically 5-10x faster to write/read and about\n'
            'half the size for numeric-heavy DataFrames.  The sidecar\n'
            'filename_to_id and config files are always written as CSV.'
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
    CLI entry point for :class:`SpeciesLabeler`.
    """
    args = _parse_args()

    labeler = SpeciesLabeler(
        inputs      = args.inputs,
        out_csv     = args.out_csv,
        recursive   = args.recursive,
        done_csv    = args.done_csv,
        id_map_csv  = args.id_map_csv,
        use_feather = args.use_feather,
    )

    result = labeler.run()
    log.info(result.summary())
    sys.exit(0 if result.n_chirps_labeled > 0 else 1)


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()