#!/usr/bin/env python
# ******************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-21 11:15:48
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-22 18:46:22
# ******************************************

'''
sb_measures_add_daytime_columns.py

Adds was_daytime (bool) and time_of_day_pactime (ISO datetime string)
columns to a bat measures parquet file and its sibling noise parquet file.

For the measures file the columns are populated by looking up each row's
file_id in the recordings SQLite table to obtain the original recording
filename, extracting the timestamp from that filename, and then asking
DaytimeFileSelector whether that moment falls within daylight hours at
Jasper Ridge Biological Preserve.

For the noise file the columns are added but always left empty (None).
The recordings table was built only from files that produced measures,
so noise file_ids are not present there; timestamp resolution is
therefore not possible for noise rows.

Recording filename formats have varied over time.  At startup the script
samples filenames from the database, tallies which of the known patterns
match, and logs the breakdown so the coverage is visible.  During
processing each filename is tried against all patterns in order; the
first match wins.

Known patterns (see _FNAME_PATTERNS):
  - SonoBat D/T/m  : barn1_D20220205T192049m784-HiF.wav
  - YYYYMMDD_HHMMSS: barn-20200228_181107_2secs
  - YYYY-MM-DD_HH-MM-SS: site_2022-02-05_19-20-49
  - 14 consecutive digits YYYYMMDDHHMMSS

New parquet files are written next to their originals with a fresh
timestamp in the name:

    bats_2026-04-21T14_03_27.123456.parquet
    bats_noise_2026-04-21T14_03_27.123456.parquet

Both output files carry the full BatsData envelope (file_map, normalizer
state) copied from their originals, so Utils.read_df_file() works on them
without any KeyError.

run() returns the two new paths as a tuple so callers are never left
holding a stale filename.

Usage
-----
    python sb_measures_add_daytime_columns.py \\
        /data/bats_2026-04-14T23_44_31.660585.parquet \\
        /data/chirp_meta.db
'''
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

import pandas as pd

from data_calcs.daytime_file_selection import DaytimeFileSelector
from logging_service import LoggingService
from sonobat_utils.sb_measures_postprocessing import BatsData
from sonobat_utils.utils import Utils


# ---------------------------------------------------------------------------
# Filename pattern registry
# ---------------------------------------------------------------------------

class _FnamePat(NamedTuple):
    '''
    One entry in the filename-pattern registry.

    :param name: human-readable label used in log output
    :param regex: compiled pattern; must produce groups that *extractor*
        can interpret
    :param extractor: callable that receives ``re.Match.groups()`` and
        returns ``(year, month, day, hour, minute, second)`` as ints
    '''
    name:      str
    regex:     re.Pattern
    extractor: Callable[[tuple], tuple[int, int, int, int, int, int]]


def _ymd_hms(date8: str, time6: str) -> tuple[int, int, int, int, int, int]:
    '''
    Split a compact YYYYMMDD string and a compact HHMMSS string into
    six integer components.

    :param date8: eight-digit date string YYYYMMDD
    :param time6: six-digit time string HHMMSS
    :return: (year, month, day, hour, minute, second)
    '''
    return (int(date8[:4]), int(date8[4:6]), int(date8[6:]),
            int(time6[:2]), int(time6[2:4]), int(time6[4:]))


# Ordered list: more-specific patterns first so that a more specialised
# pattern wins when multiple could match the same filename.
_FNAME_PATTERNS: list[_FnamePat] = [

    _FnamePat(
        name='SonoBat D{YYYYMMDD}T{HHMMSS}m{ms}',
        # e.g.  barn1_D20220205T192049m784-HiF.wav
        regex=re.compile(r'D(\d{8})T(\d{6})m\d{3}'),
        extractor=lambda g: _ymd_hms(g[0], g[1]),
    ),

    _FnamePat(
        name='YYYY-MM-DD_HH-MM-SS',
        # e.g.  site_2022-02-05_19-20-49
        regex=re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})'),
        extractor=lambda g: tuple(int(x) for x in g),   # already 6 groups
    ),

    _FnamePat(
        name='YYYYMMDD_HHMMSS',
        # e.g.  barn-20200228_181107_2secs  or  SITE_20220205_192049
        regex=re.compile(r'(\d{8})_(\d{6})'),
        extractor=lambda g: _ymd_hms(g[0], g[1]),
    ),

    _FnamePat(
        name='14-digit YYYYMMDDHHMMSS',
        # e.g.  prefix_20220205192049_suffix  (no separator between date/time)
        regex=re.compile(r'(\d{14})'),
        extractor=lambda g: _ymd_hms(g[0][:8], g[0][8:]),
    ),
]

# How many filenames to pull from the DB for the startup pattern survey
_SURVEY_SAMPLE_SIZE = 500


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DaytimeColumnAdder:
    '''
    Reads a bat-measures parquet file and its sibling noise parquet,
    appends was_daytime and time_of_day_pactime columns, and writes new
    parquet files with current-timestamp names alongside the originals.

    The recording datetime for each measures row is resolved by joining
    on file_id to the recordings table in the SQLite database and parsing
    the stored filename.  All sunrise/sunset logic is delegated to
    DaytimeFileSelector.

    Noise rows receive both new columns set to None.  The recordings
    table was built exclusively from files that produced measures, so
    noise file_ids are not present there.

    Both output files preserve the full BatsData envelope (file_map,
    normalizer state) from their originals so that Utils.read_df_file()
    works on them without a KeyError.

    :param measures_path: path to the bat measures parquet file
    :type measures_path: str
    :param db_path: path to chirp_meta.db SQLite database
    :type db_path: str
    '''

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, measures_path: str, db_path: str) -> None:
        self.log           = LoggingService()
        self.measures_path = Path(measures_path)
        self.db_path       = Path(db_path)

        if not self.measures_path.exists():
            self.log.warn(f"Measures file not found: {self.measures_path}")
            sys.exit(1)
        if not self.db_path.exists():
            self.log.warn(f"Database not found: {self.db_path}")
            sys.exit(1)

        # DaytimeFileSelector owns the Jasper Ridge astral location and
        # the Pacific Daylight Time zone object; we reuse both.
        self.selector = DaytimeFileSelector()

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self) -> tuple[Path, Path | None]:
        '''
        Orchestrate the full augmentation workflow:

        1. Locate the sibling noise parquet.
        2. Sample the DB and log which filename patterns are present.
        3. Load both dataframes.
        4. Build a file_id → (was_daytime, time_of_day_pactime) lookup.
        5. Append the two new columns to each dataframe.
        6. Write new parquet files next to their originals, preserving
           the BatsData envelope (file_map, normalizer state) from the
           originals.

        :return: Tuple of (new_measures_path, new_noise_path).  The noise
                 path is None when no sibling noise file was found.
        :rtype: tuple[Path, Path | None]
        '''
        noise_path = self._derive_noise_path(self.measures_path)
        if not noise_path.exists():
            self.log.warn(
                f"Sibling noise file not found at {noise_path}; "
                f"noise output will be skipped.")
            noise_path = None

        # ---- survey filename patterns present in the DB ------------------
        self._survey_filename_patterns()

        # ---- load --------------------------------------------------------
        self.log.info(f"Loading measures: {self.measures_path}")
        measures_df: pd.DataFrame = Utils.read_df_file(str(self.measures_path))
        self.log.info(f"  {len(measures_df):,} rows, "
                      f"{len(measures_df.columns)} columns")

        noise_df: pd.DataFrame | None = None
        if noise_path is not None:
            self.log.info(f"Loading noise: {noise_path}")
            noise_df = Utils.read_df_file(str(noise_path))
            self.log.info(f"  {len(noise_df):,} rows")

        # ---- build file_id lookup ----------------------------------------
        unique_ids = list(measures_df['file_id'].unique())
        self.log.info(
            f"Resolving {len(unique_ids):,} unique file_ids via recordings table ...")
        fid_map = self._build_fid_map(unique_ids)

        # ---- augment measures (populated) --------------------------------
        self.log.info("Appending daytime columns to measures ...")
        measures_aug = self._append_columns(measures_df, fid_map, populate=True)

        # ---- augment noise (always empty) --------------------------------
        noise_aug: pd.DataFrame | None = None
        if noise_df is not None:
            self.log.info("Appending empty daytime columns to noise ...")
            noise_aug = self._append_columns(noise_df, fid_map={}, populate=False)

        # ---- write -------------------------------------------------------
        # Re-use the BatsData envelope from the originals so the file_map and
        # normalizer state are preserved.  Plain DataFrame.to_parquet() would
        # strip the bats_metadata key, breaking Utils.read_df_file() on the
        # output.
        ts = datetime.now().strftime('%Y-%m-%dT%H_%M_%S.%f')

        out_measures = self.measures_path.parent / f"bats_{ts}.parquet"
        self.log.info(f"Writing measures -> {out_measures}")
        bats_m = BatsData.read_parquet(str(self.measures_path))
        bats_m.df = measures_aug
        bats_m.to_parquet(out_measures)

        out_noise: Path | None = None
        if noise_aug is not None:
            out_noise = noise_path.parent / f"bats_noise_{ts}.parquet"
            self.log.info(f"Writing noise   -> {out_noise}")
            bats_n = BatsData.read_parquet(str(noise_path))
            bats_n.df = noise_aug
            bats_n.to_parquet(out_noise)

        self.log.info("Done.")
        return out_measures, out_noise

    # ------------------------------------------------------------------
    # _survey_filename_patterns
    # ------------------------------------------------------------------

    def _survey_filename_patterns(self) -> None:
        '''
        Sample up to ``_SURVEY_SAMPLE_SIZE`` filenames from the recordings
        table and log how many match each known pattern, plus how many
        matched nothing.  This makes coverage (and any gaps) immediately
        visible in the run log without adding per-row overhead to the
        main processing loop.
        '''
        sql = f"SELECT filename FROM recordings LIMIT {_SURVEY_SAMPLE_SIZE}"
        with sqlite3.connect(str(self.db_path)) as conn:
            sample = [row[0] for row in conn.execute(sql).fetchall()]

        if not sample:
            self.log.warn("recordings table appears to be empty — cannot survey patterns.")
            return

        counts: dict[str, int] = defaultdict(int)
        n_unmatched = 0

        n_null = 0
        for raw in sample:
            if raw is None:
                n_null += 1
                continue
            fname   = raw.split('|')[0]
            matched = False
            for pat in _FNAME_PATTERNS:
                if pat.regex.search(fname):
                    counts[pat.name] += 1
                    matched = True
                    break       # first match wins; don't double-count
            if not matched:
                n_unmatched += 1

        self.log.info(
            f"Filename pattern survey (sample of {len(sample)} out of "
            f"{self._recordings_total()} total recordings -- percentages are of sample):")
        for pat in _FNAME_PATTERNS:
            n   = counts.get(pat.name, 0)
            pct = 100.0 * n / len(sample)
            self.log.info(f"  {pat.name:45s}  {n:5d} / {len(sample)}  ({pct:.1f}% of sample)")
        if n_null:
            self.log.warn(
                f"  {'NULL filename':45s}  {n_null:5d} / {len(sample)}  "
                f"({100.0 * n_null / len(sample):.1f}% of sample) "
                f"-- recordings rows with no filename; columns will be None.")
        if n_unmatched:
            self.log.warn(
                f"  {'NO PATTERN MATCHED':45s}  {n_unmatched:5d} / {len(sample)}  "
                f"({100.0 * n_unmatched / len(sample):.1f}% of sample) "
                f"-- consider adding a new entry to _FNAME_PATTERNS")

    # ------------------------------------------------------------------
    # _recordings_total
    # ------------------------------------------------------------------

    def _recordings_total(self) -> int:
        '''
        Return the total number of rows in the recordings table.
        Used to make the survey header informative about sample size
        vs. total population.

        :return: row count of the recordings table
        :rtype: int
        '''
        with sqlite3.connect(str(self.db_path)) as conn:
            return conn.execute('SELECT COUNT(*) FROM recordings').fetchone()[0]

    # ------------------------------------------------------------------
    # _build_fid_map
    # ------------------------------------------------------------------

    def _build_fid_map(
            self,
            file_ids: list[int],
    ) -> dict[int, tuple[bool | None, str | None]]:
        '''
        Query the recordings table for the given file_ids and return a
        dict mapping each to a ``(was_daytime, time_of_day_pactime)`` pair.

        The timestamp is extracted by trying every pattern in
        ``_FNAME_PATTERNS`` in order; the first match is used.

        :param file_ids: unique file_id values to resolve
        :type file_ids: list[int]
        :return: mapping from file_id to (bool | None, str | None)
        :rtype: dict
        '''
        # SQLite allows at most 999 bound variables per statement.
        # Chunk the id list and merge the results.
        _CHUNK = 900
        rows: list[tuple] = []
        with sqlite3.connect(str(self.db_path)) as conn:
            for start in range(0, len(file_ids), _CHUNK):
                chunk = file_ids[start:start + _CHUNK]
                placeholders = ','.join('?' * len(chunk))
                sql = (f"SELECT file_id, filename FROM recordings "
                       f"WHERE file_id IN ({placeholders})")
                rows.extend(conn.execute(sql, chunk).fetchall())

        self.log.info(f"  recordings table returned {len(rows):,} rows "
                      f"for {len(file_ids):,} requested ids.")

        n_failed: int = 0
        pat_counts: dict[str, int] = defaultdict(int)
        result: dict[int, tuple[bool | None, str | None]] = {}

        for fid, raw_filename in rows:
            if raw_filename is None:
                self.log.warn(f"file_id {fid}: NULL filename in recordings table -- columns set to None.")
                n_failed += 1
                result[fid] = (None, None)
                continue
            fname = raw_filename.split('|')[0]
            try:
                rec_time, pat_name = self._time_from_filename(fname)
                pat_counts[pat_name] += 1
                daytime = self.selector.is_daytime_recording(rec_time)
                iso_str = rec_time.isoformat()
            except ValueError as exc:
                self.log.warn(f"file_id {fid}: {exc} — columns set to None.")
                n_failed += 1
                daytime = None
                iso_str = None

            result[fid] = (daytime, iso_str)

        # Log which patterns were actually used
        self.log.info("Patterns used during file_id resolution:")
        for pat in _FNAME_PATTERNS:
            n = pat_counts.get(pat.name, 0)
            if n:
                self.log.info(f"  {pat.name}: {n:,}")
        if n_failed:
            self.log.warn(
                f"  {n_failed:,} filename(s) matched no pattern — "
                f"those rows will have None in both new columns.")

        unreachable = len(file_ids) - len(rows)
        if unreachable:
            self.log.warn(
                f"  {unreachable:,} file_id(s) had no entry in the "
                f"recordings table — those rows will have None in both "
                f"new columns.")

        return result

    # ------------------------------------------------------------------
    # _time_from_filename
    # ------------------------------------------------------------------

    def _time_from_filename(self, filename: str) -> tuple[datetime, str]:
        '''
        Try each pattern in ``_FNAME_PATTERNS`` in order and return the
        first successfully parsed timezone-aware datetime together with
        the name of the matching pattern.

        Recorders were configured to use Pacific Standard Time year-round
        (UTC-8, fixed offset — no DST adjustment).  The datetime is
        therefore constructed with a fixed UTC-8 timezone so that
        sunrise/sunset comparisons in DaytimeFileSelector are correct
        regardless of the time of year.

        :param filename: recording filename (pipe-suffix already removed)
        :type filename: str
        :return: (timezone-aware datetime in fixed PST, matched-pattern name)
        :rtype: tuple[datetime, str]
        :raises ValueError: if no pattern in _FNAME_PATTERNS matches
        '''
        import pytz
        # Fixed UTC-8 offset (PST year-round, as the recorders were set).
        # Using pytz.FixedOffset rather than self.selector.timezone avoids
        # the LMT-offset artifact that pytz produces when a tzinfo is passed
        # directly to the datetime constructor instead of via .localize().
        PST = pytz.FixedOffset(-8 * 60)

        for pat in _FNAME_PATTERNS:
            m = pat.regex.search(filename)
            if m is None:
                continue
            try:
                yr, mo, dy, hr, mi, sc = pat.extractor(m.groups())
                dt = PST.localize(datetime(yr, mo, dy,
                                           hour=hr, minute=mi, second=sc))
                return dt, pat.name
            except (ValueError, IndexError) as exc:
                # Groups matched but produced an invalid date/time value
                # (e.g. month=99); fall through and try the next pattern.
                self.log.warn(
                    f"Pattern '{pat.name}' matched '{filename}' "
                    f"but produced invalid datetime ({exc}); trying next.")
                continue

        raise ValueError(
            f"No pattern in _FNAME_PATTERNS matched '{filename}'")

    # ------------------------------------------------------------------
    # _append_columns
    # ------------------------------------------------------------------

    def _append_columns(self,
                        df: pd.DataFrame,
                        fid_map: dict[int, tuple],
                        populate: bool) -> pd.DataFrame:
        '''
        Return a copy of *df* with was_daytime and time_of_day_pactime
        appended.

        :param df: source dataframe
        :type df: pd.DataFrame
        :param fid_map: mapping from file_id to (was_daytime, iso_str)
        :type fid_map: dict
        :param populate: when True, values are drawn from fid_map;
            when False, both columns are set to None regardless
        :type populate: bool
        :return: augmented copy of df
        :rtype: pd.DataFrame
        '''
        df = df.copy()
        if populate:
            df['was_daytime'] = df['file_id'].map(
                lambda fid: fid_map.get(fid, (None, None))[0])
            df['time_of_day_pactime'] = df['file_id'].map(
                lambda fid: fid_map.get(fid, (None, None))[1])
        else:
            df['was_daytime']         = None
            df['time_of_day_pactime'] = None
        return df

    # ------------------------------------------------------------------
    # _derive_noise_path
    # ------------------------------------------------------------------

    def _derive_noise_path(self, measures_path: Path) -> Path:
        '''
        Infer the sibling noise parquet path from the measures path.

        Given a stem ``bats_TIMESTAMP`` the returned path has stem
        ``bats_noise_TIMESTAMP``.  For an unexpected stem format
        ``_noise`` is appended as a fallback.

        :param measures_path: path to the measures parquet file
        :type measures_path: Path
        :return: expected path of the sibling noise parquet file
        :rtype: Path
        '''
        stem  = measures_path.stem   # e.g. bats_2026-04-14T23_44_31.660585
        match = re.match(r'^([^_]+)_(.+)$', stem)
        if match:
            prefix, rest = match.groups()
            noise_stem = f"{prefix}_noise_{rest}"
        else:
            noise_stem = f"{stem}_noise"
        return measures_path.parent / f"{noise_stem}.parquet"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            'Add was_daytime and time_of_day_pactime columns to bat '
            'measures (and sibling noise) parquet files.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        'measures_file',
        help='Path to the bat measures parquet file '
             '(e.g. bats_2026-04-14T23_44_31.660585.parquet)',
    )
    ap.add_argument(
        'db_path',
        help='Path to the chirp_meta.db SQLite database',
    )
    return ap


if __name__ == '__main__':
    args = _build_parser().parse_args()
    new_measures, new_noise = DaytimeColumnAdder(
        measures_path=args.measures_file,
        db_path=args.db_path,
    ).run()
    print(f"Measures: {new_measures}")
    if new_noise:
        print(f"Noise:    {new_noise}")
