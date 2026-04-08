#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-04 15:36:10
# @File:   normalize_chop_stems.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-04 20:16:12
# **********************************************************

"""
normalize_chop_stems.py — Normalize sextus chop filenames to the quintus
canonical convention and stage them for transfer.

Background
----------
SonoBat on the sextus Windows VMs produces chop filenames in the ISO form::

    barn1_D20200228T180314m000.wav

The quintus pipeline expects the canonical convention::

    barn-20200228_180314_2secs.wav

This script:

1. Loads the two stem-map JSONs produced by bat_chops_deduping.py into
   DataFrames, parses ``(date, time)`` vectorially, and classifies every
   sextus stem as ``sextus_only``, ``content_differs``, or ``duplicate``
   via a DataFrame merge — no Python-level loops over stems.

2. For each ``sextus_only`` or ``content_differs`` chop .wav file, copies
   it to a staging directory under ``--staging-root`` with the canonical
   filename.  ``content_differs`` files receive a ``_1`` recorder-index
   suffix to coexist with the quintus copy.

3. For each batch's two Cumulative txt files
   (``*_CumulativeParameters_*.txt`` and ``*_CumulativeSonoBatch_*.txt``),
   reads them into DataFrames, rewrites ``Filename`` and ``Path`` columns
   via vectorised ``map()`` / ``str.replace()``, and writes them to
   staging.  The ``Path`` column is rewritten to the full quintus
   destination path so that ``sb_measures_postprocessing.py``'s
   PathEncoder produces useful file_map entries.

4. Writes an audit CSV (``normalization_audit.csv``) to the staging root —
   the classification DataFrame itself, requiring no separate accumulation.

5. Prints an advisory ``rclone`` command for transferring the staged data
   to quintus.

The quintus destination root is constructed as::

    <dest-parent>/<site>_sonobat<sb-version>_processed_<TIMESTAMP>

where TIMESTAMP is an ISO datetime (``YYYYMMDDTHHMMSS``) stamped at script
start.

Prefix handling
---------------
``--site`` supplied
    All output filenames use ``<site>-`` as prefix (trailing ``_``/``-``
    stripped, single ``-`` appended).  Original per-file prefix discarded.

``--site`` omitted
    Prefix extracted per file from the source stem (same normalisation).
    Files with no detectable prefix get no prefix in the output name.

Recorder-index suffix for content-differs collisions
-----------------------------------------------------
Overlap stems confirmed by ``bat_chops_deduping.py --audio-check`` to have
different audio content receive a ``_1`` suffix::

    quintus (existing):  barn-20220723_215745_2secs.wav
    sextus  (new):       barn-20220723_215745_2secs_1.wav

Pass ``--transfer-list`` pointing at ``confirmed_safe_to_transfer.txt``
to activate this logic.  Lines with ``# RENAME`` are the content-differs
stems; plain lines are true duplicates and are dropped.

Stem conventions handled
------------------------
* ``barn1_D20200228T180314m000``  — new SonoBat ISO form (sextus)
* ``barn_-20200228_180314``       — accidental double-separator (sextus)
* ``barn-20200228_180314``        — already canonical (some batch2 files)
* ``barn-20200228_180314_2secs``  — quintus form (canonical + suffix)

Typical usage
-------------
::

    python normalize_chop_stems.py \\
        --sextus-stems  /data/win_share/sextus_chops_file_stems.json \\
        --quintus-stems /data/win_share/quintus_chops_file_stems.json \\
        --win-share     /data/win_share \\
        --staging-root  /data/BatsTmp \\
        --dest-parent   /qnap/bats \\
        --site          barn \\
        --sb-version    3_2 \\
        --transfer-list /data/win_share/confirmed_safe_to_transfer.txt

Dry-run (no files written, advisory printed)::

    python normalize_chop_stems.py ... --dry-run
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from logging_service import LoggingService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Extracts date and time from any stem convention:
#   barn1_D20200228T180314m000  → date=20200228, time=180314
#   barn_-20200228_180314       → date=20200228, time=180314
#   barn-20200228_180314_2secs  → date=20200228, time=180314
_STEM_RE = re.compile(r'D?(\d{8})[T_](\d{6})')

# str.extract() pattern — same groups, used on a whole Series at once
_STEM_EXTRACT = r'D?(\d{8})[T_](\d{6})'

# Suffix present on quintus chop filenames, stripped when building stem keys
_DEDUP_STRIP = '_2secs'

# Column indices in Cumulative txt files (0-based, tab-separated).
# Both CumulativeParameters and CumulativeSonoBatch share:
#   col 0 = Path, col 1 = Filename
_COL_PATH     = 0
_COL_FILENAME = 1

# Status constants used in the audit / classification DataFrame
_STATUS_SEXTUS_ONLY     = 'sextus_only'
_STATUS_CONTENT_DIFFERS = 'content_differs'
_STATUS_DUPLICATE       = 'duplicate'
_STATUS_UNPARSEABLE     = 'unparseable'


# ---------------------------------------------------------------------------
# Class StemNormalizer
# ---------------------------------------------------------------------------

class StemNormalizer:
    """
    Normalize sextus chop filenames to the quintus canonical convention
    and stage them for rclone transfer to quintus.

    All stem classification is performed with vectorised Pandas operations;
    no Python-level loops over individual stems.

    :param sextus_stems_path: Path to sextus_chops_file_stems.json.
    :param quintus_stems_path: Path to quintus_chops_file_stems.json.
    :param win_share: Root of the sextus Windows share mount
                      (e.g. /data/win_share).
    :param staging_root: Root directory for staged output
                         (e.g. /data/BatsTmp).
    :param dest_parent: Parent directory on quintus where the run directory
                        will be created (e.g. /qnap/bats).
    :param site: Optional canonical prefix override (e.g. 'barn').
                 When None, prefix is inferred per stem.
    :param sb_version: SonoBat version string (e.g. '3_2').
    :param transfer_list_path: Optional path to confirmed_safe_to_transfer.txt
                               from bat_chops_deduping.py --audio-check.
    :param dry_run: If True, log actions but write nothing to disk.
    """

    def __init__(
        self,
        sextus_stems_path:  Path,
        quintus_stems_path: Path,
        win_share:          Path,
        staging_root:       Path,
        dest_parent:        Path,
        site:               Optional[str],
        sb_version:         str,
        transfer_list_path: Optional[Path] = None,
        dry_run:            bool = False,
    ) -> None:
        self.log = LoggingService()

        self.sextus_stems_path  = sextus_stems_path
        self.quintus_stems_path = quintus_stems_path
        self.win_share          = win_share
        self.staging_root       = staging_root
        self.dest_parent        = dest_parent
        self.sb_version         = sb_version
        self.transfer_list_path = transfer_list_path
        self.dry_run            = dry_run

        # Normalised override prefix (with trailing dash) or None
        self.site_prefix: Optional[str] = (
            self._normalize_prefix(site) if site else None
        )
        # Site label for directory naming — resolved in run() if not given
        self._site_label: Optional[str] = (
            site.strip('_-') if site else None
        )

        self.run_ts = datetime.now().strftime('%Y%m%dT%H%M%S')

        # Finalised in run() after site label is known
        self.dest_dir_name:     str  = ''
        self.quintus_dest_root: Path = Path('.')
        self.staging_dest:      Path = Path('.')

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full normalisation workflow.

        :return: None
        """
        # ── Load stem maps into DataFrames ────────────────────────────
        sextus_df  = self._load_stem_map(self.sextus_stems_path,  'sextus')
        quintus_df = self._load_stem_map(self.quintus_stems_path, 'quintus')

        # ── Parse (date, time) vectorially ───────────────────────────
        sextus_df  = self._parse_dt_into_df(sextus_df,  'sextus')
        quintus_df = self._parse_dt_into_df(quintus_df, 'quintus')

        # ── Resolve site label ────────────────────────────────────────
        if self._site_label is None:
            self._site_label = self._infer_site_label(sextus_df)
            self.log.info(
                f'Site label inferred from stems: {self._site_label!r}'
            )

        # ── Finalise destination paths ────────────────────────────────
        self.dest_dir_name = (
            f'{self._site_label}_sonobat{self.sb_version}'
            f'_processed_{self.run_ts}'
        )
        self.quintus_dest_root = self.dest_parent / self.dest_dir_name
        self.staging_dest      = self.staging_root / self.dest_dir_name

        self.log.info(f'Run timestamp       : {self.run_ts}')
        self.log.info(f'Site label          : {self._site_label}')
        self.log.info(
            f'Site prefix override: {self.site_prefix!r} '
            f'({"explicit --site" if self.site_prefix else "inferred per stem"})'
        )
        self.log.info(f'Quintus dest root   : {self.quintus_dest_root}')
        self.log.info(f'Staging dest        : {self.staging_dest}')
        self.log.info(f'Dry run             : {self.dry_run}')

        # ── Load transfer list ────────────────────────────────────────
        rename_paths: set[str] = set()
        if self.transfer_list_path:
            rename_paths = self._load_transfer_list(self.transfer_list_path)
            self.log.info(
                f'Transfer list: {len(rename_paths):,} content-differs '
                f'paths from {self.transfer_list_path}'
            )
        else:
            self.log.info(
                'No --transfer-list; all overlaps treated as duplicates'
            )

        # ── Classify sextus stems ─────────────────────────────────────
        sextus_df = self._classify(sextus_df, quintus_df, rename_paths)

        counts = sextus_df['status'].value_counts()
        self.log.info(f'Classification results:')
        for status, n in counts.items():
            self.log.info(f'  {status:20s}: {n:,}')

        # ── Build canonical stems and destination paths ───────────────
        sextus_df = self._build_canonical_cols(sextus_df)

        # ── Stage files and rewrite txt per batch ─────────────────────
        to_stage = sextus_df[
            sextus_df['status'].isin(
                [_STATUS_SEXTUS_ONLY, _STATUS_CONTENT_DIFFERS]
            )
        ]
        for batch_num in sorted(to_stage['batch'].unique()):
            self._process_batch(
                batch_num,
                to_stage[to_stage['batch'] == batch_num],
            )

        # ── Write audit CSV ───────────────────────────────────────────
        audit_path = self.staging_root / 'normalization_audit.csv'
        self._write_audit(sextus_df, audit_path)

        # ── Advisory rclone command ───────────────────────────────────
        self._print_rclone_advisory()

    # ------------------------------------------------------------------
    # Stem map loading
    # ------------------------------------------------------------------

    def _load_stem_map(self, path: Path, label: str) -> pd.DataFrame:
        """
        Load a stem-map JSON into a DataFrame with columns
        ``stem``, ``path``, ``batch``.

        :param path: Path to the JSON file.
        :param label: Human label for logging.
        :return: DataFrame with columns ['stem', 'path', 'batch'].
        :raises SystemExit: If the file cannot be read.
        """
        try:
            with path.open() as fh:
                payload = json.load(fh)
        except Exception as exc:
            self.log.err(f'Cannot load {label} stems from {path}: {exc}')
            sys.exit(1)

        stems_dict = payload.get('stems', {})
        df = pd.DataFrame(
            [
                {'stem': stem, 'path': rec['path'], 'batch': rec['batch']}
                for stem, rec in stems_dict.items()
            ],
            columns=['stem', 'path', 'batch'],
        )
        self.log.info(f'Loaded {len(df):,} stems from {label} ({path})')
        return df

    # ------------------------------------------------------------------
    # Vectorised date/time parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dt_into_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Parse ``date`` (YYYYMMDD) and ``time`` (HHMMSS) from the ``stem``
        column using a vectorised ``str.extract()``.

        Rows where the stem does not match are marked with NaN in both
        columns and logged.

        :param df: DataFrame with a ``stem`` column.
        :param label: Human label for logging.
        :return: DataFrame with ``date`` and ``time`` columns added.
        """
        extracted = df['stem'].str.extract(_STEM_EXTRACT, expand=True)
        extracted.columns = ['date', 'time']
        df = pd.concat([df, extracted], axis=1)

        n_bad = df['date'].isna().sum()
        if n_bad:
            LoggingService().warn(
                f'{label}: {n_bad:,} stems could not be parsed '
                f'(date/time NaN) — will be marked unparseable'
            )
        return df

    # ------------------------------------------------------------------
    # Prefix helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_prefix(raw: str) -> str:
        """
        Strip trailing underscores/dashes from *raw* and append one dash.

        :param raw: Raw prefix string.
        :return: Normalised prefix ending with exactly one dash.
        """
        return raw.rstrip('_-') + '-'

    @staticmethod
    def _extract_raw_prefix_series(stems: pd.Series) -> pd.Series:
        """
        Vectorised extraction of the raw prefix from a Series of stems.

        Returns everything before the date/time match, with a lone
        trailing ISO ``D`` marker stripped.

        :param stems: Series of stem strings.
        :return: Series of raw prefix strings (may be empty string).
        """
        # Everything before the date match; strip trailing ISO 'D' marker
        # (e.g. 'barn1_D' → 'barn1_') so it is not treated as part of prefix.
        raw = stems.str.extract(r'^(.*?)D?(\d{8})[T_]\d{6}', expand=True)[0]
        raw = raw.fillna('')
        raw = raw.str.replace(r'[_\-]D$', lambda m: m.group(0)[:-1],
                              regex=True)
        return raw.fillna('')

    def _build_prefix_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Build the normalised output prefix for every row in *df*.

        Uses the explicit override when ``self.site_prefix`` is set;
        otherwise extracts and normalises the prefix from each stem.

        :param df: DataFrame with a ``stem`` column.
        :return: Series of normalised prefix strings (e.g. 'barn-' or '').
        """
        if self.site_prefix is not None:
            return pd.Series(self.site_prefix, index=df.index)

        raw = self._extract_raw_prefix_series(df['stem'])
        # Strip all leading/trailing separators, then append a single dash
        # to non-empty values:  'barn_-' → 'barn-',  '' → ''
        prefix = raw.str.strip('_-')
        prefix = prefix.where(prefix == '', prefix + '-')
        return prefix

    def _infer_site_label(self, df: pd.DataFrame) -> str:
        """
        Infer a site label from the first parseable stem in *df*.

        :param df: Sextus DataFrame with ``stem`` column.
        :return: Inferred site label string, or 'unknown'.
        """
        raw = self._extract_raw_prefix_series(df['stem'])
        labels = raw.str.strip('_-')
        non_empty = labels[labels != '']
        if len(non_empty):
            return non_empty.iloc[0]
        return 'unknown'

    # ------------------------------------------------------------------
    # Transfer list
    # ------------------------------------------------------------------

    @staticmethod
    def _load_transfer_list(path: Path) -> set[str]:
        """
        Parse ``confirmed_safe_to_transfer.txt`` and return the set of
        sextus source paths annotated with ``# RENAME`` (content-differs).

        Plain lines (true duplicates) are not returned.

        :param path: Path to confirmed_safe_to_transfer.txt.
        :return: Set of absolute sextus path strings.
        :raises RuntimeError: If the file cannot be read.
        """
        rename_paths: set[str] = set()
        try:
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '# RENAME' in line:
                        source_path = line.split('# RENAME')[0].strip()
                        rename_paths.add(source_path)
        except Exception as exc:
            raise RuntimeError(
                f'Cannot read transfer list {path}: {exc}'
            ) from exc
        return rename_paths

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        sextus_df:    pd.DataFrame,
        quintus_df:   pd.DataFrame,
        rename_paths: set[str],
    ) -> pd.DataFrame:
        """
        Classify every sextus stem as ``sextus_only``, ``content_differs``,
        ``duplicate``, or ``unparseable`` via DataFrame merge operations.

        :param sextus_df: Sextus DataFrame with date/time columns.
        :param quintus_df: Quintus DataFrame with date/time columns.
        :param rename_paths: Sextus paths confirmed as content-differs.
        :return: Sextus DataFrame with a ``status`` column added.
        """
        # Mark unparseable rows first
        bad_mask = sextus_df['date'].isna()
        sextus_df = sextus_df.copy()
        sextus_df['status'] = _STATUS_UNPARSEABLE

        # Build quintus (date, time) key set — drop unparseable quintus rows
        quintus_keys = (
            quintus_df.dropna(subset=['date', 'time'])[['date', 'time']]
            .drop_duplicates()
            .assign(_in_quintus=True)
        )

        # Left-join sextus onto quintus keys
        merged = sextus_df[~bad_mask].merge(
            quintus_keys, on=['date', 'time'], how='left'
        )

        in_quintus  = (
            merged['_in_quintus']
            .fillna(False)
            .infer_objects(copy=False)
            .astype(bool)
        )
        is_rename   = merged['path'].isin(rename_paths)

        # Assign status
        status = pd.Series(_STATUS_SEXTUS_ONLY, index=merged.index)
        status[in_quintus &  is_rename] = _STATUS_CONTENT_DIFFERS
        status[in_quintus & ~is_rename] = _STATUS_DUPLICATE
        merged['status'] = status
        merged.drop(columns=['_in_quintus'], inplace=True)

        # Recombine with unparseable rows
        result = pd.concat(
            [merged, sextus_df[bad_mask]], ignore_index=True
        )
        return result

    # ------------------------------------------------------------------
    # Canonical stem and path columns
    # ------------------------------------------------------------------

    def _build_canonical_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ``canonical_stem``, ``staging_path``, and ``quintus_path``
        columns to the classification DataFrame.

        Only rows with status ``sextus_only`` or ``content_differs`` get
        non-empty path columns (others receive empty string).

        :param df: Classification DataFrame.
        :return: DataFrame with three new columns added.
        """
        df = df.copy()

        prefix = self._build_prefix_series(df)

        # Base canonical stem: <prefix><date>_<time>_2secs
        base = (
            prefix
            + df['date'].fillna('')
            + '_'
            + df['time'].fillna('')
            + '_2secs'
        )
        # content_differs gets _1 suffix
        is_cd = df['status'] == _STATUS_CONTENT_DIFFERS
        df['canonical_stem'] = base.where(~is_cd, base + '_1')
        # Clear canonical_stem for rows with no date (unparseable)
        df.loc[df['date'].isna(), 'canonical_stem'] = ''

        # Subdirectory names
        prefix_root = prefix.str.rstrip('-')
        chop_subdir = (prefix_root + '_chopped_files').where(
            prefix_root != '', 'chopped_files'
        )
        parsed_subdir = (prefix_root + '_Parsed_Files_' + df['date'].fillna('')).where(
            prefix_root != '',
            'Parsed_Files_' + df['date'].fillna('')
        )

        # Only compute paths for rows that will be staged
        to_stage_mask = df['status'].isin(
            [_STATUS_SEXTUS_ONLY, _STATUS_CONTENT_DIFFERS]
        )

        src_path_col  = pd.Series('', index=df.index)
        dest_path_col = pd.Series('', index=df.index)

        if to_stage_mask.any():
            sub           = df[to_stage_mask]
            canonical_wav = df['canonical_stem'] + '.wav'

            src_path_series = (
                pd.Series(str(self.staging_dest), index=sub.index)
                .str.cat('batch' + sub['batch'].astype(str), sep='/')
                .str.cat(chop_subdir,                        sep='/')
                .str.cat(parsed_subdir,                      sep='/')
                .str.cat(canonical_wav,                      sep='/')
            )
            dest_path_series = (
                pd.Series(str(self.quintus_dest_root), index=sub.index)
                .str.cat('batch' + sub['batch'].astype(str), sep='/')
                .str.cat(chop_subdir,                        sep='/')
                .str.cat(parsed_subdir,                      sep='/')
                .str.cat(canonical_wav,                      sep='/')
            )
            src_path_col[to_stage_mask]  = src_path_series
            dest_path_col[to_stage_mask] = dest_path_series

        df['src_path']  = src_path_col
        df['dest_path'] = dest_path_col
        return df

    # ------------------------------------------------------------------
    # Per-batch processing
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        batch_num: int,
        batch_df:  pd.DataFrame,
    ) -> None:
        """
        Copy .wav files and rewrite Cumulative txt files for one batch.

        :param batch_num: Integer batch number.
        :param batch_df: Rows to stage for this batch.
        :return: None
        """
        self.log.info(
            f'--- batch{batch_num}: {len(batch_df):,} files to stage ---'
        )
        self._stage_wav_files(batch_num, batch_df)
        self._rewrite_cumulative_files(batch_num, batch_df)

    def _stage_wav_files(
        self,
        batch_num: int,
        batch_df:  pd.DataFrame,
    ) -> None:
        """
        Copy sextus .wav files to staging with canonical filenames.

        :param batch_num: Integer batch number.
        :param batch_df: Rows to stage (status sextus_only or content_differs).
        :return: None
        """
        n_copied  = 0
        n_missing = 0

        for _, row in batch_df.iterrows():
            source_wav = Path(row['path'])
            staged_wav = Path(row['src_path'])

            if not source_wav.exists():
                self.log.warn(f'  Source missing: {source_wav}')
                n_missing += 1
                continue

            if not self.dry_run:
                staged_wav.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_wav, staged_wav)
            n_copied += 1

        self.log.info(
            f'  batch{batch_num} .wav: {n_copied} '
            f'{"would copy" if self.dry_run else "copied"}, '
            f'{n_missing} missing source'
        )

    def _rewrite_cumulative_files(
        self,
        batch_num: int,
        batch_df:  pd.DataFrame,
    ) -> None:
        """
        Rewrite Filename and Path columns in the two Cumulative txt files
        for one batch using vectorised DataFrame operations, then write to
        staging.

        Rows whose Filename does not appear in *batch_df* are passed
        through unchanged (overlap/duplicate rows stay as-is).

        :param batch_num: Integer batch number.
        :return: None
        """
        batch_dir = self.win_share / f'batch{batch_num}'
        site_root = (
            self.site_prefix.rstrip('-') if self.site_prefix
            else self._site_label
        )

        # Build lookup: bare_old_stem → (new_filename, dest_path)
        # Using dicts for O(1) map() lookups inside pandas
        bare_to_new_filename: dict[str, str] = {}
        bare_to_dest_path:    dict[str, str] = {}

        for _, row in batch_df.iterrows():
            bare = self._bare_stem(row['stem'])
            bare_to_new_filename[bare] = f'{row["canonical_stem"]}.wav'
            bare_to_dest_path[bare]    = row['dest_path']

        for kind in ('Parameters', 'SonoBatch'):
            all_matches = sorted(batch_dir.glob(f'*_Cumulative{kind}_*.txt'))
            # Exclude summary files — they have a different schema
            src_files = [
                p for p in all_matches
                if not any(kw in p.name
                           for kw in ('BatchSummary', 'NightlySummary'))
            ]
            if not src_files:
                self.log.warn(
                    f'  No Cumulative{kind} file found in {batch_dir}'
                )
                continue
            if len(src_files) > 1:
                self.log.warn(
                    f'  Multiple Cumulative{kind} files — '
                    f'using first: {src_files[0].name}'
                )
            src_file = src_files[0]

            dest_filename = f'{site_root}_{src_file.name}'
            dest_path = (
                self.staging_dest / f'batch{batch_num}' / dest_filename
            )

            self._rewrite_one_cumulative(
                src_file             = src_file,
                dest_path            = dest_path,
                bare_to_new_filename = bare_to_new_filename,
                bare_to_dest_path    = bare_to_dest_path,
                batch_num            = batch_num,
                kind                 = kind,
            )

    def _rewrite_one_cumulative(
        self,
        src_file:             Path,
        dest_path:            Path,
        bare_to_new_filename: dict[str, str],
        bare_to_dest_path:    dict[str, str],
        batch_num:            int,
        kind:                 str,
        chunksize:            int = 200_000,
    ) -> None:
        """
        Read one Cumulative txt file in chunks, rewrite Path and Filename
        columns via vectorised map(), and write to staging incrementally.

        Processing one chunk at a time keeps peak memory proportional to
        *chunksize* rows rather than the full file size.

        :param src_file: Source Cumulative txt file.
        :param dest_path: Destination path in staging.
        :param bare_to_new_filename: Bare stem → canonical filename mapping.
        :param bare_to_dest_path: Bare stem → quintus destination path mapping.
        :param batch_num: Batch number (for logging).
        :param kind: 'Parameters' or 'SonoBatch' (for logging).
        :param chunksize: Rows per chunk (default 200,000).
        :return: None
        """
        self.log.info(
            f'  Rewriting {kind} for batch{batch_num}: {src_file.name}'
        )

        try:
            chunk_iter = pd.read_csv(
                src_file,
                sep='\t',
                dtype=str,
                keep_default_na=False,
                low_memory=False,
                chunksize=chunksize,
            )
        except Exception as exc:
            self.log.err(f'  Cannot read {src_file}: {exc}')
            return

        if not self.dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        n_rewritten  = 0
        n_unchanged  = 0
        first_chunk  = True

        for chunk in chunk_iter:
            if chunk.empty:
                continue

            col_names    = list(chunk.columns)
            path_col     = col_names[_COL_PATH]
            filename_col = col_names[_COL_FILENAME]

            # Bare stem from Filename column (strip .wav and _2secs)
            bare = (
                chunk[filename_col]
                .str.replace(r'\.wav$', '', case=False, regex=True)
                .str.replace(r'_2secs$', '',            regex=True)
            )

            new_filename  = bare.map(bare_to_new_filename)
            new_dest_path = bare.map(bare_to_dest_path)

            n_rewritten += int(new_filename.notna().sum())
            n_unchanged += int(new_filename.isna().sum())

            chunk[filename_col] = new_filename.fillna(chunk[filename_col])
            chunk[path_col]     = new_dest_path.fillna(chunk[path_col])

            if not self.dry_run:
                chunk.to_csv(
                    dest_path,
                    sep='\t',
                    index=False,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    na_rep='',
                )

            first_chunk = False
            del chunk      # explicit release — important for large files

        self.log.info(
            f'    {n_rewritten:,} rows rewritten, '
            f'{n_unchanged:,} rows unchanged'
        )
        if self.dry_run:
            self.log.info(f'    [dry-run] Would write to {dest_path}')
        else:
            self.log.info(f'    Written to {dest_path}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bare_stem(stem: str) -> str:
        """
        Strip .wav extension and _2secs suffix from a stem string.

        :param stem: Raw stem string.
        :return: Bare stem for lookup against Filename column.
        """
        s = stem
        if s.lower().endswith('.wav'):
            s = s[:-4]
        if s.endswith(_DEDUP_STRIP):
            s = s[: -len(_DEDUP_STRIP)]
        return s

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _write_audit(self, df: pd.DataFrame, audit_path: Path) -> None:
        """
        Write the classification DataFrame to a CSV audit file.

        :param df: Full classification DataFrame.
        :param audit_path: Destination CSV path.
        :return: None
        """
        audit_cols = [
            'stem', 'path', 'batch', 'date', 'time',
            'canonical_stem', 'status', 'src_path', 'dest_path',
        ]
        out = df[[c for c in audit_cols if c in df.columns]]

        if not self.dry_run:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(audit_path, index=False)
            self.log.info(
                f'Audit CSV: {len(out):,} rows → {audit_path}'
            )
        else:
            self.log.info(
                f'[dry-run] Would write audit CSV '
                f'with {len(out):,} rows to {audit_path}'
            )

    # ------------------------------------------------------------------
    # Advisory rclone command
    # ------------------------------------------------------------------

    def _print_rclone_advisory(self) -> None:
        """
        Print the rclone command to transfer staged data to quintus.

        :return: None
        """
        src  = self.staging_dest
        dest = self.quintus_dest_root
        msg = (
            f'\n'
            f'{"=" * 70}\n'
            f'Staging complete.  Run timestamp : {self.run_ts}\n'
            f'Site label                       : {self._site_label}\n'
            f'\n'
            f'Quintus destination directory:\n'
            f'  {dest}\n'
            f'\n'
            f'To transfer staged data to quintus:\n'
            f'\n'
            f'  rclone copy \\\n'
            f'      {src} \\\n'
            f'      stanford:{dest} \\\n'
            f'      --transfers 16 --checkers 32 --buffer-size 32M \\\n'
            f'      --progress\n'
            f'\n'
            f'To verify before transferring (dry-run):\n'
            f'\n'
            f'  rclone copy \\\n'
            f'      {src} \\\n'
            f'      stanford:{dest} \\\n'
            f'      --transfers 16 --checkers 32 --buffer-size 32M \\\n'
            f'      --dry-run\n'
            f'{"=" * 70}\n'
        )
        print(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for :class:`StemNormalizer`.

    :return: Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog='normalize_chop_stems.py',
        description=(
            'Normalize sextus chop filenames to the quintus canonical\n'
            'convention and stage them for rclone transfer.\n\n'
            'All stem classification uses vectorised Pandas operations.\n\n'
            'Produces:\n'
            '  * Renamed .wav files under --staging-root\n'
            '  * Rewritten Cumulative txt files under --staging-root\n'
            '  * normalization_audit.csv under --staging-root\n'
            '  * Advisory rclone command printed to stdout\n\n'
            'The quintus destination directory is named:\n'
            '  <site>_sonobat<sb-version>_processed_<TIMESTAMP>\n'
            'where TIMESTAMP is stamped at script start.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--sextus-stems',
        required=True, metavar='JSON', type=Path,
        help='sextus_chops_file_stems.json from bat_chops_deduping.py.',
    )
    parser.add_argument(
        '--quintus-stems',
        required=True, metavar='JSON', type=Path,
        help='quintus_chops_file_stems.json from bat_chops_deduping.py.',
    )
    parser.add_argument(
        '--win-share',
        required=True, metavar='DIR', type=Path,
        help='Root of the sextus Windows share (e.g. /data/win_share).',
    )
    parser.add_argument(
        '--staging-root',
        required=True, metavar='DIR', type=Path,
        help='Root for staged output (e.g. /data/BatsTmp).',
    )
    parser.add_argument(
        '--dest-parent',
        required=True, metavar='DIR', type=Path,
        help=(
            'Parent directory on quintus for the run directory\n'
            '(e.g. /qnap/bats).  Used to construct quintus Path values\n'
            'in rewritten txt files.'
        ),
    )
    parser.add_argument(
        '--site',
        default=None, metavar='SITE',
        help=(
            "Optional canonical prefix override (e.g. 'barn').\n"
            'Trailing _/- stripped, single - appended.\n'
            'Also sets the site label for directory naming.\n'
            'When omitted, prefix and label inferred from stems.'
        ),
    )
    parser.add_argument(
        '--sb-version',
        required=True, metavar='VER',
        help=(
            "SonoBat version string (e.g. '3_2').\n"
            'Used in: <site>_sonobat<ver>_processed_<TIMESTAMP>.'
        ),
    )
    parser.add_argument(
        '--transfer-list',
        default=None, metavar='FILE', type=Path,
        help=(
            'confirmed_safe_to_transfer.txt from bat_chops_deduping.py\n'
            '--audio-check.  Lines with "# RENAME" are content-differs\n'
            'collision files; staged with _1 recorder-index suffix.\n'
            'Plain lines (true duplicates) are dropped.\n'
            'When omitted, all overlaps are dropped.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log all actions but write nothing to disk.',
    )

    args = parser.parse_args()

    for attr, flag in [
        ('sextus_stems',  '--sextus-stems'),
        ('quintus_stems', '--quintus-stems'),
        ('win_share',     '--win-share'),
        ('staging_root',  '--staging-root'),
    ]:
        p = getattr(args, attr)
        if not p.exists():
            parser.error(f'{flag} path does not exist: {p}')

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point for :class:`StemNormalizer`.

    :return: None
    """
    args = _parse_args()

    normalizer = StemNormalizer(
        sextus_stems_path  = args.sextus_stems,
        quintus_stems_path = args.quintus_stems,
        win_share          = args.win_share,
        staging_root       = args.staging_root,
        dest_parent        = args.dest_parent,
        site               = args.site,
        sb_version         = args.sb_version,
        transfer_list_path = args.transfer_list,
        dry_run            = args.dry_run,
    )
    normalizer.run()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    