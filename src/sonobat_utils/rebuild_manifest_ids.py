#!/usr/bin/env python3
 # **********************************************************
 #
 # @Author: Andreas Paepcke
 # @Date:   2026-04-11 19:04:34
 # @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/rebuild_manifest_ids.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-04-11 19:09:44
 #
 # **********************************************************
"""
rebuild_manifest_ids.py
=======================
After re-running sb_measures_postprocessing.py, the parquet's file_id and
chirp_idx are authoritative.  This script rebuilds those two columns in the
manifest CSV so that (file_id, chirp_idx) is the same unique key in both
files, enabling direct joins for the SQLite metadata database.

Steps
-----
1. Read the parquet's embedded file_map (stem → file_id).
2. Derive new file_id in manifest by mapping Filename stem → parquet file_id.
3. Build a (file_id, TimeInFile) → chirp_idx lookup from the parquet.
4. Derive chirp_idx in manifest by joining on (new_file_id, time_in_file_ms).
5. Write corrected manifest (backup of original written first).

Rows whose Filename stem is absent from the parquet file_map (i.e. fragments
that were filtered out of the clean parquet by the confidence threshold) will
have file_id and chirp_idx set to -1 and are flagged in the log.  They remain
in the manifest so no PNG references are lost — they just won't join to the
measures parquet.

Usage
-----
    python rebuild_manifest_ids.py \\
        --parquet  /qnap/bats/all_data/bats_<timestamp>.parquet \\
        --manifest /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv

The corrected manifest is written in-place after a .bak backup is made.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from logging_service import LoggingService

log = LoggingService()

_THRIFT_LIMIT = 1_000_000_000


class ManifestRebuilder:
    """
    Rebuild file_id and chirp_idx columns in a manifest CSV to match
    the authoritative values in a parquet produced by
    sb_measures_postprocessing.py.

    :param parquet_path:  Path to the bats_<timestamp>.parquet file.
    :param manifest_path: Path to manifest.csv from chirps_to_spectros.py.
    """

    def __init__(self, parquet_path: str | Path, manifest_path: str | Path):
        self.parquet_path  = Path(parquet_path)
        self.manifest_path = Path(manifest_path)

    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full rebuild.  Writes a .bak backup of the original
        manifest, then overwrites the manifest with corrected columns.

        :return: None
        """
        # ── Step 1: load parquet file_map from schema metadata ─────────
        log.info(f'Reading parquet metadata: {self.parquet_path}')
        pf       = pq.ParquetFile(
            self.parquet_path,
            thrift_string_size_limit    = _THRIFT_LIMIT,
            thrift_container_size_limit = _THRIFT_LIMIT,
        )
        meta_raw = pf.schema_arrow.metadata or {}
        meta_key = b'bats_metadata'
        if meta_key not in meta_raw:
            log.err(
                f'{self.parquet_path} has no bats_metadata — '
                f'was it written by BatsData.to_parquet()?'
            )
            sys.exit(1)

        file_map: dict[int, str] = {
            int(k): v
            for k, v in json.loads(meta_raw[meta_key].decode())['file_map'].items()
        }
        # Invert: stem → file_id
        stem_to_fid: dict[str, int] = {v: k for k, v in file_map.items()}
        log.info(f'  {len(stem_to_fid):,} stems in parquet file_map')

        # ── Step 2: load parquet (file_id, TimeInFile, chirp_idx) ──────
        log.info('Reading parquet (file_id, TimeInFile, chirp_idx) ...')
        df_pq = pf.read(columns=['file_id', 'TimeInFile', 'chirp_idx']).to_pandas()
        log.info(f'  {len(df_pq):,} parquet rows')

        # Build lookup: (file_id, TimeInFile_ms_int) → chirp_idx
        # TimeInFile is stored as raw ms (integer) — cast to int for safety.
        df_pq['tif_key'] = df_pq['TimeInFile'].round().astype(int)
        chirp_lookup: dict[tuple[int, int], int] = {
            (int(row.file_id), int(row.tif_key)): int(row.chirp_idx)
            for row in df_pq.itertuples(index=False)
        }
        log.info(f'  {len(chirp_lookup):,} (file_id, TimeInFile) keys in lookup')

        # ── Step 3: load manifest ───────────────────────────────────────
        log.info(f'Reading manifest: {self.manifest_path}')
        df_man = pd.read_csv(self.manifest_path, low_memory=False)
        log.info(f'  {len(df_man):,} manifest rows')

        # ── Step 4: backup ──────────────────────────────────────────────
        bak_path = self.manifest_path.with_suffix('.csv.bak')
        shutil.copy2(self.manifest_path, bak_path)
        log.info(f'  Backup written: {bak_path}')

        # ── Step 5: rebuild file_id ─────────────────────────────────────
        df_man['file_id'] = df_man['Filename'].map(stem_to_fid)
        n_unmapped_fid = df_man['file_id'].isna().sum()
        if n_unmapped_fid:
            log.warn(
                f'  {n_unmapped_fid:,} manifest rows have a Filename stem '
                f'absent from the parquet file_map (filtered by conf threshold '
                f'or from old data). file_id set to -1 for these rows.'
            )
        df_man['file_id'] = df_man['file_id'].fillna(-1).astype(int)

        # ── Step 6: rebuild chirp_idx ───────────────────────────────────
        tif_key = df_man['time_in_file_ms'].round().astype('Int64')
        fid_key = df_man['file_id']

        def _lookup_chirp_idx(row_fid, row_tif):
            if row_fid == -1 or pd.isna(row_tif):
                return -1
            return chirp_lookup.get((int(row_fid), int(row_tif)), -1)

        log.info('Assigning chirp_idx (vectorized lookup) ...')
        df_man['chirp_idx'] = [
            _lookup_chirp_idx(fid, tif)
            for fid, tif in zip(fid_key, tif_key)
        ]

        n_unmapped_cidx = (df_man['chirp_idx'] == -1).sum()
        n_already_bad   = n_unmapped_fid   # already counted
        n_new_missing   = n_unmapped_cidx - n_already_bad
        if n_new_missing > 0:
            log.warn(
                f'  {n_new_missing:,} rows have a valid file_id but '
                f'(file_id, time_in_file_ms) not found in parquet — '
                f'possibly outlier-filtered rows. chirp_idx set to -1.'
            )

        # ── Step 7: report coverage ─────────────────────────────────────
        n_total   = len(df_man)
        n_matched = (df_man['chirp_idx'] >= 0).sum()
        log.info(
            f'Coverage: {n_matched:,}/{n_total:,} manifest rows '
            f'({100*n_matched/n_total:.1f}%) have valid (file_id, chirp_idx). '
            f'{n_total - n_matched:,} rows set to -1.'
        )

        # ── Step 8: write corrected manifest ───────────────────────────
        df_man.to_csv(self.manifest_path, index=False)
        log.info(f'Corrected manifest written: {self.manifest_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Rebuild file_id and chirp_idx in manifest.csv to match the '
            'authoritative values in the bats_*.parquet produced by '
            'sb_measures_postprocessing.py.'
        )
    )
    parser.add_argument(
        '--parquet', required=True, metavar='PATH',
        help='Path to bats_<timestamp>.parquet'
    )
    parser.add_argument(
        '--manifest', required=True, metavar='PATH',
        help='Path to manifest.csv (edited in-place; .bak backup written first)'
    )
    args = parser.parse_args()

    if not Path(args.parquet).exists():
        parser.error(f'Parquet not found: {args.parquet}')
    if not Path(args.manifest).exists():
        parser.error(f'Manifest not found: {args.manifest}')

    ManifestRebuilder(
        parquet_path  = args.parquet,
        manifest_path = args.manifest,
    ).run()


if __name__ == '__main__':
    main()
