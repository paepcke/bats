#!/usr/bin/env python
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-24 18:08:33
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-24 18:08:59
#!/usr/bin/env python3
# **********************************************************

"""
merge_into_main.py
===================
Merge a new-site ``bats_<ts>.parquet`` produced by
:mod:`from_scratch_postprocessing` into the main corpus parquet, and
propagate the merge into the spectrogram manifest and the SQLite
``chirp_meta.db``.

This is the final step for integrating a new recording site (e.g. marsh)
into the existing barn/lake2 corpus so that all downstream consumers —
RF/CNN training, ``bat_db_builder``, ``sb_measures_add_daytime_columns``
— see a single unified dataset.

Prerequisite: stable file_ids
------------------------------
For this merge to be safe, ``from_scratch_postprocessing.py`` must have
been run with ``--existing-parquet <main_parquet>`` so that the new-site
parquet's file_ids were assigned by extending the existing PathEncoder
rather than starting fresh from 0.  This script verifies that no
file_id collision exists and aborts if one is found.

What gets merged
-----------------
``bats_<ts>.parquet``  (clean measures, main corpus)
    New rows are appended.  The merged DataFrame is written as a new
    ``bats_<merged_ts>.parquet``; the originals are not modified.

``bats_noise_<ts>.parquet``  (noise measures, main corpus)
    Same treatment if ``--new-noise-parquet`` is supplied.

``manifest.csv``
    New rows (one per spectrogram PNG crop) are appended.  Existing rows
    are preserved exactly.  Duplicate ``(file_id, chirp_idx, harmonic_idx)``
    rows are dropped (keeping the existing row) in case of accidental
    re-runs.  Crop paths in the new manifest are rewritten if
    ``--remap-crop-root`` is supplied (useful when the new-site crops
    live under a different root than the main corpus crops).

``chirp_meta.db``  (SQLite)
    New rows are inserted into the ``recordings`` table via
    ``INSERT OR IGNORE`` so re-runs are safe.  ``chirp_info`` and
    ``chirp_spectrograms`` rows are left to ``bat_db_builder.py``, which
    populates them from the merged manifest — run it after this script.

Outputs
-------
All output files are written to ``--dest-dir``.  Input files are never
modified.  The merged parquet filename carries a fresh timestamp so the
run is fully traceable.

CLI usage
---------
Typical usage after processing a new marsh batch::

    python merge_into_main.py \\
        --main-parquet   /qnap/bats/all_data/bats_2026-04-22T16_40_00.parquet \\
        --new-parquet    /qnap/src/marsh_stanford_processed/bats_<ts>.parquet \\
        --main-manifest  /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --new-manifest   /qnap/src/marsh_stanford_processed/manifest.csv \\
        --dest-dir       /qnap/bats/all_data \\
        --db-path        /qnap/bats/chirp_meta.db

With noise parquet and crop-root remapping::

    python merge_into_main.py \\
        --main-parquet      /qnap/bats/all_data/bats_2026-04-22T16_40_00.parquet \\
        --new-parquet       /qnap/src/marsh_stanford_processed/bats_<ts>.parquet \\
        --main-noise-parquet /qnap/bats/all_data/bats_noise_2026-04-22T16_40_00.parquet \\
        --new-noise-parquet  /qnap/src/marsh_stanford_processed/bats_noise_<ts>.parquet \\
        --main-manifest     /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --new-manifest      /qnap/src/marsh_stanford_processed/manifest.csv \\
        --remap-crop-root   /qnap/src/marsh_stanford_processed/crops \\
                            /qnap/bats/jr_pipeline/data/bat_crops \\
        --dest-dir          /qnap/bats/all_data \\
        --db-path           /qnap/bats/chirp_meta.db
"""

import argparse
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from logging_service import LoggingService
from sonobat_utils.utils import Utils
from sonobat_utils.sb_measures_postprocessing import BatsData, _PST


class MainCorpusMerger:
    """
    Merge a new-site parquet into the main corpus parquet, manifest, and
    SQLite database.

    :param main_parquet:       Path to the existing main ``bats_*.parquet``.
    :param new_parquet:        Path to the new-site ``bats_*.parquet`` from
                               ``from_scratch_postprocessing.py``.
    :param dest_dir:           Directory where merged output files are written.
    :param main_noise_parquet: Path to the existing main noise parquet.
                               Optional.
    :param new_noise_parquet:  Path to the new-site noise parquet.
                               Optional; if supplied, ``main_noise_parquet``
                               must also be supplied.
    :param main_manifest:      Path to the existing ``manifest.csv``.
                               Optional.
    :param new_manifest:       Path to the new-site ``manifest.csv``.
                               Optional; if supplied, ``main_manifest`` must
                               also be supplied.
    :param remap_crop_root:    ``(old_root, new_root)`` tuple.  If given,
                               every crop path in ``new_manifest`` that
                               starts with ``old_root`` is rewritten to
                               start with ``new_root`` before appending.
                               Useful when the new-site crops live under a
                               different tree than the main corpus crops.
    :param db_path:            Path to ``chirp_meta.db``.  Optional; if
                               supplied, new ``recordings`` rows are
                               inserted.
    """

    def __init__(
        self,
        main_parquet:        str | Path,
        new_parquet:         str | Path,
        dest_dir:            str | Path,
        main_noise_parquet:  str | Path | None = None,
        new_noise_parquet:   str | Path | None = None,
        main_manifest:       str | Path | None = None,
        new_manifest:        str | Path | None = None,
        remap_crop_root:     tuple[str, str] | None = None,
        db_path:             str | Path | None = None,
    ) -> None:
        self.log                = LoggingService()
        self.main_parquet       = Path(main_parquet)
        self.new_parquet        = Path(new_parquet)
        self.dest_dir           = Path(dest_dir)
        self.main_noise_parquet = (Path(main_noise_parquet)
                                   if main_noise_parquet else None)
        self.new_noise_parquet  = (Path(new_noise_parquet)
                                   if new_noise_parquet else None)
        self.main_manifest      = (Path(main_manifest)
                                   if main_manifest else None)
        self.new_manifest       = (Path(new_manifest)
                                   if new_manifest else None)
        self.remap_crop_root    = remap_crop_root
        self.db_path            = Path(db_path) if db_path else None
        self.timestamp          = datetime.now(_PST).isoformat().replace(':', '_')

        self.dest_dir.mkdir(parents=True, exist_ok=True)

        self._validate_args()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full merge workflow.

        :return: None
        :raises SystemExit: On file_id collisions or missing files.
        """
        # ── 1. Load both clean parquets ───────────────────────────────
        self.log.info(f"Loading main parquet:    {self.main_parquet}")
        main_bats = Utils.read_df_file(str(self.main_parquet))
        self.log.info(
            f"  {len(main_bats.df):,} rows, "
            f"{len(main_bats.file_map):,} recordings"
        )

        self.log.info(f"Loading new parquet:     {self.new_parquet}")
        new_bats = Utils.read_df_file(str(self.new_parquet))
        self.log.info(
            f"  {len(new_bats.df):,} rows, "
            f"{len(new_bats.file_map):,} recordings"
        )

        # ── 2. Verify no file_id collisions ───────────────────────────
        self._check_no_collision(main_bats.file_map, new_bats.file_map,
                                 label='clean')

        # ── 3. Merge clean DataFrames and file_maps ───────────────────
        merged_df       = self._merge_dataframes(main_bats.df, new_bats.df,
                                                  label='clean')
        merged_file_map = {**main_bats.file_map, **new_bats.file_map}

        # ── 4. Write merged clean parquet ─────────────────────────────
        merged_bats = BatsData(
            df         = merged_df,
            file_map   = merged_file_map,
            normalizer = main_bats.normalizer,   # existing scaler is authoritative
            timestamp  = self.timestamp,
        )
        out_clean = self.dest_dir / f"bats_{self.timestamp}.parquet"
        merged_bats.to_parquet(out_clean)
        self.log.info(
            f"Wrote merged clean parquet → {out_clean}  "
            f"({len(merged_df):,} rows)"
        )

        # ── 5. Optionally merge noise parquets ────────────────────────
        if self.main_noise_parquet and self.new_noise_parquet:
            self.log.info(
                f"Loading main noise parquet:  {self.main_noise_parquet}"
            )
            main_noise = Utils.read_df_file(str(self.main_noise_parquet))
            self.log.info(f"  {len(main_noise.df):,} rows")

            self.log.info(
                f"Loading new noise parquet:   {self.new_noise_parquet}"
            )
            new_noise = Utils.read_df_file(str(self.new_noise_parquet))
            self.log.info(f"  {len(new_noise.df):,} rows")

            self._check_no_collision(main_noise.file_map,
                                     new_noise.file_map, label='noise')

            merged_noise_df  = self._merge_dataframes(
                main_noise.df, new_noise.df, label='noise'
            )
            merged_noise_map = {**main_noise.file_map, **new_noise.file_map}

            merged_noise = BatsData(
                df         = merged_noise_df,
                file_map   = merged_noise_map,
                normalizer = main_noise.normalizer,
                timestamp  = self.timestamp,
            )
            out_noise = self.dest_dir / f"bats_noise_{self.timestamp}.parquet"
            merged_noise.to_parquet(out_noise)
            self.log.info(
                f"Wrote merged noise parquet  → {out_noise}  "
                f"({len(merged_noise_df):,} rows)"
            )

        # ── 6. Optionally merge manifests ─────────────────────────────
        if self.main_manifest and self.new_manifest:
            out_manifest = self.dest_dir / 'manifest.csv'
            self._merge_manifests(out_manifest)

        # ── 7. Optionally update chirp_meta.db recordings table ───────
        if self.db_path:
            self._update_db(new_bats.file_map)

        self.log.info("Merge complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_args(self) -> None:
        """
        Validate path existence and argument consistency.

        :raises SystemExit: On missing files or inconsistent arguments.
        """
        for p, label in [
            (self.main_parquet, '--main-parquet'),
            (self.new_parquet,  '--new-parquet'),
        ]:
            if not p.exists():
                self.log.err(f"{label} not found: {p}")
                sys.exit(1)

        if bool(self.main_noise_parquet) != bool(self.new_noise_parquet):
            self.log.err(
                "--main-noise-parquet and --new-noise-parquet must be "
                "supplied together or not at all."
            )
            sys.exit(1)

        if bool(self.main_manifest) != bool(self.new_manifest):
            self.log.err(
                "--main-manifest and --new-manifest must be supplied "
                "together or not at all."
            )
            sys.exit(1)

        for p, label in [
            (self.main_noise_parquet, '--main-noise-parquet'),
            (self.new_noise_parquet,  '--new-noise-parquet'),
            (self.main_manifest,      '--main-manifest'),
            (self.new_manifest,       '--new-manifest'),
            (self.db_path,            '--db-path'),
        ]:
            if p is not None and not p.exists():
                self.log.err(f"{label} not found: {p}")
                sys.exit(1)

    def _check_no_collision(
        self,
        main_file_map: dict[int, str],
        new_file_map:  dict[int, str],
        label:         str,
    ) -> None:
        """
        Abort if any file_id integer key appears in both file_maps.

        This would indicate that ``from_scratch_postprocessing.py`` was
        run without ``--existing-parquet``, creating a fresh encoder that
        collides with the main corpus.

        :param main_file_map: file_map from the main parquet.
        :param new_file_map:  file_map from the new-site parquet.
        :param label:         'clean' or 'noise', for error messages.
        :raises SystemExit: On collision.
        """
        collisions = set(main_file_map.keys()) & set(new_file_map.keys())
        if collisions:
            sample = sorted(collisions)[:5]
            self.log.err(
                f"file_id collision in {label} parquets — "
                f"{len(collisions):,} integer IDs appear in both: "
                f"{sample} …\n"
                f"This means from_scratch_postprocessing.py was run "
                f"WITHOUT --existing-parquet.  Re-run it with "
                f"--existing-parquet {self.main_parquet} to assign "
                f"non-colliding file_ids to the new recordings."
            )
            sys.exit(1)

        # Also check stem (filename) collisions — same recording in both.
        main_stems = set(main_file_map.values())
        new_stems  = set(new_file_map.values())
        stem_collisions = main_stems & new_stems
        if stem_collisions:
            sample = sorted(stem_collisions)[:5]
            self.log.err(
                f"Recording stem collision in {label} parquets — "
                f"{len(stem_collisions):,} stems already in main corpus: "
                f"{sample} …\n"
                f"These recordings are already merged.  "
                f"Do not merge the same site batch twice."
            )
            sys.exit(1)

        self.log.info(
            f"  {label}: no file_id or stem collisions — "
            f"{len(main_file_map):,} existing + "
            f"{len(new_file_map):,} new = "
            f"{len(main_file_map) + len(new_file_map):,} total recordings"
        )

    def _merge_dataframes(
        self,
        main_df: pd.DataFrame,
        new_df:  pd.DataFrame,
        label:   str,
    ) -> pd.DataFrame:
        """
        Concatenate main and new DataFrames, checking column schema
        compatibility.

        :param main_df: DataFrame from the main parquet.
        :param new_df:  DataFrame from the new-site parquet.
        :param label:   'clean' or 'noise', for log messages.
        :return:        Concatenated DataFrame, index reset.
        """
        main_cols = set(main_df.columns)
        new_cols  = set(new_df.columns)

        missing_in_new  = main_cols - new_cols
        missing_in_main = new_cols  - main_cols

        if missing_in_new:
            self.log.warn(
                f"{label}: new parquet is missing columns present in "
                f"main: {sorted(missing_in_new)}.  "
                f"They will be NaN in the new rows."
            )
        if missing_in_main:
            self.log.warn(
                f"{label}: new parquet has extra columns not in "
                f"main: {sorted(missing_in_main)}.  "
                f"They will be NaN in the main rows."
            )

        merged = pd.concat([main_df, new_df], ignore_index=True)
        self.log.info(
            f"  {label}: concatenated "
            f"{len(main_df):,} + {len(new_df):,} = {len(merged):,} rows"
        )
        return merged

    def _merge_manifests(self, out_path: Path) -> None:
        """
        Append new manifest rows to the main manifest, remapping crop
        paths if requested, and deduplicating on
        ``(file_id, chirp_idx, harmonic_idx)``.

        :param out_path: Destination path for the merged manifest CSV.
        :return: None
        """
        self.log.info(f"Loading main manifest:  {self.main_manifest}")
        main_manifest = Utils.read_df_file(str(self.main_manifest))
        self.log.info(f"  {len(main_manifest):,} rows")

        self.log.info(f"Loading new manifest:   {self.new_manifest}")
        new_manifest = Utils.read_df_file(str(self.new_manifest))
        self.log.info(f"  {len(new_manifest):,} rows")

        # Optional crop-path remapping.
        if self.remap_crop_root is not None:
            old_root, new_root = self.remap_crop_root
            crop_col = self._find_crop_path_column(new_manifest)
            if crop_col:
                before = len(new_manifest)
                new_manifest[crop_col] = (
                    new_manifest[crop_col]
                    .astype(str)
                    .str.replace(old_root, new_root, n=1, regex=False)
                )
                remapped = (
                    new_manifest[crop_col].str.startswith(new_root)
                ).sum()
                self.log.info(
                    f"  Remapped {remapped:,}/{before:,} crop paths "
                    f"from '{old_root}' → '{new_root}'"
                )
            else:
                self.log.warn(
                    "--remap-crop-root specified but no crop-path column "
                    "found in new manifest — remapping skipped."
                )

        merged = pd.concat([main_manifest, new_manifest], ignore_index=True)
        n_before = len(merged)

        # Dedup on composite key if all three columns are present.
        key_cols = [c for c in ('file_id', 'chirp_idx', 'harmonic_idx')
                    if c in merged.columns]
        if key_cols:
            merged = merged.drop_duplicates(subset=key_cols, keep='first')
            n_dropped = n_before - len(merged)
            if n_dropped:
                self.log.warn(
                    f"  Manifest dedup: dropped {n_dropped:,} duplicate "
                    f"rows on {key_cols} (likely a re-run)."
                )

        merged.to_csv(out_path, index=False)
        self.log.info(
            f"Wrote merged manifest → {out_path}  ({len(merged):,} rows)"
        )

    def _find_crop_path_column(self, df: pd.DataFrame) -> str | None:
        """
        Heuristically identify the crop-path column in a manifest DataFrame.

        Tries common names in order: ``crop_path``, ``png_path``,
        ``path``, ``filename``.

        :param df: Manifest DataFrame.
        :return:   Column name, or ``None`` if not found.
        """
        for candidate in ('crop_path', 'png_path', 'path', 'filename'):
            if candidate in df.columns:
                return candidate
        return None

    def _update_db(self, new_file_map: dict[int, str]) -> None:
        """
        Insert new recording rows into ``chirp_meta.db``'s ``recordings``
        table.  ``INSERT OR IGNORE`` makes re-runs safe.

        ``rec_site`` is inferred from the recording stem using the same
        naming conventions as ``sb_measures_add_daytime_columns.py``
        (first path component of the stem, e.g. ``marsh`` from
        ``marsh1_D20220723T215745m074``).

        ``rec_period`` is left NULL; ``bat_db_builder.py`` fills it
        when the merged manifest is available.

        :param new_file_map: file_map from the new-site parquet
                             (integer file_id → recording stem string).
        :return: None
        """
        rows = []
        for fid, stem in sorted(new_file_map.items()):
            # Infer rec_site from the stem prefix (e.g. 'marsh1_D...' → 'marsh').
            import re
            m = re.match(r'^([A-Za-z]+)', stem)
            rec_site = m.group(1).rstrip('0123456789') if m else 'unknown'
            rows.append((fid, stem, rec_site, None))

        self.log.info(
            f"Updating recordings table in {self.db_path} "
            f"({len(rows):,} new rows) …"
        )
        with sqlite3.connect(str(self.db_path)) as con:
            con.executemany(
                "INSERT OR IGNORE INTO recordings "
                "(file_id, filename, rec_site, rec_period) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.log.info("  recordings table updated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for :class:`MainCorpusMerger`.

    :return: Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog='merge_into_main.py',
        description=(
            'Merge a new-site bats_<ts>.parquet into the main corpus '
            'parquet, manifest, and SQLite database.\n\n'
            'PREREQUISITE: from_scratch_postprocessing.py must have been '
            'run with --existing-parquet pointing at the main parquet, '
            'so that new file_ids were assigned by extending the existing '
            'PathEncoder.  This script will abort if file_id collisions '
            'are detected.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Typical usage:
              python merge_into_main.py \\
                  --main-parquet    /qnap/bats/all_data/bats_2026-04-22T16_40_00.parquet \\
                  --new-parquet     /qnap/src/marsh_stanford_processed/bats_<ts>.parquet \\
                  --main-manifest   /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
                  --new-manifest    /qnap/src/marsh_stanford_processed/manifest.csv \\
                  --dest-dir        /qnap/bats/all_data \\
                  --db-path         /qnap/bats/chirp_meta.db

            After this script completes, run bat_db_builder.py against the
            merged manifest to populate chirp_info and chirp_spectrograms.
        """),
    )
    parser.add_argument(
        '--main-parquet',
        required=True, type=Path, metavar='PARQUET',
        help='Existing main bats_*.parquet (SB-originated or previously merged).',
    )
    parser.add_argument(
        '--new-parquet',
        required=True, type=Path, metavar='PARQUET',
        help='New-site bats_*.parquet from from_scratch_postprocessing.py.',
    )
    parser.add_argument(
        '--dest-dir',
        required=True, type=Path, metavar='DIR',
        help='Directory where merged output files are written.',
    )
    parser.add_argument(
        '--main-noise-parquet',
        default=None, type=Path, metavar='PARQUET',
        help='Existing main bats_noise_*.parquet.  Must be paired with --new-noise-parquet.',
    )
    parser.add_argument(
        '--new-noise-parquet',
        default=None, type=Path, metavar='PARQUET',
        help='New-site bats_noise_*.parquet.  Must be paired with --main-noise-parquet.',
    )
    parser.add_argument(
        '--main-manifest',
        default=None, type=Path, metavar='CSV',
        help='Existing manifest.csv.  Must be paired with --new-manifest.',
    )
    parser.add_argument(
        '--new-manifest',
        default=None, type=Path, metavar='CSV',
        help='New-site manifest.csv.  Must be paired with --main-manifest.',
    )
    parser.add_argument(
        '--remap-crop-root',
        default=None, nargs=2, metavar=('OLD_ROOT', 'NEW_ROOT'),
        help=(
            'Rewrite crop paths in the new manifest: replace OLD_ROOT '
            'prefix with NEW_ROOT.  Use when new-site crops live under '
            'a different tree than the main corpus crops.'
        ),
    )
    parser.add_argument(
        '--db-path',
        default=None, type=Path, metavar='DB',
        help='chirp_meta.db to update with new recordings rows.',
    )

    args = parser.parse_args()

    for p, flag in [
        (args.main_parquet, '--main-parquet'),
        (args.new_parquet,  '--new-parquet'),
    ]:
        if not p.exists():
            parser.error(f'{flag} not found: {p}')

    if bool(args.main_noise_parquet) != bool(args.new_noise_parquet):
        parser.error(
            '--main-noise-parquet and --new-noise-parquet must be '
            'supplied together or not at all.'
        )
    if bool(args.main_manifest) != bool(args.new_manifest):
        parser.error(
            '--main-manifest and --new-manifest must be supplied '
            'together or not at all.'
        )

    return args


def main() -> None:
    """
    CLI entry point for :class:`MainCorpusMerger`.

    :return: None
    """
    args = _parse_args()
    MainCorpusMerger(
        main_parquet        = args.main_parquet,
        new_parquet         = args.new_parquet,
        dest_dir            = args.dest_dir,
        main_noise_parquet  = args.main_noise_parquet,
        new_noise_parquet   = args.new_noise_parquet,
        main_manifest       = args.main_manifest,
        new_manifest        = args.new_manifest,
        remap_crop_root     = (tuple(args.remap_crop_root)
                               if args.remap_crop_root else None),
        db_path             = args.db_path,
    ).run()


if __name__ == '__main__':
    main()
