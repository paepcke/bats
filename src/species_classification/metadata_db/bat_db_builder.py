#!/usr/bin/env python
# #############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-11 10:45:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-11 20:00:49
# #############################################

"""
Builder for the bat chirp metadata SQLite database.

Schema
------
png_dirs     : unique parent directories of spectrogram PNGs
recordings   : one row per source 2-sec WAV segment (file_id)
chirp_info   : one row per chirp; links PNG, parquet row, species

Usage
-----
python bat_db_builder.py \\
    --measures-file /qnap/bats/all_data/bats_2026-04-08T23_40_36.627058.parquet \\
    --spectros-manifest /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
    --db-out-file /qnap/bats/chirp_meta.db \\
    [--on-inconsistency {warn|strict}]
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from logging_service import LoggingService

from sonobat_utils.utils import Utils


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS png_dirs (
    dir_id    INTEGER PRIMARY KEY,
    dir_path  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS recordings (
    file_id    INTEGER PRIMARY KEY,
    filename   TEXT,
    rec_site   TEXT,
    rec_period TEXT
);

CREATE TABLE IF NOT EXISTS chirp_info (
    file_id      INTEGER NOT NULL REFERENCES recordings(file_id),
    chirp_idx    INTEGER NOT NULL,
    dir_id       INTEGER NOT NULL REFERENCES png_dirs(dir_id),
    png_filename TEXT NOT NULL,
    measures_row INTEGER NOT NULL,
    species      TEXT,
    confidence   REAL,
    PRIMARY KEY (file_id, chirp_idx),
    UNIQUE (dir_id, png_filename)
);

CREATE INDEX IF NOT EXISTS idx_chirp_species ON chirp_info(species);
CREATE INDEX IF NOT EXISTS idx_chirp_dir     ON chirp_info(dir_id);
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class BatDbBuilder:
    """
    Builds the bat chirp metadata SQLite database from a parquet measures
    file and a CSV manifest of spectrogram PNG crops.

    :param parquet_path: Path to the measures parquet file.
    :param manifest_path: Path to the manifest CSV file.
    :param db_path: Destination path for the SQLite database file.
    :param on_inconsistency: ``'warn'`` to log and continue on data
        mismatches; ``'strict'`` to abort on the first mismatch.
    """

    # Tolerance for time matching (ms); exact match expected but allow
    # for float representation noise after the int cast.
    TIME_TOLERANCE_MS: int = 0

    def __init__(
        self,
        parquet_path: str,
        manifest_path: str,
        db_path: str,
        on_inconsistency: str = "warn",
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.manifest_path = Path(manifest_path)
        self.db_path = Path(db_path)
        self.on_inconsistency = on_inconsistency
        self.log = LoggingService()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        Load source data, validate cross-file consistency, and write
        the SQLite database.
        """
        self.log.info("Loading parquet measures ...")
        measures = self._load_parquet()

        self.log.info("Loading manifest CSV ...")
        manifest = self._load_manifest()

        self.log.info("Merging and validating ...")
        merged = self._merge_and_validate(measures, manifest)

        self.log.info(f"Writing database to {self.db_path} ...")
        self._write_db(measures, manifest, merged)

        self.log.info("Done.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_parquet(self) -> pd.DataFrame:
        """
        Load the measures parquet file and normalise column names needed
        for the build.

        :return: DataFrame with at minimum columns
            ``file_id``, ``chirp_idx``, ``TimeInFile``,
            ``rec_site``, ``species``, ``confidence``.
        """
        df = Utils.read_df_file(self.parquet_path)
        required = {"file_id", "chirp_idx", "TimeInFile", "rec_site",
                    "species", "confidence"}
        missing = required - set(df.columns)
        if missing:
            self.log.err(f"Parquet missing columns: {missing}")
            sys.exit(1)
        df["TimeInFile"] = df["TimeInFile"].astype(int)
        # Preserve original 0-based iloc row number before any sorting.
        df = df.reset_index(drop=True)
        df["measures_row"] = df.index
        return df

    def _load_manifest(self) -> pd.DataFrame:
        """
        Load the manifest CSV and normalise columns.

        :return: DataFrame with at minimum columns
            ``crop_path``, ``file_id``, ``chirp_idx``,
            ``time_in_file_ms``, ``species``, ``confidence``,
            ``partition``.
        """
        df = pd.read_csv(self.manifest_path, low_memory=False)
        required = {"crop_path", "file_id", "chirp_idx", "time_in_file_ms",
                    "species", "confidence", "partition"}
        missing = required - set(df.columns)
        if missing:
            self.log.err(f"Manifest missing columns: {missing}")
            sys.exit(1)
        df["time_in_file_ms"] = df["time_in_file_ms"].astype(int)
        df["chirp_idx"] = df["chirp_idx"].astype(int)
        return df

    # ------------------------------------------------------------------
    # Merge and validate
    # ------------------------------------------------------------------

    def _merge_and_validate(
        self,
        measures: pd.DataFrame,
        manifest: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inner-join measures and manifest on ``(file_id, chirp_idx)`` and
        check that ``species``, ``confidence``, and ``TimeInFile`` /
        ``time_in_file_ms`` agree between sources.

        ``chirp_idx`` is taken directly from both files; no derivation
        is performed here.

        :param measures: Loaded parquet DataFrame.
        :param manifest: Manifest DataFrame with pre-assigned ``chirp_idx``.
        :return: Merged DataFrame with columns needed for ``chirp_info``.
        """
        left = measures[
            ["file_id", "chirp_idx", "TimeInFile",
             "species", "confidence", "measures_row"]
        ].rename(columns={
            "TimeInFile": "time_parquet",
            "species": "species_parquet",
            "confidence": "confidence_parquet",
        })

        right = manifest[
            ["file_id", "chirp_idx", "time_in_file_ms",
             "species", "confidence", "crop_path", "partition"]
        ].rename(columns={
            "time_in_file_ms": "time_manifest",
            "species": "species_manifest",
            "confidence": "confidence_manifest",
        })

        merged = pd.merge(left, right, on=["file_id", "chirp_idx"],
                          how="inner")

        n_left = len(left)
        n_right = len(right)
        n_merged = len(merged)
        if n_merged < n_left or n_merged < n_right:
            self.log.warn(
                f"Join coverage: parquet={n_left}, manifest={n_right}, "
                f"merged={n_merged}. "
                f"{n_left - n_merged} parquet rows and "
                f"{n_right - n_merged} manifest rows unmatched."
            )

        inconsistencies: list[str] = []

        # --- species ---
        mismatch = merged["species_parquet"] != merged["species_manifest"]
        if mismatch.any():
            rows = merged[mismatch][
                ["file_id", "chirp_idx",
                 "species_parquet", "species_manifest"]
            ]
            msg = (
                f"species mismatch in {mismatch.sum()} rows:\n"
                f"{rows.to_string(index=False)}"
            )
            inconsistencies.append(msg)

        # --- confidence ---
        conf_diff = (
            merged["confidence_parquet"] - merged["confidence_manifest"]
        ).abs()
        bad_conf = conf_diff > 1e-6
        if bad_conf.any():
            rows = merged[bad_conf][
                ["file_id", "chirp_idx",
                 "confidence_parquet", "confidence_manifest"]
            ]
            msg = (
                f"confidence mismatch in {bad_conf.sum()} rows:\n"
                f"{rows.to_string(index=False)}"
            )
            inconsistencies.append(msg)

        # --- time ---
        time_diff = (
            merged["time_parquet"] - merged["time_manifest"]
        ).abs()
        bad_time = time_diff > self.TIME_TOLERANCE_MS
        if bad_time.any():
            rows = merged[bad_time][
                ["file_id", "chirp_idx",
                 "time_parquet", "time_manifest"]
            ]
            msg = (
                f"time mismatch in {bad_time.sum()} rows:\n"
                f"{rows.to_string(index=False)}"
            )
            inconsistencies.append(msg)

        for msg in inconsistencies:
            if self.on_inconsistency == "strict":
                self.log.err(f"Inconsistency (strict mode — aborting): {msg}")
                sys.exit(1)
            else:
                self.log.warn(f"Inconsistency (warn mode — continuing): {msg}")

        # Use parquet species/confidence as authoritative (they went
        # through the ML pipeline); warn already emitted if they differ.
        merged["species"] = merged["species_parquet"]
        merged["confidence"] = merged["confidence_parquet"]

        return merged

    # ------------------------------------------------------------------
    # Database writing
    # ------------------------------------------------------------------

    def _write_db(
        self,
        measures: pd.DataFrame,
        manifest: pd.DataFrame,
        merged: pd.DataFrame,
    ) -> None:
        """
        Create (or replace) the SQLite database and populate all tables.

        :param measures: Full measures DataFrame (for rec_site).
        :param manifest: Full manifest DataFrame (for rec_period / partition).
        :param merged: Inner-joined and validated DataFrame.
        """
        if self.db_path.exists():
            self.log.warn(f"Overwriting existing database: {self.db_path}")
            self.db_path.unlink()

        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA_SQL)
            self._insert_recordings(con, measures, manifest)
            self._insert_png_dirs_and_chirps(con, merged)
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _insert_recordings(
        self,
        con: sqlite3.Connection,
        measures: pd.DataFrame,
        manifest: pd.DataFrame,
    ) -> None:
        """
        Populate the ``recordings`` table.

        ``rec_site`` comes from the parquet; ``rec_period`` (partition)
        comes from the manifest.

        :param con: Open SQLite connection.
        :param measures: Parquet DataFrame.
        :param manifest: Manifest DataFrame.
        """
        parquet_rec = (
            measures[["file_id", "rec_site"]]
            .drop_duplicates("file_id")
            .set_index("file_id")
        )
        manifest_rec = (
            manifest[["file_id", "Filename", "partition"]]
            .drop_duplicates("file_id")
            .set_index("file_id")
        )
        all_file_ids = parquet_rec.index.union(manifest_rec.index)
        rows = []
        for fid in all_file_ids:
            filename = (
                manifest_rec.loc[fid, "Filename"]
                if fid in manifest_rec.index else None
            )
            rec_site = (
                parquet_rec.loc[fid, "rec_site"]
                if fid in parquet_rec.index else None
            )
            rec_period = (
                manifest_rec.loc[fid, "partition"]
                if fid in manifest_rec.index else None
            )
            rows.append((int(fid), filename, rec_site, rec_period))

        con.executemany(
            "INSERT INTO recordings (file_id, filename, rec_site, rec_period) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self.log.info(f"Inserted {len(rows)} recordings.")

    def _insert_png_dirs_and_chirps(
        self,
        con: sqlite3.Connection,
        merged: pd.DataFrame,
    ) -> None:
        """
        Populate ``png_dirs`` and ``chirp_info`` tables.

        Directories are collected first so each gets a stable ``dir_id``
        before chirp rows reference them.

        :param con: Open SQLite connection.
        :param merged: Merged and validated DataFrame.
        """
        # --- png_dirs ---
        merged["_png_dir"] = merged["crop_path"].apply(
            lambda p: str(Path(p).parent)
        )
        merged["_png_file"] = merged["crop_path"].apply(
            lambda p: Path(p).name
        )

        unique_dirs = sorted(merged["_png_dir"].unique())
        con.executemany(
            "INSERT INTO png_dirs (dir_path) VALUES (?)",
            [(d,) for d in unique_dirs],
        )
        self.log.info(f"Inserted {len(unique_dirs)} png_dirs.")

        # Build dir_path → dir_id lookup
        cur = con.execute("SELECT dir_id, dir_path FROM png_dirs")
        dir_id_map: dict[str, int] = {row[1]: row[0] for row in cur}

        # --- chirp_info ---
        chirp_rows = []
        for _, row in merged.iterrows():
            chirp_rows.append((
                int(row["file_id"]),
                int(row["chirp_idx"]),
                dir_id_map[row["_png_dir"]],
                row["_png_file"],
                int(row["measures_row"]),
                row["species"],
                float(row["confidence"]) if pd.notna(row["confidence"]) else None,
            ))

        con.executemany(
            "INSERT INTO chirp_info "
            "(file_id, chirp_idx, dir_id, png_filename, measures_row, "
            " species, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            chirp_rows,
        )
        self.log.info(f"Inserted {len(chirp_rows)} chirp_info rows.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the bat chirp metadata SQLite database."
    )
    parser.add_argument(
        "--measures-file", required=True,
        dest="measures_file",
        help="Path to the measures parquet file.",
    )
    parser.add_argument(
        "--spectros-manifest", required=True,
        dest="spectros_manifest",
        help="Path to the spectrogram manifest CSV file.",
    )
    parser.add_argument(
        "--db-out-file", required=True,
        dest="db_out_file",
        help="Destination path for the SQLite database.",
    )
    parser.add_argument(
        "--on-inconsistency",
        choices=["warn", "strict"],
        default="warn",
        dest="on_inconsistency",
        help=(
            "How to handle data mismatches between measures and manifest. "
            "'warn' logs and continues; 'strict' aborts immediately. "
            "Default: warn."
        ),
    )
    args = parser.parse_args()

    builder = BatDbBuilder(
        parquet_path=args.measures_file,
        manifest_path=args.spectros_manifest,
        db_path=args.db_out_file,
        on_inconsistency=args.on_inconsistency,
    )
    builder.build()


if __name__ == "__main__":
    main()
