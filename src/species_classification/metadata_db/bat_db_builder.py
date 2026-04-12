#!/usr/bin/env python
# #############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-11 10:45:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-12 09:46:23
# #############################################

"""
Builder for the bat chirp metadata SQLite database.

Schema
------
png_dirs          : unique parent directories of spectrogram PNGs
recordings        : one row per source 2-sec WAV segment (file_id)
chirp_info        : one row per chirp; links to chirp measures parquet
                    row and carries species/confidence
chirp_spectrograms: one row per spectrogram PNG; child of chirp_info,
                    keyed by (file_id, chirp_idx, harmonic_idx).
                    Multiple harmonics of the same chirp each get their
                    own PNG but share a single chirp_info row and
                    measures_row.

The chirp measures parquet file has one row per (file_id, chirp_idx);
harmonic duplicates were removed upstream because their measure values
are identical. The chirp spectrograms manifest has one row per PNG,
with harmonic_idx distinguishing multiple harmonics of the same chirp.

Usage
-----
python bat_db_builder.py \\
    --measures-file /qnap/bats/all_data/bats_2026-04-12T01_13_10.661235.parquet \\
    --spectros-manifest /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
    --db-out-file /qnap/bats/chirp_meta.db \\
    [--on-inconsistency {warn|strict}]
"""

import argparse
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
    measures_row INTEGER NOT NULL,
    species      TEXT,
    confidence   REAL,
    PRIMARY KEY (file_id, chirp_idx)
);

CREATE TABLE IF NOT EXISTS chirp_spectrograms (
    file_id      INTEGER NOT NULL,
    chirp_idx    INTEGER NOT NULL,
    harmonic_idx INTEGER NOT NULL,
    dir_id       INTEGER NOT NULL REFERENCES png_dirs(dir_id),
    png_filename TEXT NOT NULL,
    PRIMARY KEY (file_id, chirp_idx, harmonic_idx),
    FOREIGN KEY (file_id, chirp_idx) REFERENCES chirp_info(file_id, chirp_idx),
    UNIQUE (dir_id, png_filename)
);

CREATE INDEX IF NOT EXISTS idx_chirp_species ON chirp_info(species);
CREATE INDEX IF NOT EXISTS idx_spec_dir      ON chirp_spectrograms(dir_id);
CREATE INDEX IF NOT EXISTS idx_spec_harmonic ON chirp_spectrograms(harmonic_idx);
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class BatDbBuilder:
    """
    Builds the bat chirp metadata SQLite database from a chirp measures
    parquet file and a chirp spectrograms manifest CSV.

    The chirp measures parquet has one row per ``(file_id, chirp_idx)``;
    harmonic duplicates are absent because their measure values are
    identical and were removed upstream.

    The chirp spectrograms manifest has one row per PNG file, with
    ``harmonic_idx`` distinguishing multiple harmonics of the same chirp.
    Multiple harmonics share the same ``(file_id, chirp_idx)`` and thus
    the same ``measures_row`` in the chirp measures parquet.

    :param parquet_path: Path to the chirp measures parquet file.
    :param manifest_path: Path to the chirp spectrograms manifest CSV.
    :param db_path: Destination path for the SQLite database file.
    :param on_inconsistency: ``'warn'`` to log and continue on data
        mismatches between sources; ``'strict'`` to abort immediately.
    """

    # Tolerance for time validation (ms); exact match expected but allow
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
        self.log.info("Loading chirp measures parquet ...")
        measures = self._load_parquet()

        self.log.info("Loading chirp spectrograms manifest ...")
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
        Load the chirp measures parquet file and normalise columns
        needed for the build.

        The parquet has one row per ``(file_id, chirp_idx)``; harmonic
        duplicates have been removed upstream.

        :return: DataFrame with at minimum columns
            ``file_id``, ``chirp_idx``, ``TimeInFile``,
            ``rec_site``, ``species``, ``confidence``,
            plus a derived ``measures_row`` (0-based iloc index).
        """
        df = Utils.read_df_file(self.parquet_path)
        required = {"file_id", "chirp_idx", "TimeInFile", "rec_site",
                    "species", "confidence"}
        missing = required - set(df.columns)
        if missing:
            self.log.err(f"Chirp measures parquet missing columns: {missing}")
            sys.exit(1)
        df["TimeInFile"] = df["TimeInFile"].astype(int)
        # Preserve original 0-based iloc row number before any sorting.
        df = df.reset_index(drop=True)
        df["measures_row"] = df.index
        return df

    def _load_manifest(self) -> pd.DataFrame:
        """
        Load the chirp spectrograms manifest CSV and normalise columns.

        The manifest has one row per PNG file. ``harmonic_idx``
        distinguishes multiple harmonics of the same chirp; all harmonics
        of a chirp share the same ``(file_id, chirp_idx)``.

        :return: DataFrame with at minimum columns
            ``crop_path``, ``file_id``, ``chirp_idx``, ``harmonic_idx``,
            ``time_in_file_ms``, ``species``, ``confidence``, ``partition``.
        """
        df = pd.read_csv(self.manifest_path, low_memory=False)
        required = {"crop_path", "file_id", "chirp_idx", "harmonic_idx",
                    "time_in_file_ms", "species", "confidence", "partition"}
        missing = required - set(df.columns)
        if missing:
            self.log.err(
                f"Chirp spectrograms manifest missing columns: {missing}"
            )
            sys.exit(1)
        df["time_in_file_ms"] = df["time_in_file_ms"].astype(int)
        df["chirp_idx"]       = df["chirp_idx"].astype(int)
        df["harmonic_idx"]    = df["harmonic_idx"].astype(int)
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
        Inner-join the chirp measures parquet and the chirp spectrograms
        manifest on ``(file_id, chirp_idx)``, then validate that
        ``species``, ``confidence``, and timing agree between sources.

        Because the manifest may have multiple rows per ``(file_id,
        chirp_idx)`` (one per harmonic), the join fans out: each parquet
        row may match N manifest rows. ``measures_row`` is the same for
        all harmonics of a chirp.

        Validation is performed on ``harmonic_idx == 0`` rows only,
        since all harmonics of a chirp share the same values for those
        fields.

        :param measures: Chirp measures parquet DataFrame.
        :param manifest: Chirp spectrograms manifest DataFrame.
        :return: Merged DataFrame with all columns needed for
            ``chirp_info`` and ``chirp_spectrograms``.
        """
        left = measures[
            ["file_id", "chirp_idx", "TimeInFile",
             "species", "confidence", "measures_row"]
        ].rename(columns={
            "TimeInFile":  "time_parquet",
            "species":     "species_parquet",
            "confidence":  "confidence_parquet",
        })

        right = manifest[
            ["file_id", "chirp_idx", "harmonic_idx", "time_in_file_ms",
             "species", "confidence", "crop_path", "partition"]
        ].rename(columns={
            "time_in_file_ms": "time_manifest",
            "species":         "species_manifest",
            "confidence":      "confidence_manifest",
        })

        merged = pd.merge(left, right, on=["file_id", "chirp_idx"],
                          how="inner")

        n_left   = len(left)
        n_right  = len(manifest)
        n_merged = len(merged)
        if n_merged == 0:
            self.log.err(
                "Merge produced zero rows — check that chirp_idx values "
                "align between the chirp measures parquet and the chirp "
                "spectrograms manifest."
            )
            sys.exit(1)
        self.log.info(
            f"Join coverage: measures={n_left:,}, "
            f"manifest={n_right:,}, merged={n_merged:,}. "
            f"{n_left - n_merged:,} measures rows unmatched; "
            f"{n_right - n_merged:,} manifest rows unmatched."
        )

        inconsistencies: list[str] = []

        # Validate on harmonic_idx=0 rows only (one row per chirp).
        primary = merged[merged["harmonic_idx"] == 0]

        # --- species ---
        mismatch = primary["species_parquet"] != primary["species_manifest"]
        if mismatch.any():
            rows = primary[mismatch][
                ["file_id", "chirp_idx",
                 "species_parquet", "species_manifest"]
            ]
            inconsistencies.append(
                f"species mismatch in {mismatch.sum():,} chirps:\n"
                f"{rows.head(20).to_string(index=False)}"
            )

        # --- confidence ---
        conf_diff = (
            primary["confidence_parquet"] - primary["confidence_manifest"]
        ).abs()
        bad_conf = conf_diff > 1e-6
        if bad_conf.any():
            rows = primary[bad_conf][
                ["file_id", "chirp_idx",
                 "confidence_parquet", "confidence_manifest"]
            ]
            inconsistencies.append(
                f"confidence mismatch in {bad_conf.sum():,} chirps:\n"
                f"{rows.head(20).to_string(index=False)}"
            )

        # --- time ---
        time_diff = (
            primary["time_parquet"] - primary["time_manifest"]
        ).abs()
        bad_time = time_diff > self.TIME_TOLERANCE_MS
        if bad_time.any():
            rows = primary[bad_time][
                ["file_id", "chirp_idx",
                 "time_parquet", "time_manifest"]
            ]
            inconsistencies.append(
                f"time mismatch in {bad_time.sum():,} chirps:\n"
                f"{rows.head(20).to_string(index=False)}"
            )

        for msg in inconsistencies:
            if self.on_inconsistency == "strict":
                self.log.err(f"Inconsistency (strict — aborting): {msg}")
                sys.exit(1)
            else:
                self.log.warn(f"Inconsistency (warn — continuing): {msg}")

        # Use chirp measures parquet species/confidence as authoritative;
        # warn already emitted above if they differ.
        merged["species"]    = merged["species_parquet"]
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

        :param measures: Chirp measures parquet DataFrame.
        :param manifest: Chirp spectrograms manifest DataFrame.
        :param merged: Inner-joined and validated DataFrame.
        """
        if self.db_path.exists():
            self.log.warn(f"Overwriting existing database: {self.db_path}")
            self.db_path.unlink()

        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA_SQL)
            self._insert_recordings(con, measures, manifest)
            self._insert_chirp_info(con, merged)
            self._insert_spectrograms(con, merged)
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

        ``rec_site`` comes from the chirp measures parquet;
        ``rec_period`` (partition) and ``filename`` come from the
        chirp spectrograms manifest.

        :param con: Open SQLite connection.
        :param measures: Chirp measures parquet DataFrame.
        :param manifest: Chirp spectrograms manifest DataFrame.
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
        self.log.info(f"Inserted {len(rows):,} recordings.")

    def _insert_chirp_info(
        self,
        con: sqlite3.Connection,
        merged: pd.DataFrame,
    ) -> None:
        """
        Populate the ``chirp_info`` table.

        One row per ``(file_id, chirp_idx)``. All harmonics of a chirp
        share the same ``measures_row`` in the chirp measures parquet,
        so we deduplicate on ``(file_id, chirp_idx)`` taking
        ``harmonic_idx == 0`` (the primary harmonic).

        :param con: Open SQLite connection.
        :param merged: Merged and validated DataFrame.
        """
        chirp_rows = (
            merged[merged["harmonic_idx"] == 0]
            .drop_duplicates(subset=["file_id", "chirp_idx"])
            [["file_id", "chirp_idx", "measures_row",
              "species", "confidence"]]
        )

        rows = [
            (
                int(r.file_id),
                int(r.chirp_idx),
                int(r.measures_row),
                r.species,
                float(r.confidence) if pd.notna(r.confidence) else None,
            )
            for r in chirp_rows.itertuples(index=False)
        ]

        con.executemany(
            "INSERT INTO chirp_info "
            "(file_id, chirp_idx, measures_row, species, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.log.info(f"Inserted {len(rows):,} chirp_info rows.")

    def _insert_spectrograms(
        self,
        con: sqlite3.Connection,
        merged: pd.DataFrame,
    ) -> None:
        """
        Populate the ``png_dirs`` and ``chirp_spectrograms`` tables.

        Directories are collected first so each gets a stable ``dir_id``
        before spectrogram rows reference them. One row is inserted per
        PNG file, preserving all harmonics.

        :param con: Open SQLite connection.
        :param merged: Merged and validated DataFrame.
        """
        merged = merged.copy()
        merged["png_dir"]  = merged["crop_path"].apply(
            lambda p: str(Path(p).parent)
        )
        merged["png_file"] = merged["crop_path"].apply(
            lambda p: Path(p).name
        )

        # --- png_dirs ---
        unique_dirs = sorted(merged["png_dir"].unique())
        con.executemany(
            "INSERT INTO png_dirs (dir_path) VALUES (?)",
            [(d,) for d in unique_dirs],
        )
        self.log.info(f"Inserted {len(unique_dirs):,} png_dirs.")

        # Build dir_path → dir_id lookup
        cur = con.execute("SELECT dir_id, dir_path FROM png_dirs")
        dir_id_map: dict[str, int] = {row[1]: row[0] for row in cur}

        # --- chirp_spectrograms ---
        rows = [
            (
                int(r.file_id),
                int(r.chirp_idx),
                int(r.harmonic_idx),
                dir_id_map[r.png_dir],
                r.png_file,
            )
            for r in merged.itertuples(index=False)
        ]

        con.executemany(
            "INSERT INTO chirp_spectrograms "
            "(file_id, chirp_idx, harmonic_idx, dir_id, png_filename) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.log.info(f"Inserted {len(rows):,} chirp_spectrograms rows.")


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
        help="Path to the chirp measures parquet file.",
    )
    parser.add_argument(
        "--spectros-manifest", required=True,
        dest="spectros_manifest",
        help="Path to the chirp spectrograms manifest CSV.",
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
            "How to handle data mismatches between the chirp measures "
            "parquet and the chirp spectrograms manifest. "
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
