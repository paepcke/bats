#!/usr/bin/env python
# #############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-11 10:46:17
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-13 16:55:17
# #############################################


"""
CLI query wrapper for the bat chirp metadata SQLite database.

The database holds two kinds of data:
  - chirp_info        : one row per chirp, linked to the chirp measures
                        parquet via measures_row
  - chirp_spectrograms: one row per spectrogram PNG; multiple harmonics
                        of the same chirp each have their own row,
                        distinguished by harmonic_idx

Queries
-------
--spectrograms-for-file FILE_ID
    Full paths to all spectrogram PNGs for a recording.

--measures-for-file FILE_ID
    All (file_id, chirp_idx, measures_row) entries for a recording.

--species FILE_ID
    Species (or composite species) associated with a file_id.

--chirp FILE_ID CHIRP_IDX
    measures_row and PNG path(s) for a single chirp.

--random-spectrogram SPECIES
    A random PNG path + file_id for the given species string.

--relocate-pngs OLD_DIR NEW_DIR
    Update the parent directory of PNGs (in-place DB update).

Flags
-----
--pure-species
    Restrict results to unambiguously identified species
    (i.e. no '/' in the species string).

--primary-harmonic
    Restrict spectrogram results to harmonic_idx = 0 only,
    giving one PNG per chirp. Without this flag, all harmonics
    are returned.

--db PATH
    Path to the SQLite database (required for all queries).

Schema
------
png_dirs (
    dir_id    INTEGER PRIMARY KEY,
    dir_path  TEXT NOT NULL UNIQUE
)

recordings (
    file_id    INTEGER PRIMARY KEY,
    filename   TEXT,
    rec_site   TEXT,
    rec_period TEXT
)

chirp_info (
    file_id      INTEGER NOT NULL REFERENCES recordings(file_id),
    chirp_idx    INTEGER NOT NULL,
    measures_row INTEGER NOT NULL,     -- 0-based iloc index into chirp measures parquet
    species      TEXT,
    confidence   REAL,
    PRIMARY KEY (file_id, chirp_idx)
)

chirp_spectrograms (
    file_id      INTEGER NOT NULL,
    chirp_idx    INTEGER NOT NULL,
    harmonic_idx INTEGER NOT NULL,     -- 0 = primary harmonic
    dir_id       INTEGER NOT NULL REFERENCES png_dirs(dir_id),
    png_filename TEXT NOT NULL,
    PRIMARY KEY (file_id, chirp_idx, harmonic_idx),
    FOREIGN KEY (file_id, chirp_idx) REFERENCES chirp_info(file_id, chirp_idx),
    UNIQUE (dir_id, png_filename)
)

Examples
--------
bat_db_query.py --db chirp_meta.db \\
    --spectrograms-for-file 1212612

bat_db_query.py --db chirp_meta.db --pure-species \\
    --random-spectrogram Myyu

bat_db_query.py --db chirp_meta.db --primary-harmonic \\
    --chirp 1212612 7

bat_db_query.py --db chirp_meta.db \\
    --relocate-pngs /old/path/20220706_lake2 /new/path/20220706_lake2
"""

import argparse
import random
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from logging_service import LoggingService


# ---------------------------------------------------------------------------
# Querier
# ---------------------------------------------------------------------------

class BatDbQuerier:
    """
    Queries the bat chirp metadata SQLite database.

    :param db_path: Path to the SQLite database file.
    :param pure_species: If ``True``, restrict all results to rows
        whose species string contains no ``'/'`` (unambiguous IDs only).
    :param primary_harmonic: If ``True``, restrict spectrogram results
        to ``harmonic_idx = 0`` only, returning one PNG per chirp.
        If ``False`` (default), all harmonics are returned.
    """

    def __init__(
        self,
        db_path: str,
        pure_species: bool = False,
        primary_harmonic: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.pure_species = pure_species
        self.primary_harmonic = primary_harmonic
        self.log = LoggingService()

        if not self.db_path.exists():
            self.log.err(f"Database not found: {self.db_path}")
            sys.exit(1)

        self._con = sqlite3.connect(self.db_path)
        self._con.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _pure_species_clause(self) -> str:
        """
        SQL fragment (AND-prefixed) filtering to unambiguous species
        when ``--pure-species`` is set.

        :return: SQL snippet string, or empty string.
        """
        return "AND ci.species NOT LIKE '%/%'" if self.pure_species else ""

    @property
    def _primary_harmonic_clause(self) -> str:
        """
        SQL fragment (AND-prefixed) filtering to ``harmonic_idx = 0``
        when ``--primary-harmonic`` is set.

        :return: SQL snippet string, or empty string.
        """
        return "AND cs.harmonic_idx = 0" if self.primary_harmonic else ""

    def _full_png_path(self, row: sqlite3.Row) -> str:
        """
        Reconstruct the full PNG path from a row containing
        ``dir_path`` and ``png_filename``.

        :param row: A sqlite3.Row with ``dir_path`` and ``png_filename``.
        :return: Absolute path string.
        """
        return str(Path(row["dir_path"]) / row["png_filename"])

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def spectrograms_for_file(self, file_id: int) -> list[str]:
        """
        Return full paths to all spectrogram PNGs for a given file_id.

        :param file_id: Recording identifier.
        :return: List of PNG path strings, ordered by chirp_idx then
            harmonic_idx.
        """
        sql = f"""
            SELECT pd.dir_path, cs.png_filename
            FROM   chirp_info ci
            JOIN   chirp_spectrograms cs
                     ON cs.file_id  = ci.file_id
                    AND cs.chirp_idx = ci.chirp_idx
            JOIN   png_dirs pd ON pd.dir_id = cs.dir_id
            WHERE  ci.file_id = ?
            {self._pure_species_clause}
            {self._primary_harmonic_clause}
            ORDER  BY ci.chirp_idx, cs.harmonic_idx
        """
        rows = self._con.execute(sql, (file_id,)).fetchall()
        return [self._full_png_path(r) for r in rows]

    def measures_for_file(self, file_id: int) -> list[dict]:
        """
        Return chirp metadata rows for a given file_id.

        Each row corresponds to one unique chirp (one row in the chirp
        measures parquet). Use ``measures_row`` as the iloc index into
        the parquet to retrieve the full acoustic measures.

        :param file_id: Recording identifier.
        :return: List of dicts with keys
            ``file_id``, ``chirp_idx``, ``measures_row``,
            ``species``, ``confidence``.
        """
        sql = f"""
            SELECT ci.file_id, ci.chirp_idx, ci.measures_row,
                   ci.species, ci.confidence
            FROM   chirp_info ci
            WHERE  ci.file_id = ?
            {self._pure_species_clause}
            ORDER  BY ci.chirp_idx
        """
        rows = self._con.execute(sql, (file_id,)).fetchall()
        return [dict(r) for r in rows]

    def species_for_file(self, file_id: int) -> list[str]:
        """
        Return the distinct species values associated with a file_id.

        :param file_id: Recording identifier.
        :return: Sorted list of species strings.
        """
        sql = f"""
            SELECT DISTINCT ci.species
            FROM   chirp_info ci
            WHERE  ci.file_id = ?
            {self._pure_species_clause}
            ORDER  BY ci.species
        """
        rows = self._con.execute(sql, (file_id,)).fetchall()
        return [r["species"] for r in rows]

    def chirp(
        self, file_id: int, chirp_idx: int
    ) -> Optional[dict]:
        """
        Return the measures_row and PNG path(s) for a single chirp.

        ``measures_row`` is the 0-based iloc index into the chirp
        measures parquet. If ``--primary-harmonic`` is set, one PNG
        path is returned; otherwise all harmonics are listed.

        :param file_id: Recording identifier.
        :param chirp_idx: Chirp sequence index within the recording.
        :return: Dict with keys ``file_id``, ``chirp_idx``,
            ``measures_row``, ``species``, ``confidence``,
            ``png_paths`` (list of path strings),
            or ``None`` if not found.
        """
        sql = f"""
            SELECT ci.file_id, ci.chirp_idx, ci.measures_row,
                   ci.species, ci.confidence,
                   pd.dir_path, cs.png_filename, cs.harmonic_idx
            FROM   chirp_info ci
            JOIN   chirp_spectrograms cs
                     ON cs.file_id   = ci.file_id
                    AND cs.chirp_idx = ci.chirp_idx
            JOIN   png_dirs pd ON pd.dir_id = cs.dir_id
            WHERE  ci.file_id  = ?
              AND  ci.chirp_idx = ?
            {self._pure_species_clause}
            {self._primary_harmonic_clause}
            ORDER  BY cs.harmonic_idx
        """
        rows = self._con.execute(sql, (file_id, chirp_idx)).fetchall()
        if not rows:
            return None
        first = dict(rows[0])
        result = {
            "file_id":      first["file_id"],
            "chirp_idx":    first["chirp_idx"],
            "measures_row": first["measures_row"],
            "species":      first["species"],
            "confidence":   first["confidence"],
            "png_paths":    [self._full_png_path(r) for r in rows],
        }
        return result

    def random_spectrogram(self, species: str) -> Optional[dict]:
        """
        Return a random spectrogram PNG path and its file_id for the
        given species string.

        The match is exact against the ``species`` column in
        ``chirp_info``. If ``--primary-harmonic`` is set, only
        ``harmonic_idx = 0`` PNGs are candidates.

        :param species: Species string to match exactly.
        :return: Dict with keys ``file_id``, ``chirp_idx``,
            ``measures_row``, ``png_path``, ``harmonic_idx``,
            ``species``, ``confidence``,
            or ``None`` if no matching chirps exist.
        """
        sql = f"""
            SELECT ci.file_id, ci.chirp_idx, ci.measures_row,
                   ci.species, ci.confidence,
                   pd.dir_path, cs.png_filename, cs.harmonic_idx
            FROM   chirp_info ci
            JOIN   chirp_spectrograms cs
                     ON cs.file_id   = ci.file_id
                    AND cs.chirp_idx = ci.chirp_idx
            JOIN   png_dirs pd ON pd.dir_id = cs.dir_id
            WHERE  ci.species = ?
            {self._pure_species_clause}
            {self._primary_harmonic_clause}
        """
        rows = self._con.execute(sql, (species,)).fetchall()
        if not rows:
            return None
        row = random.choice(rows)
        result = dict(row)
        result["png_path"] = self._full_png_path(row)
        del result["dir_path"]
        del result["png_filename"]
        return result

    def relocate_pngs(self, old_dir: str, new_dir: str) -> int:
        """
        Update the stored directory path for PNGs that have moved.

        :param old_dir: Existing ``dir_path`` value in ``png_dirs``.
        :param new_dir: Replacement path.
        :return: Number of rows updated (0 or 1).
        """
        cur = self._con.execute(
            "UPDATE png_dirs SET dir_path = ? WHERE dir_path = ?",
            (new_dir, old_dir),
        )
        self._con.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result) -> None:
    """Pretty-print a query result to stdout."""
    if result is None:
        print("(no result)")
    elif isinstance(result, list):
        if not result:
            print("(empty)")
        else:
            for item in result:
                print(item)
    elif isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"    {item}")
            else:
                print(f"  {k}: {v}")
    else:
        print(result)


def main() -> None:
    log = LoggingService()

    parser = argparse.ArgumentParser(
        description="Query the bat chirp metadata SQLite database."
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--pure-species", action="store_true", default=False,
        dest="pure_species",
        help="Restrict results to unambiguously identified species "
             "(no '/' in species string).",
    )
    parser.add_argument(
        "--primary-harmonic", action="store_true", default=False,
        dest="primary_harmonic",
        help="Restrict spectrogram results to harmonic_idx = 0 only, "
             "returning one PNG per chirp. Default: return all harmonics.",
    )

    # Mutually exclusive query group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--spectrograms-for-file", metavar="FILE_ID", type=int,
        help="List PNG paths for all chirps of a file_id.",
    )
    group.add_argument(
        "--measures-for-file", metavar="FILE_ID", type=int,
        help="List chirp metadata rows for a file_id.",
    )
    group.add_argument(
        "--species", metavar="FILE_ID", type=int,
        help="Show species for a file_id.",
    )
    group.add_argument(
        "--chirp", nargs=2, metavar=("FILE_ID", "CHIRP_IDX"), type=int,
        help="Show measures_row and PNG path(s) for a single chirp.",
    )
    group.add_argument(
        "--random-spectrogram", metavar="SPECIES",
        help="Return a random PNG path + file_id for a species.",
    )
    group.add_argument(
        "--relocate-pngs", nargs=2, metavar=("OLD_DIR", "NEW_DIR"),
        help="Update stored PNG directory path in the database.",
    )

    args = parser.parse_args()

    querier = BatDbQuerier(
        db_path=args.db,
        pure_species=args.pure_species,
        primary_harmonic=args.primary_harmonic,
    )

    try:
        if args.spectrograms_for_file is not None:
            result = querier.spectrograms_for_file(args.spectrograms_for_file)
            _print_result(result)

        elif args.measures_for_file is not None:
            result = querier.measures_for_file(args.measures_for_file)
            _print_result(result)

        elif args.species is not None:
            result = querier.species_for_file(args.species)
            _print_result(result)

        elif args.chirp is not None:
            file_id, chirp_idx = args.chirp
            result = querier.chirp(file_id, chirp_idx)
            if result is None:
                log.warn(f"No chirp found for file_id={file_id}, "
                         f"chirp_idx={chirp_idx}.")
            _print_result(result)

        elif args.random_spectrogram is not None:
            result = querier.random_spectrogram(args.random_spectrogram)
            if result is None:
                log.warn(f"No spectrograms found for species "
                         f"'{args.random_spectrogram}'.")
            _print_result(result)

        elif args.relocate_pngs is not None:
            old_dir, new_dir = args.relocate_pngs
            n = querier.relocate_pngs(old_dir, new_dir)
            if n == 0:
                log.warn(f"No png_dirs row matched '{old_dir}'.")
            else:
                log.info(
                    f"Updated {n} png_dirs row: '{old_dir}' → '{new_dir}'."
                )

    finally:
        querier.close()


if __name__ == "__main__":
    main()