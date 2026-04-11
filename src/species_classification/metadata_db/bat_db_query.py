#!/usr/bin/env python
# #############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-11 10:46:17
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-11 10:47:05
# #############################################

"""
CLI query wrapper for the bat chirp metadata SQLite database.

Queries
-------
--spectrograms-for-file FILE_ID
    Full paths to all spectrogram PNGs for a recording.

--measures-for-file FILE_ID
    All (file_id, chirp_idx, measures_row) entries for a recording.

--species FILE_ID
    Species (or composite species) associated with a file_id.

--chirp FILE_ID CHIRP_IDX
    measures_row and PNG path for a single chirp.

--random-spectrogram SPECIES
    A random PNG path + file_id for the given species string.

--relocate-pngs OLD_DIR NEW_DIR
    Update the parent directory of PNGs (in-place DB update).

Flags
-----
--pure-species
    Restrict results to unambiguously identified species
    (i.e. no '/' in the species string).

--db PATH
    Path to the SQLite database (required for all queries).

Examples
--------
python bat_db_query.py --db chirp_meta.db \\
    --spectrograms-for-file 1212612

python bat_db_query.py --db chirp_meta.db --pure-species \\
    --random-spectrogram Myyu

python bat_db_query.py --db chirp_meta.db \\
    --chirp 1212612 7

python bat_db_query.py --db chirp_meta.db \\
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
    """

    def __init__(self, db_path: str, pure_species: bool = False) -> None:
        self.db_path = Path(db_path)
        self.pure_species = pure_species
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
        SQL fragment appended (with AND) when ``--pure-species`` is set.

        :return: SQL snippet, empty string if pure_species is False.
        """
        return "AND ci.species NOT LIKE '%/%'" if self.pure_species else ""

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
        :return: Sorted list of PNG path strings.
        """
        sql = f"""
            SELECT pd.dir_path, ci.png_filename
            FROM   chirp_info ci
            JOIN   png_dirs pd ON pd.dir_id = ci.dir_id
            WHERE  ci.file_id = ?
            {self._pure_species_clause}
            ORDER  BY ci.chirp_idx
        """
        rows = self._con.execute(sql, (file_id,)).fetchall()
        return [self._full_png_path(r) for r in rows]

    def measures_for_file(self, file_id: int) -> list[dict]:
        """
        Return chirp metadata rows for a given file_id.

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
        Return the measures row number and PNG path for a single chirp.

        :param file_id: Recording identifier.
        :param chirp_idx: Chirp sequence index within the recording.
        :return: Dict with keys ``file_id``, ``chirp_idx``,
            ``measures_row``, ``png_path``, ``species``, ``confidence``,
            or ``None`` if not found.
        """
        sql = f"""
            SELECT ci.file_id, ci.chirp_idx, ci.measures_row,
                   ci.species, ci.confidence,
                   pd.dir_path, ci.png_filename
            FROM   chirp_info ci
            JOIN   png_dirs pd ON pd.dir_id = ci.dir_id
            WHERE  ci.file_id = ?
              AND  ci.chirp_idx = ?
            {self._pure_species_clause}
        """
        row = self._con.execute(sql, (file_id, chirp_idx)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["png_path"] = self._full_png_path(row)
        del result["dir_path"]
        del result["png_filename"]
        return result

    def random_spectrogram(self, species: str) -> Optional[dict]:
        """
        Return a random PNG path and file_id for the given species string.

        The match is exact against the ``species`` column; if you want
        all Myyu including composite calls, pass ``'Myyu'`` and omit
        ``--pure-species``.

        :param species: Species string to match exactly.
        :return: Dict with keys ``file_id``, ``chirp_idx``,
            ``measures_row``, ``png_path``, ``species``, ``confidence``,
            or ``None`` if no matching chirps exist.
        """
        sql = f"""
            SELECT ci.file_id, ci.chirp_idx, ci.measures_row,
                   ci.species, ci.confidence,
                   pd.dir_path, ci.png_filename
            FROM   chirp_info ci
            JOIN   png_dirs pd ON pd.dir_id = ci.dir_id
            WHERE  ci.species = ?
            {self._pure_species_clause}
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
        elif isinstance(result[0], str):
            for item in result:
                print(item)
        else:
            for item in result:
                print(item)
    elif isinstance(result, dict):
        for k, v in result.items():
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
        help="Show measures_row and PNG path for a single chirp.",
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

    querier = BatDbQuerier(db_path=args.db, pure_species=args.pure_species)

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
                log.info(f"Updated {n} png_dirs row: '{old_dir}' → '{new_dir}'.")

    finally:
        querier.close()


if __name__ == "__main__":
    main()
