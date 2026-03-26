#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-25 19:42:48
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-25 19:52:34

"""
day_date_dirs_to_numeric_dirs.py -- Rename week-range directories to numeric date format.

Directories named with human-readable date ranges like ``Apr16-23``,
``Apr30-May8``, or ``Aug13-19`` are renamed to the ISO-like numeric form
``YYYYMMDD``, where the date corresponds to the **first day** of the range
encoded in the original directory name.

Usage::

    python day_date_dirs_to_numeric_dirs.py [--dry-run] <year> <root_dir>

Arguments:
    year        Four-digit year that applies to all directory names (e.g. 2022).
    root_dir    Path to the parent directory that contains the date-range dirs.

Options:
    --dry-run   Print the planned renames without touching the filesystem.

Examples::

    # Perform actual renames
    python day_date_dirs_to_numeric_dirs.py 2022 /data/bat_surveys

    # Preview what would happen
    python day_date_dirs_to_numeric_dirs.py --dry-run 2022 /data/bat_surveys
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Month-name → month-number lookup (abbreviated, title-cased)
# ---------------------------------------------------------------------------
MONTH_ABBREVS: dict[str, int] = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Matches patterns such as:
#   Apr16-23        → month=Apr, start_day=16
#   Apr30-May8      → month=Apr, start_day=30
#   Aug13-19        → month=Aug, start_day=13
_DIR_PATTERN = re.compile(
    r"^(?P<month>[A-Za-z]{3})(?P<day>\d{1,2})"   # leading month + start day
    r"-"                                            # separator
    r"(?:[A-Za-z]{3})?\d{1,2}$"                   # optional end-month + end day
)


class DirRenamer:
    """Rename human-readable date-range directories to numeric ``YYYYMMDD`` names.

    :param year: The calendar year (e.g. 2022) to use when constructing dates.
    :param root: Path to the parent directory containing the date-range subdirs.
    :param dry_run: When ``True``, log planned renames but make no filesystem changes.
    """

    def __init__(self, year: int, root: Path, *, dry_run: bool = False) -> None:
        self.year = year
        self.root = root
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Discover and rename (or preview) all matching subdirectories.

        Iterates over every immediate child of :attr:`root`, attempts to parse
        its name as a date-range directory, and renames those that match.
        Non-matching names and non-directory entries are silently skipped.
        """
        if not self.root.is_dir():
            print(f"ERROR: '{self.root}' is not a directory or does not exist.",
                  file=sys.stderr)
            sys.exit(1)

        entries = sorted(self.root.iterdir())
        renamed = 0
        skipped = 0

        for entry in entries:
            if not entry.is_dir():
                continue  # ignore plain files

            new_name = self._parse_dir_name(entry.name)
            if new_name is None:
                print(f"  skip  : '{entry.name}'  (unrecognised pattern)")
                skipped += 1
                continue

            dest = entry.parent / new_name
            self._rename(entry, dest)
            renamed += 1

        # Summary line
        mode_tag = "[DRY-RUN] " if self.dry_run else ""
        print(f"\n{mode_tag}Done — {renamed} renamed, {skipped} skipped.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_dir_name(self, name: str) -> str | None:
        """Parse a date-range directory name and return the target ``YYYYMMDD`` string.

        :param name: The bare directory name to parse (e.g. ``"Apr16-23"``).
        :return: A string like ``"20220416"``, or ``None`` if ``name`` does not
                 match the expected pattern or contains an invalid date.
        """
        match = _DIR_PATTERN.match(name)
        if match is None:
            return None

        month_str = match.group("month").capitalize()  # normalise case
        day = int(match.group("day"))
        month = MONTH_ABBREVS.get(month_str)

        if month is None:
            return None  # unrecognised month abbreviation

        try:
            parsed_date = date(self.year, month, day)
        except ValueError:
            # day/month combination is not a real calendar date
            return None

        return parsed_date.strftime("%Y%m%d")

    def _rename(self, src: Path, dest: Path) -> None:
        """Rename ``src`` to ``dest``, respecting the dry-run flag.

        :param src: Existing directory path.
        :param dest: Desired new directory path.
        """
        arrow = f"  {src.name}  →  {dest.name}"

        if self.dry_run:
            print(f"[DRY-RUN] would rename: {arrow}")
            return

        if dest.exists():
            print(f"  ERROR: destination already exists, skipping: {arrow}",
                  file=sys.stderr)
            return

        src.rename(dest)
        print(f"  renamed: {arrow}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser for the CLI.

    :return: Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="day_date_dirs_to_numeric_dirs.py",
        description=(
            "Rename week-range directories (e.g. Apr16-23) to numeric "
            "YYYYMMDD names based on the first day of the range."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without modifying the filesystem.",
    )
    parser.add_argument(
        "year",
        type=int,
        help="Four-digit year to apply to all directory dates (e.g. 2022).",
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing the date-range subdirectories.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and execute the rename operation."""
    parser = _build_parser()
    args = parser.parse_args()

    renamer = DirRenamer(
        year=args.year,
        root=args.root_dir,
        dry_run=args.dry_run,
    )
    renamer.run()


if __name__ == "__main__":
    main()
