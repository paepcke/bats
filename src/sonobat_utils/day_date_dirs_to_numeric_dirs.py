#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-25 19:42:48
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-26 22:42:46
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
# Month-name → month-number lookups
# ---------------------------------------------------------------------------

# Abbreviated month names (3 letters, title-cased)
# Canonical 3-letter abbreviations plus common variants found in the wild:
#   "Sept" (4-letter) and full names used without spaces (June, July, etc.)
MONTH_ABBREVS: dict[str, int] = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    # 4-letter variant
    "Sept": 9,
    # Full names used as prefixes (June11-18, July1, etc.)
    "January": 1, "February": 2, "March": 3, "April": 4,
    "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# Full month names — used by the ``N-MonthName`` format
MONTH_FULL: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# ---------------------------------------------------------------------------
# Directory-name patterns, tried in order.  All formats seen so far:
#
#  Pattern A — alpha-prefix, no space:
#    Apr16-23 / Apr30-May8 / June11-18 / June25-July1 / Sept10-17 / Sept30-Oct7
#    ^<MonToken><day>-[<MonToken>]<day>$
#    MonToken may be 3 letters (Apr), 4 letters (Sept), or a full name (June).
#
#  Pattern B — alpha-prefix, space before digits:
#    Nov 13-20  /  Oct 30-Nov 7  /  Sep 25-Oct 2
#    ^<Mon3> <day>-[<Mon3> ]<day>$   (space variant, always 3-letter prefix)
#
#  Pattern C — numeric month + full name: 3-March / 8-August
#    ^<N>-<MonthName>$
# ---------------------------------------------------------------------------

# A: any alphabetic month token (3-letter, 4-letter "Sept", or full name like
#    "June"/"July") immediately followed by digits; optional end-month token
#    and end-day.  The token is greedy-longest so "June" beats "Jun".
_PAT_A = re.compile(
    r"^(?P<month>[A-Za-z]+?)(?P<day>\d{1,2})"    # Mon-token + start-day
    r"-"
    r"(?:[A-Za-z]+)?\d{1,2}$"                     # optional end-Mon + end-day
)

# B: exactly 3-letter abbreviated month, a space, then digits; optional
#    end-month (with or without a trailing space before the end-day).
_PAT_B = re.compile(
    r"^(?P<month>[A-Za-z]{3})\s(?P<day>\d{1,2})"  # Mon3<space>start-day
    r"-"
    r"(?:[A-Za-z]{3}\s?)?\d{1,2}$"                 # optional end-Mon(+space?) + end-day
)

# C: bare month number, hyphen, full month name  (e.g. "3-March", "11-November")
_PAT_C = re.compile(
    r"^(?P<month_num>\d{1,2})-(?P<month_name>[A-Za-z]+)$"
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

        Four directory-name formats are recognised:

        * **Format A** (alpha prefix, no space): ``Apr16-23``, ``June11-18``,
          ``June25-July1``, ``Sept10-17``, ``Sept30-Oct7``
        * **Format B** (3-letter abbrev + space): ``Nov 13-20``, ``Oct 30-Nov 7``
        * **Format C** (numeric month + full name): ``3-March``, ``8-August``

        In formats A and B the *start* month and day are extracted; the end
        portion of the range is discarded.  In format C the month number is
        used directly and the first day of that month is assumed (day = 1).
        Unrecognised names (e.g. ``"Nov-Feb Bat Calls"``) return ``None``.

        :param name: The bare directory name to parse (e.g. ``"June11-18"``).
        :return: A string like ``"20210611"``, or ``None`` if ``name`` does not
                 match any recognised pattern or encodes an invalid calendar date.
        """
        # --- Format A: AprDD-... / JuneDD-... / SeptDD-... / June25-July1 ---
        # MONTH_ABBREVS covers 3-letter, 4-letter (Sept), and full-name prefixes.
        m = _PAT_A.match(name)
        if m:
            month_str = m.group("month").capitalize()
            day = int(m.group("day"))
            month = MONTH_ABBREVS.get(month_str)
            if month is not None:
                return self._make_date_str(month, day)

        # --- Format B: Apr DD-...  or  Oct 30-Nov 7 ---
        m = _PAT_B.match(name)
        if m:
            month_str = m.group("month").capitalize()
            day = int(m.group("day"))
            month = MONTH_ABBREVS.get(month_str)
            if month is not None:
                return self._make_date_str(month, day)

        # --- Format C: 3-March, 11-November ---
        m = _PAT_C.match(name)
        if m:
            month_num = int(m.group("month_num"))
            month_name = m.group("month_name").capitalize()
            # Accept if the numeric month matches the spelled-out name
            expected = MONTH_FULL.get(month_name)
            if expected is not None and expected == month_num:
                return self._make_date_str(month_num, 1)  # use day 1

        return None

    def _make_date_str(self, month: int, day: int) -> str | None:
        """Build a ``YYYYMMDD`` string from month and day, using :attr:`year`.

        :param month: Calendar month number (1-12).
        :param day: Calendar day number (1-31).
        :return: Formatted date string, or ``None`` if the combination is invalid.
        """
        try:
            return date(self.year, month, day).strftime("%Y%m%d")
        except ValueError:
            return None

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
