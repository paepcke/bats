#!/usr/bin/env python3
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-29 18:10:46
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/chopped_file_organizer.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-29 18:12:05
#
# **********************************************************

"""
Crawl a batch<n>/input directory for all '*_Parsed Files*' subdirectories,
rename chopped .wav files to SonoBat-compatible naming convention, and
distribute them into chopped_files/chopped1, chopped_files/chopped2, ...
alongside the input directory.

Usage:
To preview:
    python chopped_file_organizer.py /data/win_share/batch1/input
Then for real:
    python chopped_file_organizer.py /data/win_share/batch1/input --no-dry-run        
"""

import re
import shutil
import argparse
from math import ceil
from pathlib import Path


class ChoppedFileOrganizer:
    """Collect, rename, and bucket-distribute chopped wav files from a batch input tree."""

    # barn1_D-20200327_213406.wav  or  barn-20200403_185443.wav
    DASH_PAT = re.compile(r'^(.+)-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(\[\d+\])?\.wav$')

    # barn1_D20200327T055143m838.wav  (already in SonoBat format -- pass through)
    SONOBAT_PAT = re.compile(r'^.+_D\d{8}T\d{6}m\d+\.wav$')

    BUCKET_SIZE = 10_000

    def __init__(self, input_dir: str, dry_run: bool = True):
        """
        :param input_dir: Path to the batch<n>/input directory.
        :param dry_run: If True, only print actions without executing.
        """
        self.input_dir   = Path(input_dir).resolve()
        self.chopped_dir = self.input_dir.parent / 'chopped_files'
        self.dry_run     = dry_run

    def _to_sonobat_name(self, filename: str) -> str | None:
        """
        Convert a Long-File-Parser filename to SonoBat-compatible format.
        Returns None if the file is already in SonoBat format or unrecognized.

        :param filename: Original filename (stem + suffix).
        :return: Renamed filename, original if already valid, or None if unrecognized.
        """
        if self.SONOBAT_PAT.match(filename):
            return filename  # already fine
        m = self.DASH_PAT.match(filename)
        if m:
            prefix, yyyy, mo, dd, hh, mm, ss, suffix = m.groups()
            suffix = suffix or ''
            return f"{prefix}_D{yyyy}{mo}{dd}T{hh}{mm}{ss}m000{suffix}.wav"
        return None

    def _collect_wavs(self) -> list[Path]:
        """
        Find all .wav files nested under '*_Parsed Files*' directories.

        :return: Sorted list of Path objects.
        """
        wavs = []
        for parsed_dir in self.input_dir.rglob('*_Parsed Files*'):
            if parsed_dir.is_dir():
                wavs.extend(parsed_dir.glob('*.wav'))
        return sorted(wavs)

    def run(self):
        """
        :return: None
        """
        print(f"Scanning {self.input_dir} ...")
        wavs = self._collect_wavs()
        total = len(wavs)
        n_buckets = ceil(total / self.BUCKET_SIZE) if total else 0
        print(f"Found {total} chopped wav files -> {n_buckets} buckets of up to {self.BUCKET_SIZE}")
        print(f"Output root: {self.chopped_dir}")
        if self.dry_run:
            print("DRY RUN — no files will be moved\n")

        if not self.dry_run:
            self.chopped_dir.mkdir(exist_ok=True)

        skipped, moved = 0, 0
        for i, wav in enumerate(wavs):
            new_name = self._to_sonobat_name(wav.name)
            if new_name is None:
                if i % 1000 == 0 or True:  # always show skips
                    print(f"  SKIP (unrecognized): {wav.name}")
                skipped += 1
                continue

            bucket_num = moved // self.BUCKET_SIZE + 1
            bucket_dir = self.chopped_dir / f"chopped{bucket_num}"

            if not self.dry_run:
                bucket_dir.mkdir(exist_ok=True)
                shutil.move(str(wav), str(bucket_dir / new_name))

            moved += 1
            if moved % 1000 == 0:
                print(f"  {moved}/{total - skipped} (bucket {bucket_num}/{n_buckets})...")

        print(f"\nDone. Moved: {moved}  Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize chopped bat wav files from a batch input tree into chopped_files/chopped<n> buckets."
    )
    parser.add_argument('input_dir', help="Path to the batch<n>/input directory")
    parser.add_argument('--no-dry-run', action='store_true', help="Actually move files (default is dry run)")
    args = parser.parse_args()

    organizer = ChoppedFileOrganizer(
        input_dir=args.input_dir,
        dry_run=not args.no_dry_run
    )
    organizer.run()


if __name__ == '__main__':
    main()
