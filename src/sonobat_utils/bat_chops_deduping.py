#!/usr/bin/env python3
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-30 10:38:02
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/bat_chops_deduping.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-30 10:38:45
#
# **********************************************************

"""
bat_chops_deduping.py — Bat chop deduplication helper.

Run on each machine to produce a stem-list file, then transfer one list
to the other machine and run with --compare to find safe-to-transfer
chops and true duplicates.

Workflow
--------
  1. On quintus:
       python chop_dedup.py --machine quintus \\
                            --root-dir /qnap/bats/barn_sonobat3_2_processed \\
                            --output quintus_stems.json

  2. On sextus:
       python chop_dedup.py --machine sextus \\
                            --root-dir /data/win_share \\
                            --output sextus_stems.json

  3. Copy one JSON to the other machine, then run:
       python chop_dedup.py --compare \\
                            --stems-a quintus_stems.json \\
                            --stems-b sextus_stems.json \\
                            --audio-check          # optional: hash audio payload
                            --transfer-list sextus_to_transfer.txt

Module docstring note: the '_2secs' suffix present on quintus filenames is
stripped automatically so stems match between machines.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# WAV header size to skip when comparing audio payload
_WAV_HEADER_BYTES = 44


# ---------------------------------------------------------------------------
# ChopDeduplicator
# ---------------------------------------------------------------------------

class ChopDeduplicator:
    """Scan a directory tree for .wav chop files and emit a stem→path map.

    A *stem* is the filename without extension and without the ``_2secs``
    suffix that quintus files carry, e.g.::

        barn-20220723_220147_2secs.wav  →  barn-20220723_220147
        barn-20220723_220147.wav        →  barn-20220723_220147

    Both sides therefore produce identical stems for the same recording,
    enabling direct set arithmetic across machines.

    :param machine: Human-readable machine label (e.g. ``"quintus"``).
    :param root_dir: Root directory to walk recursively.
    """

    _SUFFIX_TO_STRIP = "_2secs"

    def __init__(self, machine: str, root_dir: Path) -> None:
        self.machine = machine
        self.root_dir = root_dir
        # stem → absolute path string
        self._stem_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> dict[str, str]:
        """Walk *root_dir* and build the stem→path mapping.

        :return: Dict mapping stem to absolute file path.
        """
        log.info("[%s] Scanning %s …", self.machine, self.root_dir)
        count = 0
        for wav_path in self.root_dir.rglob("*.wav"):
            stem = self._to_stem(wav_path.name)
            if stem in self._stem_map:
                log.warning(
                    "[%s] Duplicate stem %r — keeping first hit, skipping %s",
                    self.machine, stem, wav_path,
                )
            else:
                self._stem_map[stem] = str(wav_path)
            count += 1
            if count % 50_000 == 0:
                log.info("[%s]   … %d files scanned so far", self.machine, count)

        log.info("[%s] Done — %d .wav files, %d unique stems",
                 self.machine, count, len(self._stem_map))
        return self._stem_map

    def save(self, output_path: Path) -> None:
        """Serialise the stem map to JSON.

        :param output_path: Destination file path.
        """
        payload = {
            "machine": self.machine,
            "root_dir": str(self.root_dir),
            "stems": self._stem_map,
        }
        with output_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        log.info("[%s] Stem map written to %s", self.machine, output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _to_stem(cls, filename: str) -> str:
        """Strip extension and optional *_2secs* suffix.

        :param filename: Bare filename (not a full path).
        :return: Normalised stem string.
        """
        stem = Path(filename).stem          # strip .wav
        if stem.endswith(cls._SUFFIX_TO_STRIP):
            stem = stem[: -len(cls._SUFFIX_TO_STRIP)]
        return stem


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

class ChopComparator:
    """Compare two stem maps and classify files for safe transfer.

    Classification
    --------------
    * **sextus-only** — safe to copy to quintus (no stem overlap).
    * **overlap** — same stem on both sides; may or may not be true
      duplicates depending on audio payload.
    * **quintus-only** — already on quintus, not relevant for transfer.

    If *audio_check* is True, overlapping files are further resolved by
    comparing the MD5 of their audio payload (bytes after the WAV header).

    :param map_a: Stem map from machine A (quintus, dict ``stem→path``).
    :param map_b: Stem map from machine B (sextus, dict ``stem→path``).
    :param label_a: Label for machine A.
    :param label_b: Label for machine B.
    :param audio_check: When True, hash audio payloads for overlaps.
    """

    def __init__(
        self,
        map_a: dict[str, str],
        map_b: dict[str, str],
        label_a: str = "quintus",
        label_b: str = "sextus",
        audio_check: bool = False,
    ) -> None:
        self._map_a = map_a
        self._map_b = map_b
        self._label_a = label_a
        self._label_b = label_b
        self._audio_check = audio_check

        self.stems_a: set[str] = set(map_a)
        self.stems_b: set[str] = set(map_b)

        self.only_a: set[str] = self.stems_a - self.stems_b
        self.only_b: set[str] = self.stems_b - self.stems_a
        self.overlap: set[str] = self.stems_a & self.stems_b

        # Filled by resolve_overlaps()
        self.true_duplicates: list[str] = []
        self.content_differs: list[str] = []
        self.unresolved: list[str] = []   # one/both files missing locally

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print a summary of the comparison to stdout."""
        log.info("=== Comparison Summary ===")
        log.info("  %-12s only : %d", self._label_a, len(self.only_a))
        log.info("  %-12s only : %d", self._label_b, len(self.only_b))
        log.info("  Overlap (same stem) : %d", len(self.overlap))

    def resolve_overlaps(self) -> None:
        """Classify overlapping stems by audio-payload hash.

        Populates :attr:`true_duplicates`, :attr:`content_differs`, and
        :attr:`unresolved`.  Skips hashing if *audio_check* is False —
        all overlaps go into *unresolved*.
        """
        if not self._audio_check:
            log.info("--audio-check not set; %d overlaps left unresolved",
                     len(self.overlap))
            self.unresolved = sorted(self.overlap)
            return

        log.info("Hashing audio payload for %d overlapping stems …",
                 len(self.overlap))
        for i, stem in enumerate(sorted(self.overlap), 1):
            path_a = self._map_a[stem]
            path_b = self._map_b[stem]
            hash_a = self._audio_hash(path_a)
            hash_b = self._audio_hash(path_b)

            if hash_a is None or hash_b is None:
                self.unresolved.append(stem)
            elif hash_a == hash_b:
                self.true_duplicates.append(stem)
            else:
                self.content_differs.append(stem)

            if i % 1000 == 0:
                log.info("  … %d / %d hashed", i, len(self.overlap))

        log.info("  True duplicates       : %d", len(self.true_duplicates))
        log.info("  Content differs       : %d", len(self.content_differs))
        log.info("  Unresolved (missing)  : %d", len(self.unresolved))

    def write_transfer_list(self, output_path: Path) -> None:
        """Write a newline-separated list of *label_b* paths to transfer.

        Includes all *label_b*-only paths plus any overlaps where content
        differs (those need a rename to avoid a silent overwrite).

        :param output_path: Destination file.
        """
        lines: list[str] = []

        for stem in sorted(self.only_b):
            lines.append(self._map_b[stem])

        if self.content_differs:
            log.warning(
                "%d overlapping stems have DIFFERENT audio content — "
                "they will be included in the transfer list with a "
                "'_sb' rename annotation.",
                len(self.content_differs),
            )
            for stem in sorted(self.content_differs):
                # Annotate so the caller knows to rename on arrival
                lines.append(f"{self._map_b[stem]}  # RENAME: {stem}_sb.wav")

        with output_path.open("w") as fh:
            fh.write("\n".join(lines) + "\n")

        log.info("Transfer list written to %s  (%d entries)", output_path, len(lines))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _audio_hash(wav_path: str) -> Optional[str]:
        """Return MD5 hex digest of the audio payload (post-header bytes).

        :param wav_path: Absolute path to the .wav file.
        :return: Hex digest string, or None if the file is unreadable.
        """
        path = Path(wav_path)
        if not path.exists():
            log.warning("File not found locally, skipping hash: %s", wav_path)
            return None
        md5 = hashlib.md5()
        try:
            with path.open("rb") as fh:
                fh.seek(_WAV_HEADER_BYTES)
                while chunk := fh.read(1 << 20):   # 1 MiB chunks
                    md5.update(chunk)
        except OSError as exc:
            log.error("Cannot read %s: %s", wav_path, exc)
            return None
        return md5.hexdigest()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _load_stem_file(path: Path) -> tuple[str, dict[str, str]]:
    """Load a JSON stem file produced by --scan mode.

    :param path: Path to the JSON file.
    :return: Tuple of (machine_label, stem_map).
    """
    with path.open() as fh:
        data = json.load(fh)
    return data["machine"], data["stems"]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    :return: Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Bat chop deduplication helper.\n\n"
            "Run in --scan mode on each machine, then --compare on either."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--scan",
        action="store_true",
        help="Walk root-dir and produce a stem-map JSON file.",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="Compare two stem-map JSON files produced by --scan.",
    )

    # Scan options
    scan_grp = parser.add_argument_group("Scan options")
    scan_grp.add_argument(
        "--machine",
        metavar="NAME",
        help="Label for this machine (e.g. 'quintus' or 'sextus').",
    )
    scan_grp.add_argument(
        "--root-dir",
        metavar="DIR",
        type=Path,
        help="Root directory to scan recursively.",
    )
    scan_grp.add_argument(
        "--output",
        metavar="FILE",
        type=Path,
        default=None,
        help="Output JSON file (default: <machine>_stems.json).",
    )

    # Compare options
    cmp_grp = parser.add_argument_group("Compare options")
    cmp_grp.add_argument(
        "--stems-a",
        metavar="FILE",
        type=Path,
        help="Stem-map JSON from machine A (e.g. quintus).",
    )
    cmp_grp.add_argument(
        "--stems-b",
        metavar="FILE",
        type=Path,
        help="Stem-map JSON from machine B (e.g. sextus).",
    )
    cmp_grp.add_argument(
        "--audio-check",
        action="store_true",
        help=(
            "For overlapping stems, compare audio payload MD5 to distinguish "
            "true duplicates from files with different content. "
            "Both files must be locally accessible."
        ),
    )
    cmp_grp.add_argument(
        "--transfer-list",
        metavar="FILE",
        type=Path,
        default=Path("transfer_list.txt"),
        help="Output file listing paths from stems-b that are safe to transfer "
             "(default: transfer_list.txt).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class Runner:
    """Top-level orchestrator; parses args and dispatches to scan or compare.

    :param argv: Argument list (defaults to sys.argv[1:]).
    """

    def __init__(self, argv: Optional[list[str]] = None) -> None:
        self._parser = _build_arg_parser()
        self._args = self._parser.parse_args(argv)

    def run(self) -> int:
        """Execute the requested mode.

        :return: Exit code (0 = success).
        """
        if self._args.scan:
            return self._do_scan()
        return self._do_compare()

    # ------------------------------------------------------------------

    def _do_scan(self) -> int:
        """Execute scan mode.

        :return: Exit code.
        """
        args = self._args
        if not args.machine:
            self._parser.error("--machine is required in scan mode")
        if not args.root_dir:
            self._parser.error("--root-dir is required in scan mode")
        if not args.root_dir.is_dir():
            log.error("root-dir does not exist: %s", args.root_dir)
            return 1

        output = args.output or Path(f"{args.machine}_stems.json")

        dedup = ChopDeduplicator(machine=args.machine, root_dir=args.root_dir)
        dedup.scan()
        dedup.save(output)
        return 0

    def _do_compare(self) -> int:
        """Execute compare mode.

        :return: Exit code.
        """
        args = self._args
        if not args.stems_a or not args.stems_b:
            self._parser.error("--stems-a and --stems-b are required in compare mode")

        label_a, map_a = _load_stem_file(args.stems_a)
        label_b, map_b = _load_stem_file(args.stems_b)

        log.info("Loaded %d stems from %s (%s)", len(map_a), args.stems_a, label_a)
        log.info("Loaded %d stems from %s (%s)", len(map_b), args.stems_b, label_b)

        cmp = ChopComparator(
            map_a=map_a,
            map_b=map_b,
            label_a=label_a,
            label_b=label_b,
            audio_check=args.audio_check,
        )
        cmp.report()
        cmp.resolve_overlaps()
        cmp.write_transfer_list(args.transfer_list)
        return 0


if __name__ == "__main__":
    sys.exit(Runner().run())
