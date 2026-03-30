#!/usr/bin/env python3
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-30 10:38:02
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/bat_chops_deduping.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-30 12:15:55
#
# **********************************************************

"""
bat_chops_deduping.py — Bat chop deduplication helper.

Three actions drive the full workflow:

**scan** — walk batch subdirectories on one machine and write a stem-map JSON.

**compare** — load two stem-map JSONs, report set overlap, and write a
transfer list of files that exist only on machine B.

**copy-overlaps** — load two stem-map JSONs, compute the common ancestor of
all overlapping source paths, write a ``--files-from`` list of paths relative
to that ancestor, then execute a single ``rclone copy`` call that copies those
files to *dest-dir*, preserving the batch subdirectory structure from the
source.

Workflow
--------
  1. On quintus::

       python src/sonobat_utils/bat_chops_deduping.py --scan --machine quintus \\
           --root-dir /qnap/bats/barn_sonobat3_2_processed \\
           --output /qnap/bats/quintus_chops_file_stems.json

  2. On sextus::

       python src/sonobat_utils/bat_chops_deduping.py --scan --machine sextus \\
           --root-dir /data/win_share/chopped_files \\
           --output /data/win_share/sextus_chops_file_stems.json

  3. Copy one JSON to the other machine, then compare::

       python src/sonobat_utils/bat_chops_deduping.py --compare \\
           --stems-a /qnap/bats/quintus_chops_file_stems.json \\
           --stems-b /data/win_share/sextus_chops_file_stems.json \\
           --transfer-list /data/win_share/sextus_to_transfer.txt

  4. Copy overlapping chops from quintus to a quarantine dir on sextus.
     The rclone remote name for the source machine must be supplied via
     ``--rclone-remote`` (here ``stanford`` as configured in rclone.conf)::

       python src/sonobat_utils/bat_chops_deduping.py --copy-overlaps \\
           --stems-a /data/win_share/quintus_chops_file_stems.json \\
           --stems-b /data/win_share/sextus_chops_file_stems.json \\
           --from-machine quintus \\
           --to-machine sextus \\
           --rclone-remote stanford \\
           --dest-dir /raid/bat_wavs/dedup_temp_overlapping_chops

  5. Once the quarantine copies are local, re-run --compare with
     --audio-check to resolve true duplicates (both paths now local)::

       python src/sonobat_utils/bat_chops_deduping.py --compare \\
           --stems-a /data/win_share/quintus_overlap_chops_stems.json \\
           --stems-b /data/win_share/sextus_chops_file_stems.json \\
           --audio-check \\
           --transfer-list /data/win_share/confirmed_safe_to_transfer.txt

Batch discovery
---------------
The scanner looks for subdirectories of *root-dir* whose names match
``<batch-root-nm><n>`` for n = 1, 2, 3, … (consecutive integers starting
at 1).  Scanning stops at the first gap.  Any other subdirectories of
*root-dir* are ignored.  The default batch root name is ``"batch"``, so
the scanner finds ``batch1``, ``batch2``, ``batch3``, ``batch4``, etc.

JSON format
-----------
Each entry in the ``"stems"`` dict maps a normalised stem to a record::

    {
      "path":  "/data/win_share/chopped_files/batch4/chopped34/barn-20220723_220147.wav",
      "batch": 4
    }

The ``"_2secs"`` suffix present on quintus filenames is stripped
automatically so stems match between machines.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
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

# Bytes to skip at the start of a WAV file when comparing audio payload
_WAV_HEADER_BYTES = 44

# Default prefix for batch subdirectory names
_DEFAULT_BATCH_ROOT_NM = "batch"


# ---------------------------------------------------------------------------
# StemRecord
# ---------------------------------------------------------------------------

class StemRecord:
    """Holds the path and batch number for a single chop file.

    :param path: Absolute path to the .wav file.
    :param batch: Batch number (integer) under which the file was found.
    """

    __slots__ = ("path", "batch")

    def __init__(self, path: str, batch: int) -> None:
        self.path = path
        self.batch = batch

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output.

        :return: ``{"path": ..., "batch": ...}`` dict.
        """
        return {"path": self.path, "batch": self.batch}

    @classmethod
    def from_dict(cls, data: dict) -> "StemRecord":
        """Deserialise from a plain dict loaded from JSON.

        :param data: Dict with ``"path"`` and ``"batch"`` keys.
        :return: New :class:`StemRecord` instance.
        """
        return cls(path=data["path"], batch=data["batch"])


# ---------------------------------------------------------------------------
# ChopDeduplicator
# ---------------------------------------------------------------------------

class ChopDeduplicator:
    """Scan batch subdirectories under *root_dir* for .wav chop files.

    Only subdirectories named ``<batch_root_nm>1``, ``<batch_root_nm>2``,
    … are scanned; scanning stops at the first gap (e.g. if ``batch3``
    does not exist, ``batch4`` is not checked).  All other subdirectories
    of *root_dir* are ignored.

    A *stem* is the filename without extension and without the ``_2secs``
    suffix that quintus files carry, e.g.::

        barn-20220723_220147_2secs.wav  →  barn-20220723_220147
        barn-20220723_220147.wav        →  barn-20220723_220147

    Both sides therefore produce identical stems for the same recording,
    enabling direct set arithmetic across machines.

    :param machine: Human-readable machine label (e.g. ``"quintus"``).
    :param root_dir: Parent directory that contains the batch subdirectories.
    :param batch_root_nm: Common name prefix for batch dirs (default ``"batch"``).
    """

    _SUFFIX_TO_STRIP = "_2secs"

    def __init__(
        self,
        machine: str,
        root_dir: Path,
        batch_root_nm: str = _DEFAULT_BATCH_ROOT_NM,
    ) -> None:
        self.machine = machine
        self.root_dir = root_dir
        self.batch_root_nm = batch_root_nm
        # stem → StemRecord
        self._stem_map: dict[str, StemRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> dict[str, StemRecord]:
        """Discover batch dirs and build the stem→StemRecord mapping.

        :return: Dict mapping normalised stem to :class:`StemRecord`.
        """
        batch_dirs = self._discover_batch_dirs()
        if not batch_dirs:
            log.warning(
                "[%s] No batch directories found under %s with prefix %r",
                self.machine, self.root_dir, self.batch_root_nm,
            )
            return self._stem_map

        for batch_num, batch_dir in batch_dirs:
            self._scan_batch(batch_num, batch_dir)

        log.info(
            "[%s] Grand total: %d unique stems across %d batches",
            self.machine, len(self._stem_map), len(batch_dirs),
        )
        return self._stem_map

    def save(self, output_path: Path) -> None:
        """Serialise the stem map to JSON.

        :param output_path: Destination file path.
        """
        payload = {
            "machine": self.machine,
            "root_dir": str(self.root_dir),
            "batch_root_nm": self.batch_root_nm,
            "stems": {stem: rec.to_dict() for stem, rec in self._stem_map.items()},
        }
        with output_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        log.info(
            "[%s] Stem map written to %s  (%d entries)",
            self.machine, output_path, len(self._stem_map),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_batch_dirs(self) -> list[tuple[int, Path]]:
        """Return an ordered list of (batch_number, path) for existing batch dirs.

        Probes ``<batch_root_nm>1``, ``<batch_root_nm>2``, … and stops at
        the first gap.

        :return: List of (int, Path) tuples in ascending batch order.
        """
        found: list[tuple[int, Path]] = []
        n = 1
        while True:
            candidate = self.root_dir / f"{self.batch_root_nm}{n}"
            if candidate.is_dir():
                found.append((n, candidate))
                log.info("[%s] Found batch dir: %s", self.machine, candidate)
                n += 1
            else:
                if n == 1:
                    log.warning(
                        "[%s] %s does not exist — no batches found",
                        self.machine, candidate,
                    )
                else:
                    log.info(
                        "[%s] %s not found — stopping batch search at %d batches",
                        self.machine, candidate, len(found),
                    )
                break
        return found

    def _scan_batch(self, batch_num: int, batch_dir: Path) -> None:
        """Walk one batch directory and add its stems to the map.

        :param batch_num: Integer batch number (for the StemRecord).
        :param batch_dir: Absolute path to the batch subdirectory.
        """
        log.info("[%s] Scanning batch%d at %s …", self.machine, batch_num, batch_dir)
        count = 0
        for wav_path in batch_dir.rglob("*.wav"):
            stem = self._to_stem(wav_path.name)
            if stem in self._stem_map:
                existing = self._stem_map[stem]
                log.warning(
                    "[%s] Duplicate stem %r — already seen in batch%d (%s), "
                    "skipping batch%d path %s",
                    self.machine, stem, existing.batch, existing.path,
                    batch_num, wav_path,
                )
            else:
                self._stem_map[stem] = StemRecord(
                    path=str(wav_path), batch=batch_num
                )
            count += 1
            if count % 50_000 == 0:
                log.info(
                    "[%s]   batch%d … %d files scanned so far",
                    self.machine, batch_num, count,
                )

        log.info(
            "[%s] batch%d done — %d .wav files, running total %d unique stems",
            self.machine, batch_num, count, len(self._stem_map),
        )

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
# ChopComparator
# ---------------------------------------------------------------------------

class ChopComparator:
    """Compare two stem maps and classify files for safe transfer.

    Classification
    --------------
    * **b-only** — stem present only in map B (sextus); safe to copy.
    * **overlap** — same stem on both sides; may or may not be true
      duplicates depending on audio payload.
    * **a-only** — stem present only in map A (quintus); not relevant
      for the transfer direction.

    If *audio_check* is True, overlapping files are further resolved by
    comparing the MD5 of their audio payload (bytes after the WAV header).

    :param map_a: Stem map from machine A (quintus).
    :param map_b: Stem map from machine B (sextus).
    :param label_a: Label for machine A.
    :param label_b: Label for machine B.
    :param audio_check: When True, hash audio payloads for overlapping stems.
    """

    def __init__(
        self,
        map_a: dict[str, StemRecord],
        map_b: dict[str, StemRecord],
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

        # Populated by resolve_overlaps()
        self.true_duplicates: list[str] = []
        self.content_differs: list[str] = []
        self.unresolved: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Log a summary of the set comparison."""
        log.info("=== Comparison Summary ===")
        log.info("  %-12s only : %d", self._label_a, len(self.only_a))
        log.info("  %-12s only : %d", self._label_b, len(self.only_b))
        log.info("  Overlap (same stem) : %d", len(self.overlap))

    def resolve_overlaps(self) -> None:
        """Classify overlapping stems by audio-payload MD5.

        Populates :attr:`true_duplicates`, :attr:`content_differs`, and
        :attr:`unresolved`.  If *audio_check* is False all overlaps are
        placed in *unresolved* without any file I/O.
        """
        if not self._audio_check:
            log.info(
                "--audio-check not set; %d overlaps left unresolved",
                len(self.overlap),
            )
            self.unresolved = sorted(self.overlap)
            return

        log.info(
            "Hashing audio payload for %d overlapping stems …", len(self.overlap)
        )
        for i, stem in enumerate(sorted(self.overlap), 1):
            path_a = self._map_a[stem].path
            path_b = self._map_b[stem].path
            hash_a = self._audio_hash(path_a)
            hash_b = self._audio_hash(path_b)

            if hash_a is None or hash_b is None:
                self.unresolved.append(stem)
            elif hash_a == hash_b:
                self.true_duplicates.append(stem)
            else:
                self.content_differs.append(stem)

            if i % 1_000 == 0:
                log.info("  … %d / %d hashed", i, len(self.overlap))

        log.info("  True duplicates       : %d", len(self.true_duplicates))
        log.info("  Content differs       : %d", len(self.content_differs))
        log.info("  Unresolved (missing)  : %d", len(self.unresolved))

    def write_transfer_list(self, output_path: Path) -> None:
        """Write a newline-separated list of *label_b* paths to transfer.

        Includes all *label_b*-only paths.  Overlapping stems where audio
        content differs are also included, annotated with a ``# RENAME``
        comment so they can be handled without silently overwriting the
        existing quintus copy.

        :param output_path: Destination file.
        """
        lines: list[str] = []

        for stem in sorted(self.only_b):
            lines.append(self._map_b[stem].path)

        if self.content_differs:
            log.warning(
                "%d overlapping stems have DIFFERENT audio content — "
                "included in transfer list with a '# RENAME' annotation.",
                len(self.content_differs),
            )
            for stem in sorted(self.content_differs):
                lines.append(
                    f"{self._map_b[stem].path}  # RENAME: {stem}_sb.wav"
                )

        with output_path.open("w") as fh:
            fh.write("\n".join(lines) + "\n")

        log.info(
            "Transfer list written to %s  (%d entries)", output_path, len(lines)
        )

    # ------------------------------------------------------------------
    # Private helpers
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
# OverlapCopier
# ---------------------------------------------------------------------------

class OverlapCopier:
    """Copy overlapping chops from one machine to a quarantine dir on another.

    Strategy: compute the longest common ancestor of all overlapping source
    paths, write a ``--files-from`` list of paths *relative to that ancestor*,
    then execute a single ``rclone copy`` call::

        rclone copy <rclone_remote>:<common_ancestor> <dest_dir> \\
            --files-from=<tmp> \\
            --transfers 16 --checkers 32 --buffer-size 32M \\
            --progress --ignore-checksum [--dry-run]

    The batch subdirectory structure below the common ancestor is preserved
    verbatim under *dest_dir*.

    :param map_a: Stem map from machine A.
    :param map_b: Stem map from machine B.
    :param label_a: Label for machine A (as stored in the JSON).
    :param label_b: Label for machine B (as stored in the JSON).
    :param from_machine: Machine label to copy from (must match a JSON label).
    :param to_machine: Machine label to copy to.
    :param rclone_remote: rclone remote name for *from_machine*
                          (e.g. ``"stanford"``).
    :param dest_dir: Absolute local path where overlapping chops will land.
    :param dry_run: When True, pass ``--dry-run`` to rclone.
    """

    # Proven performance settings — hardcoded for this hardware pair
    _RCLONE_TRANSFERS = 16
    _RCLONE_CHECKERS  = 32
    _RCLONE_BUFFER    = "32M"

    def __init__(
        self,
        map_a: dict[str, StemRecord],
        map_b: dict[str, StemRecord],
        label_a: str,
        label_b: str,
        from_machine: str,
        to_machine: str,
        rclone_remote: str,
        dest_dir: Path,
        dry_run: bool = False,
    ) -> None:
        self._map_a = map_a
        self._map_b = map_b
        self._label_a = label_a
        self._label_b = label_b
        self._from_machine = from_machine
        self._to_machine = to_machine
        self._rclone_remote = rclone_remote
        self._dest_dir = dest_dir
        self._dry_run = dry_run

        self._overlap: set[str] = set(map_a) & set(map_b)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def copy(self) -> int:
        """Compute common ancestor, write files-from list, invoke rclone.

        :return: rclone exit code (0 = success).
        """
        if not self._overlap:
            log.info("No overlapping stems — nothing to copy.")
            return 0

        log.info(
            "Preparing to copy %d overlapping chops from %s to %s:%s",
            len(self._overlap), self._from_machine,
            self._to_machine, self._dest_dir,
        )

        source_map = self._select_source_map()
        common_ancestor, files_from_path = self._write_files_from(source_map)
        return self._run_rclone(common_ancestor, files_from_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_source_map(self) -> dict[str, StemRecord]:
        """Return the stem map whose label matches *from_machine*.

        Falls back to *map_a* with a warning if neither label matches
        exactly (e.g. short hostname vs FQDN).

        :return: The stem map to use as the rclone source.
        """
        if self._from_machine == self._label_a:
            log.info("Source map: %s", self._label_a)
            return self._map_a
        if self._from_machine == self._label_b:
            log.info("Source map: %s", self._label_b)
            return self._map_b
        log.warning(
            "--from-machine %r does not exactly match either JSON label "
            "(%r, %r) — defaulting to stems-a (%s).",
            self._from_machine, self._label_a, self._label_b, self._label_a,
        )
        return self._map_a

    def _write_files_from(
        self, source_map: dict[str, StemRecord]
    ) -> tuple[str, Path]:
        """Compute the common ancestor and write a relative files-from list.

        rclone's ``--files-from`` expects paths relative to the source
        directory passed to ``rclone copy``.  We derive that source directory
        as the longest common path prefix across all overlapping files using
        :func:`os.path.commonpath`, then strip it from each absolute path to
        produce the relative entries.

        :param source_map: Stem map whose overlap paths form the copy list.
        :return: Tuple of (common_ancestor_str, path_to_tmp_files_from_file).
        """
        overlap_paths = [source_map[stem].path for stem in self._overlap]
        common_ancestor = os.path.commonpath(overlap_paths)
        log.info("Common ancestor of overlap paths: %s", common_ancestor)

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="bat_chops_overlap_",
            suffix=".txt",
            delete=False,
        )
        for stem in sorted(self._overlap):
            abs_path = source_map[stem].path
            rel_path = os.path.relpath(abs_path, common_ancestor)
            tmp.write(rel_path + "\n")
        tmp.close()
        log.info(
            "files-from list written to %s  (%d entries)",
            tmp.name, len(self._overlap),
        )
        return common_ancestor, Path(tmp.name)

    def _run_rclone(self, common_ancestor: str, files_from_path: Path) -> int:
        """Execute a single rclone copy call and stream its output.

        Command structure::

            rclone copy <remote>:<common_ancestor> <dest_dir> \\
                --files-from=<list> \\
                --transfers 16 --checkers 32 --buffer-size 32M \\
                --progress --ignore-checksum [--dry-run]

        :param common_ancestor: Absolute path on the source machine used as
                                the rclone source directory.
        :param files_from_path: Path to the temporary files-from list.
        :return: rclone exit code.
        """
        source = f"{self._rclone_remote}:{common_ancestor}"
        dest   = str(self._dest_dir)

        cmd = [
            "rclone", "copy",
            source, dest,
            f"--files-from={files_from_path}",
            f"--transfers={self._RCLONE_TRANSFERS}",
            f"--checkers={self._RCLONE_CHECKERS}",
            f"--buffer-size={self._RCLONE_BUFFER}",
            "--progress",
            "--ignore-checksum",
        ]
        if self._dry_run:
            cmd.append("--dry-run")
            log.info("DRY RUN — no files will be transferred.")

        log.info("rclone command:\n  %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, check=False)
        finally:
            files_from_path.unlink(missing_ok=True)

        if result.returncode == 0:
            log.info("rclone completed successfully.")
        else:
            log.error("rclone exited with code %d.", result.returncode)
        return result.returncode


# ---------------------------------------------------------------------------
# JSON I/O helper
# ---------------------------------------------------------------------------

def _load_stem_file(path: Path) -> tuple[str, dict[str, StemRecord]]:
    """Load a JSON stem file produced by --scan mode.

    :param path: Path to the JSON file.
    :return: Tuple of (machine_label, stem_map) where stem_map maps each
             stem string to a :class:`StemRecord`.
    """
    with path.open() as fh:
        data = json.load(fh)
    stem_map: dict[str, StemRecord] = {
        stem: StemRecord.from_dict(rec)
        for stem, rec in data["stems"].items()
    }
    return data["machine"], stem_map


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    :return: Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Bat chop deduplication helper.\n\n"
            "Three actions: --scan, --compare, --copy-overlaps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--scan",
        action="store_true",
        help="Walk batch subdirs under --root-dir and produce a stem-map JSON.",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="Compare two stem-map JSONs and write a transfer list.",
    )
    mode.add_argument(
        "--copy-overlaps",
        action="store_true",
        help=(
            "Copy chops that appear in both stem maps from --from-machine "
            "to --dest-dir on --to-machine via a single rsync call."
        ),
    )

    # ---- Scan options ------------------------------------------------
    scan_grp = parser.add_argument_group("Scan options  (--scan)")
    scan_grp.add_argument(
        "--machine",
        metavar="NAME",
        help="Label for this machine (e.g. 'quintus' or 'sextus').",
    )
    scan_grp.add_argument(
        "--root-dir",
        metavar="DIR",
        type=Path,
        help="Parent directory that contains the batch subdirectories.",
    )
    scan_grp.add_argument(
        "--batch-root-nm",
        metavar="PREFIX",
        default=_DEFAULT_BATCH_ROOT_NM,
        help=(
            f"Common name prefix for batch subdirectories "
            f"(default: '{_DEFAULT_BATCH_ROOT_NM}'). "
            f"The scanner probes <PREFIX>1, <PREFIX>2, … stopping at the "
            f"first gap."
        ),
    )
    scan_grp.add_argument(
        "--output",
        metavar="FILE",
        type=Path,
        default=None,
        help="Output JSON file (default: <machine>_chops_file_stems.json).",
    )

    # ---- Shared: compare + copy-overlaps -----------------------------
    shared_grp = parser.add_argument_group(
        "Shared options  (--compare and --copy-overlaps)"
    )
    shared_grp.add_argument(
        "--stems-a",
        metavar="FILE",
        type=Path,
        help="Stem-map JSON from machine A (typically quintus).",
    )
    shared_grp.add_argument(
        "--stems-b",
        metavar="FILE",
        type=Path,
        help="Stem-map JSON from machine B (typically sextus).",
    )

    # ---- Compare options ---------------------------------------------
    cmp_grp = parser.add_argument_group("Compare options  (--compare)")
    cmp_grp.add_argument(
        "--audio-check",
        action="store_true",
        help=(
            "For overlapping stems, compare audio-payload MD5 to distinguish "
            "true duplicates from files with genuinely different content. "
            "Both files must be locally accessible."
        ),
    )
    cmp_grp.add_argument(
        "--transfer-list",
        metavar="FILE",
        type=Path,
        default=Path("transfer_list.txt"),
        help=(
            "Output file listing stems-b paths that are safe to transfer "
            "(default: transfer_list.txt)."
        ),
    )

    # ---- Copy-overlaps options ---------------------------------------
    cp_grp = parser.add_argument_group("Copy-overlaps options  (--copy-overlaps)")
    cp_grp.add_argument(
        "--from-machine",
        metavar="HOST",
        help="Machine label to copy overlapping chops FROM (must match JSON label).",
    )
    cp_grp.add_argument(
        "--to-machine",
        metavar="HOST",
        help="Machine label to copy overlapping chops TO.",
    )
    cp_grp.add_argument(
        "--rclone-remote",
        metavar="REMOTE",
        help=(
            "rclone remote name for --from-machine as configured in rclone.conf "
            "(e.g. 'stanford')."
        ),
    )
    cp_grp.add_argument(
        "--dest-dir",
        metavar="DIR",
        type=Path,
        help=(
            "Absolute local path where overlapping chops land, with batch "
            "subdirectory structure preserved from the source."
        ),
    )
    cp_grp.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to rclone; log the command without transferring data.",
    )

    return parser


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """Top-level orchestrator: parse args and dispatch to the chosen action.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).
    """

    def __init__(self, argv: Optional[list[str]] = None) -> None:
        self._parser = _build_arg_parser()
        self._args = self._parser.parse_args(argv)

    def run(self) -> int:
        """Execute the requested action.

        :return: Exit code (0 = success, 1 = error).
        """
        if self._args.scan:
            return self._do_scan()
        if self._args.compare:
            return self._do_compare()
        return self._do_copy_overlaps()

    # ------------------------------------------------------------------

    def _do_scan(self) -> int:
        """Execute scan mode.

        :return: Exit code.
        """
        args = self._args
        if not args.machine:
            self._parser.error("--machine is required with --scan")
        if not args.root_dir:
            self._parser.error("--root-dir is required with --scan")
        if not args.root_dir.is_dir():
            log.error("root-dir does not exist: %s", args.root_dir)
            return 1

        output = args.output or Path(f"{args.machine}_chops_file_stems.json")

        dedup = ChopDeduplicator(
            machine=args.machine,
            root_dir=args.root_dir,
            batch_root_nm=args.batch_root_nm,
        )
        dedup.scan()
        dedup.save(output)
        return 0

    def _do_compare(self) -> int:
        """Execute compare mode.

        :return: Exit code.
        """
        args = self._args
        if not args.stems_a or not args.stems_b:
            self._parser.error("--stems-a and --stems-b are required with --compare")

        label_a, map_a = _load_stem_file(args.stems_a)
        label_b, map_b = _load_stem_file(args.stems_b)

        log.info("Loaded %d stems from %s  (machine: %s)", len(map_a), args.stems_a, label_a)
        log.info("Loaded %d stems from %s  (machine: %s)", len(map_b), args.stems_b, label_b)

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

    def _do_copy_overlaps(self) -> int:
        """Execute copy-overlaps mode.

        :return: rclone exit code, or 1 on argument error.
        """
        args = self._args
        missing = [
            opt for opt, val in [
                ("--stems-a",       args.stems_a),
                ("--stems-b",       args.stems_b),
                ("--from-machine",  args.from_machine),
                ("--to-machine",    args.to_machine),
                ("--rclone-remote", args.rclone_remote),
                ("--dest-dir",      args.dest_dir),
            ] if not val
        ]
        if missing:
            self._parser.error(
                f"--copy-overlaps requires: {', '.join(missing)}"
            )

        label_a, map_a = _load_stem_file(args.stems_a)
        label_b, map_b = _load_stem_file(args.stems_b)

        log.info("Loaded %d stems from %s  (machine: %s)", len(map_a), args.stems_a, label_a)
        log.info("Loaded %d stems from %s  (machine: %s)", len(map_b), args.stems_b, label_b)

        copier = OverlapCopier(
            map_a=map_a,
            map_b=map_b,
            label_a=label_a,
            label_b=label_b,
            from_machine=args.from_machine,
            to_machine=args.to_machine,
            rclone_remote=args.rclone_remote,
            dest_dir=args.dest_dir,
            dry_run=args.dry_run,
        )
        return copier.copy()


if __name__ == "__main__":
    sys.exit(Runner().run())
