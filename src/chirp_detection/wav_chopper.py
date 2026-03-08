#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-08 15:19:14
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/chirp_detection/wav_chopper.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-08 15:35:07
#
# **********************************************************
"""
wav_chopper.py
==============
Chop a collection of ultrasound bat-detector ``.wav`` files into fixed-duration
segments ("chunks") suitable for downstream analysis by SonoBat or similar tools.

Each chunk is a self-contained ``.wav`` file whose filename encodes:

* the **file_id** — the stem of the original recording (globally unique by
  detector timestamp), and
* the **TimeInFile** offset — the chunk's start position within the original
  recording in milliseconds.

Chunk filename format::

    {file_id}_t{offset_ms:07d}ms.wav

    Example:
        barn1_D20210819T024235m074_t0000000ms.wav
        barn1_D20210819T024235m074_t0002000ms.wav
        barn1_D20210819T024235m074_t0004000ms.wav

The original absolute path is also embedded in the ``.wav`` file's RIFF INFO
chunk (``INAM`` subchunk) as a belt-and-suspenders backup, but the filename
alone is sufficient to reconstruct all provenance:

    file_id  = Path(chunk).stem.rsplit('_t', 1)[0]
    time_ms  = int(Path(chunk).stem.rsplit('_t', 1)[1].rstrip('ms'))
    orig_wav = next(Path(root).rglob(file_id + '.wav'))   # 'find' equivalent

Parallelism
-----------
:class:`WavChopper` uses :class:`concurrent.futures.ProcessPoolExecutor` with
the same ``n_workers`` convention as :class:`WavScrubber`: default reserves 4
cores for the OS; pass ``n_workers=0`` to use every available core.

Checkpointing
-------------
Results are appended to a checkpoint CSV row-by-row as each source file
completes.  A default timestamped checkpoint is created in ``cwd()`` if no
explicit path is given, so an interrupted run can always be resumed.

Typical usage
-------------
::

    from wav_scrubber import WavScrubber
    from wav_chopper  import WavChopper

    retained = WavScrubber(wav_paths).run().retained

    chopper = WavChopper(retained, out_dir='/data/chunks')
    result  = chopper.run()

    print(result.summary())
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import logging
import os
import shlex
import struct
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import scipy.io.wavfile as wavfile

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

try:
    from enum import StrEnum          # Python 3.11+
except ImportError:
    from strenum import StrEnum       # pip install strenum

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time-expansion detection — mirrors wav_scrubber constants
# ---------------------------------------------------------------------------

#: Header sample rates below this value indicate a time-expanded file.
_TE_SR_THRESHOLD_HZ: int = 80_000

#: Factor applied to a TE file's header rate to recover the true ultrasound SR.
_TIME_EXPAND_FACTOR: int = 10


# ---------------------------------------------------------------------------
# StrEnum: per-file chop outcome
# ---------------------------------------------------------------------------

class ChopReason(StrEnum):
    """Outcome of attempting to chop a single source file."""
    CHOPPED     = 'chopped'      #: File was chopped into one or more chunks.
    UNREADABLE  = 'unreadable'   #: File could not be read as a WAV.
    EMPTY       = 'empty'        #: File had zero usable samples after loading.
    WRITE_ERROR = 'write_error'  #: Chunk could not be written to the output dir.


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ChopInterrupted(Exception):
    """Raised by :meth:`WavChopper.run` when the user presses Ctrl-C.

    :param partial_result: :class:`ChopResult` for all files that finished.
    :param checkpoint_csv: Path to the checkpoint CSV, or ``None``.
    """

    def __init__(self, partial_result: ChopResult, checkpoint_csv: Optional[Path]) -> None:
        """
        :param partial_result: :class:`ChopResult` with completed records.
        :param checkpoint_csv: Checkpoint path or ``None``.
        """
        self.partial_result = partial_result
        self.checkpoint_csv = checkpoint_csv
        super().__init__("Chop run interrupted by user")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ChopRecord:
    """
    Chop outcome for a single source ``.wav`` file.

    :param source_path:  Absolute path to the original file.
    :param reason:       :attr:`ChopReason.CHOPPED` or an error reason.
    :param n_chunks:     Number of chunks written (0 on error).
    :param chunk_paths:  Absolute paths of all written chunk files.
    :param sample_rate:  True sample rate after TE correction (Hz), or ``None``.
    :param duration_s:   True duration of the source file (s), or ``None``.
    :param detail:       Free-form detail string (e.g. exception message).
    """
    source_path:  Path
    reason:       ChopReason
    n_chunks:     int               = 0
    chunk_paths:  List[Path]        = dataclasses.field(default_factory=list)
    sample_rate:  Optional[int]     = None
    duration_s:   Optional[float]   = None
    detail:       str               = ""

    @property
    def ok(self) -> bool:
        """:return: ``True`` if the file was successfully chopped."""
        return self.reason == ChopReason.CHOPPED


@dataclasses.dataclass
class ChopResult:
    """
    Aggregate result of a :meth:`WavChopper.run` call.

    :param records:    One :class:`ChopRecord` per source file, in input order.
    :param out_dir:    Directory where chunks were written.
    :param chunk_dur:  Chunk duration used (seconds).
    """
    records:    List[ChopRecord]
    out_dir:    Path
    chunk_dur:  float

    # ------------------------------------------------------------------ #
    #  Convenience views                                                   #
    # ------------------------------------------------------------------ #

    @property
    def chopped(self) -> List[ChopRecord]:
        """:return: Records for files that were successfully chopped."""
        return [r for r in self.records if r.ok]

    @property
    def failed(self) -> List[ChopRecord]:
        """:return: Records for files that could not be chopped."""
        return [r for r in self.records if not r.ok]

    @property
    def all_chunks(self) -> List[Path]:
        """:return: Flat list of every chunk path produced."""
        out: List[Path] = []
        for rec in self.chopped:
            out.extend(rec.chunk_paths)
        return out

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """
        Return a human-readable summary string.

        :return: Multi-line summary of chop outcomes.
        """
        n_src    = len(self.records)
        n_ok     = len(self.chopped)
        n_fail   = len(self.failed)
        n_chunks = sum(r.n_chunks for r in self.records)
        lines = [
            f"WavChopper results  (chunk duration: {self.chunk_dur:.1f} s)",
            f"  Source files  : {n_src}",
            f"  Successfully chopped : {n_ok}",
            f"  Failed               : {n_fail}",
            f"  Total chunks written : {n_chunks}",
            f"  Output directory     : {self.out_dir}",
        ]
        if n_fail:
            lines.append(f"\nFailed files:")
            for rec in self.failed:
                lines.append(f"  [{rec.reason}]  {rec.source_path}  — {rec.detail}")
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> Path:
        """
        Write a per-chunk CSV report (one row per chunk, not per source file).

        Columns: ``source_path``, ``chunk_path``, ``file_id``, ``offset_ms``,
        ``sample_rate``, ``source_duration_s``, ``reason``, ``detail``.

        :param path: Destination CSV path.
        :return:     Resolved path of the written file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow([
                'source_path', 'chunk_path', 'file_id',
                'offset_ms', 'sample_rate', 'source_duration_s',
                'reason', 'detail',
            ])
            for rec in self.records:
                if rec.ok:
                    for chunk in rec.chunk_paths:
                        file_id, offset_ms = _parse_chunk_stem(chunk.stem)
                        w.writerow([
                            str(rec.source_path), str(chunk),
                            file_id, offset_ms,
                            rec.sample_rate, f'{rec.duration_s:.3f}',
                            str(rec.reason), rec.detail,
                        ])
                else:
                    w.writerow([
                        str(rec.source_path), '', '', '',
                        rec.sample_rate or '',
                        f'{rec.duration_s:.3f}' if rec.duration_s else '',
                        str(rec.reason), rec.detail,
                    ])
        return p.resolve()


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _chunk_stem(file_id: str, offset_ms: int) -> str:
    """
    Build the chunk filename stem (without extension).

    :param file_id:   Stem of the original source file.
    :param offset_ms: Start offset of this chunk in the source file (ms).
    :return:          Stem string, e.g. ``barn1_D20210819T024235m074_t0002000ms``.
    """
    return f"{file_id}_t{offset_ms:07d}ms"


def _parse_chunk_stem(stem: str) -> tuple[str, int]:
    """
    Recover ``(file_id, offset_ms)`` from a chunk filename stem.

    :param stem: Chunk filename stem, e.g. ``barn1_D20210819T024235m074_t0002000ms``.
    :return:     ``(file_id, offset_ms)`` tuple.
    :raises ValueError: If the stem does not match the expected format.
    """
    try:
        file_id, tail = stem.rsplit('_t', 1)
        offset_ms = int(tail.rstrip('ms'))
        return file_id, offset_ms
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Cannot parse chunk stem '{stem}': expected "
            f"'<file_id>_t<offset>ms'"
        ) from exc


# ---------------------------------------------------------------------------
# RIFF INFO chunk writer
# ---------------------------------------------------------------------------

def _pack_riff_info(fields: dict[str, str]) -> bytes:
    """
    Pack a RIFF LIST/INFO chunk containing the given key→value pairs.

    Each field is a 4-character FourCC (e.g. ``'INAM'``) mapped to a string
    value.  Values are null-terminated and padded to an even byte count.

    :param fields: Mapping of FourCC → string value.
    :return:       Raw bytes for the complete LIST/INFO chunk.
    """
    inner = b''
    for fourcc, value in fields.items():
        assert len(fourcc) == 4, f"FourCC must be exactly 4 characters: {fourcc!r}"
        data = value.encode('utf-8', errors='replace') + b'\x00'
        if len(data) % 2:
            data += b'\x00'          # pad to even
        inner += fourcc.encode('ascii') + struct.pack('<I', len(data)) + data

    # LIST chunk: 'LIST' + size + 'INFO' + inner subchunks
    list_data = b'INFO' + inner
    return b'LIST' + struct.pack('<I', len(list_data)) + list_data


def _write_wav_with_info(
    path: Path,
    audio: np.ndarray,
    sr: int,
    info: dict[str, str],
) -> None:
    """
    Write a ``.wav`` file and append a RIFF LIST/INFO metadata chunk.

    :class:`scipy.io.wavfile.write` produces a minimal RIFF file; we append
    the INFO chunk directly to the file bytes after writing.

    :param path:  Destination path.
    :param audio: Sample data (int16 or float32 array).
    :param sr:    Sample rate to write in the WAV header.
    :param info:  FourCC → string metadata to embed in the INFO chunk.
    """
    # Write standard WAV first.
    wavfile.write(str(path), sr, audio)

    # Read back the raw bytes and patch the RIFF size.
    raw = bytearray(path.read_bytes())

    info_chunk = _pack_riff_info(info)

    # Append INFO chunk and update the top-level RIFF size field (bytes 4–7).
    raw += info_chunk
    riff_size = len(raw) - 8          # RIFF header is 8 bytes
    struct.pack_into('<I', raw, 4, riff_size)

    path.write_bytes(bytes(raw))


# ---------------------------------------------------------------------------
# Per-file chop logic (module-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _chop_one(
    source_path:    Path,
    out_dir:        Path,
    chunk_dur_s:    float,
    te_threshold:   int,
    te_factor:      int,
) -> ChopRecord:
    """
    Chop a single ``.wav`` file into fixed-duration chunks and write them to
    *out_dir*.

    Module-level so that :class:`ProcessPoolExecutor` can pickle it for
    dispatch to worker processes.

    :param source_path:  Path to the source ``.wav`` file.
    :param out_dir:      Directory to write chunk files into.
    :param chunk_dur_s:  Desired chunk duration in seconds.
    :param te_threshold: Sample-rate boundary for TE detection (Hz).
    :param te_factor:    Expansion factor for time-expanded files.
    :return:             :class:`ChopRecord` with outcome and chunk paths.
    """
    rec = ChopRecord(source_path=source_path.resolve(), reason=ChopReason.CHOPPED)

    # ── 1. Load ────────────────────────────────────────────────────────────
    try:
        sr_header, data = wavfile.read(str(source_path))
    except Exception as exc:
        rec.reason = ChopReason.UNREADABLE
        rec.detail = str(exc)
        return rec

    # Flatten to mono by averaging channels.
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)

    if len(data) == 0:
        rec.reason = ChopReason.EMPTY
        rec.detail = "zero samples after loading"
        return rec

    # ── 2. TE correction (mirrors wav_scrubber logic) ──────────────────────
    sr = sr_header * te_factor if sr_header < te_threshold else sr_header
    rec.sample_rate = sr
    rec.duration_s  = len(data) / sr

    # ── 3. Compute chunk boundaries in samples ─────────────────────────────
    chunk_samples = int(round(chunk_dur_s * sr))
    if chunk_samples <= 0:
        rec.reason = ChopReason.UNREADABLE
        rec.detail = f"chunk_dur_s={chunk_dur_s} produced zero samples at {sr} Hz"
        return rec

    file_id    = source_path.stem
    orig_path  = str(source_path.resolve())
    chunks_written: List[Path] = []

    n_samples = len(data)
    offset    = 0

    while offset < n_samples:
        end        = min(offset + chunk_samples, n_samples)
        chunk_data = data[offset:end]

        offset_ms = round(offset / sr * 1000)
        out_name  = _chunk_stem(file_id, offset_ms) + '.wav'
        out_path  = out_dir / out_name

        info = {
            'INAM': orig_path,              # original absolute path
            'IPRD': file_id,                # file_id for quick lookup
            'ICMT': f"offset_ms={offset_ms}",  # TimeInFile in ms
        }

        try:
            # Write at the *header* sample rate so the WAV is playable at the
            # original speed — but store the true SR in the INFO chunk so
            # downstream tools can correct if needed.
            # For direct recordings sr == sr_header; for TE files writing at
            # sr_header preserves the original playback rate of the recording.
            _write_wav_with_info(out_path, chunk_data, sr_header, info)
        except Exception as exc:
            rec.reason = ChopReason.WRITE_ERROR
            rec.detail = f"chunk {out_name}: {exc}"
            # Return partial progress: record chunks already written.
            rec.n_chunks    = len(chunks_written)
            rec.chunk_paths = chunks_written
            return rec

        chunks_written.append(out_path.resolve())
        offset += chunk_samples

    rec.n_chunks    = len(chunks_written)
    rec.chunk_paths = chunks_written
    return rec


# ---------------------------------------------------------------------------
# WavChopper
# ---------------------------------------------------------------------------

class WavChopper:
    """
    Chop a list of ``.wav`` files into fixed-duration segments in parallel.

    Each chunk's filename encodes the source file stem (``file_id``) and the
    start offset within the original recording::

        {file_id}_t{offset_ms:07d}ms.wav

    Provenance is also embedded in a RIFF LIST/INFO chunk inside the ``.wav``
    file itself (``INAM`` = original absolute path, ``IPRD`` = file_id,
    ``ICMT`` = ``offset_ms=…``).

    :param wav_paths:      Paths to source ``.wav`` files to chop.
    :param out_dir:        Directory where chunk files are written.
    :param chunk_dur_s:    Duration of each chunk in seconds.
    :param n_workers:      Worker processes.  ``None`` (default) reserves 4
                           cores for the OS; pass ``0`` to use every core.
    :param show_progress:  Show a tqdm progress bar.
    :param checkpoint_csv: Path to a checkpoint CSV for incremental progress.
                           If ``None``, a timestamped default is created in
                           the current working directory.
    :param worker_timeout: Seconds before a stalled worker is abandoned.
    """

    # ------------------------------------------------------------------ #
    #  Class-level tunables                                               #
    # ------------------------------------------------------------------ #

    #: Default chunk duration in seconds.
    _CHUNK_DURATION_S: float = 2.0

    #: Header SR boundary below which a file is treated as time-expanded.
    _TE_SR_THRESHOLD_HZ: int = _TE_SR_THRESHOLD_HZ

    #: Expansion factor for time-expanded files.
    _TIME_EXPAND_FACTOR: int = _TIME_EXPAND_FACTOR

    def __init__(
        self,
        wav_paths:       Sequence[str | Path],
        out_dir:         str | Path,
        chunk_dur_s:     float           = _CHUNK_DURATION_S,
        n_workers:       Optional[int]   = None,
        show_progress:   bool            = True,
        checkpoint_csv:  Optional[str | Path] = None,
        worker_timeout:  Optional[float] = 120.0,
    ) -> None:
        """
        :param wav_paths:      Source ``.wav`` paths to chop.
        :param out_dir:        Destination directory for chunk files.
        :param chunk_dur_s:    Chunk duration in seconds (default 2.0).
        :param n_workers:      Worker processes (``None`` = cpu_count − 4;
                               ``0`` = all cores).
        :param show_progress:  Show tqdm progress bar.
        :param checkpoint_csv: Checkpoint CSV path.  Auto-generated in cwd()
                               if ``None``.
        :param worker_timeout: Per-file worker timeout in seconds.
        """
        self.wav_paths    = [Path(p) for p in wav_paths]
        self.out_dir      = Path(out_dir)
        self.chunk_dur_s  = chunk_dur_s

        if n_workers is None:
            self.n_workers = max(1, (os.cpu_count() or 1) - 4)
        elif n_workers == 0:
            self.n_workers = os.cpu_count() or 1
        else:
            self.n_workers = n_workers

        self.show_progress  = show_progress
        self.worker_timeout = worker_timeout

        if checkpoint_csv is not None:
            self.checkpoint_csv = Path(checkpoint_csv)
        else:
            ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            self.checkpoint_csv = Path.cwd() / f'chop_checkpoint_{ts}.csv'

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> ChopResult:
        """
        Chop all source files in parallel and return a :class:`ChopResult`.

        Checkpointing
        -------------
        Source files already present in the checkpoint CSV are skipped; their
        records are loaded back into the result so the final :class:`ChopResult`
        is complete.  New records are appended row-by-row as workers complete.

        :return: :class:`ChopResult` with one :class:`ChopRecord` per source.
        :raises ChopInterrupted: If the user presses Ctrl-C.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── load checkpoint ────────────────────────────────────────────────
        completed: dict[Path, ChopRecord] = {}
        self.checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.checkpoint_csv.exists()
            or self.checkpoint_csv.stat().st_size == 0
        )
        checkpoint_fh = self.checkpoint_csv.open('a', newline='')
        ckpt_fields   = [f.name for f in dataclasses.fields(ChopRecord)
                         if f.name != 'chunk_paths']
        ckpt_fields.append('chunk_paths_joined')
        ckpt_writer = csv.DictWriter(checkpoint_fh, fieldnames=ckpt_fields)
        if write_header:
            ckpt_writer.writeheader()
            checkpoint_fh.flush()
        else:
            completed = self._load_checkpoint(self.checkpoint_csv)
            if completed:
                log.info('WavChopper: resuming — %d files already in checkpoint',
                         len(completed))

        # ── partition todo vs done ─────────────────────────────────────────
        todo  = [p for p in self.wav_paths if p.resolve() not in completed]
        total = len(self.wav_paths)
        skip  = total - len(todo)
        if skip:
            log.info('WavChopper: skipping %d already-completed files', skip)

        # ── progress bar ───────────────────────────────────────────────────
        if self.show_progress and _TQDM_AVAILABLE:
            pbar = _tqdm(total=total, unit='file', desc='Chopping')
            pbar.update(skip)
        else:
            pbar = None

        records_map: dict[Path, ChopRecord] = dict(completed)

        # ── bounded submission window ──────────────────────────────────────
        window     = self.n_workers * 4
        future_map = {}   # future → source_path

        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                it       = iter(todo)
                pending  = 0

                # Seed the initial window.
                for src in it:
                    fut = pool.submit(
                        _chop_one,
                        src,
                        self.out_dir,
                        self.chunk_dur_s,
                        self._TE_SR_THRESHOLD_HZ,
                        self._TIME_EXPAND_FACTOR,
                    )
                    future_map[fut] = src
                    pending += 1
                    if pending >= window:
                        break

                while future_map:
                    from concurrent.futures import FIRST_COMPLETED, wait as _wait
                    done_set, _ = _wait(
                        future_map,
                        timeout=self.worker_timeout,
                        return_when=FIRST_COMPLETED,
                    )

                    if not done_set:
                        # Timeout: mark every still-pending future as UNREADABLE.
                        for fut, src in list(future_map.items()):
                            if not fut.done():
                                fut.cancel()
                                rec = ChopRecord(
                                    source_path=src.resolve(),
                                    reason=ChopReason.UNREADABLE,
                                    detail=f'worker timeout after {self.worker_timeout}s',
                                )
                                records_map[src.resolve()] = rec
                                self._write_ckpt_row(ckpt_writer, checkpoint_fh, rec)
                                if pbar:
                                    pbar.update(1)
                        future_map.clear()
                        break

                    for fut in done_set:
                        src = future_map.pop(fut)
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = ChopRecord(
                                source_path=src.resolve(),
                                reason=ChopReason.UNREADABLE,
                                detail=str(exc),
                            )
                        records_map[src.resolve()] = rec
                        self._write_ckpt_row(ckpt_writer, checkpoint_fh, rec)
                        if pbar:
                            pbar.update(1)

                        # Refill window.
                        for next_src in it:
                            fut2 = pool.submit(
                                _chop_one,
                                next_src,
                                self.out_dir,
                                self.chunk_dur_s,
                                self._TE_SR_THRESHOLD_HZ,
                                self._TIME_EXPAND_FACTOR,
                            )
                            future_map[fut2] = next_src
                            break

        except KeyboardInterrupt:
            if pbar:
                pbar.close()
            checkpoint_fh.close()
            ordered = [records_map.get(p.resolve()) for p in self.wav_paths
                       if p.resolve() in records_map]
            partial = ChopResult(
                records=ordered,
                out_dir=self.out_dir,
                chunk_dur=self.chunk_dur_s,
            )
            raise ChopInterrupted(partial, self.checkpoint_csv)

        if pbar:
            pbar.close()
        checkpoint_fh.close()

        ordered = [records_map[p.resolve()] for p in self.wav_paths
                   if p.resolve() in records_map]
        return ChopResult(
            records=ordered,
            out_dir=self.out_dir,
            chunk_dur=self.chunk_dur_s,
        )

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_ckpt_row(
        writer: csv.DictWriter,
        fh,
        rec: ChopRecord,
    ) -> None:
        """
        Append a single :class:`ChopRecord` to the checkpoint CSV.

        :param writer: DictWriter targeting the checkpoint file.
        :param fh:     Open file handle (flushed after write).
        :param rec:    Record to append.
        """
        row = {f.name: getattr(rec, f.name)
               for f in dataclasses.fields(rec)
               if f.name != 'chunk_paths'}
        row['source_path']       = str(rec.source_path)
        row['chunk_paths_joined'] = '|'.join(str(p) for p in rec.chunk_paths)
        writer.writerow(row)
        fh.flush()

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[Path, ChopRecord]:
        """
        Read an existing checkpoint CSV and return a ``{source_path: ChopRecord}``
        mapping for all completed files.

        :param path: Path to the checkpoint CSV.
        :return:     Mapping of resolved source path → :class:`ChopRecord`.
        """
        completed: dict[Path, ChopRecord] = {}
        if not path.exists():
            return completed
        try:
            with path.open(newline='') as fh:
                for row in csv.DictReader(fh):
                    chunk_paths = [
                        Path(s) for s in row.get('chunk_paths_joined', '').split('|')
                        if s
                    ]
                    rec = ChopRecord(
                        source_path = Path(row['source_path']),
                        reason      = ChopReason(row['reason']),
                        n_chunks    = int(row.get('n_chunks', 0)),
                        chunk_paths = chunk_paths,
                        sample_rate = int(row['sample_rate']) if row.get('sample_rate') else None,
                        duration_s  = float(row['duration_s']) if row.get('duration_s') else None,
                        detail      = row.get('detail', ''),
                    )
                    completed[rec.source_path.resolve()] = rec
        except Exception as exc:
            warnings.warn(f'WavChopper: could not read checkpoint {path}: {exc}')
        return completed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments for the WavChopper CLI.

    :return: ``(args, paths)`` tuple.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='wav_chopper',
        description=(
            'Chop bat-detector .wav files into fixed-duration segments.\n\n'
            'Chunk filenames encode the source file ID and time offset:\n'
            '  {file_id}_t{offset_ms:07d}ms.wav'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'input',
        nargs='+',
        help=(
            'one or more .wav files, shell globs (e.g. wav_dir/*.wav), '
            'or directories.\nDirectories are searched at the top level only; '
            'use -r/--recursive to descend into subdirectories.'
        ),
    )
    parser.add_argument(
        '-o', '--out-dir',
        required=True,
        help='directory where chunk .wav files are written',
    )
    parser.add_argument(
        '-d', '--chunk-dur',
        type=float,
        default=WavChopper._CHUNK_DURATION_S,
        metavar='SECS',
        help=f'chunk duration in seconds (default: {WavChopper._CHUNK_DURATION_S})',
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='descend into subdirectories when a directory is given (default: top level only)',
    )
    parser.add_argument(
        '-w', '--workers',
        type=int, default=None,
        help=(
            'number of worker processes.  Default: cpu_count − 4.\n'
            'Pass 0 to use every available core.'
        ),
    )
    parser.add_argument(
        '-s', '--summary',
        default=None,
        dest='out_csv',
        help='write per-chunk report to this CSV path',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        metavar='CSV',
        help=(
            'checkpoint CSV for incremental progress.  '
            'Auto-generated in cwd() if omitted.\n'
            'Re-run with the same arguments to resume after interruption.'
        ),
    )
    parser.add_argument(
        '--timeout',
        type=float, default=120.0,
        metavar='SECS',
        help='seconds before a stalled worker is abandoned (default: 120)',
    )
    args = parser.parse_args()

    # Collect .wav paths — no recursion by default; -r/--recursive to descend.
    recurse = args.recursive
    seen:  set[Path]  = set()
    paths: list[Path] = []
    for item in args.input:
        p = Path(item)
        if p.is_dir():
            glob_fn  = p.rglob if recurse else p.glob
            for w in sorted(glob_fn('*.wav')):
                if w not in seen:
                    seen.add(w)
                    paths.append(w)
        elif p.suffix.lower() == '.wav':
            if p not in seen:
                seen.add(p)
                paths.append(p)
        else:
            print(f"Warning: skipping non-WAV input '{item}'", file=sys.stderr)

    if not paths:
        parser.error('No .wav files found in the given inputs.')

    if args.chunk_dur <= 0:
        parser.error(f'--chunk-dur must be positive, got {args.chunk_dur}')

    return args, paths


def main() -> None:
    """
    CLI entry point: chop .wav files and optionally write a report.
    """
    args, paths = _parse_args()

    print(f'WavChopper: {len(paths)} source files  →  '
          f'{args.chunk_dur:.1f}s chunks  →  {args.out_dir}')

    if args.checkpoint is None:
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        auto_ckpt = Path.cwd() / f'chop_checkpoint_{ts}.csv'
        print(f'  Checkpoint : {auto_ckpt}  '
              f'(auto-generated; pass --checkpoint <path> to choose your own)')

    chopper = WavChopper(
        wav_paths      = paths,
        out_dir        = args.out_dir,
        chunk_dur_s    = args.chunk_dur,
        n_workers      = args.workers,
        show_progress  = True,
        checkpoint_csv = args.checkpoint,
        worker_timeout = args.timeout,
    )

    try:
        result = chopper.run()
    except ChopInterrupted as exc:
        partial = exc.partial_result
        ckpt    = exc.checkpoint_csv
        n_done  = len(partial.records)
        n_total = len(paths)
        print(
            f'\n*** Interrupted after {n_done}/{n_total} files '
            f'({100*n_done/max(n_total,1):.0f}%) ***',
            file=sys.stderr,
        )
        if ckpt:
            resume_argv = list(sys.argv)
            if '--checkpoint' not in resume_argv:
                resume_argv.extend(['--checkpoint', str(ckpt)])
            print(f'\nCheckpoint saved: {ckpt}', file=sys.stderr)
            print(f'  {n_done} source files safe.  '
                  f'{n_total - n_done} remain.', file=sys.stderr)
            print('\nTo resume, re-run the same command:', file=sys.stderr)
            print(f'  {shlex.join(resume_argv)}', file=sys.stderr)
        sys.exit(2)

    print(result.summary())

    if args.out_csv:
        p = result.to_csv(args.out_csv)
        print(f'Chunk report written to {p}')

    sys.exit(0 if result.chopped else 1)


if __name__ == '__main__':
    main()