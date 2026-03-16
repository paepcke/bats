# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-16 09:51:39
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/wav_file_info.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-16 10:03:51
#
# **********************************************************

"""
Determine the true acoustic properties of a bat detector ``.wav`` file,
handling time-expansion (TE), direct-recording (DR), and unknown formats
gracefully.

The authoritative source is the **GUANO metadata** embedded in the file when
present.  When GUANO is absent or the ``guano`` library is not installed, a
heuristic based on the WAV header sample rate is applied.

Recording types
---------------
``DR``  Direct recording.
    The file sample rate equals the true ultrasound sample rate.
    ``sr_true == sr_header``, ``te == 1``.
    Typical: ``sr_header >= 80,000 Hz``.

``TE``  Time-expanded recording.
    The detector records at true ultrasound rate but stores audio slowed
    down by factor ``te`` (commonly 10).  The file sample rate is
    ``sr_header = sr_true / te`` (e.g. 25,000 Hz for a 250 kHz detector
    with te=10).  True duration = ``file_frames / sr_header / te``.

``FD``  Frequency-division recording.
    The detector mixes the ultrasound signal down to audio frequencies by
    dividing by a fixed factor.  The file sample rate is at audio rates
    but the signal is NOT slowed — it is frequency-shifted.  Full
    reconstruction requires knowing the division factor.  This type is
    detected and flagged but **not** corrected automatically; callers
    should log a warning and skip such files.

``UNKNOWN``
    Cannot determine recording type from available metadata.

GUANO priority
--------------
When the ``guano`` library is available and the file contains a GUANO chunk:

*  ``TE`` field (integer) gives the expansion factor directly.
*  ``Samplerate`` field gives the true sample rate.
*  ``Length`` field gives the true duration in seconds.

These fields are written by SonoBat when it annotates a recording, and are
the most reliable source of truth.

Heuristic fallback
------------------
When GUANO is absent or ``guano`` is not installed:

*  ``sr_header >= 80,000 Hz``  → assumed DR  (te=1)
*  ``sr_header <  80,000 Hz``  → assumed TE  (te=10, sr_true=sr_header×10)

The heuristic te=10 covers the vast majority of field deployments using
Wildlife Acoustics or Pettersson detectors in TE mode.  If a site uses a
different TE factor and no GUANO header is present, ``WavInfo`` will be
wrong; the caller can override via ``WavInfo.from_path(..., te_override=N)``.

Usage
-----
::

    from wav_file_info import WavInfo, RecordingType

    info = WavInfo.from_path('/path/to/recording.wav')
    print(info)
    # WavInfo(path=..., type=TE, sr_header=25000, te=10,
    #         sr_true=250000, true_duration_s=5.0, source=guano)

    # Seek to true-time offset t_ms in the file:
    file_frame = info.true_ms_to_file_frame(t_ms)

    # Check whether the file can contain ultrasonic bat calls:
    if not info.is_ultrasonic:
        log.warn(f'Skipping non-ultrasonic file: {info.path}')
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional

import soundfile as sf

import guano as _guano
_GUANO_AVAILABLE = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sample rate threshold below which a file is assumed to be TE (heuristic).
_TE_SR_THRESHOLD_HZ: int = 80_000

#: Default TE factor assumed when GUANO is absent and sr < threshold.
_DEFAULT_TE_FACTOR: int = 10

#: Minimum true sample rate (Hz) to be considered ultrasonic.
_MIN_ULTRASONIC_SR_HZ: int = 80_000


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RecordingType(StrEnum):
    """Recording storage format."""
    DR      = 'DR'       #: Direct recording — file SR == true SR
    TE      = 'TE'       #: Time-expanded — file SR < true SR
    FD      = 'FD'       #: Frequency-division — not correctable automatically
    UNKNOWN = 'UNKNOWN'  #: Cannot determine from available metadata


class WavInfoSource(StrEnum):
    """How ``WavInfo`` was determined."""
    GUANO     = 'guano'      #: Read from embedded GUANO metadata
    HEURISTIC = 'heuristic'  #: Inferred from WAV header sample rate
    OVERRIDE  = 'override'   #: Caller-supplied ``te_override``


# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------

@dataclass
class WavInfo:
    """
    True acoustic properties of a bat detector ``.wav`` file.

    Do not construct directly — use :meth:`from_path`.

    :param path:            Absolute path to the ``.wav`` file.
    :param rec_type:        Recording storage type.
    :param sr_header:       Sample rate in the WAV header (Hz).
    :param te:              Time-expansion factor (1 for DR/FD).
    :param sr_true:         True ultrasound sample rate (Hz).
    :param true_duration_s: True acoustic duration (seconds).
    :param file_frames:     Total frames in the file.
    :param source:          How the type was determined.
    :param note:            Free-form note from GUANO or heuristic reasoning.
    """
    path:            Path
    rec_type:        RecordingType
    sr_header:       int
    te:              int
    sr_true:         int
    true_duration_s: float
    file_frames:     int
    source:          WavInfoSource
    note:            str = ''

    # ------------------------------------------------------------------ #
    #  Factory                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_path(
        cls,
        path:        str | Path,
        te_override: Optional[int] = None,
    ) -> 'WavInfo':
        """
        Construct a :class:`WavInfo` by reading a ``.wav`` file's headers.

        Resolution order:

        1. ``te_override`` — caller-supplied factor, highest priority.
        2. GUANO ``TE`` field (requires ``guano`` library).
        3. Heuristic: ``sr_header < 80,000 Hz`` → TE×10.

        :param path:        Path to the ``.wav`` file.
        :param te_override: If provided, treat this as the TE factor
                            regardless of header content.
        :return:            Populated :class:`WavInfo`.
        :raises FileNotFoundError: If the file does not exist.
        :raises RuntimeError:      If the file cannot be opened by soundfile.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f'WAV file not found: {path}')

        # ── Read WAV header ───────────────────────────────────────────
        try:
            with sf.SoundFile(str(path)) as f:
                sr_header   = f.samplerate
                file_frames = len(f)
        except Exception as exc:
            raise RuntimeError(f'Cannot open {path}: {exc}') from exc

        # ── Caller override ───────────────────────────────────────────
        if te_override is not None:
            te         = te_override
            sr_true    = sr_header * te
            true_dur   = file_frames / sr_header / te
            rec_type   = RecordingType.DR if te == 1 else RecordingType.TE
            return cls(
                path            = path,
                rec_type        = rec_type,
                sr_header       = sr_header,
                te              = te,
                sr_true         = sr_true,
                true_duration_s = true_dur,
                file_frames     = file_frames,
                source          = WavInfoSource.OVERRIDE,
                note            = f'te_override={te_override}',
            )

        # ── Try GUANO ─────────────────────────────────────────────────
        guano_result = cls._from_guano(path, sr_header, file_frames)
        if guano_result is not None:
            return guano_result

        # ── Heuristic fallback ────────────────────────────────────────
        return cls._from_heuristic(path, sr_header, file_frames)

    # ------------------------------------------------------------------ #
    #  GUANO reader                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def _from_guano(
        cls,
        path:        Path,
        sr_header:   int,
        file_frames: int,
    ) -> Optional['WavInfo']:
        """
        Attempt to read recording properties from embedded GUANO metadata.

        :param path:        File path.
        :param sr_header:   WAV header sample rate.
        :param file_frames: Total WAV frames.
        :return:            :class:`WavInfo` or ``None`` if no GUANO found.
        """
        try:
            gf = _guano.GuanoFile(str(path))
        except Exception:
            return None

        # TE field: integer expansion factor.
        te_raw = gf.get('TE') or gf.get('', {}).get('TE')
        if te_raw is None:
            # Also check nested namespace (different GUANO writers vary).
            for ns in gf._md.values():
                if hasattr(ns, 'get') and ns.get('TE') is not None:
                    te_raw = ns.get('TE')
                    break

        if te_raw is None:
            return None

        try:
            te = int(te_raw)
        except (TypeError, ValueError):
            return None

        # True sample rate from GUANO 'Samplerate' if present, else infer.
        sr_true_raw = gf.get('Samplerate') or gf.get('', {}).get('Samplerate')
        if sr_true_raw is not None:
            try:
                sr_true = int(sr_true_raw)
            except (TypeError, ValueError):
                sr_true = sr_header * te
        else:
            sr_true = sr_header * te

        # True duration from GUANO 'Length' if present, else compute.
        length_raw = gf.get('Length') or gf.get('', {}).get('Length')
        if length_raw is not None:
            try:
                true_dur = float(length_raw)
            except (TypeError, ValueError):
                true_dur = file_frames / sr_header / te
        else:
            true_dur = file_frames / sr_header / te

        rec_type = RecordingType.DR if te == 1 else RecordingType.TE

        return cls(
            path            = path,
            rec_type        = rec_type,
            sr_header       = sr_header,
            te              = te,
            sr_true         = sr_true,
            true_duration_s = true_dur,
            file_frames     = file_frames,
            source          = WavInfoSource.GUANO,
            note            = f'GUANO TE={te}, Samplerate={sr_true}, Length={true_dur:.3f}s',
        )

    # ------------------------------------------------------------------ #
    #  Heuristic fallback                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def _from_heuristic(
        cls,
        path:        Path,
        sr_header:   int,
        file_frames: int,
    ) -> 'WavInfo':
        """
        Infer recording type from WAV header sample rate alone.

        :param path:        File path.
        :param sr_header:   WAV header sample rate.
        :param file_frames: Total WAV frames.
        :return:            :class:`WavInfo` with ``source=HEURISTIC``.
        """
        if sr_header >= _TE_SR_THRESHOLD_HZ:
            te       = 1
            sr_true  = sr_header
            true_dur = file_frames / sr_header
            rec_type = RecordingType.DR
            note     = f'sr_header={sr_header} >= {_TE_SR_THRESHOLD_HZ} → DR'
        else:
            te       = _DEFAULT_TE_FACTOR
            sr_true  = sr_header * te
            true_dur = file_frames / sr_header / te
            rec_type = RecordingType.TE
            note     = (
                f'sr_header={sr_header} < {_TE_SR_THRESHOLD_HZ} → '
                f'assumed TE×{te}; true SR={sr_true} Hz'
            )

        return cls(
            path            = path,
            rec_type        = rec_type,
            sr_header       = sr_header,
            te              = te,
            sr_true         = sr_true,
            true_duration_s = true_dur,
            file_frames     = file_frames,
            source          = WavInfoSource.HEURISTIC,
            note            = note,
        )

    # ------------------------------------------------------------------ #
    #  Derived properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def is_ultrasonic(self) -> bool:
        """
        Return ``True`` if the true sample rate is high enough to contain
        bat echolocation calls (>= 80 kHz).

        :return: ``True`` for DR and TE recordings; ``False`` for FD and
                 recordings with very low true SR.
        """
        return self.sr_true >= _MIN_ULTRASONIC_SR_HZ

    @property
    def nyquist_hz(self) -> float:
        """
        True Nyquist frequency in Hz.

        :return: ``sr_true / 2``.
        """
        return self.sr_true / 2.0

    # ------------------------------------------------------------------ #
    #  Seek helpers                                                       #
    # ------------------------------------------------------------------ #

    def true_ms_to_file_frame(self, true_ms: float) -> int:
        """
        Convert a true-time offset (ms from recording start) to the
        corresponding frame index in the ``.wav`` file.

        For TE recordings: ``file_frame = true_ms / 1000 / te * sr_header``

        :param true_ms: True-time offset in milliseconds.
        :return:        File frame index (clamped to [0, file_frames - 1]).
        """
        file_s = true_ms / 1000.0 / self.te
        frame  = int(file_s * self.sr_header)
        return max(0, min(frame, self.file_frames - 1))

    def true_ms_to_file_frames_count(self, duration_ms: float) -> int:
        """
        Convert a true-time duration (ms) to the equivalent number of
        file-domain frames.

        For TE recordings: ``n_frames = duration_ms / 1000 / te * sr_header``

        :param duration_ms: Duration in true milliseconds.
        :return:            Number of file frames.
        """
        return max(1, int(duration_ms / 1000.0 / self.te * self.sr_header))

    def file_frame_to_true_ms(self, frame: int) -> float:
        """
        Convert a file frame index to true-time milliseconds from recording
        start.

        :param frame: File frame index.
        :return:      True-time offset in milliseconds.
        """
        return frame / self.sr_header * self.te * 1000.0

    def true_duration_window_s(self) -> float:
        """
        Return the true acoustic duration of the recording in seconds,
        for use as a match window in :class:`WavPathResolver`.

        :return: True duration in seconds.
        """
        return self.true_duration_s

    # ------------------------------------------------------------------ #
    #  Display                                                            #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        return (
            f'WavInfo({self.path.name}, type={self.rec_type}, '
            f'sr_header={self.sr_header:,}, te={self.te}, '
            f'sr_true={self.sr_true:,}, '
            f'true_dur={self.true_duration_s:.2f}s, '
            f'source={self.source})'
        )
