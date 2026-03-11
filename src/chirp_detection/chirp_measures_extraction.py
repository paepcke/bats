#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-08 16:27:19
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/chirp_detection/chirp_measures_extraction.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-11 12:13:57
#
# **********************************************************#

"""
chirp_measures_extraction.py
=============================
Extract acoustic measures from chopped bat-detector ``.wav`` chunks, producing
a CSV that mirrors the subset of SonoBat's per-chirp output used by the
downstream analysis pipeline.

Each row in the output describes one chirp (pulse) detected within a chunk.
Chunks are the 2-second segments produced by :class:`~wav_chopper.WavChopper`.

Output columns
--------------
``file_id``, ``chirp_idx``, ``is_last``, ``species``, ``TimeInFile``, ``PrecedingIntrvl``,
``HiFreq``, ``Bndwdth``, ``FreqMaxPwr``, ``PrcntMaxAmpDur``,
``FreqKnee``, ``PrcntKneeDur``, ``StartF``, ``UpprKnFreq``,
``HiFtoUpprKnAmp``, ``HiFtoKnAmp``, ``HiFtoFcAmp``,
``UpprKnToKnAmp``, ``KnToFcAmp``, ``LdgToFcAmp``,
``FreqCtr``, ``FFwd32dB``, ``FFwd20dB``, ``FFwd15dB``,
``FBak5dB``, ``FFwd5dB``, ``Bndw32dB``,
``Amp1stQrtl``, ``Amp2ndQrtl``, ``Amp3rdQrtl``, ``Amp4thQrtl``,
``1st10kHzSlp``, ``1st5to15kHzSlp``, ``1st10kHzExp``, ``1st5to15kHzExp``,
``AmpK@start``

Notes
-----
* ``chirp_idx`` is a 0-origin counter of chirps within one recording file,
  assigned in ascending ``TimeInFile`` order.  It is computed as a
  post-processing pass after all workers complete.
* ``is_last`` is ``1`` for the final chirp in each recording file, ``0``
  otherwise.  Stored as integer so ``pd.read_csv(...,
  dtype={'is_last': bool})`` round-trips to a boolean column without
  ambiguity.
* ``TimeInFile`` is the chirp onset in milliseconds measured from the start of
  the **original full recording**, computed as the chunk's own ``_t{offset}ms``
  plus the within-chunk onset.
* ``PrecedingIntrvl`` is the gap (ms) from the previous chirp's offset to this
  chirp's onset, within the same chunk.  It is ``NaN`` for the first chirp in
  each chunk; post-processing can fill cross-chunk gaps from the sorted
  ``TimeInFile`` sequence.
* Column names are kept as strings matching SonoBat's naming convention,
  including names that begin with a digit (``1st10kHzSlp`` etc.).
* Any measure that cannot be computed for a given chirp is stored as ``NaN``.
  ``pandas.DataFrame.to_csv()`` writes ``NaN`` as an empty field, which
  ``pandas.read_csv()`` reads back as ``NaN`` by default.

Parallelism
-----------
:class:`MeasureExtractor` fans out one worker per chunk file using
:class:`concurrent.futures.ProcessPoolExecutor`.  Each worker calls the
module-level ``_extract_one()`` function and returns a list of row dicts.
The main process concatenates all rows into a single DataFrame and writes the
CSV.  No ordering guarantees are made on the rows.

Typical usage
-------------
::

    from chirp_measures_extraction import MeasureExtractor

    extractor = MeasureExtractor(chunk_paths, out_csv='measures.csv')
    result    = extractor.run()
    print(result.summary())
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import logging
import math
import os
import shlex
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, wait as _wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.stats import kurtosis as scipy_kurtosis

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from enum import StrEnum

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Column names — ordered exactly as they appear in the output CSV
# ---------------------------------------------------------------------------

COLUMNS: List[str] = [
    'file_id',
    'chirp_idx',
    'is_last',
    'species',
    'TimeInFile',
    'PrecedingIntrvl',
    'HiFreq',
    'Bndwdth',
    'FreqMaxPwr',
    'PrcntMaxAmpDur',
    'FreqKnee',
    'PrcntKneeDur',
    'StartF',
    'UpprKnFreq',
    'HiFtoUpprKnAmp',
    'HiFtoKnAmp',
    'HiFtoFcAmp',
    'UpprKnToKnAmp',
    'KnToFcAmp',
    'LdgToFcAmp',
    'FreqCtr',
    'FFwd32dB',
    'FFwd20dB',
    'FFwd15dB',
    'FBak5dB',
    'FFwd5dB',
    'Bndw32dB',
    'Amp1stQrtl',
    'Amp2ndQrtl',
    'Amp3rdQrtl',
    'Amp4thQrtl',
    '1st10kHzSlp',
    '1st5to15kHzSlp',
    '1st10kHzExp',
    '1st5to15kHzExp',
    'AmpK@start',
]

# ---------------------------------------------------------------------------
# Constants shared by module-level workers
# ---------------------------------------------------------------------------

#: Header sample rates below this value indicate a time-expanded file.
_TE_SR_THRESHOLD_HZ: int = 80_000

#: Expansion factor for time-expanded files.
_TIME_EXPAND_FACTOR: int = 10

#: Lower bound of the bat echolocation band (Hz).
_BAT_BAND_LO_HZ: float = 15_000.0

#: Upper bound of the bat echolocation band (Hz).
_BAT_BAND_HI_HZ: float = 120_000.0

#: Energy threshold for pulse detection (dBFS).
_PULSE_THRESHOLD_DBFS: float = -60.0

#: Minimum valid pulse duration (ms).
_PULSE_DUR_MIN_MS: float = 0.5

#: Maximum valid pulse duration (ms).
_PULSE_DUR_MAX_MS: float = 35.0

#: Minimum spectral bandwidth at −20 dB from peak (kHz).
_MIN_BANDWIDTH_KHZ: float = 5.0

#: Maximum intra-pulse fragment gap to merge (ms).
_PULSE_MERGE_GAP_MS: float = 5.0

#: STFT window duration (ms) — finer than the scrubber's 0.5 ms for
#: accurate ridge extraction.
_STFT_WINDOW_MS: float = 0.25

#: Minimum duration (ms) for a ledge to be considered real.
_LEDGE_MIN_DUR_MS: float = 0.3


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

class ExtractionStatus(StrEnum):
    """Per-file extraction outcome."""
    OK          = 'ok'           #: At least one chirp was extracted.
    NO_CHIRPS   = 'no_chirps'    #: File loaded but no valid chirps found.
    UNREADABLE  = 'unreadable'   #: File could not be read.


@dataclasses.dataclass
class FileRecord:
    """
    Extraction outcome for a single chunk file.

    :param path:       Absolute path to the chunk.
    :param status:     Outcome code.
    :param n_chirps:   Number of chirp rows extracted.
    :param detail:     Free-form error message (empty on success).
    """
    path:     Path
    status:   ExtractionStatus
    n_chirps: int  = 0
    detail:   str  = ''


@dataclasses.dataclass
class ExtractionResult:
    """
    Aggregate result of a :meth:`MeasureExtractor.run` call.

    :param file_records: One :class:`FileRecord` per input chunk.
    :param out_csv:      Path to the written CSV, or ``None`` if not yet written.
    :param n_rows:       Total chirp rows in the output.
    """
    file_records: List[FileRecord]
    out_csv:      Optional[Path]
    n_rows:       int

    def summary(self) -> str:
        """
        Return a human-readable summary string.

        :return: Multi-line summary.
        """
        n_files   = len(self.file_records)
        n_ok      = sum(1 for r in self.file_records if r.status == ExtractionStatus.OK)
        n_empty   = sum(1 for r in self.file_records if r.status == ExtractionStatus.NO_CHIRPS)
        n_err     = sum(1 for r in self.file_records if r.status == ExtractionStatus.UNREADABLE)
        lines = [
            'MeasureExtractor results',
            f'  Chunk files processed : {n_files}',
            f'  Files with chirps     : {n_ok}',
            f'  Files with no chirps  : {n_empty}',
            f'  Unreadable files      : {n_err}',
            f'  Total chirp rows      : {self.n_rows}',
        ]
        if self.out_csv:
            lines.append(f'  Output CSV            : {self.out_csv}')
        if n_err:
            lines.append('\nUnreadable files:')
            for r in self.file_records:
                if r.status == ExtractionStatus.UNREADABLE:
                    lines.append(f'  {r.path}  — {r.detail}')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Low-level signal helpers
# ---------------------------------------------------------------------------

def _te_correct(sr_header: int) -> int:
    """
    Return the true sample rate, applying TE correction if needed.

    :param sr_header: Sample rate from the WAV header.
    :return:          True ultrasound sample rate (Hz).
    """
    return sr_header * _TIME_EXPAND_FACTOR if sr_header < _TE_SR_THRESHOLD_HZ else sr_header


def _compute_stft(
    audio: np.ndarray,
    sr: int,
    window_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a power spectrogram with a Hann window.

    :param audio:     Float32 mono waveform in [−1, 1].
    :param sr:        True sample rate (Hz).
    :param window_ms: STFT window duration (ms).
    :return:          ``(freqs_hz, times_s, Sxx)`` — Sxx is linear power,
                      shape ``(n_freqs, n_times)``.
    """
    nperseg  = max(64, int(round(window_ms / 1000.0 * sr)))
    noverlap = nperseg * 3 // 4
    freqs, times, Sxx = signal.spectrogram(
        audio, fs=sr,
        window='hann', nperseg=nperseg, noverlap=noverlap,
        scaling='spectrum',
    )
    return freqs.astype(np.float64), times.astype(np.float64), Sxx.astype(np.float64)


def _extract_ridge(
    Sxx_bat: np.ndarray,
    freqs_bat: np.ndarray,
) -> np.ndarray:
    """
    Extract the frequency ridge from a bat-band spectrogram slice.

    The ridge is the peak-power frequency at each time frame (argmax along
    the frequency axis).

    :param Sxx_bat:   Power spectrogram restricted to the bat band,
                      shape ``(n_bat_freqs, n_times)``.
    :param freqs_bat: Frequency axis (Hz), length ``n_bat_freqs``.
    :return:          Ridge frequencies (Hz), shape ``(n_times,)``.
    """
    idx = np.argmax(Sxx_bat, axis=0)
    return freqs_bat[idx]


def _find_segments(
    above: np.ndarray,
    times_s: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Find contiguous ``True`` runs and return ``(onset_s, offset_s)`` pairs.

    :param above:   Boolean array, ``True`` where energy exceeds threshold.
    :param times_s: Time axis (s), same length as *above*.
    :return:        List of ``(onset_s, offset_s)`` tuples.
    """
    segs: List[Tuple[float, float]] = []
    in_seg = False
    onset  = 0.0
    for i, val in enumerate(above):
        if val and not in_seg:
            onset  = float(times_s[i])
            in_seg = True
        elif not val and in_seg:
            segs.append((onset, float(times_s[i - 1])))
            in_seg = False
    if in_seg:
        segs.append((onset, float(times_s[-1])))
    return segs


def _merge_segments(
    segs: List[Tuple[float, float]],
    max_gap_s: float,
) -> List[Tuple[float, float]]:
    """
    Merge consecutive segments separated by ≤ *max_gap_s* seconds.

    :param segs:      Sorted ``(onset_s, offset_s)`` list.
    :param max_gap_s: Maximum bridgeable gap (s).
    :return:          Merged segment list.
    """
    if not segs or max_gap_s <= 0.0:
        return segs
    merged = [segs[0]]
    for onset, offset in segs[1:]:
        if onset - merged[-1][1] <= max_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], offset))
        else:
            merged.append((onset, offset))
    return merged


def _interp_amplitude(
    ridge_hz:  np.ndarray,
    ridge_amp: np.ndarray,
    target_hz: float,
) -> float:
    """
    Interpolate amplitude on the ridge at a target frequency.

    Searches for the time frame whose ridge frequency is closest to
    *target_hz* and returns the corresponding amplitude.

    :param ridge_hz:  Ridge frequencies (Hz).
    :param ridge_amp: Ridge amplitudes (linear power), same shape.
    :param target_hz: Target frequency (Hz).
    :return:          Interpolated amplitude, or ``NaN`` if ridge is empty.
    """
    if len(ridge_hz) == 0:
        return math.nan
    idx = int(np.argmin(np.abs(ridge_hz - target_hz)))
    return float(ridge_amp[idx])


def _safe_ratio(num: float, den: float) -> float:
    """
    Return ``num / den``, or ``NaN`` if *den* is zero or either is ``NaN``.

    :param num: Numerator.
    :param den: Denominator.
    :return:    Ratio or ``NaN``.
    """
    if math.isnan(num) or math.isnan(den) or den == 0.0:
        return math.nan
    return num / den


# ---------------------------------------------------------------------------
# Per-chirp measure computation
# ---------------------------------------------------------------------------

def _measures_from_chirp(
    audio:     np.ndarray,
    sr:        int,
    onset_s:   float,
    offset_s:  float,
    freqs_hz:  np.ndarray,
    times_s:   np.ndarray,
    Sxx:       np.ndarray,
    chunk_offset_ms: float,
    prev_offset_s:   Optional[float],
    window_ms: float,
) -> Dict[str, object]:
    """
    Compute all acoustic measures for a single detected chirp.

    :param audio:            Full-chunk float32 waveform.
    :param sr:               True sample rate (Hz).
    :param onset_s:          Chirp onset within the chunk (s).
    :param offset_s:         Chirp offset within the chunk (s).
    :param freqs_hz:         Full spectrogram frequency axis (Hz).
    :param times_s:          Full spectrogram time axis (s).
    :param Sxx:              Full-chunk power spectrogram ``(n_freqs, n_times)``.
    :param chunk_offset_ms:  Start of this chunk within the original recording (ms).
    :param prev_offset_s:    Offset of the previous chirp within the chunk (s),
                             or ``None`` if this is the first chirp.
    :param window_ms:        STFT window duration used (ms).
    :return:                 Dict mapping column name → value (float or NaN).
    """
    NaN = math.nan
    row: Dict[str, object] = {col: NaN for col in COLUMNS}

    dur_s  = offset_s - onset_s
    dur_ms = dur_s * 1000.0
    if dur_ms < _PULSE_DUR_MIN_MS:
        return row   # degenerate — should not normally happen after filtering

    # ── Bat-band slice ─────────────────────────────────────────────────────
    bat_mask  = (freqs_hz >= _BAT_BAND_LO_HZ) & (freqs_hz <= _BAT_BAND_HI_HZ)
    freqs_bat = freqs_hz[bat_mask]

    t_mask = (times_s >= onset_s) & (times_s <= offset_s)
    if t_mask.sum() < 2 or bat_mask.sum() == 0:
        return row

    Sxx_chirp  = Sxx[np.ix_(bat_mask, t_mask)]   # (n_bat_freqs, n_chirp_times)
    times_chirp = times_s[t_mask]
    n_t = Sxx_chirp.shape[1]

    # ── Ridge ──────────────────────────────────────────────────────────────
    ridge_idx = np.argmax(Sxx_chirp, axis=0)       # (n_chirp_times,)
    ridge_hz  = freqs_bat[ridge_idx]               # (n_chirp_times,)
    ridge_amp = Sxx_chirp[ridge_idx, np.arange(n_t)]  # peak power at each t

    # ── Timing ─────────────────────────────────────────────────────────────
    time_in_file_ms = chunk_offset_ms + onset_s * 1000.0
    preceding_ms    = (onset_s - prev_offset_s) * 1000.0 if prev_offset_s is not None else NaN

    # ── Frequency landmarks ────────────────────────────────────────────────
    hi_freq_hz  = float(ridge_hz.max())
    lo_freq_hz  = float(ridge_hz.min())
    bndwdth_khz = (hi_freq_hz - lo_freq_hz) / 1000.0
    start_f_hz  = float(ridge_hz[0])
    freq_ctr_hz = (hi_freq_hz + lo_freq_hz) / 2.0

    # Fc = characteristic frequency = frequency of peak power over entire chirp
    peak_frame  = int(np.argmax(ridge_amp))
    fc_hz       = float(ridge_hz[peak_frame])
    prcnt_max_amp_dur = (times_chirp[peak_frame] - onset_s) / dur_s * 100.0

    # ── Knee detection ─────────────────────────────────────────────────────
    # The knee is the point of maximum curvature on the ridge.
    # Smooth the ridge first to suppress STFT quantisation noise.
    if n_t >= 5:
        smooth_len  = min(5, n_t if n_t % 2 == 1 else n_t - 1)
        ridge_smooth = signal.savgol_filter(ridge_hz, smooth_len, 2) \
                       if smooth_len >= 3 else ridge_hz.copy()
    else:
        ridge_smooth = ridge_hz.copy()

    # Second derivative of smoothed ridge → maximum = knee
    if len(ridge_smooth) >= 3:
        d2 = np.gradient(np.gradient(ridge_smooth))
        knee_idx = int(np.argmax(np.abs(d2)))
    else:
        knee_idx = len(ridge_smooth) // 2

    knee_hz       = float(ridge_smooth[knee_idx])
    prcnt_knee_dur = (times_chirp[knee_idx] - onset_s) / dur_s * 100.0

    # Upper knee: secondary inflection between HiFreq and FreqKnee.
    # Look in the first half (high-frequency portion) of the ridge for the
    # frame of maximum |d2| that is above the knee frequency.
    upper_kn_hz = NaN
    upper_kn_idx: Optional[int] = None
    if knee_idx > 0:
        pre_knee = ridge_smooth[:knee_idx]
        if len(pre_knee) >= 3:
            d2_pre = np.gradient(np.gradient(pre_knee))
            uk_idx = int(np.argmax(np.abs(d2_pre)))
            upper_kn_hz  = float(pre_knee[uk_idx])
            upper_kn_idx = uk_idx

    # ── Amplitude at frequency landmarks ───────────────────────────────────
    amp_at_hi    = _interp_amplitude(ridge_hz, ridge_amp, hi_freq_hz)
    amp_at_uk    = _interp_amplitude(ridge_hz, ridge_amp, upper_kn_hz) \
                   if not math.isnan(upper_kn_hz) else NaN
    amp_at_knee  = _interp_amplitude(ridge_hz, ridge_amp, knee_hz)
    amp_at_fc    = _interp_amplitude(ridge_hz, ridge_amp, fc_hz)
    peak_amp     = float(ridge_amp.max())

    hi_to_uk_amp  = _safe_ratio(amp_at_uk,   amp_at_hi)
    hi_to_kn_amp  = _safe_ratio(amp_at_knee, amp_at_hi)
    hi_to_fc_amp  = _safe_ratio(amp_at_fc,   amp_at_hi)
    uk_to_kn_amp  = _safe_ratio(amp_at_knee, amp_at_uk)
    kn_to_fc_amp  = _safe_ratio(amp_at_fc,   amp_at_knee)

    # ── Ledge detection ────────────────────────────────────────────────────
    # A ledge is a transient plateau in the FM sweep: a region where the
    # instantaneous slope is near-zero for ≥ _LEDGE_MIN_DUR_MS.
    ldg_to_fc_amp = NaN
    if n_t >= 4:
        dt_s        = float(np.mean(np.diff(times_chirp)))
        slope_hz_ms = np.gradient(ridge_smooth) / (dt_s * 1000.0)  # kHz/ms
        # Plateau = |slope| < 10% of the mean absolute slope
        mean_abs_slope = float(np.mean(np.abs(slope_hz_ms))) + 1e-9
        plateau_mask   = np.abs(slope_hz_ms) < 0.10 * mean_abs_slope
        # Find contiguous plateau runs
        plateau_segs = _find_segments(plateau_mask, times_chirp - onset_s)
        ledge_segs   = [s for s in plateau_segs
                        if (s[1] - s[0]) * 1000.0 >= _LEDGE_MIN_DUR_MS]
        if ledge_segs:
            # Use the first (highest-frequency) ledge
            ldg_onset_rel = ledge_segs[0][0]
            ldg_idx = int(np.argmin(np.abs((times_chirp - onset_s) - ldg_onset_rel)))
            amp_at_ldg  = float(ridge_amp[ldg_idx])
            ldg_to_fc_amp = _safe_ratio(amp_at_fc, amp_at_ldg)

    # ── Bandwidth-at-dB measures ───────────────────────────────────────────
    # Use the spectrum at the peak-power frame (Fc frame).
    spec_at_fc = Sxx_chirp[:, peak_frame]           # power vs freq in bat band
    peak_p     = float(spec_at_fc.max())

    def _freq_at_db_drop(db: float, direction: str) -> float:
        """
        Find frequency where spectrum drops *db* dB from peak, using linear
        interpolation between the two bins that straddle the threshold.

        Uses the full bat-band spectrum column at the peak-power frame rather
        than only the slice to one side of the peak bin.  This handles
        narrow-band chirps (e.g. *Laci* quasi-CF calls) where the peak bin
        sits at the edge of the chirp's frequency extent and the one-sided
        slice would be empty.

        :param db:        Drop in dB (positive value).
        :param direction: ``'fwd'`` = toward lower frequencies (forward in
                          sweep), ``'bak'`` = toward higher frequencies.
        :return:          Frequency (kHz) or NaN.
        """
        if peak_p <= 0:
            return NaN
        threshold = peak_p * 10 ** (-db / 10.0)
        pk_idx    = int(np.argmax(spec_at_fc))
        if direction == 'fwd':
            # Search below peak index (lower frequencies)
            if pk_idx == 0:
                # Peak is at lowest bat-band bin — no room to drop; use that freq
                return float(freqs_bat[0]) / 1000.0
            region       = spec_at_fc[:pk_idx]
            region_freqs = freqs_bat[:pk_idx]
            above = np.where(region >= threshold)[0]
            if len(above) == 0:
                return float(freqs_bat[0]) / 1000.0
            last_above = above[-1]
            if last_above + 1 >= len(region):
                return float(region_freqs[last_above]) / 1000.0
            f0, f1 = float(region_freqs[last_above]), float(region_freqs[last_above + 1])
            p0, p1 = float(region[last_above]),        float(region[last_above + 1])
            if p0 == p1:
                return f0 / 1000.0
            t = (threshold - p0) / (p1 - p0)
            return (f0 + t * (f1 - f0)) / 1000.0
        else:
            # Search above peak index (higher frequencies)
            if pk_idx >= len(spec_at_fc) - 1:
                return float(freqs_bat[-1]) / 1000.0
            region       = spec_at_fc[pk_idx + 1:]
            region_freqs = freqs_bat[pk_idx + 1:]
            above = np.where(region >= threshold)[0]
            if len(above) == 0:
                return float(freqs_bat[-1]) / 1000.0
            first_above = above[0]
            if first_above == 0:
                return float(region_freqs[0]) / 1000.0
            f0, f1 = float(region_freqs[first_above - 1]), float(region_freqs[first_above])
            p0, p1 = float(region[first_above - 1]),        float(region[first_above])
            if p0 == p1:
                return f1 / 1000.0
            t = (threshold - p0) / (p1 - p0)
            return (f0 + t * (f1 - f0)) / 1000.0

    ffwd32 = _freq_at_db_drop(32.0, 'fwd')
    ffwd20 = _freq_at_db_drop(20.0, 'fwd')
    ffwd15 = _freq_at_db_drop(15.0, 'fwd')
    ffwd5  = _freq_at_db_drop(5.0,  'fwd')
    fbak5  = _freq_at_db_drop(5.0,  'bak')

    bndw32 = NaN
    if not math.isnan(ffwd32) and not math.isnan(fbak5):
        bndw32 = fbak5 - ffwd32   # kHz

    # ── Amplitude quartile measures ────────────────────────────────────────
    # Use the time-domain envelope (abs of analytic signal via Hilbert).
    # Restrict to samples within the chirp window.
    s_onset = int(round(onset_s * sr))
    s_offset = min(int(round(offset_s * sr)), len(audio))
    if s_offset > s_onset:
        chirp_samples = audio[s_onset:s_offset]
        analytic      = signal.hilbert(chirp_samples)
        envelope      = np.abs(analytic).astype(np.float64)
        n_env         = len(envelope)
        pk_env        = float(envelope.max()) if n_env > 0 else 1.0
        if pk_env == 0.0:
            pk_env = 1.0

        q = n_env // 4
        def _qmean(a, b):
            seg = envelope[a:b]
            return float(seg.mean()) / pk_env if len(seg) > 0 else NaN

        amp_q1 = _qmean(0,       q)
        amp_q2 = _qmean(q,       2 * q)
        amp_q3 = _qmean(2 * q,   3 * q)
        amp_q4 = _qmean(3 * q,   n_env)

        # AmpK@start: kurtosis of the first 25% of the envelope
        first_q = envelope[:q]
        amp_k_start = float(scipy_kurtosis(first_q)) if len(first_q) >= 4 else NaN
    else:
        amp_q1 = amp_q2 = amp_q3 = amp_q4 = amp_k_start = NaN

    # ── Slope measures (ridge in kHz vs time in ms) ────────────────────────
    # Use the smoothed ridge so that shallow-sweep species (e.g. Laci, Tabr)
    # whose raw ridge is a coarse staircase of STFT bins don't produce NaN
    # from searchsorted monotonicity failures.  For steep-sweep species the
    # smoothed and raw ridges are nearly identical (sub-bin difference).
    ridge_smooth_khz = ridge_smooth / 1000.0
    times_ms         = (times_chirp - onset_s) * 1000.0   # relative to chirp onset

    def _slope_and_exp_in_range(
        delta_khz_start: float,
        delta_khz_end: float,
    ) -> Tuple[float, float]:
        """
        Compute the linear slope (kHz/ms) and exponential decay constant
        over the portion of the smoothed ridge that covers a specified
        frequency drop relative to HiFreq.

        Uses the Savitzky-Golay-smoothed ridge rather than the raw ridge to
        avoid ``searchsorted`` failures on shallow-sweep chirps where the raw
        ridge is a staircase of coarse STFT bins rather than a smooth curve.

        The segment starts where the ridge first drops below
        ``hi_freq_khz - delta_khz_start`` and ends where it first drops below
        ``hi_freq_khz - delta_khz_end``.

        :param delta_khz_start: kHz below HiFreq at which the segment starts.
        :param delta_khz_end:   kHz below HiFreq at which the segment ends.
        :return: ``(slope_kHz_per_ms, exp_decay_constant)`` or ``(NaN, NaN)``.
        """
        # Use the smoothed ridge's own hi/lo for threshold consistency.
        hi_khz  = float(ridge_smooth_khz.max())
        lo_khz  = float(ridge_smooth_khz.min())
        f_start = hi_khz - delta_khz_start
        f_end   = hi_khz - delta_khz_end
        # Clamp f_end to the actual lo so narrow-band species still get an
        # estimate over whatever range is available.
        f_end   = max(f_end, lo_khz)
        if f_start <= f_end or f_start <= lo_khz:
            return NaN, NaN
        # Locate the segment of the ridge that lies between f_start and f_end
        # using nearest-value lookup rather than searchsorted.  searchsorted
        # requires a monotone-decreasing ridge, which fails for non-monotone
        # or ascending calls (e.g. Coto, some Tabr).  Instead we find the
        # frame closest to f_start and the frame closest to f_end, then take
        # the time-ordered slice between them.
        i_peak   = int(np.argmax(ridge_smooth_khz))   # frame at hi_khz
        i_f_start = int(np.argmin(np.abs(ridge_smooth_khz - f_start)))
        i_f_end   = int(np.argmin(np.abs(ridge_smooth_khz - f_end)))
        i_lo, i_hi = (min(i_f_start, i_f_end), max(i_f_start, i_f_end))
        # Ensure the peak lies within or adjacent to the segment — if the
        # ridge is non-monotone and both f_start and f_end are on the same
        # side of the peak, the segment is ill-defined; skip it.
        if i_hi <= i_lo:
            return NaN, NaN
        seg_t = times_ms[i_lo:i_hi + 1]
        seg_f = ridge_smooth_khz[i_lo:i_hi + 1]
        if len(seg_t) < 2:
            return NaN, NaN
        # Linear slope via least-squares
        try:
            coeffs  = np.polyfit(seg_t, seg_f, 1)
            slope   = float(coeffs[0])           # kHz/ms
        except (np.linalg.LinAlgError, ValueError):
            slope   = NaN
        # Exponential decay constant: fit log(f) = a*t + b
        pos_mask = seg_f > 0
        if pos_mask.sum() < 2:
            return slope, NaN
        try:
            exp_coeffs = np.polyfit(seg_t[pos_mask], np.log(seg_f[pos_mask]), 1)
            exp_const  = float(exp_coeffs[0])    # per ms
        except (np.linalg.LinAlgError, ValueError):
            exp_const  = NaN
        return slope, exp_const

    slp_10,  exp_10  = _slope_and_exp_in_range(0.0,  10.0)
    slp_5_15, exp_5_15 = _slope_and_exp_in_range(5.0, 15.0)

    # ── Assemble row ───────────────────────────────────────────────────────
    row['TimeInFile']       = round(time_in_file_ms, 3)
    row['PrecedingIntrvl']  = round(preceding_ms, 3) if not math.isnan(preceding_ms) else NaN
    row['HiFreq']           = round(hi_freq_hz  / 1000.0, 6)
    row['Bndwdth']          = round(bndwdth_khz,           6)
    row['FreqMaxPwr']       = round(fc_hz        / 1000.0, 6)
    row['PrcntMaxAmpDur']   = round(prcnt_max_amp_dur,     6)
    row['FreqKnee']         = round(knee_hz      / 1000.0, 6)
    row['PrcntKneeDur']     = round(prcnt_knee_dur,        6)
    row['StartF']           = round(start_f_hz   / 1000.0, 6)
    row['UpprKnFreq']       = round(upper_kn_hz  / 1000.0, 6) if not math.isnan(upper_kn_hz) else NaN
    row['HiFtoUpprKnAmp']   = round(hi_to_uk_amp,  6) if not math.isnan(hi_to_uk_amp)  else NaN
    row['HiFtoKnAmp']       = round(hi_to_kn_amp,  6) if not math.isnan(hi_to_kn_amp)  else NaN
    row['HiFtoFcAmp']       = round(hi_to_fc_amp,  6) if not math.isnan(hi_to_fc_amp)  else NaN
    row['UpprKnToKnAmp']    = round(uk_to_kn_amp,  6) if not math.isnan(uk_to_kn_amp)  else NaN
    row['KnToFcAmp']        = round(kn_to_fc_amp,  6) if not math.isnan(kn_to_fc_amp)  else NaN
    row['LdgToFcAmp']       = round(ldg_to_fc_amp, 6) if not math.isnan(ldg_to_fc_amp) else NaN
    row['FreqCtr']          = round(freq_ctr_hz  / 1000.0, 6)
    row['FFwd32dB']         = round(ffwd32, 6) if not math.isnan(ffwd32) else NaN
    row['FFwd20dB']         = round(ffwd20, 6) if not math.isnan(ffwd20) else NaN
    row['FFwd15dB']         = round(ffwd15, 6) if not math.isnan(ffwd15) else NaN
    row['FBak5dB']          = round(fbak5,  6) if not math.isnan(fbak5)  else NaN
    row['FFwd5dB']          = round(ffwd5,  6) if not math.isnan(ffwd5)  else NaN
    row['Bndw32dB']         = round(bndw32, 6) if not math.isnan(bndw32) else NaN
    row['Amp1stQrtl']       = round(amp_q1, 6) if not math.isnan(amp_q1) else NaN
    row['Amp2ndQrtl']       = round(amp_q2, 6) if not math.isnan(amp_q2) else NaN
    row['Amp3rdQrtl']       = round(amp_q3, 6) if not math.isnan(amp_q3) else NaN
    row['Amp4thQrtl']       = round(amp_q4, 6) if not math.isnan(amp_q4) else NaN
    row['1st10kHzSlp']      = round(slp_10,   6) if not math.isnan(slp_10)   else NaN
    row['1st5to15kHzSlp']   = round(slp_5_15, 6) if not math.isnan(slp_5_15) else NaN
    row['1st10kHzExp']      = round(exp_10,   6) if not math.isnan(exp_10)   else NaN
    row['1st5to15kHzExp']   = round(exp_5_15, 6) if not math.isnan(exp_5_15) else NaN
    row['AmpK@start']       = round(amp_k_start, 6) if not math.isnan(amp_k_start) else NaN

    return row


# ---------------------------------------------------------------------------
# Per-file worker (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _extract_one(
    chunk_path: Path,
    window_ms:  float,
    species:    Optional[str] = None,
) -> Tuple[List[Dict[str, object]], FileRecord]:
    """
    Load a chunk ``.wav``, detect chirps, and compute all measures.

    Module-level so :class:`ProcessPoolExecutor` can pickle it.

    :param chunk_path: Path to the chunk ``.wav`` file.
    :param window_ms:  STFT window duration (ms).
    :param species:    Four-letter species code from DB lookup, or ``None``
                       when the source is a plain path/directory input.
    :return:           ``(rows, FileRecord)`` where *rows* is a list of dicts
                       (one per chirp) ready for DataFrame construction.
    """
    rec = FileRecord(path=chunk_path.resolve(), status=ExtractionStatus.OK)

    # ── 1. Load ────────────────────────────────────────────────────────────
    try:
        sr_header, data = wavfile.read(str(chunk_path))
    except Exception as exc:
        rec.status = ExtractionStatus.UNREADABLE
        rec.detail = str(exc)
        return [], rec

    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) == 0:
        rec.status = ExtractionStatus.NO_CHIRPS
        rec.detail = 'zero samples'
        return [], rec

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2_147_483_648.0
    else:
        audio = data.astype(np.float32)

    sr = _te_correct(sr_header)

    # ── 2. Chunk offset from filename ──────────────────────────────────────
    # _chunk_stem format: {file_id}_t{offset_ms:07d}ms
    try:
        stem     = chunk_path.stem
        file_id, offset_ms_str = stem.rsplit('_t', 1)
        chunk_offset_ms = float(offset_ms_str.rstrip('ms'))
    except Exception:
        # Filename doesn't follow the convention — use 0 offset, keep going.
        file_id         = chunk_path.stem
        chunk_offset_ms = 0.0

    # ── 3. STFT ────────────────────────────────────────────────────────────
    freqs_hz, times_s, Sxx = _compute_stft(audio, sr, window_ms)

    bat_mask  = (freqs_hz >= _BAT_BAND_LO_HZ) & (freqs_hz <= _BAT_BAND_HI_HZ)
    Sxx_bat   = Sxx[bat_mask, :]
    if Sxx_bat.size == 0:
        rec.status = ExtractionStatus.NO_CHIRPS
        rec.detail = 'bat band above Nyquist'
        return [], rec

    # ── 4. Pulse detection ─────────────────────────────────────────────────
    threshold_linear = 10.0 ** (_PULSE_THRESHOLD_DBFS / 10.0) * Sxx_bat.shape[0]
    energy           = Sxx_bat.sum(axis=0)
    above            = energy >= threshold_linear

    raw_segs   = _find_segments(above, times_s)
    merged_segs = _merge_segments(raw_segs, _PULSE_MERGE_GAP_MS / 1000.0)

    # Apply duration + bandwidth filter
    valid_segs: List[Tuple[float, float]] = []
    freqs_bat_arr = freqs_hz[bat_mask]
    for onset_s, offset_s in merged_segs:
        dur_ms = (offset_s - onset_s) * 1000.0
        if not (_PULSE_DUR_MIN_MS <= dur_ms <= _PULSE_DUR_MAX_MS):
            continue
        t_mask     = (times_s >= onset_s) & (times_s <= offset_s)
        if t_mask.sum() == 0:
            continue
        pulse_spec = Sxx_bat[:, t_mask].max(axis=1)
        peak_p     = pulse_spec.max()
        if peak_p <= 0:
            continue
        bw_mask = pulse_spec >= peak_p * 10 ** (-20.0 / 10.0)
        bw_khz  = (freqs_bat_arr[bw_mask].max() - freqs_bat_arr[bw_mask].min()) / 1000.0 \
                  if bw_mask.sum() >= 2 else 0.0
        if bw_khz < _MIN_BANDWIDTH_KHZ:
            continue
        valid_segs.append((onset_s, offset_s))

    if not valid_segs:
        rec.status = ExtractionStatus.NO_CHIRPS
        rec.detail = 'no valid pulses after filtering'
        return [], rec

    # ── 5. Measure extraction ──────────────────────────────────────────────
    rows: List[Dict[str, object]] = []
    prev_offset_s: Optional[float] = None

    for onset_s, offset_s in valid_segs:
        row = _measures_from_chirp(
            audio          = audio,
            sr             = sr,
            onset_s        = onset_s,
            offset_s       = offset_s,
            freqs_hz       = freqs_hz,
            times_s        = times_s,
            Sxx            = Sxx,
            chunk_offset_ms= chunk_offset_ms,
            prev_offset_s  = prev_offset_s,
            window_ms      = window_ms,
        )
        row['file_id'] = file_id
        row['species'] = species  # None → empty CSV cell for path/dir inputs
        rows.append(row)
        prev_offset_s = offset_s

    rec.n_chirps = len(rows)
    return rows, rec


# ---------------------------------------------------------------------------
# MeasureExtractor
# ---------------------------------------------------------------------------

class MeasureExtractor:
    """
    Extract acoustic measures from a list of chopped ``.wav`` chunks in
    parallel, writing one CSV row per detected chirp.

    :param chunk_paths:    Paths to ``.wav`` chunk files (from
                           :class:`~wav_chopper.WavChopper`).
    :param out_csv:        Destination CSV path.
    :param n_workers:      Worker processes.  ``None`` = cpu_count − 4;
                           ``0`` = all cores.
    :param show_progress:  Show a tqdm progress bar.
    :param worker_timeout: Per-file timeout (s).
    """

    # ------------------------------------------------------------------ #
    #  Class-level tunables                                               #
    # ------------------------------------------------------------------ #

    #: STFT window duration (ms) used for all spectral analysis.
    #: Finer than the scrubber's 0.5 ms for accurate ridge extraction.
    _STFT_WINDOW_MS: float = _STFT_WINDOW_MS

    #: Header SR below which a file is treated as time-expanded.
    _TE_SR_THRESHOLD_HZ: int = _TE_SR_THRESHOLD_HZ

    #: Expansion factor for time-expanded files.
    _TIME_EXPAND_FACTOR: int = _TIME_EXPAND_FACTOR

    def __init__(
        self,
        inputs:         Sequence[str | Path],
        out_csv:        str | Path,
        recursive:      bool            = False,
        n_workers:      Optional[int]   = None,
        show_progress:  bool            = True,
        worker_timeout: Optional[float] = 120.0,
    ) -> None:
        """
        :param inputs:         Source items — any mix of ``.wav`` files,
                               directories, or ``.db`` SQLite databases built
                               by ``create_wav_file_db.py``.
        :param out_csv:        Output CSV path.
        :param recursive:      Descend into subdirectories when a directory
                               is given.
        :param n_workers:      Worker count (``None`` = cpu_count − 4; ``0`` = all).
        :param show_progress:  Show tqdm progress bar.
        :param worker_timeout: Per-file timeout in seconds.
        """
        self.inputs      = [Path(p) for p in inputs]
        self.out_csv     = Path(out_csv)
        self.recursive   = recursive

        if n_workers is None:
            self.n_workers = max(1, (os.cpu_count() or 1) - 4)
        elif n_workers == 0:
            self.n_workers = os.cpu_count() or 1
        else:
            self.n_workers = n_workers

        self.show_progress  = show_progress
        self.worker_timeout = worker_timeout

    # ------------------------------------------------------------------ #
    #  Path iterator                                                       #
    # ------------------------------------------------------------------ #

    def _iter_paths(self):
        """
        Yield ``(path, species)`` tuples from all inputs.

        Dispatch rules:

        * ``.db`` file  — query the SQLite database produced by
          ``create_wav_file_db.py``; yields ``(full_path, species_code)``
          for every row.
        * directory     — glob ``*.wav`` (recursive if ``self.recursive``);
          yields ``(path, None)``.
        * ``.wav`` file — yields ``(path, None)`` directly.
        * anything else — logs a warning and skips.

        Duplicate paths (across all input types) are suppressed via a
        ``seen`` set.

        :yields: ``(Path, str | None)`` tuples.
        """
        import sqlite3

        seen: set[Path] = set()

        def _emit(p: Path, sp):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                return rp, sp
            return None, None

        for inp in self.inputs:
            if inp.suffix.lower() == '.db':
                if not inp.exists():
                    log.warn(f'DB file not found, skipping: {inp}')
                    continue
                try:
                    conn = sqlite3.connect(str(inp))
                    cur  = conn.execute(
                        'SELECT l.folder_path, s.filename, s.species_code '
                        'FROM Samples s JOIN Locations l ON s.location_id = l.id'
                    )
                    for folder_path, filename, species_code in cur:
                        full = Path(folder_path) / filename
                        p, sp = _emit(full, species_code or None)
                        if p is not None:
                            yield p, sp
                    conn.close()
                except Exception as exc:
                    log.warn(f'Cannot read DB {inp}: {exc}')

            elif inp.is_dir():
                glob_fn = inp.rglob if self.recursive else inp.glob
                for wav in sorted(glob_fn('*.wav')):
                    p, sp = _emit(wav, None)
                    if p is not None:
                        yield p, sp

            elif inp.suffix.lower() == '.wav':
                if not inp.exists():
                    log.warn(f'.wav file not found, skipping: {inp}')
                    continue
                p, sp = _emit(inp, None)
                if p is not None:
                    yield p, sp

            else:
                log.warn(f'Unrecognised input type, skipping: {inp}')

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> ExtractionResult:
        """
        Fan out measure extraction across all chunks and write the CSV.

        Rows are written as they arrive from workers; no ordering guarantee
        is made.  The CSV header is written once at the start.

        :return: :class:`ExtractionResult` with per-file outcomes.
        """
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

        path_iter  = self._iter_paths()
        window_ms  = self._STFT_WINDOW_MS
        file_recs: List[FileRecord] = []
        n_rows     = 0

        # tqdm needs a total — not known upfront when DB is involved, so
        # leave total=None (shows a spinner rather than a percentage bar).
        if self.show_progress and _TQDM_AVAILABLE:
            pbar = _tqdm(total=None, unit='chunk', desc='Extracting')
        else:
            pbar = None

        with self.out_csv.open('w', newline='') as csv_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=COLUMNS,
                                    extrasaction='ignore',
                                    restval='')   # NaN written as empty = pandas NaN on read
            writer.writeheader()

            window   = self.n_workers * 4
            fut_map: Dict = {}
            it = path_iter

            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                # Seed window
                for path, species in it:
                    fut = pool.submit(_extract_one, path, window_ms, species)
                    fut_map[fut] = path
                    if len(fut_map) >= window:
                        break

                while fut_map:
                    done_set, _ = _wait(
                        fut_map,
                        timeout=self.worker_timeout,
                        return_when=FIRST_COMPLETED,
                    )

                    if not done_set:
                        # Timeout — cancel remaining
                        for fut, path in list(fut_map.items()):
                            if not fut.done():
                                fut.cancel()
                                file_recs.append(FileRecord(
                                    path   = path.resolve(),
                                    status = ExtractionStatus.UNREADABLE,
                                    detail = f'worker timeout after {self.worker_timeout}s',
                                ))
                                if pbar is not None:
                                    pbar.update(1)
                        fut_map.clear()
                        break

                    for fut in done_set:
                        path = fut_map.pop(fut)
                        try:
                            rows, frec = fut.result()
                        except Exception as exc:
                            rows = []
                            frec = FileRecord(
                                path   = path.resolve(),
                                status = ExtractionStatus.UNREADABLE,
                                detail = str(exc),
                            )

                        # Write rows immediately — NaN → empty cell
                        for row in rows:
                            out_row = {
                                k: ('' if (isinstance(v, float) and math.isnan(v)) else v)
                                for k, v in row.items()
                            }
                            writer.writerow(out_row)
                        n_rows   += len(rows)
                        file_recs.append(frec)
                        csv_fh.flush()
                        if pbar is not None:
                            pbar.update(1)

                        # Refill window
                        for next_path, next_species in it:
                            fut2 = pool.submit(_extract_one, next_path, window_ms, next_species)
                            fut_map[fut2] = next_path
                            break

        if pbar is not None:
            pbar.close()

        # ── Post-processing: assign chirp_idx and is_last ──────────────────
        # These require the full sequence per file_id, so they can only be
        # computed after all workers have returned.  is_last is written as
        # 1/0 so pd.read_csv(..., dtype={'is_last': bool}) round-trips cleanly.
        if n_rows > 0:
            import pandas as pd
            df = pd.read_csv(self.out_csv)
            df.sort_values(['file_id', 'TimeInFile'], inplace=True)
            df['chirp_idx'] = df.groupby('file_id').cumcount()
            df['is_last']   = (~df.duplicated(subset='file_id', keep='last')).astype(int)
            # Re-order columns so chirp_idx and is_last sit right after file_id
            other_cols = [c for c in df.columns
                          if c not in ('file_id', 'chirp_idx', 'is_last', 'species')]
            df = df[['file_id', 'chirp_idx', 'is_last', 'species'] + other_cols]
            df.to_csv(self.out_csv, index=False)

        return ExtractionResult(
            file_records = file_recs,
            out_csv      = self.out_csv.resolve(),
            n_rows       = n_rows,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments.

    :return: ``args`` namespace (``args.inputs`` is a list of raw Path objects).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='chirp_measures_extraction',
        description=(
            'Extract acoustic measures from bat .wav files.\n\n'
            'Inputs can be any mix of:\n'
            '  • individual .wav files\n'
            '  • directories (searched for .wav files; use -r to recurse)\n'
            '  • .db SQLite databases built by create_wav_file_db.py\n'
            'Output: CSV with one row per detected chirp'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'input',
        nargs='+',
        help=(
            'one or more .wav files, directories, or .db SQLite databases.\n'
            'Directories are searched at top level only; use -r to recurse.'
        ),
    )
    parser.add_argument(
        '-o', '--out-csv',
        required=True,
        help='destination CSV path for extracted measures',
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='descend into subdirectories when a directory is given',
    )
    parser.add_argument(
        '-w', '--workers',
        type=int, default=None,
        help='worker processes (default: cpu_count − 4; 0 = all cores)',
    )
    parser.add_argument(
        '--timeout',
        type=float, default=120.0,
        metavar='SECS',
        help='per-file worker timeout in seconds (default: 120)',
    )
    args = parser.parse_args()

    # Validate inputs exist; warn on unrecognised extensions but still pass
    # them through so _iter_paths() can emit its own warning with full context.
    inputs: list[Path] = []
    for item in args.input:
        p = Path(item)
        suffix = p.suffix.lower()
        if not p.exists() and suffix not in ('.wav', '.db'):
            print(f"Warning: '{item}' does not exist and is not a recognised type — skipping",
                  file=sys.stderr)
            continue
        inputs.append(p)

    if not inputs:
        parser.error('No valid inputs found.')

    args.inputs = inputs
    return args


def main() -> None:
    """
    CLI entry point: extract measures and write CSV.
    """
    args = _parse_args()
    log.info(f'MeasureExtractor: {len(args.inputs)} input(s)  →  {args.out_csv}')

    extractor = MeasureExtractor(
        inputs         = args.inputs,
        out_csv        = args.out_csv,
        recursive      = args.recursive,
        n_workers      = args.workers,
        show_progress  = True,
        worker_timeout = args.timeout,
    )
    result = extractor.run()
    print(result.summary())
    sys.exit(0 if result.n_rows > 0 else 1)


if __name__ == '__main__':
    main()