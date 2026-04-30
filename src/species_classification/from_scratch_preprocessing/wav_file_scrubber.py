#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-07 16:37:48
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/chirp_detection/wav_file_scrubber.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-30 09:55:57
#
# **********************************************************

"""
wav_file_scrubber.py
===============
Filter a collection of ultrasound bat-detector ``.wav`` files, retaining only
those that contain at least *min_pulses* plausible bat echolocation pulses.

The scrubber replicates and extends SonoBat's Batch Scrubber logic:

SonoBat criteria (replicated)
------------------------------
* Minimum pulse count above –60 dBFS in the bat-band (15–120 kHz).
* Rejection of distorted / clipped recordings.
* Rejection of low-quality calls (too short, too long, insufficient bandwidth).

Additional criteria (extensions)
----------------------------------
* Explicit sample-rate check: files below 80 kHz are not ultrasound recordings.
* Duration sanity: files longer than ``max_duration_s`` are flagged (default
  60 s; your 10-second files will never hit this).
* Pulse duration gate: candidate pulses outside [2, 35] ms are excluded.
* Inter-Pulse Interval (IPI) regularity: a bat pass has a rhythmic pulse train
  (IPI typically 30–500 ms in search phase).  When the detected pulse onsets
  are present but their timing is highly irregular (coefficient of variation
  > ``max_ipi_cv``) the file is rejected as noise.
* Bandwidth gate: each retained pulse must have a spectral bandwidth > 5 kHz
  at −20 dB from its peak; flat tones (electrical interference, insects) are
  excluded.
* Clipping gate: if more than ``max_clip_fraction`` of samples are at ±full-
  scale the recording is too distorted to trust.

Parallelism
-----------
``WavScrubber.run()`` uses :class:`concurrent.futures.ProcessPoolExecutor`
with ``n_workers`` worker processes (default: all logical CPUs).  Each worker
calls ``_scrub_one()`` on a single file.

GPU acceleration
----------------
When PyTorch is available **and** a CUDA device is present, the STFT used
for the bat-band energy envelope is computed on the GPU via
``torchaudio.transforms.Spectrogram``.  If no GPU is found the code falls
back transparently to ``scipy.signal.spectrogram`` on the CPU.  Pass
``use_gpu=False`` to force CPU mode.

Typical usage
-------------
    from wav_scrubber import WavScrubber

    scrubber = WavScrubber(wav_paths, min_pulses=5, n_workers=8)
    result   = scrubber.run()

    print(result.summary())
    result.to_csv("scrub_report.csv")

    retained = result.retained   # list[Path]
    scrubbed = result.scrubbed   # list[ScrubRecord]
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import auto
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from enum import StrEnum          # Python 3.11+

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# StrEnum: rejection reasons
# ---------------------------------------------------------------------------

class ScrubReason(StrEnum):
    """Structured reason a file was moved to the scrubbed list."""
    RETAINED          = auto()   # file passed all checks
    UNREADABLE        = auto()   # corrupt or unreadable WAV
    LOW_SAMPLE_RATE   = auto()   # sample rate < 80 kHz → not ultrasound
    TOO_LONG          = auto()   # duration > max_duration_s
    CLIPPED           = auto()   # > max_clip_fraction at ±full-scale
    NO_BAT_BAND_ENERGY = auto()  # peak amplitude in bat band < –60 dBFS
    TOO_FEW_PULSES    = auto()   # fewer than min_pulses valid pulses found
    IRREGULAR_IPI     = auto()   # pulse timing too random (CV > max_ipi_cv)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ScrubInterrupted(Exception):
    """Raised by WavScrubber.run when interrupted by Ctrl-C.

    The partial results completed before the interrupt are available via
    the ``partial_result`` attribute.

    :param partial_result: ScrubResult for all files that finished.
    :param checkpoint_csv: Path to the checkpoint CSV, or None.
    """

    def __init__(self, partial_result, checkpoint_csv):
        """
        :param partial_result: :class:`ScrubResult` with completed records.
        :param checkpoint_csv: Checkpoint path or ``None``.
        """
        self.partial_result = partial_result
        self.checkpoint_csv = checkpoint_csv
        super().__init__("Scrub run interrupted by user")


# ---------------------------------------------------------------------------
# Dataclasses: per-file record and aggregate result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScrubRecord:
    """
    Scrub outcome for a single ``.wav`` file.

    :param path:        Absolute path to the file.
    :param verdict:     :attr:`ScrubReason.RETAINED` or a rejection reason.
    :param pulse_count: Number of valid bat-band pulses detected (0 if rejected
                        before pulse detection).
    :param peak_db:     Peak amplitude in the bat band, dBFS.  ``None`` if the
                        file could not be read.
    :param ipi_cv:      Coefficient of variation of inter-pulse intervals.
                        ``None`` when fewer than two pulses were found.
    :param sample_rate: Sample rate of the file in Hz.  ``None`` if unreadable.
    :param duration_s:  Duration of the file in seconds.  ``None`` if unreadable.
    :param detail:      Free-form detail string (e.g. exception message).
    """
    path:        Path
    verdict:     ScrubReason
    pulse_count: int               = 0
    peak_db:     Optional[float]   = None
    ipi_cv:      Optional[float]   = None
    sample_rate: Optional[int]     = None
    duration_s:  Optional[float]   = None
    detail:      str               = ""

    @property
    def retained(self) -> bool:
        """:return: ``True`` if this file passed all scrub checks."""
        return self.verdict == ScrubReason.RETAINED


@dataclasses.dataclass
class ScrubResult:
    """
    Aggregate outcome of a :class:`WavScrubber` run.

    :param records:  One :class:`ScrubRecord` per input file.
    """
    records: List[ScrubRecord]

    # ------------------------------------------------------------------ #
    #  Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    @property
    def retained(self) -> List[Path]:
        """
        :return: Paths of files that passed all scrub checks.
        """
        return [r.path for r in self.records if r.retained]

    @property
    def scrubbed(self) -> List[ScrubRecord]:
        """
        :return: Records for files that were rejected.
        """
        return [r for r in self.records if not r.retained]

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """
        Return a human-readable one-paragraph summary of scrub outcomes.

        :return: Multi-line summary string.
        """
        total    = len(self.records)
        n_kept   = len(self.retained)
        n_scrub  = total - n_kept
        by_reason: dict[str, int] = {}
        for r in self.scrubbed:
            by_reason[r.verdict] = by_reason.get(r.verdict, 0) + 1
        lines = [
            f"WavScrubber: {total} files processed",
            f"  Retained : {n_kept}",
            f"  Scrubbed : {n_scrub}",
        ]
        for reason, count in sorted(by_reason.items()):
            lines.append(f"    {reason:<25} {count}")
        return "\n".join(lines)

    def to_csv(self, outfile: str | Path) -> Path:
        """
        Write all records to a CSV file.

        Columns: path, verdict, pulse_count, peak_db, ipi_cv,
                 sample_rate, duration_s, detail.

        :param outfile: Destination path (created / overwritten).
        :return:        Resolved output path.
        """
        p = Path(outfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        fields = [f.name for f in dataclasses.fields(ScrubRecord)]
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for rec in self.records:
                row = dataclasses.asdict(rec)
                row["path"] = str(rec.path)
                w.writerow(row)
        return p.resolve()


# ---------------------------------------------------------------------------
# Per-file scrub logic (module-level so it is picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

# These constants are intentionally module-level so worker processes do not
# need to import a class or carry state.

#: Lowest frequency (kHz) considered part of the bat echolocation band.
_BAT_BAND_LO_KHZ: float = 15.0

#: Highest frequency (kHz) considered part of the bat echolocation band.
_BAT_BAND_HI_KHZ: float = 120.0

#: Amplitude threshold below which a candidate pulse is ignored (dBFS).
_PULSE_THRESHOLD_DBFS: float = -60.0

#: Minimum valid pulse duration in milliseconds.
#: Small Myotis species produce FM sweeps as short as 0.3 ms; 0.5 ms is a
#: conservative floor that rejects single-frame STFT noise spikes without
#: losing genuine short calls.
_PULSE_DUR_MIN_MS: float = 0.5

#: Maximum valid pulse duration in milliseconds.
_PULSE_DUR_MAX_MS: float = 35.0

#: Maximum gap between consecutive above-threshold segments (ms) that is
#: still considered part of the same pulse.  Short FM sweeps often dip below
#: the energy threshold between harmonics or during a rapid frequency drop,
#: fragmenting a single call into several micro-segments.  Segments separated
#: by less than this gap are merged before the duration/bandwidth filter is
#: applied.  Set to 0.0 to disable merging.
_PULSE_MERGE_GAP_MS: float = 5.0

#: Minimum spectral bandwidth (kHz) a pulse must have at −20 dB from its peak.
_MIN_BANDWIDTH_KHZ: float = 5.0

#: Maximum fraction of samples allowed at ±full-scale before flagging clipping.
_MAX_CLIP_FRACTION: float = 0.005   # 0.5 %

#: Sample rates (Hz) at or above this value are treated as direct-recording
#: ultrasound.  Rates below this threshold are assumed to be time-expanded
#: files whose header SR has been divided by ``_TIME_EXPAND_FACTOR``.
#: Known direct-recording rates (256k, 384k, 500k) are well above 80 kHz;
#: known TE rates (8k, 22.05k, 25k, 44.1k, 48k, 50k) are well below it,
#: so the boundary is unambiguous for all off-the-shelf bat detectors.
_TE_SR_THRESHOLD_HZ: int = 80_000

#: Factor by which time-expanded files slow the recording.  The header
#: sample rate is multiplied by this value to recover the true ultrasound
#: rate before any analysis.
#: Override at the class level (WavScrubber._TIME_EXPAND_FACTOR) if your
#: detector uses a different expansion ratio.
_TIME_EXPAND_FACTOR: int = 10

#: Maximum file duration in seconds (measured at the *true* sample rate
#: after any time-expansion correction).
_MAX_DURATION_S: float = 60.0


def _scrub_one(
    path: Path,
    min_pulses: int,
    max_ipi_cv: float,
    use_gpu: bool,
    ipi_freq_floor_hz: float = 800.0,
) -> ScrubRecord:
    """
    Apply all scrub checks to a single ``.wav`` file and return a
    :class:`ScrubRecord`.

    This function is deliberately module-level (not a method) so that
    :class:`ProcessPoolExecutor` can pickle it for dispatch to worker
    processes.

    :param path:               Path to the ``.wav`` file.
    :param min_pulses:         Minimum number of valid bat-band pulses required.
    :param max_ipi_cv:         Maximum allowed coefficient of variation for
                               inter-pulse intervals.
    :param use_gpu:            If ``True``, attempt GPU-accelerated STFT; fall
                               back to CPU on failure.
    :param ipi_freq_floor_hz:  Low-frequency cutoff (Hz, at the *true* sample
                               rate after any TE correction) applied to the
                               energy envelope used for IPI pulse detection.
                               Frequencies below this value are excluded from
                               the envelope sum so that sub-bat interference
                               (e.g. frog choruses at marsh sites) cannot
                               corrupt the IPI regularity check.  The full
                               bat-band spectrogram is still used for bandwidth
                               filtering and all other checks.
                               Default: 800 Hz (true), safely below the lowest
                               bat species in the label set (~12 kHz for Anpa,
                               ~20 kHz for Coto) and well above the frog
                               interference observed in marsh recordings
                               (< 600 Hz true).
    :return:                   :class:`ScrubRecord` with verdict and diagnostics.
    """
    rec = ScrubRecord(path=path.resolve(), verdict=ScrubReason.RETAINED)

    # ------------------------------------------------------------------ #
    #  1. Load WAV                                                         #
    # ------------------------------------------------------------------ #
    try:
        sr, data = wavfile.read(str(path))
    except Exception as exc:
        rec.verdict = ScrubReason.UNREADABLE
        rec.detail  = str(exc)
        return rec

    rec.sample_rate = int(sr)

    # Flatten to mono by averaging channels
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalise to float32 in [−1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2_147_483_648.0
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32)
    else:
        audio = data.astype(np.float32)
        peak  = np.abs(audio).max()
        if peak > 0:
            audio /= peak

    # ------------------------------------------------------------------ #
    #  2. Time-expansion detection and sample-rate correction             #
    # ------------------------------------------------------------------ #
    # Off-the-shelf bat detectors write time-expanded (TE) files with the
    # header SR divided by _TIME_EXPAND_FACTOR (e.g. 250 kHz → 25 kHz).
    # The PCM samples are correct; only the header rate is scaled down.
    # We detect TE files by the header SR being below _TE_SR_THRESHOLD_HZ
    # and recover the true ultrasound rate by multiplying back up.
    # If the corrected rate is still below the threshold the file is
    # genuinely not an ultrasound recording and we reject it.
    time_expanded = False
    if sr < _TE_SR_THRESHOLD_HZ:
        corrected_sr = sr * _TIME_EXPAND_FACTOR
        if corrected_sr < _TE_SR_THRESHOLD_HZ:
            rec.verdict = ScrubReason.LOW_SAMPLE_RATE
            rec.detail  = (
                f"sample_rate={sr} Hz; even after ×{_TIME_EXPAND_FACTOR} "
                f"correction ({corrected_sr} Hz) below "
                f"{_TE_SR_THRESHOLD_HZ} Hz threshold"
            )
            return rec
        sr           = corrected_sr
        time_expanded = True

    rec.sample_rate = sr   # store corrected rate
    rec.duration_s  = len(audio) / sr

    # ------------------------------------------------------------------ #
    #  3. Duration gate                                                    #
    # ------------------------------------------------------------------ #
    if rec.duration_s > _MAX_DURATION_S:
        rec.verdict = ScrubReason.TOO_LONG
        rec.detail  = (
            f"duration={rec.duration_s:.1f} s > {_MAX_DURATION_S} s"
            + (" (after TE correction)" if time_expanded else "")
        )
        return rec

    # ------------------------------------------------------------------ #
    #  4. Clipping gate                                                    #
    # ------------------------------------------------------------------ #
    clip_frac = float(np.mean(np.abs(audio) >= 0.9999))
    if clip_frac > _MAX_CLIP_FRACTION:
        rec.verdict = ScrubReason.CLIPPED
        rec.detail  = f"clip_fraction={clip_frac:.4f}"
        return rec

    # ------------------------------------------------------------------ #
    #  5. Bat-band STFT                                                    #
    # ------------------------------------------------------------------ #
    # Window: ~0.5 ms gives good time resolution for sub-ms to 35 ms calls.
    nperseg  = max(64, int(round(0.0005 * sr)))
    noverlap = nperseg * 3 // 4

    freqs_hz, times_s, Sxx = _compute_spectrogram(
        audio, sr, nperseg, noverlap, use_gpu
    )

    # Restrict to bat band
    bat_mask  = (freqs_hz >= _BAT_BAND_LO_KHZ * 1000) & \
                (freqs_hz <= _BAT_BAND_HI_KHZ * 1000)
    Sxx_bat   = Sxx[bat_mask, :]           # shape: (n_bat_freqs, n_times)
    freqs_bat = freqs_hz[bat_mask]

    if Sxx_bat.size == 0:
        rec.verdict = ScrubReason.NO_BAT_BAND_ENERGY
        rec.detail  = "bat band above Nyquist for this sample rate"
        return rec

    # ------------------------------------------------------------------ #
    #  6. Peak amplitude in bat band                                       #
    # ------------------------------------------------------------------ #
    # Convert power spectrum to dBFS (relative to a full-scale sine)
    peak_power  = float(Sxx_bat.max())
    peak_db     = 10.0 * np.log10(peak_power + 1e-12)
    rec.peak_db = peak_db

    if peak_db < _PULSE_THRESHOLD_DBFS:
        rec.verdict = ScrubReason.NO_BAT_BAND_ENERGY
        rec.detail  = f"peak_bat_band={peak_db:.1f} dBFS < {_PULSE_THRESHOLD_DBFS} dBFS"
        return rec

    # ------------------------------------------------------------------ #
    #  7. Pulse detection on bat-band energy envelope                     #
    # ------------------------------------------------------------------ #
    # For IPI detection, restrict to frequencies above ipi_freq_floor_hz.
    # This excludes sub-bat interference (e.g. frog choruses in marsh
    # habitat that sit solidly below 600 Hz true) without affecting the
    # bandwidth check in step 8 which still uses the full Sxx_bat.
    ipi_mask  = freqs_bat >= ipi_freq_floor_hz
    Sxx_ipi   = Sxx_bat[ipi_mask, :] if ipi_mask.any() else Sxx_bat
    # Collapse spectrogram to 1-D energy envelope (sum across IPI freqs)
    energy     = Sxx_ipi.sum(axis=0)                   # (n_times,)
    energy_db  = 10.0 * np.log10(energy + 1e-12)
 
    # Scale threshold by Sxx_ipi bin count, not Sxx_bat — the energy being
    # compared is the sum over IPI-band bins only, so the threshold must
    # match that reduced bin count.  Using Sxx_bat.shape[0] here would set
    # the bar too high and suppress detection of real bat pulses.
    threshold_linear = 10.0 ** (_PULSE_THRESHOLD_DBFS / 10.0) * Sxx_ipi.shape[0]    

    above            = energy >= threshold_linear

    # Find contiguous ON-segments, then merge fragments separated by a short
    # gap (e.g. a single sweep whose energy dips momentarily between harmonics).
    pulses = _find_segments(above, times_s)
    pulses = _merge_segments(pulses, _PULSE_MERGE_GAP_MS / 1000.0)
    # pulses: list of (onset_s, offset_s)

    # ------------------------------------------------------------------ #
    #  8. Pulse duration and bandwidth filter                              #
    # ------------------------------------------------------------------ #
    valid_pulses: list[tuple[float, float]] = []
    for onset_s, offset_s in pulses:
        dur_ms = (offset_s - onset_s) * 1000.0
        if not (_PULSE_DUR_MIN_MS <= dur_ms <= _PULSE_DUR_MAX_MS):
            continue

        # Bandwidth check: find freq extent at −20 dB from peak in this window
        t_mask  = (times_s >= onset_s) & (times_s <= offset_s)
        if t_mask.sum() == 0:
            continue
        pulse_spec = Sxx_bat[:, t_mask].max(axis=1)   # max power per freq bin
        peak_p     = pulse_spec.max()
        if peak_p <= 0:
            continue
        bw_mask    = pulse_spec >= peak_p * 10 ** (-20.0 / 10.0)  # −20 dB
        if bw_mask.sum() == 0:
            continue
        bw_khz = (freqs_bat[bw_mask].max() - freqs_bat[bw_mask].min()) / 1000.0
        if bw_khz < _MIN_BANDWIDTH_KHZ:
            continue

        valid_pulses.append((onset_s, offset_s))

    rec.pulse_count = len(valid_pulses)

    if rec.pulse_count < min_pulses:
        rec.verdict = ScrubReason.TOO_FEW_PULSES
        rec.detail  = (
            f"valid_pulses={rec.pulse_count} < min_pulses={min_pulses}"
        )
        return rec

    # ------------------------------------------------------------------ #
    #  9. IPI regularity check (requires ≥ 2 pulses)                      #
    # ------------------------------------------------------------------ #
    if len(valid_pulses) >= 2:
        onsets = np.array([p[0] for p in valid_pulses])
        ipis   = np.diff(onsets)             # inter-pulse intervals (s)
        mean_ipi = ipis.mean()
        if mean_ipi > 0:
            ipi_cv     = float(ipis.std() / mean_ipi)
            rec.ipi_cv = ipi_cv
            if ipi_cv > max_ipi_cv:
                rec.verdict = ScrubReason.IRREGULAR_IPI
                rec.detail  = (
                    f"ipi_cv={ipi_cv:.2f} > max_ipi_cv={max_ipi_cv:.2f}"
                )
                return rec

    # ------------------------------------------------------------------ #
    #  All checks passed                                                   #
    # ------------------------------------------------------------------ #
    rec.verdict = ScrubReason.RETAINED
    return rec


def _compute_spectrogram(
    audio: np.ndarray,
    sr: int,
    nperseg: int,
    noverlap: int,
    use_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a power spectrogram, using GPU if available and requested.

    :param audio:    Float32 waveform in [−1, 1].
    :param sr:       Sample rate in Hz.
    :param nperseg:  STFT window length in samples.
    :param noverlap: Number of overlapping samples.
    :param use_gpu:  If ``True``, attempt ``torchaudio`` GPU path.
    :return:         Tuple of ``(freqs_hz, times_s, Sxx)`` where ``Sxx``
                     has shape ``(n_freqs, n_times)`` and values are linear
                     power.
    """
    if use_gpu:
        try:
            return _compute_spectrogram_gpu(audio, sr, nperseg, noverlap)
        except Exception:
            pass   # silent fall-through to CPU path

    return _compute_spectrogram_cpu(audio, sr, nperseg, noverlap)


def _compute_spectrogram_cpu(
    audio: np.ndarray,
    sr: int,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CPU path: ``scipy.signal.spectrogram`` with a Hann window.

    :param audio:    Float32 waveform.
    :param sr:       Sample rate (Hz).
    :param nperseg:  Window length (samples).
    :param noverlap: Overlap (samples).
    :return:         ``(freqs_hz, times_s, Sxx_linear_power)``
    """
    freqs, times, Sxx = signal.spectrogram(
        audio, fs=sr,
        window='hann', nperseg=nperseg, noverlap=noverlap,
        scaling='spectrum',
    )
    return freqs.astype(np.float32), times.astype(np.float32), Sxx.astype(np.float32)


def _compute_spectrogram_gpu(
    audio: np.ndarray,
    sr: int,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU path: ``torchaudio.transforms.Spectrogram`` on CUDA.

    Raises any exception on failure so the caller can fall back to CPU.

    :param audio:    Float32 waveform.
    :param sr:       Sample rate (Hz).
    :param nperseg:  Window length (samples).
    :param noverlap: Overlap (samples).
    :return:         ``(freqs_hz, times_s, Sxx_linear_power)``
    """
    import torch
    import torchaudio.transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")

    device   = torch.device("cuda")
    hop      = nperseg - noverlap
    t_audio  = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, N)

    transform = T.Spectrogram(
        n_fft=nperseg, hop_length=hop, win_length=nperseg,
        window_fn=torch.hann_window, power=2.0, normalized=False,
    ).to(device)

    with torch.no_grad():
        Sxx_t = transform(t_audio).squeeze(0)   # (n_freqs, n_times)

    Sxx    = Sxx_t.cpu().numpy()
    n_freq = Sxx.shape[0]
    freqs  = np.linspace(0, sr / 2, n_freq, dtype=np.float32)
    n_time = Sxx.shape[1]
    times  = np.arange(n_time, dtype=np.float32) * hop / sr

    return freqs, times, Sxx.astype(np.float32)


def _merge_segments(
    segments: list[tuple[float, float]],
    max_gap_s: float,
) -> list[tuple[float, float]]:
    """
    Merge consecutive segments whose inter-segment gap is ≤ *max_gap_s*.

    A single FM sweep often fragments into several above-threshold runs when
    the energy dips briefly between harmonics or at the low-frequency tail of
    the sweep.  Merging restores each physical call to one segment before the
    duration and bandwidth filters are applied.

    Note: this does **not** merge segments that are far apart in time — the
    gap threshold is intentionally small (default 5 ms) so that distinct
    pulses in a rapid sequence (inter-pulse intervals down to ~20 ms) are
    never joined together.

    :param segments:  List of ``(onset_s, offset_s)`` tuples, sorted by onset.
    :param max_gap_s: Maximum gap in seconds to bridge.  Pass ``0.0`` to
                      disable merging.
    :return:          Merged list of ``(onset_s, offset_s)`` tuples.
    """
    if not segments or max_gap_s <= 0.0:
        return segments
    merged: list[tuple[float, float]] = [segments[0]]
    for onset, offset in segments[1:]:
        prev_onset, prev_offset = merged[-1]
        if onset - prev_offset <= max_gap_s:
            merged[-1] = (prev_onset, max(prev_offset, offset))
        else:
            merged.append((onset, offset))
    return merged



def _find_segments(
    above: np.ndarray,
    times_s: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Find contiguous ``True`` runs in a boolean array and return their
    onset/offset times.

    :param above:   Boolean array, ``True`` where energy exceeds threshold.
    :param times_s: Time axis in seconds, same length as *above*.
    :return:        List of ``(onset_s, offset_s)`` tuples.
    """
    segments: list[tuple[float, float]] = []
    in_seg   = False
    onset    = 0.0
    for i, val in enumerate(above):
        if val and not in_seg:
            onset  = float(times_s[i])
            in_seg = True
        elif not val and in_seg:
            segments.append((onset, float(times_s[i - 1])))
            in_seg = False
    if in_seg:
        segments.append((onset, float(times_s[-1])))
    return segments


# ---------------------------------------------------------------------------
# WavScrubber
# ---------------------------------------------------------------------------

class WavScrubber:
    """
    Parallel scrubber that filters a list of ``.wav`` files for bat content.

    A file is **retained** when it passes all of:

    1. Readable as a WAV file.
    2. Sample rate ≥ 80 kHz (ultrasound recording).
    3. Duration ≤ ``max_duration_s`` (default 60 s).
    4. Clipping fraction ≤ 0.5 % of samples.
    5. Peak amplitude in the 15–120 kHz bat band ≥ –60 dBFS.
    6. At least ``min_pulses`` pulses with duration in [2, 35] ms and
       bandwidth > 5 kHz at –20 dB.
    7. Inter-pulse interval coefficient of variation ≤ ``max_ipi_cv``
       (rhythmic pulse train, not random noise events).

    Processing is parallelised across ``n_workers`` CPU processes.  STFT
    computation is optionally GPU-accelerated via ``torchaudio`` when
    ``use_gpu=True`` and a CUDA device is present; falls back to
    ``scipy`` on CPU otherwise.

    :param wav_paths:      Iterable of paths to ``.wav`` files.
    :param min_pulses:     Minimum valid pulses required to retain a file.
    :param max_ipi_cv:     Maximum inter-pulse interval coefficient of
                           variation (dimensionless).  Higher → more
                           tolerant of irregular spacing.
    :param n_workers:      Worker processes.  Defaults to ``os.cpu_count()``.
    :param use_gpu:        Attempt GPU-accelerated STFT (requires PyTorch +
                           torchaudio with CUDA).  Falls back silently.
    :param max_duration_s: Files longer than this are rejected immediately.
    :param show_progress:  If ``True``, print a progress line to stdout.
    """

    # ------------------------------------------------------------------ #
    #  Class-level tunables                                               #
    # ------------------------------------------------------------------ #

    #: Files with header SR below this value are treated as time-expanded.
    #: Matches the module-level ``_TE_SR_THRESHOLD_HZ`` used in workers;
    #: override here if your detector uses a non-standard boundary.
    _TE_SR_THRESHOLD_HZ: int = _TE_SR_THRESHOLD_HZ

    #: Expansion factor applied to time-expanded file header rates.
    #: Change to 2 or 4 for detectors that use a different TE ratio.
    _TIME_EXPAND_FACTOR: int = _TIME_EXPAND_FACTOR

    #: Maximum expected recording duration in seconds (at the true
    #: ultrasound rate after any TE correction).  Files longer than this
    #: are rejected as TOO_LONG.
    _MAX_DURATION_S: float = _MAX_DURATION_S

    def __init__(
        self,
        wav_paths: Sequence[str | Path],
        min_pulses:          int            = 5,
        max_ipi_cv:          float          = 1.5,
        ipi_freq_floor_hz:   float          = 800.0,
        n_workers:           Optional[int]  = None,
        use_gpu:             bool           = True,
        max_duration_s:      float          = 60.0,
        show_progress:       bool           = True,
        checkpoint_csv:      Optional[str | Path] = None,
        worker_timeout:      Optional[float] = 120.0,
    ) -> None:
        """
        :param wav_paths:           Paths to ``.wav`` files to evaluate.
        :param min_pulses:          Minimum valid bat-band pulses to retain a
                                    file.
        :param max_ipi_cv:          IPI regularity tolerance.
        :param ipi_freq_floor_hz:   Low-frequency cutoff (Hz, true rate) for
                                    the IPI pulse-detection envelope.
                                    Excludes sub-bat noise sources such as frog
                                    choruses from the regularity check without
                                    affecting bandwidth filtering or any other
                                    check.  Default: 800 Hz.
        :param n_workers:           Worker processes.  ``None`` (default)
                                    reserves 4 cores for the OS and interactive
                                    use; pass ``0`` to use every available core.
        :param use_gpu:             Enable GPU-accelerated STFT.
        :param max_duration_s:      Files longer than this are rejected.
        :param show_progress:       Show a tqdm progress bar (falls back to
                                    periodic print lines if tqdm is not
                                    installed).
        :param checkpoint_csv:      Path to a CSV file used for incremental
                                    checkpointing.  If the file already exists,
                                    any paths already present in it are skipped
                                    (resume after interruption).  New results
                                    are appended row-by-row as they complete so
                                    that progress is never lost.
        :param worker_timeout:      Seconds to wait for a single worker result
                                    before marking the file as UNREADABLE.
                                    Protects against worker processes that hang
                                    on corrupt files.  ``None`` disables the
                                    timeout.
        """

        self.wav_paths          = [Path(p) for p in wav_paths]
        self.min_pulses         = min_pulses
        self.max_ipi_cv         = max_ipi_cv
        self.ipi_freq_floor_hz  = ipi_freq_floor_hz
        # Default: leave 4 cores free for the OS, shell, and interactive
        # use.  On a 48-core machine this gives 44 workers; on an 8-core
        # laptop it gives 4.  Pass n_workers=0 to use every core.
        if n_workers is None:
            self.n_workers = max(1, (os.cpu_count() or 1) - 4)
        elif n_workers == 0:
            self.n_workers = os.cpu_count() or 1
        else:
            self.n_workers = n_workers
        self.use_gpu        = use_gpu
        self.max_duration_s = max_duration_s
        self.show_progress  = show_progress
        if checkpoint_csv is not None:
            self.checkpoint_csv = Path(checkpoint_csv)
        else:
            # Default: a timestamped CSV in the current working directory.
            # Using cwd() rather than __file__ so the file appears where
            # the user is running from, not buried in the package tree.
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            self.checkpoint_csv = Path.cwd() / f"scrub_checkpoint_{ts}.csv"
        self.worker_timeout = worker_timeout

        # Resolve GPU availability once at construction
        self._gpu_available = self._check_gpu() if use_gpu else False

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> ScrubResult:
        """
        Scrub all files in parallel and return a :class:`ScrubResult`.

        Checkpointing
        -------------
        If ``checkpoint_csv`` was set at construction, any paths already
        recorded in that file are skipped and their existing records are loaded
        back into the result.  New results are appended to the CSV as each
        worker completes, so an interrupted run can be resumed by re-running
        with the same ``checkpoint_csv`` path and the same input file list.

        :return: :class:`ScrubResult` containing one :class:`ScrubRecord`
                 per input file, in input order.
        """
        # ── load checkpoint ───────────────────────────────────────────────
        completed: dict[Path, ScrubRecord] = {}
        checkpoint_fh = None
        checkpoint_writer = None

        if self.checkpoint_csv is not None:
            completed = self._load_checkpoint(self.checkpoint_csv)
            if completed:
                log.info(f"WavScrubber: resuming — {len(completed)} files already in checkpoint")
            # Open for append so new results land immediately on disk.
            # newline='' is required by csv.DictWriter on all platforms.
            self.checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.checkpoint_csv.exists() or self.checkpoint_csv.stat().st_size == 0
            checkpoint_fh = self.checkpoint_csv.open("a", newline="")
            fields = [f.name for f in dataclasses.fields(ScrubRecord)]
            checkpoint_writer = csv.DictWriter(checkpoint_fh, fieldnames=fields)
            if write_header:
                checkpoint_writer.writeheader()
                checkpoint_fh.flush()

        # ── partition into todo vs already done ───────────────────────────
        todo  = [p for p in self.wav_paths if p.resolve() not in completed]
        total = len(self.wav_paths)
        skip  = total - len(todo)
        if skip:
            print(f"WavScrubber: skipping {skip} already-checkpointed files")

        # ── idx_map preserves original input order in the final list ──────
        idx_map = {path.resolve(): i for i, path in enumerate(self.wav_paths)}
        records: list[ScrubRecord | None] = [None] * total

        # Slot in previously completed records
        for resolved_path, rec in completed.items():
            if resolved_path in idx_map:
                records[idx_map[resolved_path]] = rec

        # ── progress bar ──────────────────────────────────────────────────
        if self.show_progress and _TQDM_AVAILABLE:
            pbar = _tqdm(
                total=total, initial=skip, unit="file",
                desc="scrubbing", dynamic_ncols=True,
            )
        else:
            pbar = None

        # How many futures to keep in flight at once.  Submitting all 600k
        # paths upfront would flood the executor's work queue with 600k
        # Future objects and 600k Path copies before a single result is read
        # back.  A window of 4× workers keeps the queue full without
        # materialising the entire todo list in memory simultaneously.
        # Ctrl-C also drains much faster with a bounded queue.
        _QUEUE_MULTIPLIER = 4
        window = self.n_workers * _QUEUE_MULTIPLIER

        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Seed the initial window of futures
                todo_iter   = iter(todo)
                in_flight: dict = {}   # future → path

                def _submit_next() -> bool:
                    """Submit one more path from todo_iter; return False if exhausted."""
                    try:
                        p = next(todo_iter)
                    except StopIteration:
                        return False
                    f = executor.submit(
                        _scrub_one, p,
                        self.min_pulses, self.max_ipi_cv, self._gpu_available,
                        self.ipi_freq_floor_hz,
                    )
                    in_flight[f] = p
                    return True

                # Fill the initial window
                for _ in range(window):
                    if not _submit_next():
                        break

                done_this_run = 0
                while in_flight:
                    # as_completed yields the next finished future
                    for future in as_completed(in_flight):
                        path = in_flight.pop(future)

                        try:
                            rec = future.result(timeout=self.worker_timeout)
                        except TimeoutError:
                            rec = ScrubRecord(
                                path    = path.resolve(),
                                verdict = ScrubReason.UNREADABLE,
                                detail  = (
                                    f"worker timed out after "
                                    f"{self.worker_timeout:.0f}s"
                                ),
                            )
                        except Exception as exc:
                            rec = ScrubRecord(
                                path    = path.resolve(),
                                verdict = ScrubReason.UNREADABLE,
                                detail  = f"worker exception: {exc}",
                            )

                        records[idx_map[path.resolve()]] = rec
                        done_this_run += 1

                        # ── checkpoint: append immediately ────────────────
                        if checkpoint_writer is not None:
                            row = dataclasses.asdict(rec)
                            row["path"] = str(rec.path)
                            checkpoint_writer.writerow(row)
                            checkpoint_fh.flush()

                        # ── refill window ────────────────────────────────
                        _submit_next()

                        # ── progress ─────────────────────────────────────
                        if pbar is not None:
                            pbar.update(1)
                        elif self.show_progress:
                            done_total = skip + done_this_run
                            if done_total % 1000 == 0 or done_total == total:
                                pct = 100 * done_total / total
                                print(
                                    f"  scrubbing {done_total}/{total}"
                                    f"  ({pct:.0f}%)",
                                    flush=True,
                                )

                        # as_completed only yields one future per call when
                        # used this way; break back to the while loop so we
                        # re-enter as_completed with the updated in_flight dict.
                        break

        except KeyboardInterrupt:
            # Close gracefully before raising so the terminal is clean
            # and no checkpoint data is lost.
            if pbar is not None:
                pbar.close()
            if checkpoint_fh is not None:
                checkpoint_fh.close()
            partial = ScrubResult(
                records=[r for r in records if r is not None]
            )
            raise ScrubInterrupted(partial, self.checkpoint_csv) from None

        finally:
            if pbar is not None:
                pbar.close()
            if checkpoint_fh is not None:
                checkpoint_fh.close()

        return ScrubResult(records=records)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_checkpoint(csv_path: Path) -> dict[Path, ScrubRecord]:
        """
        Read a checkpoint CSV and return a mapping of resolved path →
        :class:`ScrubRecord` for every row found.

        Missing or unreadable checkpoint files return an empty dict.

        :param csv_path: Path to the checkpoint CSV written by :meth:`run`.
        :return:         Dict of ``{resolved_path: ScrubRecord}``.
        """
        result: dict[Path, ScrubRecord] = {}
        if not csv_path.exists():
            return result
        fields = {f.name: f for f in dataclasses.fields(ScrubRecord)}
        try:
            with csv_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        kwargs = {}
                        for name, field in fields.items():
                            raw = row.get(name, "")
                            if raw == "" or raw is None:
                                kwargs[name] = None
                            elif field.type in ("Optional[float]", "float"):
                                kwargs[name] = float(raw)
                            elif field.type in ("Optional[int]", "int"):
                                kwargs[name] = int(raw)
                            elif name == "verdict":
                                kwargs[name] = ScrubReason(raw)
                            elif name == "path":
                                kwargs[name] = Path(raw)
                            else:
                                kwargs[name] = raw
                        rec = ScrubRecord(**kwargs)
                        result[rec.path.resolve()] = rec
                    except Exception:
                        pass  # skip malformed rows
        except Exception as exc:
            log.warn(f"WavScrubber: could not read checkpoint '{csv_path}': {exc}")
        return result

    @staticmethod
    def _check_gpu() -> bool:
        """
        Return ``True`` if PyTorch + torchaudio are installed and a CUDA
        device is available.

        :return: ``True`` if GPU path is usable.
        """
        try:
            import torch
            import torchaudio  # noqa: F401
            available = bool(torch.cuda.is_available())
            if available:
                log.info(f"WavScrubber: GPU STFT enabled ({torch.cuda.get_device_name(0)})")
            else:
                log.info("WavScrubber: PyTorch present but no CUDA device; "
                         "using CPU STFT")
            return available
        except ImportError:
            log.info("WavScrubber: PyTorch/torchaudio not found; using CPU STFT")
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments.

    :return: ``argparse.Namespace``.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="wav_scrubber",
        description=(
            "Filter bat-detector .wav files, retaining only those with "
            "at least MIN_PULSES plausible bat echolocation pulses."
        ),
    )
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "one or more .wav files, shell globs (e.g. wav_dir/*.wav), "
            "or directories.  Directories are searched for .wav files; "
            "use -r/--recursive to also descend into subdirectories."
        ),
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help=(
            "when a directory is given, recurse into subdirectories "
            "(default: only the top-level directory is searched)"
        ),
    )
    parser.add_argument(
        "-n", "--min-pulses",
        type=int, default=5,
        help="minimum valid pulses to retain a file (default: 5)",
    )
    parser.add_argument(
        "--ipi-cv",
        type=float, default=1.5,
        metavar="MAX_CV",
        help="max IPI coefficient of variation (default: 1.5)",
    )
    parser.add_argument(
        "--ipi-freq-floor",
        type=float, default=800.0,
        metavar="HZ",
        dest="ipi_freq_floor_hz",
        help=(
            "Low-frequency cutoff in Hz (true rate) for the IPI pulse-"
"            detection envelope.  Frequencies below this value are "
"            excluded from the energy envelope used to time bat pulses, "
"            preventing sub-bat interference (e.g. frog choruses) from "
"            corrupting the IPI regularity check.  The full bat-band "
"            spectrogram is still used for all other checks.  "
"            Default: 800 Hz (safely below all species in the label set, "
"            well above marsh frog interference at < 600 Hz true)."
        ),
    )
    parser.add_argument(
        "-w", "--workers",
        type=int, default=None,
        help=(
            "number of worker processes.  Default: cpu_count − 4 "
            "(leaves cores free for interactive use).  "
            "Pass 0 to use every available core."
        ),
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="disable GPU acceleration",
    )
    parser.add_argument(
        "-o", "--out-csv",
        default=None,
        help="write per-file scrub report to this CSV path",
    )
    parser.add_argument(
        "--retained-list",
        default=None,
        help="write paths of retained files (one per line) to this file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="CSV",
        help=(
            "path to a checkpoint CSV for incremental progress.  "
            "Already-processed files are skipped on resume.  "
            "Results are appended row-by-row as they complete."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float, default=120.0,
        metavar="SECS",
        help="seconds before a stalled worker is marked UNREADABLE (default: 120)",
    )
    args = parser.parse_args()

    # Expand each input item to a list of .wav paths.
    # Shell globs are already expanded by the shell before argparse sees them,
    # so each item is either a single .wav file or a directory.  Directories
    # are walked one level deep by default; -r/--recursive walks all levels.
    # A seen-set deduplicates in case the user passes overlapping globs AND
    # a parent directory, or the same file via multiple glob patterns.
    seen:  set[Path]  = set()
    paths: list[Path] = []
    for item in args.input:
        p = Path(item)
        if p.is_dir():
            glob_fn  = p.rglob if args.recursive else p.glob
            new_wavs = sorted(glob_fn("*.wav"))
            for w in new_wavs:
                if w not in seen:
                    seen.add(w)
                    paths.append(w)
        elif p.suffix.lower() == ".wav":
            if p not in seen:
                seen.add(p)
                paths.append(p)
        else:
            print(f"Warning: skipping non-WAV input '{item}'", file=sys.stderr)

    if not paths:
        parser.error("No .wav files found in the given inputs.")

    return args, paths


def main() -> None:
    """
    CLI entry point: scrub .wav files and optionally write reports.
    """
    import sys
    import shlex

    args, paths = _parse_args()

    print(f"WavScrubber: {len(paths)} files to process")
    if args.checkpoint is None:
        # Auto checkpoint path is resolved inside WavScrubber.__init__;
        # reconstruct the same name here just to show the user upfront.
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        auto_ckpt = Path.cwd() / f"scrub_checkpoint_{ts}.csv"
        print(f"  Checkpoint: {auto_ckpt}  (auto-generated; "
              f"pass --checkpoint <path> to choose your own)")

    scrubber = WavScrubber(
        wav_paths          = paths,
        min_pulses         = args.min_pulses,
        max_ipi_cv         = args.ipi_cv,
        ipi_freq_floor_hz  = args.ipi_freq_floor_hz,
        n_workers          = args.workers,
        use_gpu            = not args.no_gpu,
        show_progress      = True,
        checkpoint_csv     = args.checkpoint,
        worker_timeout     = args.timeout,
    )

    try:
        result = scrubber.run()
    except ScrubInterrupted as exc:
        partial = exc.partial_result
        ckpt    = exc.checkpoint_csv
        n_done  = len(partial.records)
        n_total = len(paths)
        print(
            f"\n*** Interrupted after {n_done}/{n_total} files "
            f"({100*n_done/n_total:.0f}%) ***",
            file=sys.stderr,
        )
        if ckpt is not None:
            resume_argv = list(sys.argv)
            if "--checkpoint" not in resume_argv:
                resume_argv.extend(["--checkpoint", str(ckpt)])
            print(f"\nCheckpoint saved: {ckpt}", file=sys.stderr)
            print(
                f"  {n_done} results safe.  "
                f"{n_total - n_done} files remain.",
                file=sys.stderr,
            )
            print(
                "\nTo resume, re-run the same command — "
                "already-processed files are skipped automatically:",
                file=sys.stderr,
            )
            print(f"  {shlex.join(resume_argv)}", file=sys.stderr)
        else:
            print(
                "\nNo checkpoint was configured — completed results are lost.",
                file=sys.stderr,
            )
            print(
                "  Always pass --checkpoint <path> for large runs:",
                file=sys.stderr,
            )
            suggested = list(sys.argv) + ["--checkpoint", "scrub_progress.csv"]
            print(f"  {shlex.join(suggested)}", file=sys.stderr)
        sys.exit(2)

    print(result.summary())

    if args.out_csv:
        p = result.to_csv(args.out_csv)
        print(f"Scrub report written to {p}")

    if args.retained_list:
        rp = Path(args.retained_list)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text("\n".join(str(p) for p in result.retained) + "\n")
        print(f"Retained file list written to {rp}")

    sys.exit(0 if result.retained else 1)


if __name__ == "__main__":
    main()
