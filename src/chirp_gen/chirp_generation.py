# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-05 15:29:10
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-05 15:51:35

"""
chirp_generator.py
==================
Generate a synthetic bat echolocation chirp from SonoBat acoustic measures
stored in a Pandas Series.

Typical usage
-------------
    from chirp_generator import ChirpGenerator

    gen = ChirpGenerator(measures_series)          # validates + synthesises
    df  = gen.spectrogram_df(outfile="spec.feather")
    gen.spectrogram_png(outfile="spec.png")
    gen.wav(outfile="chirp.wav")

Required measures
-----------------
All frequencies in kHz, durations in ms, percentages in %.

    CallDuration   – total chirp duration (ms)
    Fc             – characteristic frequency: lowest-slope point in final 40%
    HiFreq         – highest apparent frequency
    StartF         – frequency at call onset (usually == HiFreq)
    FreqKnee       – frequency at main slope-transition inflection
    PrcntKneeDur   – temporal position of knee (% of CallDuration)
    PrcntMaxAmpDur – temporal position of amplitude peak (% of CallDuration)
    FreqMaxPwr     – frequency of maximum amplitude
    Bndwdth        – total bandwidth (HiFreq − LowFreq)

Optional measures (used when present, ignored when absent/NaN)
--------------------------------------------------------------
    LowFreq, FreqCtr, UpprKnFreq
    HiFtoKnAmp, HiFtoKnExp, KnToFcAmp, KnToFcExp
    HiFtoFcAmp, HiFtoFcExp, HiFtoUpprKnAmp
    UpprKnToKnAmp, KnToFcAmp, LdgToFcAmp
    1st10kHzSlp, 1st10kHzExp, 1st5to15kHzSlp, 1st5to15kHzExp
    FFwd5dB, FFwd15dB, FFwd32dB, FBak5dB, FBak32dB, Bndw32dB
    Amp1stQrtl, Amp2ndQrtl, Amp3rdQrtl, Amp4thQrtl
    AmpK@start
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ChirpMeasureError(ValueError):
    """Raised when one or more SonoBat measures are missing or self-contradictory."""



# ---------------------------------------------------------------------------
# ChirpGenerator
# ---------------------------------------------------------------------------

class ChirpGenerator:
    """
    Reconstruct a synthetic bat echolocation chirp from SonoBat acoustic
    measures and expose the result as a waveform, spectrogram DataFrame,
    and rendered PNG.

    The synthesis pipeline runs automatically at construction time so that
    all public methods are immediately available after instantiation.

    Public instance variables (set after ``__init__``)
    --------------------------------------------------
    spectrogram : pd.DataFrame
        STFT-derived power spectrogram with columns = time bins (ms) and
        index = frequency bins (kHz).  Values are in dB (power spectrum).
    waveform : np.ndarray
        Synthesised waveform as float32 in the range [−1, 1].
    freq_trend_hz : np.ndarray
        Instantaneous frequency trajectory in Hz, one value per sample.
    amp_envelope : np.ndarray
        Amplitude envelope in [0, 1], one value per sample.
    t_ms : np.ndarray
        Time axis in ms corresponding to each sample.

    :param measures:    Pandas Series of SonoBat parameter values.
    :param sample_rate: PCM sample rate in Hz.  Default 500 000 suits bat
                        ultrasound up to 250 kHz.
    """

    # Required keys that must be present and finite
    _REQUIRED = (
        'CallDuration', 'Fc', 'HiFreq', 'StartF',
        'FreqKnee', 'PrcntKneeDur', 'PrcntMaxAmpDur',
        'FreqMaxPwr', 'Bndwdth',
    )

    # Tolerance (kHz) for frequency-consistency checks
    _FREQ_TOL = 3.0

    # ------------------------------------------------------------------ #
    #  Static helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get(s: pd.Series, key: str, default=None) -> Optional[float]:
        """
        Safely fetch a float from a Series.

        :param s:       Input Series.
        :param key:     Key to look up.
        :param default: Value to return when key is absent or NaN.
        :return:        Float value or *default*.
        """
        if key not in s.index:
            return default
        v = s[key]
        if pd.isna(v):
            return default
        return float(v)

    @staticmethod
    def _exp_segment(t: np.ndarray,
                     f_start: float,
                     f_end: float,
                     amp_param: Optional[float] = None,
                     exp_param: Optional[float] = None) -> np.ndarray:
        """
        Compute a frequency trajectory over a time array using an exponential model.

        Models the SonoBat segment formula ``f(t) = A * exp(B * t)`` where A is
        the amplitude parameter (≈ frequency at segment start) and B is the
        exponent (negative for a downward sweep).  Falls back to simple exponential
        interpolation between *f_start* and *f_end* when parameters are absent.

        :param t:         Time array in ms, **starting at 0** for this segment.
        :param f_start:   Frequency (kHz) at ``t[0]``.
        :param f_end:     Frequency (kHz) at ``t[-1]``.
        :param amp_param: A coefficient.  Overrides *f_start* as the initial value
                          when provided.
        :param exp_param: B exponent.  Derived from endpoints when absent.
        :return:          Frequency array (kHz) with the same shape as *t*.
        """
        if len(t) == 0:
            return np.array([])
        dur = t[-1]
        if dur == 0:
            return np.full_like(t, f_start)

        A = amp_param if amp_param is not None else f_start

        if exp_param is not None:
            B = exp_param
        elif f_end > 0 and A > 0 and A != f_end:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                B = np.log(f_end / A) / dur
        else:
            B = 0.0

        freq = A * np.exp(B * t)
        lo, hi = min(f_start, f_end), max(f_start, f_end)
        return np.clip(freq, lo, hi)

    def __init__(self, measures: pd.Series, sample_rate: int = 500_000) -> None:
        """
        Validate measures, synthesise the chirp, and populate instance variables.

        :param measures:    SonoBat acoustic measures as a Pandas Series.
        :param sample_rate: Audio sample rate in Hz.
        :raises ChirpMeasureError: If required keys are missing or measures
                                   are internally inconsistent.
        """
        self.measures    = measures.copy()
        self.sample_rate = sample_rate

        self._validate()
        self._run_pipeline()

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        """
        Check that required measures are present and that the set is
        internally consistent.

        Checks performed
        ----------------
        * All required keys present and non-NaN.
        * ``Bndwdth`` ≈ ``HiFreq − LowFreq`` (when LowFreq present).
        * ``FreqKnee`` is strictly between ``Fc`` and ``HiFreq``.
        * ``UpprKnFreq`` (when present) is strictly between ``FreqKnee``
          and ``HiFreq``.
        * ``PrcntKneeDur`` and ``PrcntMaxAmpDur`` are in (0, 100).
        * ``FFwd*dB`` frequencies are ≤ ``FreqMaxPwr`` (post-peak, descending).
        * ``FBak*dB`` frequencies are ≥ ``FreqMaxPwr`` (pre-peak, ascending).
        * dB-threshold frequencies are monotone:
          ``FFwd5dB ≥ FFwd15dB ≥ FFwd32dB`` and
          ``FBak5dB ≤ FBak15dB ≤ FBak32dB``.

        :raises ChirpMeasureError: On any violation.
        """
        s   = self.measures
        err = []

        # --- Required keys ---
        for key in self._REQUIRED:
            if self._get(s, key) is None:
                err.append(f"Required measure '{key}' is missing or NaN.")

        if err:
            raise ChirpMeasureError("\n".join(err))

        # Convenience locals
        hi    = self._get(s, 'HiFreq')
        fc    = self._get(s, 'Fc')
        knee  = self._get(s, 'FreqKnee')
        bw    = self._get(s, 'Bndwdth')
        lo    = self._get(s, 'LowFreq')
        ukn   = self._get(s, 'UpprKnFreq')
        fmaxp = self._get(s, 'FreqMaxPwr')
        kpct  = self._get(s, 'PrcntKneeDur')
        apct  = self._get(s, 'PrcntMaxAmpDur')

        # --- Bandwidth consistency ---
        if lo is not None:
            implied_bw = hi - lo
            if abs(implied_bw - bw) > self._FREQ_TOL:
                err.append(
                    f"Bndwdth={bw:.2f} kHz conflicts with "
                    f"HiFreq({hi:.2f}) − LowFreq({lo:.2f}) = {implied_bw:.2f} kHz "
                    f"(tolerance ±{self._FREQ_TOL} kHz)."
                )

        # --- Frequency ordering ---
        if not (fc < knee < hi):
            err.append(
                f"FreqKnee={knee:.2f} must be strictly between "
                f"Fc={fc:.2f} and HiFreq={hi:.2f} kHz."
            )

        if ukn is not None and not (knee < ukn < hi):
            err.append(
                f"UpprKnFreq={ukn:.2f} must be strictly between "
                f"FreqKnee={knee:.2f} and HiFreq={hi:.2f} kHz."
            )

        # --- Percentage bounds ---
        for label, val in [('PrcntKneeDur', kpct), ('PrcntMaxAmpDur', apct)]:
            if not (0.0 < val < 100.0):
                err.append(f"{label}={val:.2f} must be in the open interval (0, 100).")

        # --- FFwd dB-threshold frequencies (post-peak → descending freq) ---
        fwd5  = self._get(s, 'FFwd5dB')
        fwd15 = self._get(s, 'FFwd15dB')
        fwd32 = self._get(s, 'FFwd32dB')

        for label, val in [('FFwd5dB', fwd5), ('FFwd15dB', fwd15), ('FFwd32dB', fwd32)]:
            if val is not None and val > fmaxp + self._FREQ_TOL:
                err.append(
                    f"{label}={val:.2f} kHz is above FreqMaxPwr={fmaxp:.2f} kHz. "
                    f"FFwd values are post-peak and must be ≤ FreqMaxPwr."
                )

        if fwd5 is not None and fwd15 is not None and fwd5 < fwd15 - self._FREQ_TOL:
            err.append(
                f"FFwd5dB={fwd5:.2f} < FFwd15dB={fwd15:.2f}: "
                f"deeper dB drops must reach lower frequencies (FFwd5 ≥ FFwd15)."
            )
        if fwd15 is not None and fwd32 is not None and fwd15 < fwd32 - self._FREQ_TOL:
            err.append(
                f"FFwd15dB={fwd15:.2f} < FFwd32dB={fwd32:.2f}: "
                f"deeper dB drops must reach lower frequencies (FFwd15 ≥ FFwd32)."
            )

        # --- FBak dB-threshold frequencies (pre-peak → ascending freq) ---
        bak5  = self._get(s, 'FBak5dB')
        bak32 = self._get(s, 'FBak32dB')

        for label, val in [('FBak5dB', bak5), ('FBak32dB', bak32)]:
            if val is not None and val < fmaxp - self._FREQ_TOL:
                err.append(
                    f"{label}={val:.2f} kHz is below FreqMaxPwr={fmaxp:.2f} kHz. "
                    f"FBak values are pre-peak and must be ≥ FreqMaxPwr."
                )

        if bak5 is not None and bak32 is not None and bak5 > bak32 + self._FREQ_TOL:
            err.append(
                f"FBak5dB={bak5:.2f} > FBak32dB={bak32:.2f}: "
                f"deeper dB drops must reach higher (earlier) frequencies "
                f"(FBak5 ≤ FBak32)."
            )

        if err:
            raise ChirpMeasureError("\n".join(err))

    # ------------------------------------------------------------------ #
    #  Pipeline                                                            #
    # ------------------------------------------------------------------ #

    def _run_pipeline(self) -> None:
        """
        Execute the full synthesis pipeline and populate all instance variables.

        Order: frequency trend → amplitude envelope → waveform → spectrogram.
        """
        self.freq_trend_hz = self._build_freq_trend()
        self.amp_envelope  = self._build_amp_envelope()
        self.waveform      = self._synthesize()
        self.spectrogram   = self._compute_spectrogram()

    # ------------------------------------------------------------------ #
    #  Internal: frequency trend                                          #
    # ------------------------------------------------------------------ #

    def _build_freq_trend(self) -> np.ndarray:
        """
        Build the instantaneous frequency trajectory f(t) in Hz.

        The call is divided into up to three segments separated by the
        optional upper-knee, the main knee, and the characteristic frequency:

            StartF → [UpperKnee] → Knee → Fc

        Each segment is modelled as ``f(t) = A * exp(B * t)`` using the
        corresponding Amp/Exp parameters from the measures when available,
        or falling back to exponential interpolation between known endpoints.
        A gentle warp is applied at the temporal midpoint to honour FreqCtr
        when present.

        :return: Instantaneous frequency in Hz, shape ``(n_samples,)``.
        """
        s = self.measures

        hi_freq  = self._get(s, 'HiFreq')
        start_f  = self._get(s, 'StartF',   hi_freq)
        fc       = self._get(s, 'Fc')
        knee_f   = self._get(s, 'FreqKnee')
        knee_pct = self._get(s, 'PrcntKneeDur')
        dur_ms   = self._get(s, 'CallDuration')
        ukn_f    = self._get(s, 'UpprKnFreq')   # None if absent

        n = int(round(dur_ms * 1e-3 * self.sample_rate))
        self.t_ms = np.linspace(0.0, dur_ms, n, endpoint=False)
        freq_khz  = np.zeros(n)

        knee_idx = int(round(knee_pct / 100.0 * (n - 1)))
        knee_idx = int(np.clip(knee_idx, 1, n - 2))

        # Upper knee index: 30 % of the way to the main knee
        ukn_idx = max(1, int(round(knee_idx * 0.30))) if ukn_f is not None else None

        # Segment 0: StartF → UpperKnee (only when UpprKnFreq present)
        seg0_end = 0
        f_after_seg0 = start_f
        if ukn_idx is not None:
            t_seg = self.t_ms[:ukn_idx] - self.t_ms[0]
            freq_khz[:ukn_idx] = self._exp_segment(
                t_seg, start_f, ukn_f,
                amp_param=self._get(s, 'HiFtoUpprKnAmp'),
            )
            seg0_end      = ukn_idx
            f_after_seg0  = ukn_f

        # Segment 1: (UpperKnee or Start) → Knee
        t_seg = self.t_ms[seg0_end:knee_idx] - self.t_ms[seg0_end]
        freq_khz[seg0_end:knee_idx] = self._exp_segment(
            t_seg, f_after_seg0, knee_f,
            amp_param=self._get(s, 'HiFtoKnAmp'),
            exp_param=self._get(s, 'HiFtoKnExp'),
        )

        # Segment 2: Knee → Fc
        t_seg = self.t_ms[knee_idx:] - self.t_ms[knee_idx]
        freq_khz[knee_idx:] = self._exp_segment(
            t_seg, knee_f, fc,
            amp_param=self._get(s, 'KnToFcAmp'),
            exp_param=self._get(s, 'KnToFcExp'),
        )

        # Soft FreqCtr correction: warp the second half toward the midpoint target
        fc_ctr = self._get(s, 'FreqCtr')
        if fc_ctr is not None:
            mid = n // 2
            actual = freq_khz[mid]
            if abs(actual) > 1e-6:
                ratio = fc_ctr / actual
                blend = np.linspace(0.0, 1.0, n - mid)
                freq_khz[mid:] *= (1.0 + (ratio - 1.0) * blend)

        return np.clip(freq_khz * 1000.0, 1.0, self.sample_rate / 2.0 - 1.0)

    # ------------------------------------------------------------------ #
    #  Internal: amplitude envelope                                       #
    # ------------------------------------------------------------------ #

    def _build_amp_envelope(self) -> np.ndarray:
        """
        Construct the amplitude envelope A(t) in [0, 1].

        Construction strategy
        ---------------------
        1. **Attack** (start → peak): exponential rise governed by ``AmpK@start``
           (slope of log-amplitude vs normalised time).
        2. **Decay** (peak → end): exponential decay whose time-constant is
           derived from the ``FFwd5dB`` marker timing when available.
        3. **Quartile scaling**: the envelope is piecewise-rescaled so that
           total energy in each temporal quarter matches the ``Amp*Qrtl`` ratios.

        :return: Amplitude envelope, shape ``(n_samples,)``, values in [0, 1].
        """
        s      = self.measures
        n      = len(self.t_ms)
        dur_ms = self.t_ms[-1]

        peak_pct = self._get(s, 'PrcntMaxAmpDur', 40.0)
        peak_idx = int(np.clip(round(peak_pct / 100.0 * (n - 1)), 1, n - 2))

        amp_k = self._get(s, 'AmpK@start', 3.0)

        env = np.zeros(n)

        # Attack
        t_norm = np.linspace(0.0, 1.0, peak_idx + 1)
        env[:peak_idx + 1] = np.exp(amp_k * t_norm)
        env[:peak_idx + 1] /= env[peak_idx]

        # Decay time-constant from FFwd5dB timing
        decay_tau = dur_ms * 0.30    # default

        fwd5 = self._get(s, 'FFwd5dB')
        if fwd5 is not None:
            # Find the first time after the peak where f(t) ≤ FFwd5dB * 1000 Hz
            target_hz = fwd5 * 1000.0
            t_hit = None
            for i in range(peak_idx, n):
                if self.freq_trend_hz[i] <= target_hz:
                    t_hit = self.t_ms[i]
                    break
            if t_hit is not None:
                dt = t_hit - self.t_ms[peak_idx]
                if dt > 0.0:
                    # exp(−dt / tau) = 10^(−5/20)  →  tau = dt / (5/20 * ln10)
                    decay_tau = dt / (5.0 / 20.0 * np.log(10.0))

        t_decay = self.t_ms[peak_idx:] - self.t_ms[peak_idx]
        env[peak_idx:] = np.exp(-t_decay / max(decay_tau, 0.01))

        # Quartile rescaling
        q_vals = [
            self._get(s, 'Amp1stQrtl'), self._get(s, 'Amp2ndQrtl'),
            self._get(s, 'Amp3rdQrtl'), self._get(s, 'Amp4thQrtl'),
        ]
        if all(v is not None for v in q_vals):
            total  = sum(q_vals)
            targets = [v / total for v in q_vals]
            qsize   = n // 4
            env_mean = np.mean(np.abs(env)) + 1e-9
            for qi, tgt in enumerate(targets):
                sl      = slice(qi * qsize, (qi + 1) * qsize)
                current = np.sum(np.abs(env[sl]))
                if current > 1e-9:
                    env[sl] *= tgt * 4.0 / (current / env_mean)

        env = np.abs(env)
        peak = env.max()
        if peak > 0.0:
            env /= peak
        return env

    # ------------------------------------------------------------------ #
    #  Internal: waveform synthesis                                       #
    # ------------------------------------------------------------------ #

    def _synthesize(self) -> np.ndarray:
        """
        Produce the PCM waveform via cumulative phase integration.

        The instantaneous phase at sample *k* is
        ``φ(k) = 2π · Σᵢ₌₀ᵏ  f(i) / sample_rate``
        and the waveform is ``x(k) = A(k) · sin(φ(k))``.

        :return: Float32 waveform in [−1, 1], shape ``(n_samples,)``.
        """
        dt      = 1.0 / self.sample_rate
        phase   = np.cumsum(self.freq_trend_hz) * dt * 2.0 * np.pi
        wave    = self.amp_envelope * np.sin(phase)
        peak    = np.max(np.abs(wave))
        if peak > 0.0:
            wave /= peak
        return wave.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Internal: spectrogram computation                                  #
    # ------------------------------------------------------------------ #

    def _compute_spectrogram(self) -> pd.DataFrame:
        """
        Compute the STFT-based power spectrogram of the synthesised waveform.

        Window length is chosen to give ≈ 0.2 ms time resolution, which
        comfortably resolves the structural features of typical bat chirps
        (2–20 ms duration, 20–200 kHz bandwidth).

        :return: DataFrame with frequency bins (kHz) as index and time bins
                 (ms) as columns.  Values are power in dB re peak.
        """
        nperseg  = max(64, int(round(0.0002 * self.sample_rate)))  # ~0.2 ms
        noverlap = nperseg * 3 // 4

        freqs, times, Sxx = signal.spectrogram(
            self.waveform, fs=self.sample_rate,
            window='hann', nperseg=nperseg, noverlap=noverlap,
            scaling='spectrum',
        )

        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

        df = pd.DataFrame(
            Sxx_db,
            index=pd.Index(np.round(freqs / 1000.0, 4), name='frequency_kHz'),
            columns=pd.Index(np.round(times * 1000.0, 4), name='time_ms'),
        )
        return df

    # ------------------------------------------------------------------ #
    #  Public: spectrogram DataFrame                                      #
    # ------------------------------------------------------------------ #

    def spectrogram_df(self, outfile: Optional[str] = None) -> pd.DataFrame:
        """
        Return the power spectrogram as a DataFrame and optionally save it.

        The DataFrame has frequency bins (kHz) as the index and time bins (ms)
        as columns.  Values are in dB (power spectrum, relative to peak).

        :param outfile: Optional output path.  Extension determines format:
                        ``.csv`` → CSV, ``.feather`` → Apache Arrow Feather.
                        Parent directories are created as needed.
        :return: Spectrogram DataFrame.
        :raises ValueError: If *outfile* has an unsupported extension.
        """
        if outfile is not None:
            p = Path(outfile)
            p.parent.mkdir(parents=True, exist_ok=True)
            ext = p.suffix.lower()
            if ext == '.csv':
                self.spectrogram.to_csv(p)
            elif ext == '.feather':
                try:
                    import pyarrow  # noqa
                except ImportError as exc:
                    raise ImportError("Feather output requires pyarrow: pip install pyarrow") from exc
                df_out = self.spectrogram.reset_index()
                df_out.columns = df_out.columns.astype(str)
                df_out.to_feather(p)
            else:
                raise ValueError(
                    f"Unsupported spectrogram output format '{ext}'. "
                    f"Use '.csv' or '.feather'."
                )
        return self.spectrogram

    # ------------------------------------------------------------------ #
    #  Public: spectrogram PNG                                            #
    # ------------------------------------------------------------------ #

    def spectrogram_png(self, outfile: Optional[str] = None) -> Optional[Path]:
        """
        Render a publication-quality spectrogram image.

        The image shows the STFT power in the ``inferno`` colourmap on a dark
        background.  The reconstructed frequency trend is overlaid as a white
        line, and structural landmarks (StartF, Knee, UpperKnee if present,
        FreqMaxPwr) are marked with coloured dots.

        :param outfile: Optional output path (must end in ``.png``).
                        Parent directories are created as needed.  When
                        *None* the figure is rendered but not saved.
        :return: ``Path`` of the saved file, or ``None`` if *outfile* is None.
        :raises ValueError: If *outfile* does not end in ``.png``.
        """
        if outfile is not None:
            p = Path(outfile)
            if p.suffix.lower() != '.png':
                raise ValueError(
                    f"spectrogram_png requires a '.png' output path, got '{p.suffix}'."
                )
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p = None

        s  = self.measures
        sr = self.sample_rate

        # ---- frequency axis view range ----
        lo_kHz = max(0.0, (self._get(s, 'LowFreq') or self._get(s, 'Fc', 20.0)) - 15.0)
        hi_kHz = min(sr / 2000.0, (self._get(s, 'HiFreq', 100.0)) + 15.0)

        spec = self.spectrogram
        freq_idx = spec.index.values        # kHz
        time_col = spec.columns.values      # ms
        view     = (freq_idx >= lo_kHz) & (freq_idx <= hi_kHz)
        Sxx_view = spec.values[view]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0a0a1a')
        ax.set_facecolor('#0a0a1a')

        vmin = float(np.percentile(Sxx_view, 10))
        vmax = float(Sxx_view.max())

        ax.pcolormesh(
            time_col, freq_idx[view], Sxx_view,
            cmap='inferno', vmin=vmin, vmax=vmax, shading='gouraud',
        )

        # Frequency trend overlay
        ax.plot(self.t_ms, self.freq_trend_hz / 1000.0,
                color='white', lw=1.2, alpha=0.85, label='f(t) trend')

        # Structural landmark dots
        dur_ms   = self._get(s, 'CallDuration')
        kn_t     = self._get(s, 'PrcntKneeDur',   0.0) / 100.0 * dur_ms
        max_t    = self._get(s, 'PrcntMaxAmpDur', 0.0) / 100.0 * dur_ms

        landmarks = [
            (self._get(s, 'StartF'),    0.0,    '#00ffff', 'StartF'),
            (self._get(s, 'FreqKnee'),  kn_t,   '#ff9900', 'Knee'),
            (self._get(s, 'FreqMaxPwr'), max_t, '#00ff88', 'MaxPwr'),
        ]
        for f_val, t_val, color, label in landmarks:
            if f_val is not None:
                ax.scatter([t_val], [f_val], s=45, color=color,
                           zorder=6, label=label)

        ukn_f = self._get(s, 'UpprKnFreq')
        if ukn_f is not None:
            ukn_t = kn_t * 0.30
            ax.scatter([ukn_t], [ukn_f], s=45, color='#ff44ff',
                       zorder=6, label='UpperKnee')

        # Colorbar
        sm   = plt.cm.ScalarMappable(
            cmap='inferno',
            norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label('Power (dB)', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        ax.set_xlabel('Time (ms)', color='white', fontsize=11)
        ax.set_ylabel('Frequency (kHz)', color='white', fontsize=11)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        fc_v  = self._get(s, 'Fc',           0.0)
        hi_v  = self._get(s, 'HiFreq',       0.0)
        bw_v  = self._get(s, 'Bndwdth',      0.0)
        dur_v = self._get(s, 'CallDuration', 0.0)
        ax.set_title(
            f"Synthetic Chirp  |  Fc={fc_v:.1f} kHz  "
            f"HiFreq={hi_v:.1f} kHz  BW={bw_v:.1f} kHz  Dur={dur_v:.2f} ms",
            color='white', fontsize=10, pad=8,
        )
        ax.legend(loc='upper right', fontsize=8,
                  facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

        plt.tight_layout()
        if p is not None:
            plt.savefig(p, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
        plt.close(fig)
        return p

    # ------------------------------------------------------------------ #
    #  Public: wav file                                                   #
    # ------------------------------------------------------------------ #

    def wav(self, outfile: Optional[str] = None) -> Optional[Path]:
        """
        Write the synthesised waveform as a 16-bit PCM ``.wav`` file.

        :param outfile: Output path (must end in ``.wav``).  Parent
                        directories are created as needed.  When *None*
                        nothing is written and ``None`` is returned.
        :return: ``Path`` of the saved file, or ``None`` if *outfile* is None.
        :raises ValueError: If *outfile* does not end in ``.wav``.
        """
        if outfile is None:
            return None

        p = Path(outfile)
        if p.suffix.lower() != '.wav':
            raise ValueError(
                f"wav() requires a '.wav' output path, got '{p.suffix}'."
            )
        p.parent.mkdir(parents=True, exist_ok=True)

        pcm = (self.waveform * 32767.0).astype(np.int16)
        wavfile.write(str(p), self.sample_rate, pcm)
        return p

    # ------------------------------------------------------------------ #
    #  Convenience                                                        #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        s    = self.measures
        dur  = self._get(s, 'CallDuration', float('nan'))
        fc   = self._get(s, 'Fc',           float('nan'))
        hi   = self._get(s, 'HiFreq',       float('nan'))
        n    = len(self.waveform) if hasattr(self, 'waveform') else 0
        return (
            f"ChirpGenerator("
            f"Fc={fc:.1f} kHz, HiFreq={hi:.1f} kHz, "
            f"dur={dur:.1f} ms, {n} samples @ {self.sample_rate} Hz)"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Eptesicus fuscus – calibrated to the Dye Creek Ranch SonoBat screenshot
    demo = pd.Series({
        'CallDuration':   16.0,
        'Fc':             26.0,
        'HiFreq':        112.0,
        'LowFreq':        22.0,
        'StartF':        112.0,
        'Bndwdth':        90.0,
        'FreqMaxPwr':     28.0,
        'PrcntMaxAmpDur': 55.0,
        'FreqKnee':       45.0,
        'PrcntKneeDur':   18.0,
        'UpprKnFreq':     78.0,
        'FreqCtr':        35.0,
        'HiFtoKnAmp':    115.0,
        'HiFtoKnExp':     -0.18,
        'KnToFcAmp':      47.0,
        'KnToFcExp':      -0.025,
        'HiFtoFcAmp':    115.0,
        'HiFtoFcExp':     -0.055,
        'AmpK@start':      6.0,
        'Amp1stQrtl':     60.0,
        'Amp2ndQrtl':    240.0,
        'Amp3rdQrtl':    310.0,
        'Amp4thQrtl':    130.0,
        'FFwd5dB':        27.0,
        'FFwd15dB':       25.0,
        'FFwd32dB':       23.0,
        'FBak5dB':        32.0,
        'FBak32dB':       40.0,
        'Bndw32dB':       17.0,
    })

    gen = ChirpGenerator(demo, sample_rate=500_000)
    print(gen)

    gen.wav(outfile='/mnt/user-data/outputs/chirp.wav')
    gen.spectrogram_png(outfile='/mnt/user-data/outputs/chirp_spec.png')
    gen.spectrogram_df(outfile='/mnt/user-data/outputs/chirp_spec.feather')

    print(f"Spectrogram shape: {gen.spectrogram.shape}  "
          f"({len(gen.spectrogram.index)} freq bins × "
          f"{len(gen.spectrogram.columns)} time bins)")
    print("Done.")