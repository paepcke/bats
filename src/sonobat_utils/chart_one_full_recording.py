#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-16 10:08:54
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/chart_one_full_recording.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-16 10:09:14
#
# **********************************************************
"""
Diagnostic: plot a context window of a full recording centered on a chirp's
TimeInOrigRecording value, with a cyan marker at the chirp onset.

Handles time-expanded (TE) recordings where the file is stored at 1/10 speed
(header SR < 80 kHz, true SR = header SR × 10).

Key insight for TE files
------------------------
The file contains audio slowed down 10×.  All timing and frequency quantities
must be kept in one consistent domain:

* **File domain**  — uses ``sr_header`` (e.g. 25 kHz).  This is what
  ``soundfile`` uses for frame counts and seeks.
* **True domain**  — uses ``sr_true = sr_header × te`` (e.g. 250 kHz).
  This is what bat calls actually occupy in frequency and time.

We load audio in file-domain frames, run ``signal.spectrogram`` with
``fs=sr_header``, then scale the resulting ``freqs`` and ``times`` arrays
by ``te`` to convert to true-domain Hz and true-domain seconds.

Usage
-----
::

    python chart_one_full_recording.py \\
        --manifest /qnap/bats/jr_pipeline/data/bat_crops_test2/manifest.csv \\
        --feather  /qnap/bats/sonobat3_2_species_ids.feather \\
        --out      /tmp/full_rec_check.png \\
        --row      0 \\
        --window-ms 100
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
from sonobat_utils.wav_file_info import WavInfo


def plot_chirp_in_context(
    wav_path:   str,
    t_ms:       float,
    fname:      str,
    out_path:   str,
    window_ms:  float = 100.0,
    freq_lo_hz: float = 15_000.0,
    freq_hi_hz: float = 80_000.0,
) -> None:
    """
    Load ``window_ms`` of audio centred on ``t_ms`` (true-time ms from the
    recording start), compute a spectrogram, and save a PNG with a cyan
    vertical line marking the chirp onset.

    :param wav_path:   Path to the full-length ``.wav`` file.
    :param t_ms:       Chirp onset in true milliseconds from recording start.
    :param fname:      Fragment stem — used as plot title.
    :param out_path:   Destination PNG path.
    :param window_ms:  Total context in true milliseconds (default 100ms).
    :param freq_lo_hz: Lower display frequency bound in Hz.
    :param freq_hi_hz: Upper display frequency bound in Hz.
    """
    half_ms = window_ms / 2.0

    # ── Determine recording type via WavInfo ──────────────────────────
    info      = WavInfo.from_path(wav_path)
    te        = info.te
    sr_true   = info.sr_true
    sr_header = info.sr_header

    print(f"Header SR:         {sr_header:,} Hz")
    print(f"TE factor:         {te}×  →  true SR: {sr_true:,} Hz")
    print(f"True duration:     {info.true_duration_s:.2f} s  (source: {info.source})")

    # ── Seek using WavInfo helpers ────────────────────────────────────
    start_fr = info.true_ms_to_file_frame(t_ms - half_ms)
    n_frames = info.true_ms_to_file_frames_count(window_ms)
    n_frames = min(n_frames, info.file_frames - start_fr)

    start_true_s = max(0.0, (t_ms - half_ms) / 1000.0)

    print(f"True window:       {t_ms - half_ms:.1f} – {t_ms + half_ms:.1f} ms")
    print(f"File frames:       {start_fr:,} + {n_frames:,}")

    with sf.SoundFile(wav_path) as f:
        f.seek(start_fr)
        audio = f.read(n_frames, dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

    print(f"Audio samples loaded: {len(audio):,}")

    # ── Spectrogram in file domain, scaled to true domain ────────────
    nperseg_file = min(len(audio) // 2, max(256, len(audio) // 4))
    nperseg_file = max(8, nperseg_file)
    noverlap     = nperseg_file * 3 // 4

    freqs_file, times_file, Sxx = signal.spectrogram(
        audio, fs=sr_header,
        window='hann', nperseg=nperseg_file, noverlap=noverlap,
        scaling='spectrum',
    )

    freqs_true = freqs_file * te   # file Hz → true Hz
    times_true = times_file * te   # file s  → true s

    # Restrict to bat band in true Hz.
    band = (freqs_true >= freq_lo_hz) & (freqs_true <= freq_hi_hz)
    if band.sum() < 2:
        print(f"No bins in bat band {freq_lo_hz/1000:.0f}–{freq_hi_hz/1000:.0f} kHz "
              f"(true freqs range: {freqs_true[0]/1000:.1f}–{freqs_true[-1]/1000:.1f} kHz)")
        return

    Sxx_band  = Sxx[band, :]
    freqs_bat = freqs_true[band] / 1000.0   # kHz for y-axis

    # Time axis: true ms from recording start.
    t_axis_ms = times_true * 1000.0 + start_true_s * 1000.0

    print(f"Spectrogram shape:    {Sxx_band.shape}  "
          f"(freq bins × time bins)")
    print(f"Freq range:           {freqs_bat[0]:.1f} – {freqs_bat[-1]:.1f} kHz")
    print(f"Time axis:            {t_axis_ms[0]:.1f} – {t_axis_ms[-1]:.1f} ms")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.pcolormesh(
        t_axis_ms, freqs_bat,
        10.0 * np.log10(Sxx_band + 1e-12),
        shading='gouraud', cmap='inferno',
    )
    ax.axvline(
        t_ms, color='cyan', linewidth=1.5,
        label=f'TimeInOrigRecording = {t_ms:.0f} ms',
    )
    ax.set_xlabel('Time in recording (ms)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title(f'{fname}  |  {Path(wav_path).name}')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot a chirp in full-recording context with TE correction.'
    )
    parser.add_argument('--manifest', required=True,
                        help='manifest.csv from chirps_to_spectros.py')
    parser.add_argument('--feather',  required=True,
                        help='sonobat3_2_species_ids.feather')
    parser.add_argument('--out',      default='/tmp/full_rec_check.png',
                        help='Output PNG (default: /tmp/full_rec_check.png)')
    parser.add_argument('--row',      type=int, default=0,
                        help='Manifest row index to plot (default: 0)')
    parser.add_argument('--window-ms', type=float, default=100.0,
                        help='Context window in true ms (default: 100)')
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    feather  = pd.read_feather(args.feather)

    row   = manifest.iloc[args.row]
    fname = row['Filename']
    t_ms  = float(row['time_in_orig_rec_ms'])

    matches = feather.loc[feather['Filename'] == fname, 'matched_wav']
    if matches.empty or pd.isna(matches.iloc[0]):
        print(f"No matched_wav for {fname!r}")
        return
    wav_path = str(matches.iloc[0])

    print(f"Filename:            {fname}")
    print(f"time_in_orig_rec_ms: {t_ms}")
    print(f"matched_wav:         {wav_path}")
    print()

    plot_chirp_in_context(
        wav_path  = wav_path,
        t_ms      = t_ms,
        fname     = fname,
        out_path  = args.out,
        window_ms = args.window_ms,
    )


if __name__ == '__main__':
    main()
