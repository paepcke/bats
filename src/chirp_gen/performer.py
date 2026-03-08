#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-07 16:06:48
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/chirp_gen/performer.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-08 11:46:35
#
# **********************************************************

"""
perform.py
==========
Play or display synthetic bat chirp files, or generate and page through
chirps from a measures file on the fly.

Usage
-----
    # Play a .wav file (real-time or time-expanded):
    python perform.py chirp.wav

    # Play a .wav file and show its spectrogram (blocks until window is closed):
    python perform.py -s chirp.wav

    # Display a spectrogram PNG:
    python perform.py chirp_spec.png

    # Page through every row in a measures file, showing each chirp:
    python perform.py measures.csv
    python perform.py measures.feather

    # Paging with output — also save generated files alongside:
    python perform.py measures.csv -o /tmp/chirp_out

Supported player/viewer backends (tried in order)
--------------------------------------------------
Audio : cvlc, vlc, afplay (macOS), paplay, aplay
Image : open (macOS), eog, feh, display (ImageMagick), xdg-open
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt

from chirp_gen.chirp_generation import ChirpGenerator, ChirpMeasureError

# ---------------------------------------------------------------------------
# Time-expansion detection — mirrors wav_scrubber constants
# ---------------------------------------------------------------------------

#: Header sample rates below this value indicate a time-expanded file.
_TE_SR_THRESHOLD_HZ: int = 80_000

#: Factor applied to a TE file's header rate to recover the true ultrasound SR.
_TIME_EXPAND_FACTOR: int = 10


# ---------------------------------------------------------------------------
# Performer
# ---------------------------------------------------------------------------

class Performer:
    """
    Play, display, or interactively page through synthetic bat chirp files.

    Accepts three kinds of input:

    * ``.wav``                 — play immediately with the best available audio
      player.  Pass ``show_spectrogram=True`` to also render a spectrogram that
      blocks until the viewer window is closed.
    * ``.png``                 — display immediately with the best available image viewer.
    * ``.csv`` / ``.tsv`` / ``.feather`` — synthesise one chirp per row via
      :class:`~chirp_generator.ChirpGenerator`, show its spectrogram, play
      the time-expanded audio, then prompt to advance or quit.

    Audio player candidates (tried in order)
    -----------------------------------------
    ``cvlc``, ``vlc``, ``afplay`` (macOS), ``paplay``, ``aplay``

    Image viewer candidates (tried in order)
    -----------------------------------------
    ``open`` (macOS), ``eog``, ``feh``, ``display`` (ImageMagick), ``xdg-open``

    :param infile:           Path to the file to perform.
    :param outdir:           Optional directory to save generated files permanently
                             (only meaningful for measures files).
    :param show_spectrogram: When ``True`` and *infile* is a ``.wav``, render a
                             spectrogram of the recording before (or instead of)
                             playing.  Ignored for all other file types.
    """

    # Audio player candidates, tried left-to-right.
    # cvlc is VLC in headless/console mode (no GUI window).
    _AUDIO_PLAYERS = ['cvlc', 'vlc', 'afplay', 'paplay', 'aplay']

    # Image viewer candidates, tried left-to-right.
    _IMAGE_VIEWERS = ['open', 'eog', 'feh', 'display', 'xdg-open']

    #: Header sample rates below this value are treated as time-expanded.
    _TE_SR_THRESHOLD_HZ: int = _TE_SR_THRESHOLD_HZ

    #: Expansion factor for TE files.
    _TIME_EXPAND_FACTOR: int = _TIME_EXPAND_FACTOR

    def __init__(
        self,
        infile: Path,
        outdir: Optional[Path] = None,
        show_spectrogram: bool = False,
    ) -> None:
        """
        :param infile:           Path to a ``.wav``, ``.png``, ``.csv``, ``.tsv``,
                                 or ``.feather`` file.
        :param outdir:           Optional directory for permanent output files.
        :param show_spectrogram: Show a spectrogram when playing a ``.wav`` file.
        """
        self.infile           = infile
        self.outdir           = outdir
        self.show_spectrogram = show_spectrogram

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Dispatch to the appropriate handler based on the input file extension.
        """
        suffix = self.infile.suffix.lower()
        if suffix == '.wav':
            if self.show_spectrogram:
                self._show_wav_spectrogram(self.infile)
            self.play_wav(self.infile)
        elif suffix == '.png':
            self.show_png(self.infile)
        else:
            if self.outdir is None:
                print("Tip: pass -o <dir> to save generated files permanently.\n")
            self._perform_df()

    # ------------------------------------------------------------------ #
    #  Audio                                                               #
    # ------------------------------------------------------------------ #

    def play_wav(self, path: Path) -> None:
        """
        Play a ``.wav`` file using the best available audio player.

        Tries candidates in :attr:`_AUDIO_PLAYERS` order.  Blocks until
        playback finishes.

        :param path: Path to the ``.wav`` file.
        :raises FileNotFoundError: If *path* does not exist.
        :raises RuntimeError: If no supported audio player is found on PATH.
        """
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {path}")

        player = self._find_tool(self._AUDIO_PLAYERS)
        if player is None:
            raise RuntimeError(
                f"No audio player found on PATH.  Install one of: "
                f"{', '.join(self._AUDIO_PLAYERS)}"
            )

        # cvlc/vlc exit immediately by default; --play-and-exit makes them block.
        if player in ('cvlc', 'vlc'):
            cmd = [player, '--play-and-exit', str(path)]
        else:
            cmd = [player, str(path)]

        print(f"Playing {path.name}  [{player}]")
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------ #
    #  Image                                                               #
    # ------------------------------------------------------------------ #

    def show_png(self, path: Path) -> None:
        """
        Display a ``.png`` file using the best available image viewer.

        Tries candidates in :attr:`_IMAGE_VIEWERS` order.  Blocks until the
        viewer window is closed.

        :param path: Path to the ``.png`` file.
        :raises FileNotFoundError: If *path* does not exist.
        :raises RuntimeError: If no supported image viewer is found on PATH.
        """
        if not path.exists():
            raise FileNotFoundError(f"PNG file not found: {path}")

        viewer = self._find_tool(self._IMAGE_VIEWERS)
        if viewer is None:
            raise RuntimeError(
                f"No image viewer found on PATH.  Install one of: "
                f"{', '.join(self._IMAGE_VIEWERS)}"
            )

        # 'open' on macOS is non-blocking by default; -W waits for the app to quit.
        cmd = ['open', '-W', str(path)] if viewer == 'open' else [viewer, str(path)]

        print(f"Showing {path.name}  [{viewer}]")
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------ #
    #  WAV spectrogram                                                     #
    # ------------------------------------------------------------------ #

    def _show_wav_spectrogram(self, path: Path) -> None:
        """
        Render and display an interactive spectrogram for a ``.wav`` file.

        Uses the same STFT parameters as :class:`~chirp_generator.ChirpGenerator`
        (Hann window, ~0.5 ms frame, 75 % overlap) for consistency across the
        application suite.  Time-expanded files (header SR < ``_TE_SR_THRESHOLD_HZ``)
        are auto-corrected by multiplying the header rate by ``_TIME_EXPAND_FACTOR``
        so the frequency axis reflects true ultrasound frequencies.

        Blocks until the matplotlib window is closed by the user.

        :param path: Path to the ``.wav`` file.
        :raises FileNotFoundError: If *path* does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {path}")

        sr_header, data = wavfile.read(str(path))

        # Flatten to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Normalise to float32 in [−1, 1]
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2_147_483_648.0
        else:
            audio = data.astype(np.float32)

        # ── Time-expansion correction (mirrors wav_scrubber logic) ────────
        time_expanded = sr_header < self._TE_SR_THRESHOLD_HZ
        sr = sr_header * self._TIME_EXPAND_FACTOR if time_expanded else sr_header
        duration_s = len(audio) / sr

        # ── STFT — same parameters as ChirpGenerator._compute_spectrogram ─
        # ~0.5 ms window gives good time resolution for sub-ms to 35 ms calls.
        nperseg  = max(64, int(round(0.0005 * sr)))
        noverlap = nperseg * 3 // 4

        freqs_hz, times_s, Sxx = signal.spectrogram(
            audio, fs=sr,
            window='hann', nperseg=nperseg, noverlap=noverlap,
            scaling='spectrum',
        )

        # ── Plot ──────────────────────────────────────────────────────────
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
        Sxx_db = np.clip(Sxx_db, -80.0, 0.0)

        # Show 0 Hz to Nyquist (or 130 kHz ceiling for readability)
        freq_ceil_hz = min(sr / 2, 130_000)
        freq_mask    = freqs_hz <= freq_ceil_hz

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.pcolormesh(
            times_s * 1000,
            freqs_hz[freq_mask] / 1000,
            Sxx_db[freq_mask],
            cmap='inferno', vmin=-75, vmax=-20, shading='auto',
        )
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (kHz)')

        te_note = f'  [time-expanded ×{self._TIME_EXPAND_FACTOR}, corrected to {sr//1000} kHz]' \
                  if time_expanded else f'  [{sr//1000} kHz]'
        ax.set_title(f'{path.name}{te_note}  —  {duration_s*1000:.1f} ms')

        # Mark the standard bat band boundaries
        for f_khz, label in [(15, '15 kHz'), (120, '120 kHz')]:
            if f_khz <= freq_ceil_hz / 1000:
                ax.axhline(f_khz, color='cyan', lw=0.8, ls='--', alpha=0.55,
                           label=label)

        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()

        print(f"Spectrogram for {path.name} — close window to continue.")
        plt.show(block=True)

    # ------------------------------------------------------------------ #
    #  Measures-file pager                                                 #
    # ------------------------------------------------------------------ #

    def _perform_df(self) -> None:
        """
        Page through every row of the measures file, synthesising and playing
        each chirp interactively.

        For each row a :class:`~chirp_generator.ChirpGenerator` is instantiated.
        A temporary directory holds the generated ``.wav`` and ``.png`` for that
        row and is removed automatically before advancing to the next.  If
        :attr:`outdir` is set, files are also written there permanently.
        """
        import pandas as pd

        suffix = self.infile.suffix.lower()
        if suffix in ('.csv', '.tsv'):
            df = pd.read_csv(self.infile, sep='\t' if suffix == '.tsv' else ',')
        else:
            df = pd.read_feather(self.infile)

        n_rows = len(df)
        print(f"Loaded {n_rows} rows from {self.infile.name}")

        for i in range(n_rows):
            row = df.iloc[i]
            print(f"\n─── Chirp {i + 1} / {n_rows} ───")

            try:
                gen = ChirpGenerator(row)
            except ChirpMeasureError as exc:
                print(f"  Skipping row {i}: {exc}")
                continue
            except Exception as exc:
                print(f"  Unexpected error on row {i}: {exc}")
                continue

            print(f"  {gen}")

            # Fresh temp dir per chirp; deleted automatically on block exit.
            with tempfile.TemporaryDirectory(prefix='perform_chirp_') as tmpdir:
                tmp     = Path(tmpdir)
                tmp_wav = tmp / f'chirp_{i}.wav'
                tmp_png = tmp / f'chirp_{i}.png'

                gen.wav(outfile=str(tmp_wav), time_expand=True)
                gen.spectrogram_png(outfile=str(tmp_png))

                if self.outdir is not None:
                    gen.wav(outfile=str(self.outdir / f'chirp_{i}.wav'), time_expand=True)
                    gen.spectrogram_png(outfile=str(self.outdir / f'chirp_{i}.png'))
                    gen.spectrogram_df(outfile=str(self.outdir / f'chirp_{i}.csv'))
                    print(f"  Saved to {self.outdir}")

                try:
                    self.show_png(tmp_png)
                except RuntimeError as exc:
                    print(f"  [image] {exc}")

                # Play the 10× slowed version so it's audible; fall back to real-time.
                tmp_slowed = tmp / f'chirp_{i}_slowed.wav'
                target_wav = tmp_slowed if tmp_slowed.exists() else tmp_wav
                try:
                    self.play_wav(target_wav)
                except (RuntimeError, subprocess.CalledProcessError) as exc:
                    print(f"  [audio] {exc}")

                # Prompt while temp dir is still alive.
                try:
                    answer = input("  Enter for next chirp, 'q' to quit: ")
                except EOFError:
                    answer = ''

                if answer.strip().lower() == 'q':
                    print("Stopping.")
                    break

        print("\nDone.")

    # ------------------------------------------------------------------ #
    #  Static helper                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_tool(candidates: list[str]) -> Optional[str]:
        """
        Return the first candidate program found on PATH, or ``None``.

        :param candidates: Ordered list of program names to try.
        :return:           Name of the first available program, or ``None``.
        """
        for name in candidates:
            if shutil.which(name) is not None:
                return name
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    :return: Tuple of ``(infile, outdir, show_spectrogram)``.
    """
    desc = (
        "Play a .wav, display a .png, or page through chirps from a\n"
        "measures .csv/.tsv/.feather file."
    )
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawTextHelpFormatter,
        description=desc,
    )
    parser.add_argument(
        'infile',
        help="path to a .wav, .png, .csv, .tsv, or .feather file",
    )
    parser.add_argument(
        '-o', '--outdir',
        help=(
            "optional directory where generated .wav, .png, and .csv files\n"
            "are saved permanently (only meaningful for measures files)"
        ),
        default=None,
    )
    parser.add_argument(
        '-s', '--spectrogram',
        action='store_true',
        default=False,
        help=(
            "show an interactive spectrogram before playing the audio\n"
            "(only valid with a .wav input file)"
        ),
    )

    args   = parser.parse_args()
    infile = Path(args.infile)

    if not infile.exists():
        parser.error(f"File not found: {args.infile}")

    allowed = {'.wav', '.png', '.csv', '.tsv', '.feather'}
    if infile.suffix.lower() not in allowed:
        parser.error(
            f"Unsupported file type '{infile.suffix}'.  "
            f"Expected one of: {', '.join(sorted(allowed))}"
        )

    if args.spectrogram and infile.suffix.lower() != '.wav':
        parser.error(
            f"-s/--spectrogram is only valid with a .wav input file, "
            f"not '{infile.suffix}'"
        )

    outdir = None
    if args.outdir is not None:
        outdir = Path(args.outdir)
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            parser.error(f"Could not create output directory '{args.outdir}': {exc}")

    return infile, outdir, args.spectrogram


def main() -> None:
    """
    Instantiate :class:`Performer` and call :meth:`~Performer.run`.
    """
    infile, outdir, show_spectrogram = parse_args()
    try:
        Performer(infile, outdir=outdir, show_spectrogram=show_spectrogram).run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()