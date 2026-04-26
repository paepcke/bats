#!/usr/bin/env python3
# *******************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-26 16:45:50
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-26 16:46:12
# *******************************************

"""
wav2spectros.py — View or batch-export linear-frequency spectrogram(s) from WAV files.

Behaviour
---------
* **Single WAV** → spectrogram is displayed interactively.
* **2–5 WAVs** → all spectrograms are displayed in sequence.
* **>5 WAVs, no --out-dir** → the first five are displayed; a warning is printed
  that additional files exist and ``--out-dir`` should be used to export them all.
* **--out-dir supplied** → every resolved WAV is written as a ``.png`` file in that
  directory; nothing is displayed.

Input sources
-------------
Positional arguments may be any mix of ``.wav`` files and directories.
Directories are scanned recursively by default; pass ``--no-recursion`` to
restrict the scan to the top level only.

Usage::

    # display one file
    python wav2spectros.py /path/to/chirp.wav

    # display up to five files from a directory
    python wav2spectros.py /path/to/wavs/

    # export everything to PNGs
    python wav2spectros.py /path/to/wavs/ --out-dir /data/BatTmps/spectros

    # mix of paths, bat-freq range, non-recursive scan
    python wav2spectros.py dir1/ file.wav dir2/ --fmax 120000 --no-recursion

Dependencies: librosa, matplotlib  (present in the bat-pipeline conda env)
"""

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISPLAY_LIMIT = 5      # max spectrograms shown when --out-dir is absent


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------

class WavToSpectros:
    """Resolve WAV paths from a mixed list of files and directories, then either
    display linear-frequency STFT spectrograms interactively or export them as PNG files.

    :param paths:       List of Path objects — individual ``.wav`` files and/or
                        directories to search.
    :param out_dir:     If supplied, PNGs are written here and nothing is displayed.
    :param recursive:   When *True* (default), directories are searched recursively.
    :param fmax:        Maximum frequency to render (Hz).  Defaults to Nyquist.
    :param colormap:    Matplotlib colormap name (default: ``viridis``).
    :param hop_length:  Hop length for the STFT (default: 512).
    """

    def __init__(
        self,
        paths:      list[Path],
        out_dir:    Path | None  = None,
        recursive:  bool         = True,
        fmax:       float | None = None,
        colormap:   str          = "viridis",
        hop_length: int          = 512,
    ) -> None:
        self.paths      = [Path(p) for p in paths]
        self.out_dir    = Path(out_dir) if out_dir else None
        self.recursive  = recursive
        self.fmax       = fmax
        self.colormap   = colormap
        self.hop_length = hop_length

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Resolve all WAV paths and either display or export spectrograms.

        :raises SystemExit: If no ``.wav`` files are found.
        """
        wav_files = self._resolve_wav_files()
        if not wav_files:
            print("No .wav files found in the supplied paths.", file=sys.stderr)
            sys.exit(1)

        if self.out_dir is not None:
            self._export_all(wav_files)
        else:
            self._display_subset(wav_files)

    # ------------------------------------------------------------------
    # Private — path resolution
    # ------------------------------------------------------------------

    def _resolve_wav_files(self) -> list[Path]:
        """Return a sorted, deduplicated list of ``.wav`` files from ``self.paths``.

        :return: Sorted list of resolved WAV ``Path`` objects.
        """
        seen:   set[Path]  = set()
        result: list[Path] = []

        for p in self.paths:
            if p.is_file():
                if p.suffix.lower() == ".wav" and p not in seen:
                    seen.add(p)
                    result.append(p)
                elif p.suffix.lower() != ".wav":
                    print(f"Warning: skipping non-WAV file: {p}", file=sys.stderr)
            elif p.is_dir():
                pattern   = "**/*.wav" if self.recursive else "*.wav"
                dir_wavs  = sorted(p.glob(pattern))
                for w in dir_wavs:
                    if w not in seen:
                        seen.add(w)
                        result.append(w)
            else:
                print(f"Warning: path not found, skipping: {p}", file=sys.stderr)

        return sorted(result)

    # ------------------------------------------------------------------
    # Private — display mode
    # ------------------------------------------------------------------

    def _display_subset(self, wav_files: list[Path]) -> None:
        """Display up to ``DISPLAY_LIMIT`` spectrograms interactively.

        Prints a warning and advice when more files are available than the
        display limit allows.

        :param wav_files: Full list of resolved WAV files.
        """
        total    = len(wav_files)
        to_show  = wav_files[:DISPLAY_LIMIT]
        skipped  = total - len(to_show)

        print(f"Found {total} .wav file(s).")

        if skipped:
            print(
                f"\nWarning: only the first {DISPLAY_LIMIT} of {total} files will be "
                f"displayed.\n"
                f"  {skipped} file(s) are not shown.  To export all as PNG files, "
                f"re-run with:\n"
                f"    --out-dir <directory>\n",
                file=sys.stderr,
            )

        for i, wav_path in enumerate(to_show, 1):
            print(f"\n[{i}/{len(to_show)}] {wav_path}")
            self._show_one(wav_path)

    def _show_one(self, wav_path: Path) -> None:
        """Load a WAV file and display its linear STFT spectrogram.

        :param wav_path: Path to the WAV file.
        :raises RuntimeError: If librosa cannot load the file.
        """
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        print(f"  Sample rate : {sr:,} Hz")
        print(f"  Duration    : {len(y)/sr:.3f} s  ({len(y):,} samples)")

        fmax = self.fmax or sr / 2
        print(f"  Freq range  : 0 – {fmax:,.0f} Hz")

        fig, ax = self._build_figure(y, sr, fmax, wav_path.name)
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private — export mode
    # ------------------------------------------------------------------

    def _export_all(self, wav_files: list[Path]) -> None:
        """Write a PNG spectrogram for every WAV file in ``wav_files``.

        :param wav_files: Full list of resolved WAV files.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        total = len(wav_files)
        print(f"Found {total} .wav file(s).  Exporting PNGs → {self.out_dir}\n")

        # Use non-interactive backend so no window is opened.
        matplotlib.use("Agg")

        ok = err = 0
        for i, wav_path in enumerate(wav_files, 1):
            out_path = self.out_dir / (wav_path.stem + ".png")
            print(
                f"[{i:>{len(str(total))}}/{total}] {wav_path.name}"
                f"  →  {out_path.name}",
                end="  ",
            )
            try:
                self._export_one(wav_path, out_path)
                print("✓")
                ok += 1
            except Exception as exc:
                print(f"FAILED: {exc}")
                err += 1

        print(f"\nDone. {ok} succeeded, {err} failed.")

    def _export_one(self, wav_path: Path, out_path: Path) -> None:
        """Compute a linear STFT spectrogram for one WAV file and save it as PNG.

        :param wav_path: Source WAV file.
        :param out_path: Destination PNG file.
        :raises RuntimeError: If librosa cannot load the file.
        """
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        fmax  = self.fmax or sr / 2

        fig, ax = self._build_figure(y, sr, fmax, wav_path.name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private — shared figure builder
    # ------------------------------------------------------------------

    def _build_figure(
        self,
        y:     np.ndarray,
        sr:    int,
        fmax:  float,
        title: str,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Compute a linear STFT spectrogram and return a fully configured figure/axes pair.

        Frequency bins are uniformly spaced in Hz — no perceptual warping — making
        this appropriate for animal vocalizations where the hearing model is unknown.

        :param y:     Audio time series.
        :param sr:    Sample rate (Hz).
        :param fmax:  Maximum frequency to render (Hz).
        :param title: Figure / axes title string.
        :return:      ``(fig, ax)`` ready for ``plt.show()`` or ``fig.savefig()``.
        """
        D     = librosa.stft(y, hop_length=self.hop_length)
        S_db  = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 5))
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="linear",
            fmax=fmax,
            cmap=self.colormap,
            ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "View or export linear-frequency STFT spectrograms from WAV files.\n\n"
            "Positional arguments may be any mix of .wav files and directories.\n"
            "Directories are scanned recursively by default.\n"
            "With --out-dir, all resolved WAVs are exported as PNGs (no display).\n"
            "Without --out-dir, up to five spectrograms are displayed interactively."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help=".wav file(s) and/or director(ies) to process.",
    )
    p.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help=(
            "Export PNG files to this directory instead of displaying them. "
            "The directory is created if it does not exist."
        ),
    )
    p.add_argument(
        "--no-recursion",
        action="store_true",
        default=False,
        help="Scan directories one level deep only (default: recursive).",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximum frequency in Hz (default: Nyquist). Use e.g. 120000 for bats.",
    )
    p.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap (default: viridis). Try: magma, inferno, plasma.",
    )
    p.add_argument("--hop-length", type=int, default=512, help="STFT hop length (default: 512).")
    return p


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()

    processor = WavToSpectros(
        paths      = [Path(p) for p in args.paths],
        out_dir    = Path(args.out_dir) if args.out_dir else None,
        recursive  = not args.no_recursion,
        fmax       = args.fmax,
        colormap   = args.colormap,
        hop_length = args.hop_length,
    )
    processor.run()


if __name__ == "__main__":
    main()
