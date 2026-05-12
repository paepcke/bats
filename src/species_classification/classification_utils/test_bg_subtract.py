#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-12 16:26:55
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-12 16:56:27
# **********************************************************

"""
test_bg_subtract.py
===================
Visual comparison of background-suppression strategies for bat spectrogram
crops.  Two approaches are compared side by side:

1. **Per-row low-percentile subtraction** — for each frequency bin (row) the
   Nth-percentile intensity across all time columns is used as a noise-floor
   estimate and subtracted.  Low percentiles (5-10) are conservative and
   preserve narrow-band / shallow-sweep calls (Lano, Laci, Epfu); the median
   (50) is aggressive and can self-cancel persistent calls.

2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) -- divides the
   image into tiles, equalises each tile's histogram independently, then
   stitches them back with bilinear interpolation.  Recording-adaptive by
   nature: boosts contrast where the call is faint without amplifying noise in
   already-bright regions.  Does not subtract anything, so narrow-band calls
   cannot be cancelled.  Tunable via --clahe-clip and --clahe-tile.

Each output figure has one column per method tested, plus the original, with
a row-energy profile (mean intensity per frequency bin) beneath each
spectrogram and a "call energy retained %" in the title.  For CLAHE the
retained % will typically be >= 100 % since contrast is boosted, not reduced.

Requirements
------------
opencv-python must be installed for CLAHE:  pip install opencv-python

Usage
-----
    python test_bg_subtract.py \\
        --crops  /qnap/bats/.../00006557.png \\
                 /qnap/bats/.../00008887.png \\
        --percentiles 5 10 20 50 \\
        --clahe-clip 2.0 4.0 \\
        --clahe-tile 8 \\
        --out-dir /qnap/bats/jr_pipeline/gradcam_results/bg_test

One PNG per crop is written to --out-dir, named <crop_stem>_bg_test.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

from logging_service import LoggingService

log = LoggingService()


# ---------------------------------------------------------------------------
# Normalisation methods
# ---------------------------------------------------------------------------

class BgSubtractor:
    """
    Per-row low-percentile background subtraction.

    Mirrors CropPreprocessor.subtract_bg exactly so results are representative
    of what the training pipeline will produce.

    :param percentile:   Row percentile used as noise-floor estimate (0-100).
    :param min_floor_dn: Rows with an estimated floor below this value (0-255)
                         are left untouched -- they are already near-black.
    """

    def __init__(self, percentile: float = 10.0, min_floor_dn: int = 4) -> None:
        self.percentile   = percentile
        self.min_floor_dn = min_floor_dn

    @property
    def label(self) -> str:
        """:return: Short label for figure titles."""
        return f'pct={self.percentile:.0f}\nfloor_guard={self.min_floor_dn}'

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply subtraction to a grayscale PIL image.

        :param img: Grayscale PIL image (mode 'L').
        :return:    Cleaned grayscale PIL image.
        """
        arr   = np.array(img, dtype=np.int16)
        floor = np.percentile(arr, self.percentile, axis=1, keepdims=True)
        floor = np.where(floor < self.min_floor_dn, 0, floor)
        arr   = np.clip(arr - floor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='L')


class ClaheNormaliser:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) via OpenCV.

    Divides the image into tile_size x tile_size tiles, equalises each tile's
    histogram independently (clamped at clip_limit to avoid over-amplifying
    noise), then stitches with bilinear interpolation.

    Because CLAHE only redistributes existing contrast rather than subtracting
    a floor, narrow-band calls cannot be cancelled even when they occupy the
    majority of time columns in a tile.

    :param clip_limit: Contrast clip limit.  Higher values allow stronger local
                       contrast enhancement but may amplify noise.
                       Typical range: 1.0-4.0.  Default: 2.0.
    :param tile_size:  Tile grid dimension (tiles are square).  Smaller tiles
                       give finer local adaptation; larger tiles approach global
                       histogram equalisation.  Default: 8.
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8) -> None:
        if not _HAVE_CV2:
            raise ImportError(
                'opencv-python is required for CLAHE. '
                'Install with: pip install opencv-python'
            )
        self.clip_limit = clip_limit
        self.tile_size  = tile_size
        self._clahe = cv2.createCLAHE(
            clipLimit    = clip_limit,
            tileGridSize = (tile_size, tile_size),
        )

    @property
    def label(self) -> str:
        """:return: Short label for figure titles."""
        return f'CLAHE clip={self.clip_limit}\ntile={self.tile_size}'

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply CLAHE to a grayscale PIL image.

        :param img: Grayscale PIL image (mode 'L').
        :return:    Contrast-enhanced grayscale PIL image.
        """
        arr      = np.array(img, dtype=np.uint8)
        enhanced = self._clahe.apply(arr)
        return Image.fromarray(enhanced, mode='L')


# ---------------------------------------------------------------------------
# Figure renderer
# ---------------------------------------------------------------------------

class BgTestRenderer:
    """
    Render a side-by-side comparison figure for one crop across all methods.

    Columns: Original | BgSubtractor methods... | CLAHE methods...

    Each column shows:
    * The processed spectrogram (grayscale, origin=upper)
    * A row-energy profile (mean DN per frequency bin) beneath it
    * A "call energy retained %" in the column title

    :param methods: Ordered list of (label, callable) pairs.  Each callable
                    accepts a PIL image and returns a PIL image.
    :param out_dir: Directory to write output PNGs.
    """

    def __init__(
        self,
        methods: List[Tuple[str, object]],
        out_dir: Path,
    ) -> None:
        self.methods = methods
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #

    @staticmethod
    def _call_energy_retained(
        arr_orig: np.ndarray,
        arr_proc: np.ndarray,
        call_pct: float = 80.0,
    ) -> float:
        """
        Fraction of call-row energy retained (or boosted) after processing.

        Call rows are rows whose mean DN in the original exceeds the
        call_pct-th percentile of all row means.

        :param arr_orig: Original image array (H, W) uint8.
        :param arr_proc: Processed image array (H, W) uint8.
        :param call_pct: Percentile threshold for identifying call rows.
        :return:         Scalar >= 0.  Values > 1.0 indicate boosted energy
                         (expected for CLAHE).
        """
        row_means = arr_orig.mean(axis=1)
        thresh    = np.percentile(row_means, call_pct)
        call_rows = row_means >= thresh
        if not call_rows.any():
            return float('nan')
        orig_mean = arr_orig[call_rows].mean()
        proc_mean = arr_proc[call_rows].mean()
        return float(proc_mean / (orig_mean + 1e-6))

    # ----------------------------------------------------------------------- #

    def render(self, crop_path: Path) -> None:
        """
        Produce and save a comparison figure for crop_path.

        :param crop_path: Path to the crop PNG.
        """
        img_orig = Image.open(crop_path).convert('L')
        arr_orig = np.array(img_orig, dtype=np.uint8)

        n_cols = 1 + len(self.methods)
        fig, axes = plt.subplots(
            2, n_cols,
            figsize     = (4 * n_cols, 7),
            gridspec_kw = {'height_ratios': [4, 1]},
        )
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        def _plot_col(col: int, arr: np.ndarray, title: str) -> None:
            ax_img  = axes[0, col]
            ax_prof = axes[1, col]
            ax_img.imshow(arr, cmap='gray', origin='upper', vmin=0, vmax=255)
            ax_img.set_title(title, fontsize=8)
            ax_img.axis('off')
            profile = arr.mean(axis=1)
            ax_prof.plot(profile, np.arange(len(profile)), color='steelblue', lw=1)
            ax_prof.invert_yaxis()
            ax_prof.set_xlim(0, 255)
            ax_prof.set_xlabel('mean DN', fontsize=7)
            if col == 0:
                ax_prof.set_ylabel('freq bin', fontsize=7)
            ax_prof.tick_params(labelsize=6)

        # Column 0: original.
        _plot_col(0, arr_orig, 'Original')

        # Remaining columns: one per method.
        for i, (label, method) in enumerate(self.methods, start=1):
            try:
                arr_proc = np.array(method(img_orig), dtype=np.uint8)
                retained = self._call_energy_retained(arr_orig, arr_proc)
                title    = f'{label}\ncall energy retained: {retained:.0%}'
            except Exception as exc:
                arr_proc = arr_orig.copy()
                title    = f'{label}\nERROR: {exc}'
            _plot_col(i, arr_proc, title)

        fig.suptitle(crop_path.name, fontsize=10, y=1.01)
        plt.tight_layout()
        out_path = self.out_dir / f'{crop_path.stem}_bg_test.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        log.info(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='test_bg_subtract',
        description=(
            'Compare per-row percentile subtraction and CLAHE for bat '
            'spectrogram background suppression.\n'
            'One output PNG per input crop, with one column per method.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--crops', nargs='+', required=True, metavar='PNG',
        help='One or more crop PNG paths to test.',
    )
    # -- percentile subtraction args ----------------------------------------
    parser.add_argument(
        '--percentiles', nargs='+', type=float,
        default=[5.0, 10.0, 20.0, 50.0],
        metavar='PCT',
        help='Per-row percentile values to test (default: 5 10 20 50).',
    )
    parser.add_argument(
        '--min-floor-dn', type=int, default=4, metavar='DN',
        help='Silence guard: rows with floor below this DN are untouched. Default: 4.',
    )
    parser.add_argument(
        '--no-subtract', action='store_true', default=False,
        help='Skip all percentile-subtraction columns.',
    )
    # -- CLAHE args ----------------------------------------------------------
    parser.add_argument(
        '--clahe-clip', nargs='+', type=float, default=[2.0],
        metavar='CLIP',
        help='CLAHE clip limit(s) to test (default: 2.0). Typical range: 1.0-4.0.',
    )
    parser.add_argument(
        '--clahe-tile', type=int, default=8, metavar='N',
        help='CLAHE tile grid size NxN (default: 8).',
    )
    parser.add_argument(
        '--no-clahe', action='store_true', default=False,
        help='Skip all CLAHE columns.',
    )
    # -- output --------------------------------------------------------------
    parser.add_argument(
        '--out-dir', required=True, metavar='DIR',
        help='Directory to write output comparison PNGs.',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    methods: List[Tuple[str, object]] = []

    if not args.no_subtract:
        for pct in sorted(args.percentiles):
            sub = BgSubtractor(percentile=pct, min_floor_dn=args.min_floor_dn)
            methods.append((sub.label, sub))

    if not args.no_clahe:
        if not _HAVE_CV2:
            log.warn(
                'opencv-python not found -- CLAHE columns will be skipped. '
                'Install with: pip install opencv-python'
            )
        else:
            for clip in sorted(args.clahe_clip):
                norm = ClaheNormaliser(clip_limit=clip, tile_size=args.clahe_tile)
                methods.append((norm.label, norm))

    if not methods:
        log.err('No methods selected -- nothing to do.')
        sys.exit(1)

    missing = [p for p in args.crops if not Path(p).exists()]
    if missing:
        for p in missing:
            log.err(f'Crop not found: {p}')
        sys.exit(1)

    renderer = BgTestRenderer(methods=methods, out_dir=Path(args.out_dir))
    for crop_path in args.crops:
        renderer.render(Path(crop_path))

    log.info(f'Done. {len(args.crops)} figures written to {args.out_dir}')


if __name__ == '__main__':
    main()
