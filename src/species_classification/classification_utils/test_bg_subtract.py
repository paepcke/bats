#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-12 16:26:55
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-12 16:28:31
# **********************************************************

"""
test_bg_subtract.py
===================
Visual sanity-check for the per-row low-percentile background subtraction
implemented in ``CropPreprocessor.subtract_bg()``.

For each crop supplied on the command line the script renders a figure with
one column per ``--percentile`` value tested, showing the original and
cleaned spectrogram side by side.  This lets you verify that narrow-band /
shallow-sweep calls (Lano, Laci, Epfu) are preserved while background
streaks are suppressed, before committing the parameter choice to
``chirps_to_spectros.py``.

Usage
-----
::

    python test_bg_subtract.py \\
        --crops  /qnap/bats/jr_pipeline/data/bat_crops/20220706_lake2/00006557.png \\
                 /qnap/bats/jr_pipeline/data/bat_crops/20210608_barn/00008887.png \\
        --percentiles 5 10 20 50 \\
        --out-dir /qnap/bats/jr_pipeline/gradcam_results/bg_test

One PNG per crop is written to ``--out-dir``, named
``<crop_stem>_bg_test.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from logging_service import LoggingService

log = LoggingService()


# ---------------------------------------------------------------------------
# Core subtraction (mirrors CropPreprocessor.subtract_bg exactly)
# ---------------------------------------------------------------------------

class BgSubtractor:
    """
    Standalone version of ``CropPreprocessor.subtract_bg`` for testing.

    :param percentile:  Row percentile used as noise-floor estimate.
    :param min_floor_dn: Rows with estimated floor below this value are skipped.
    """

    def __init__(self, percentile: float = 10.0, min_floor_dn: int = 4) -> None:
        self.percentile   = percentile
        self.min_floor_dn = min_floor_dn

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply background subtraction to a grayscale PIL image.

        :param img: Grayscale PIL image (mode ``'L'``).
        :return:    Cleaned grayscale PIL image.
        """
        arr   = np.array(img, dtype=np.int16)
        floor = np.percentile(arr, self.percentile, axis=1, keepdims=True)
        floor = np.where(floor < self.min_floor_dn, 0, floor)
        arr   = np.clip(arr - floor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='L')


# ---------------------------------------------------------------------------
# Figure renderer
# ---------------------------------------------------------------------------

class BgTestRenderer:
    """
    Render a comparison figure for one crop across multiple percentile values.

    :param percentiles:  List of percentile values to test.
    :param min_floor_dn: Silence guard threshold (same for all percentiles).
    :param out_dir:      Directory to write output PNGs.
    """

    def __init__(
        self,
        percentiles:  List[float],
        min_floor_dn: int,
        out_dir:      Path,
    ) -> None:
        self.percentiles  = percentiles
        self.min_floor_dn = min_floor_dn
        self.out_dir      = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def render(self, crop_path: Path) -> None:
        """
        Produce and save a comparison figure for *crop_path*.

        The figure has ``1 + len(percentiles)`` columns:
        * Column 0: original spectrogram
        * Columns 1+: cleaned at each percentile

        A row-energy profile (mean intensity per frequency bin) is plotted
        below each spectrogram to make call vs. background energy visible.

        :param crop_path: Path to the crop PNG.
        """
        img_orig = Image.open(crop_path).convert('L')
        arr_orig = np.array(img_orig, dtype=np.uint8)

        n_cols  = 1 + len(self.percentiles)
        fig, axes = plt.subplots(
            2, n_cols,
            figsize      = (4 * n_cols, 7),
            gridspec_kw  = {'height_ratios': [4, 1]},
        )

        def _plot_col(col: int, arr: np.ndarray, title: str) -> None:
            ax_img  = axes[0, col]
            ax_prof = axes[1, col]

            ax_img.imshow(arr, cmap='gray', origin='upper', vmin=0, vmax=255)
            ax_img.set_title(title, fontsize=9)
            ax_img.axis('off')

            # Row-energy profile: mean intensity per frequency bin (row).
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

        # Columns 1+: cleaned at each percentile.
        for i, pct in enumerate(self.percentiles, start=1):
            subtractor = BgSubtractor(
                percentile   = pct,
                min_floor_dn = self.min_floor_dn,
            )
            arr_clean = np.array(subtractor(img_orig), dtype=np.uint8)

            # Compute fraction of original energy retained in call rows.
            # "Call rows" = rows where original mean DN > 10th pct of row means.
            row_means_orig  = arr_orig.mean(axis=1)
            call_row_thresh = np.percentile(row_means_orig, 80)
            call_rows       = row_means_orig >= call_row_thresh
            if call_rows.any():
                retained = (arr_clean[call_rows].mean() /
                            (arr_orig[call_rows].mean() + 1e-6))
            else:
                retained = float('nan')

            title = (f'pct={pct:.0f}  '
                     f'floor_guard={self.min_floor_dn}\n'
                     f'call energy retained: {retained:.0%}')
            _plot_col(i, arr_clean, title)

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
            'Visual comparison of per-row low-percentile background subtraction '
            'at multiple percentile values.  One output PNG per input crop.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--crops', nargs='+', required=True, metavar='PNG',
        help='One or more crop PNG paths to test.',
    )
    parser.add_argument(
        '--percentiles', nargs='+', type=float,
        default=[5.0, 10.0, 20.0, 50.0],
        metavar='PCT',
        help=(
            'Percentile values to test (default: 5 10 20 50). '
            '50 = median (original behaviour). '
            '10 = recommended starting point.'
        ),
    )
    parser.add_argument(
        '--min-floor-dn', type=int, default=4, metavar='DN',
        help=(
            'Rows whose estimated floor is below this value are left untouched. '
            'Default: 4.'
        ),
    )
    parser.add_argument(
        '--out-dir', required=True, metavar='DIR',
        help='Directory to write output comparison PNGs.',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args     = _parse_args()
    out_dir  = Path(args.out_dir)
    renderer = BgTestRenderer(
        percentiles  = sorted(args.percentiles),
        min_floor_dn = args.min_floor_dn,
        out_dir      = out_dir,
    )

    missing = [p for p in args.crops if not Path(p).exists()]
    if missing:
        for p in missing:
            log.err(f'Crop not found: {p}')
        sys.exit(1)

    for crop_path in args.crops:
        renderer.render(Path(crop_path))

    log.info(f'Done. {len(args.crops)} figures written to {out_dir}')


if __name__ == '__main__':
    main()
