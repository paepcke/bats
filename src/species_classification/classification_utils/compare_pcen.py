#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-13 10:45:14
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-13 11:00:32
# **********************************************************

"""
compare_pcen.py
===============
Side-by-side comparison of PCEN vs standard log-power spectrogram crops
produced by two parallel runs of ``chirps_to_spectros.py``.

Given a root directory containing two crop subdirectories (one with PCEN,
one without), the script matches crops by their relative path
(partition/filename) and renders a two-panel figure for each matched pair.

Typical layout expected::

    <root>/
        pcen_test/
            20220520_barn/
                00000001.png
                00000002.png
                ...
        no_pcen_test/
            20220520_barn/
                00000001.png
                ...

The script walks the *reference* directory (default: the one whose name
contains "no_pcen" or is named first alphabetically), finds all PNGs, then
looks for the matching PNG in the *comparison* directory.  Only matched pairs
are rendered.

Usage
-----
::

    python compare_pcen.py \\
        --root      /data/bats/jr_pipeline \\
        --pcen-dir  pcen_test \\
        --ref-dir   no_pcen_test \\
        --out-dir   /data/bats/jr_pipeline/pcen_comparison \\
        --n-samples 20 \\
        --species   Lano Laci Coto Tabr

    # On sextus (no /qnap — use /data or wherever the dirs are mounted):
    python compare_pcen.py \\
        --root      /data/bats/jr_pipeline \\
        --pcen-dir  pcen_test \\
        --ref-dir   no_pcen_test \\
        --out-dir   /data/bats/jr_pipeline/pcen_comparison

Output
------
One PNG per matched pair written to ``--out-dir``, named
``<partition>_<stem>_compare.png``.

A summary table ``comparison_summary.txt`` is written listing all matched
pairs with their mean intensity and contrast ratio (std/mean) for each
version — a quick quantitative check that PCEN is actually changing
something.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from logging_service import LoggingService

log = LoggingService()


# ---------------------------------------------------------------------------
# Pair finder
# ---------------------------------------------------------------------------

class CropPairFinder:
    """
    Find matching crop PNG pairs between two output directories.

    Matching is on relative path: ``<partition>/<filename>.png``.
    Only pairs present in both directories are returned.

    Species labels are optionally loaded from a manifest CSV and attached
    to each pair, enabling species-aware figure titles and filenames.

    :param ref_dir:       Reference directory (no-PCEN crops).
    :param pcen_dir:      Comparison directory (PCEN crops).
    :param manifest_path: Optional path to manifest.csv.  Must have columns
                          ``crop_path`` and ``species``.  The ref-dir crops
                          are matched by basename of ``crop_path``.
    """

    def __init__(
        self,
        ref_dir:       Path,
        pcen_dir:      Path,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.ref_dir  = ref_dir
        self.pcen_dir = pcen_dir
        self._species_map: dict = {}

        if manifest_path is not None and manifest_path.exists():
            import pandas as pd
            log.info(f'Loading species labels from {manifest_path}')
            mdf = pd.read_csv(manifest_path, usecols=['crop_path', 'species'],
                              low_memory=False)
            # Key: partition/filename.png  (last two path components)
            mdf['_rel'] = mdf['crop_path'].apply(
                lambda p: '/'.join(Path(p).parts[-2:])
            )
            self._species_map = dict(
                zip(mdf['_rel'], mdf['species'])
            )
            log.info(f'  {len(self._species_map):,} species labels loaded')
        elif manifest_path is not None:
            log.warn(f'Manifest not found: {manifest_path} — no species labels')

    def find_pairs(
        self,
        n_samples: int = 0,
        seed:      int = 42,
    ) -> List[Tuple[Path, Path, str, str]]:
        """
        Return a list of ``(ref_path, pcen_path, relative_path, species)``
        tuples for all matched pairs, optionally limited to ``n_samples``.

        :param n_samples: Maximum pairs to return (0 = all).
        :param seed:      Random seed for sampling.
        :return:          List of ``(ref_path, pcen_path, rel_path, species)``
                          tuples.  ``species`` is ``'?'`` when unknown.
        """
        ref_pngs = {
            p.relative_to(self.ref_dir): p
            for p in self.ref_dir.rglob('*.png')
        }
        pcen_pngs = {
            p.relative_to(self.pcen_dir): p
            for p in self.pcen_dir.rglob('*.png')
        }

        common = sorted(set(ref_pngs) & set(pcen_pngs))
        log.info(
            f'Found {len(ref_pngs):,} ref crops, {len(pcen_pngs):,} pcen crops, '
            f'{len(common):,} matched pairs'
        )

        if not common:
            log.err(
                f'No matched pairs found between:\n'
                f'  ref:  {self.ref_dir}\n'
                f'  pcen: {self.pcen_dir}\n'
                f'Check that both directories have the same partition '
                f'subdirectory structure and filenames.'
            )
            return []

        if n_samples > 0 and len(common) > n_samples:
            rng     = np.random.default_rng(seed)
            indices = rng.choice(len(common), size=n_samples, replace=False)
            common  = [common[i] for i in sorted(indices)]
            log.info(f'Sampled {len(common)} pairs')

        return [
            (ref_pngs[rel], pcen_pngs[rel], str(rel),
             self._species_map.get(str(rel), '?'))
            for rel in common
        ]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class PcenCompareRenderer:
    """
    Render side-by-side comparison figures for matched crop pairs.

    Each figure has three columns:
    * Reference (log-power normalisation)
    * PCEN
    * Difference (PCEN − ref, signed, shown with diverging colormap)

    A row-energy profile (mean DN per frequency bin) is shown beneath
    each spectrogram column.

    :param out_dir: Directory to write output figures.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #

    @staticmethod
    def _stats(arr: np.ndarray) -> dict:
        """
        Compute summary statistics for a uint8 image array.

        :param arr: 2-D uint8 array.
        :return:    Dict with keys mean, std, contrast (std/mean), p95.
        """
        f = arr.astype(np.float32)
        m = float(f.mean())
        s = float(f.std())
        return {
            'mean'     : round(m, 2),
            'std'      : round(s, 2),
            'contrast' : round(s / (m + 1e-6), 3),
            'p95'      : round(float(np.percentile(f, 95)), 2),
        }

    # ----------------------------------------------------------------------- #

    def render_pair(
        self,
        ref_path:  Path,
        pcen_path: Path,
        rel_path:  str,
        species:   str = '?',
    ) -> dict:
        """
        Render and save a comparison figure for one matched pair.

        :param ref_path:  Path to the reference (no-PCEN) crop PNG.
        :param pcen_path: Path to the PCEN crop PNG.
        :param rel_path:  Relative path string used for the figure title
                          and output filename.
        :param species:   Species label (e.g. ``'Lano'``) or ``'?'`` if unknown.
        :return:          Dict with ref and pcen stats for summary table.
        """
        arr_ref  = np.array(Image.open(ref_path).convert('L'),  dtype=np.uint8)
        arr_pcen = np.array(Image.open(pcen_path).convert('L'), dtype=np.uint8)

        diff = arr_pcen.astype(np.int16) - arr_ref.astype(np.int16)

        stats_ref  = self._stats(arr_ref)
        stats_pcen = self._stats(arr_pcen)

        fig, axes = plt.subplots(
            2, 3,
            figsize     = (13, 6),
            gridspec_kw = {'height_ratios': [4, 1]},
        )

        def _plot_col(col, arr, title, cmap='gray', vmin=0, vmax=255):
            ax_img  = axes[0, col]
            ax_prof = axes[1, col]
            im = ax_img.imshow(arr, cmap=cmap, origin='upper',
                               vmin=vmin, vmax=vmax)
            ax_img.set_title(title, fontsize=9)
            ax_img.axis('off')
            if cmap == 'gray':
                profile = arr.mean(axis=1)
                ax_prof.plot(profile, np.arange(len(profile)),
                             color='steelblue', lw=1)
                ax_prof.invert_yaxis()
                ax_prof.set_xlim(0, 255)
            else:
                # Difference column: plot signed profile
                profile = arr.mean(axis=1)
                ax_prof.plot(profile, np.arange(len(profile)),
                             color='firebrick', lw=1)
                ax_prof.invert_yaxis()
                ax_prof.axvline(0, color='gray', lw=0.5, ls='--')
            ax_prof.set_xlabel('mean DN', fontsize=7)
            if col == 0:
                ax_prof.set_ylabel('freq bin', fontsize=7)
            ax_prof.tick_params(labelsize=6)

        # Col 0: reference
        ref_label = (
            f'Log-power (ref)\n'
            f'mean={stats_ref["mean"]}  '
            f'contrast={stats_ref["contrast"]}  '
            f'p95={stats_ref["p95"]}'
        )
        _plot_col(0, arr_ref, ref_label)

        # Col 1: PCEN
        pcen_label = (
            f'PCEN\n'
            f'mean={stats_pcen["mean"]}  '
            f'contrast={stats_pcen["contrast"]}  '
            f'p95={stats_pcen["p95"]}'
        )
        _plot_col(1, arr_pcen, pcen_label)

        # Col 2: signed difference
        abs_max = max(abs(diff.min()), abs(diff.max()), 1)
        _plot_col(2, diff, 'PCEN − ref  (red=brighter, blue=darker)',
                  cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)

        fig.suptitle(f'{rel_path}   species: {species}', fontsize=9, y=1.01)
        plt.tight_layout()

        safe_name = rel_path.replace('/', '_').replace('\\', '_')
        sp_tag    = species if species != '?' else 'unknown'
        out_path  = self.out_dir / f'{sp_tag}_{safe_name}_compare.png'
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        return {
            'species'       : species,
            'rel_path'      : rel_path,
            'ref_mean'      : stats_ref['mean'],
            'ref_contrast'  : stats_ref['contrast'],
            'ref_p95'       : stats_ref['p95'],
            'pcen_mean'     : stats_pcen['mean'],
            'pcen_contrast' : stats_pcen['contrast'],
            'pcen_p95'      : stats_pcen['p95'],
        }

    # ----------------------------------------------------------------------- #

    def render_all(
        self,
        pairs: List[Tuple[Path, Path, str]],
    ) -> List[dict]:
        """
        Render comparison figures for all pairs and return summary records.

        :param pairs: List of ``(ref_path, pcen_path, rel_path)`` tuples.
        :return:      List of stats dicts, one per pair.
        """
        records = []
        for ref_path, pcen_path, rel_path, species in pairs:
            try:
                rec = self.render_pair(ref_path, pcen_path, rel_path, species)
                records.append(rec)
                log.info(
                    f'  [{species}] {rel_path}  '
                    f'ref_contrast={rec["ref_contrast"]}  '
                    f'pcen_contrast={rec["pcen_contrast"]}'
                )
            except Exception as exc:
                log.warn(f'  Failed on {rel_path}: {exc}')
        return records


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

class SummaryWriter:
    """
    Write a fixed-width text summary table from comparison records.

    :param out_dir: Directory to write ``comparison_summary.txt``.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def write(self, records: List[dict]) -> None:
        """
        Write summary table and aggregate stats to
        ``<out_dir>/comparison_summary.txt``.

        :param records: List of stats dicts from
                        :meth:`PcenCompareRenderer.render_all`.
        """
        if not records:
            return

        import pandas as pd
        df = pd.DataFrame(records)

        lines = ['PCEN vs Log-power crop comparison', '=' * 60, '']

        # Overall aggregate stats.
        lines.append('Aggregate (mean across all pairs):')
        lines.append(
            f'  ref  contrast: {df["ref_contrast"].mean():.3f}  '
            f'mean_DN: {df["ref_mean"].mean():.1f}  '
            f'p95: {df["ref_p95"].mean():.1f}'
        )
        lines.append(
            f'  pcen contrast: {df["pcen_contrast"].mean():.3f}  '
            f'mean_DN: {df["pcen_mean"].mean():.1f}  '
            f'p95: {df["pcen_p95"].mean():.1f}'
        )
        lines.append('')

        # Per-species aggregate stats.
        if 'species' in df.columns and df['species'].nunique() > 1:
            lines.append('Per-species aggregate:')
            for sp, grp in df.groupby('species'):
                lines.append(
                    f'  {sp:6s}  n={len(grp):3d}  '
                    f'ref_contrast={grp["ref_contrast"].mean():.3f}  '
                    f'pcen_contrast={grp["pcen_contrast"].mean():.3f}  '
                    f'ref_mean={grp["ref_mean"].mean():.1f}  '
                    f'pcen_mean={grp["pcen_mean"].mean():.1f}'
                )
            lines.append('')

        # Per-pair table.
        cols = ['species', 'rel_path', 'ref_mean', 'ref_contrast', 'ref_p95',
                'pcen_mean', 'pcen_contrast', 'pcen_p95']
        col_w = [max(len(c), max(len(str(r[c])) for r in records))
                 for c in cols]
        header = '  '.join(c.ljust(w) for c, w in zip(cols, col_w))
        sep    = '  '.join('-' * w for w in col_w)
        lines += [header, sep]
        for r in records:
            lines.append(
                '  '.join(str(r[c]).ljust(w) for c, w in zip(cols, col_w))
            )

        out_path = self.out_dir / 'comparison_summary.txt'
        out_path.write_text('\n'.join(lines) + '\n')
        log.info(f'Summary written to {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='compare_pcen',
        description=(
            'Side-by-side comparison of PCEN vs log-power spectrogram crops\n'
            'from two parallel chirps_to_spectros.py runs.\n\n'
            'Matches crops by relative path (partition/filename.png) and\n'
            'renders a three-panel figure: ref | PCEN | difference.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--root', required=True, metavar='DIR',
        help=(
            'Root directory containing both crop subdirectories.\n'
            'On sextus use the local mount point, e.g. /data/bats/jr_pipeline\n'
            'rather than /qnap/bats/jr_pipeline.'
        ),
    )
    parser.add_argument(
        '--ref-dir', default='no_pcen_test', metavar='NAME',
        help=(
            'Name of the reference (log-power) subdirectory under --root.\n'
            'Default: no_pcen_test'
        ),
    )
    parser.add_argument(
        '--pcen-dir', default='pcen_test', metavar='NAME',
        help=(
            'Name of the PCEN subdirectory under --root.\n'
            'Default: pcen_test'
        ),
    )
    parser.add_argument(
        '--out-dir', required=True, metavar='DIR',
        help='Directory to write comparison figures and summary.',
    )
    parser.add_argument(
        '--manifest', default=None, metavar='CSV',
        help=(
            'Path to manifest.csv from chirps_to_spectros.py.  When supplied,\n'
            'species labels are looked up per crop and included in figure\n'
            'titles and output filenames.  The ref-dir manifest is used;\n'
            'typically <root>/<ref-dir>/manifest.csv or the master manifest.'
        ),
    )
    parser.add_argument(
        '--n-samples', type=int, default=0, metavar='N',
        help='Maximum pairs to compare (default: 0 = all matched pairs).',
    )
    parser.add_argument(
        '--seed', type=int, default=42, metavar='N',
        help='Random seed for sampling (default: 42).',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args    = _parse_args()
    root    = Path(args.root)
    ref_dir = root / args.ref_dir
    pcen_dir = root / args.pcen_dir
    out_dir = Path(args.out_dir)

    for d, name in [(ref_dir, '--ref-dir'), (pcen_dir, '--pcen-dir')]:
        if not d.exists():
            log.err(f'{name} not found: {d}')
            sys.exit(1)

    manifest_path = None
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif (ref_dir / 'manifest.csv').exists():
        manifest_path = ref_dir / 'manifest.csv'
        log.info(f'Auto-detected manifest at {manifest_path}')

    finder = CropPairFinder(
        ref_dir       = ref_dir,
        pcen_dir      = pcen_dir,
        manifest_path = manifest_path,
    )
    pairs  = finder.find_pairs(n_samples=args.n_samples, seed=args.seed)

    if not pairs:
        sys.exit(1)

    renderer = PcenCompareRenderer(out_dir=out_dir)
    records  = renderer.render_all(pairs)

    SummaryWriter(out_dir).write(records)
    log.info(f'Done. {len(records)} figures written to {out_dir}')


if __name__ == '__main__':
    main()
