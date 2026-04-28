#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-27 18:51:22
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-27 19:02:41
# **********************************************************
"""
ipi_rejection_diagnostic.py
=============================
Diagnose why 49K files were rejected as ``irregular_ipi`` by examining
the distribution of IPI CV values and pulse counts in the scrub report,
and test the harmonic-doubling hypothesis.

Harmonic hypothesis
-------------------
A bat call at fundamental F also radiates energy at 2F, 3F, etc.  The
pulse detector may fire on both the fundamental and the harmonic,
producing two near-simultaneous detections per true call.  The resulting
IPI sequence has apparent IPIs alternating between ~0 ms (fundamental →
harmonic of the same call) and the true IPI (harmonic → fundamental of
the next call).  This produces a CV roughly equal to 1.0 regardless of
the true call regularity, and a pulse count roughly 2× the true count.

Signature to look for in the data
-----------------------------------
If harmonics are the dominant cause:

1. ``ipi_cv`` for irregular_ipi files should cluster tightly around 1.0
   (not scattered broadly above the 1.5 threshold).
2. ``pulse_count`` for irregular_ipi files should be roughly 2× the
   pulse count of retained files with similar recording conditions.
3. The ``ipi_cv`` distribution of irregular_ipi files should have a
   sharp peak rather than a broad tail — a broad tail would suggest
   multiple independent causes.

Outputs
-------
Printed summary + two PNG plots saved to ``--out-dir``:

``ipi_cv_distribution.png``
    Histogram of IPI CV values for ``irregular_ipi`` files, with the
    rejection threshold (default 1.5) and the harmonic CV prediction
    (~1.0) marked.

``pulse_count_comparison.png``
    Side-by-side pulse count distributions for ``irregular_ipi`` vs
    ``retained`` files.

Usage
-----
::

    python ipi_rejection_diagnostic.py \\
        --scrub-report /data2/marsh_stanford_processed/scrub_report.csv \\
        --out-dir      /data2/marsh_stanford_processed/diagnostics \\
        --ipi-cv-thresh 1.5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    """
    Run the IPI rejection diagnostic.

    :return: None
    """
    parser = argparse.ArgumentParser(
        prog='ipi_rejection_diagnostic.py',
        description='Diagnose irregular_ipi rejections in a WavScrubber report.',
    )
    parser.add_argument(
        '--scrub-report', required=True, type=Path, metavar='CSV',
        help='scrub_report.csv from wav_file_scrubber.py',
    )
    parser.add_argument(
        '--out-dir', required=True, type=Path, metavar='DIR',
        help='Directory for output plots.',
    )
    parser.add_argument(
        '--ipi-cv-thresh', type=float, default=1.5, metavar='FLOAT',
        help='IPI CV rejection threshold used in the scrub run (default 1.5).',
    )
    args = parser.parse_args()

    if not args.scrub_report.exists():
        print(f"ERROR: scrub report not found: {args.scrub_report}",
              file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load report ───────────────────────────────────────────────────
    print(f"Loading {args.scrub_report} …")
    df = pd.read_csv(args.scrub_report, low_memory=False)
    print(f"  {len(df):,} total rows")

    # Normalise verdict strings to lowercase for case-insensitive matching —
    # the scrubber writes 'retained' but earlier versions used 'RETAINED'.
    df['verdict'] = df['verdict'].astype(str).str.lower()
    ipi_df      = df[df['verdict'] == 'irregular_ipi'].copy()
    retained_df = df[df['verdict'] == 'retained'].copy()

    print(f"  irregular_ipi : {len(ipi_df):,}")
    print(f"  retained      : {len(retained_df):,}")

    if len(ipi_df) == 0:
        print("No irregular_ipi rows — nothing to diagnose.")
        sys.exit(0)

    # ── IPI CV summary ────────────────────────────────────────────────
    cv = ipi_df['ipi_cv'].dropna()
    print(f"\nIPI CV for irregular_ipi files (n={len(cv):,}):")
    print(f"  min    : {cv.min():.4f}")
    print(f"  p25    : {cv.quantile(0.25):.4f}")
    print(f"  median : {cv.median():.4f}")
    print(f"  mean   : {cv.mean():.4f}")
    print(f"  p75    : {cv.quantile(0.75):.4f}")
    print(f"  p90    : {cv.quantile(0.90):.4f}")
    print(f"  max    : {cv.max():.4f}")

    # Fraction near the harmonic prediction (CV ~ 1.0, ±0.2)
    near_harmonic = ((cv >= 0.8) & (cv <= 1.2)).mean()
    just_over_thresh = ((cv >= args.ipi_cv_thresh) &
                        (cv <= args.ipi_cv_thresh + 0.3)).mean()
    print(f"\n  Fraction with CV in [0.8, 1.2]  (harmonic zone) : "
          f"{near_harmonic*100:.1f}%")
    print(f"  Fraction with CV in [{args.ipi_cv_thresh:.1f}, "
          f"{args.ipi_cv_thresh+0.3:.1f}] (just over threshold) : "
          f"{just_over_thresh*100:.1f}%")

    if near_harmonic >= 0.50:
        print("\n  ✓ Majority of rejections cluster near CV=1.0 — "
              "HARMONIC DOUBLING is the dominant cause.")
    elif just_over_thresh >= 0.40:
        print("\n  ~ Many rejections just over the threshold — "
              "consider RAISING --ipi-cv slightly.")
    else:
        print("\n  ✗ CV values are broadly distributed — "
              "multiple causes likely; harmonic doubling alone does not explain "
              "the rejections.")

    # ── Pulse count comparison ────────────────────────────────────────
    ipi_pulses      = ipi_df['pulse_count'].dropna()
    retained_pulses = retained_df['pulse_count'].dropna()

    print(f"\nPulse count — irregular_ipi  : "
          f"median={ipi_pulses.median():.1f}  "
          f"mean={ipi_pulses.mean():.1f}")
    print(f"Pulse count — retained       : "
          f"median={retained_pulses.median():.1f}  "
          f"mean={retained_pulses.mean():.1f}")

    ratio = ipi_pulses.median() / retained_pulses.median() \
        if retained_pulses.median() > 0 else float('nan')
    print(f"Pulse count ratio (ipi/retained median): {ratio:.2f}")
    if 1.7 <= ratio <= 2.3:
        print("  ✓ Ratio near 2.0 — consistent with harmonic doubling "
              "(each call detected twice).")
    else:
        print("  Ratio not near 2.0 — harmonic doubling is not the sole "
              "explanation for elevated pulse counts.")

    # ── Plot 1: IPI CV distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(max(0, cv.min() - 0.1),
                       min(cv.max() + 0.1, 6.0), 80)
    ax.hist(cv.clip(upper=6.0), bins=bins, color='steelblue',
            alpha=0.8, edgecolor='none')
    ax.axvline(args.ipi_cv_thresh, color='red', lw=1.5, linestyle='--',
               label=f'rejection threshold ({args.ipi_cv_thresh})')
    ax.axvline(1.0, color='orange', lw=1.5, linestyle=':',
               label='harmonic-doubling prediction (CV≈1.0)')
    ax.set_xlabel('IPI coefficient of variation')
    ax.set_ylabel('File count')
    ax.set_title(f'IPI CV distribution — irregular_ipi files (n={len(cv):,})')
    ax.legend(fontsize=9)
    fig.tight_layout()
    out1 = args.out_dir / 'ipi_cv_distribution.png'
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out1}")

    # ── Plot 2: Pulse count comparison ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    max_pulse = int(max(ipi_pulses.quantile(0.99),
                        retained_pulses.quantile(0.99))) + 2
    bins_p = np.arange(0, max_pulse + 2)

    axes[0].hist(ipi_pulses.clip(upper=max_pulse), bins=bins_p,
                 color='salmon', alpha=0.8, edgecolor='none')
    axes[0].axvline(ipi_pulses.median(), color='darkred', lw=1.5,
                    linestyle='--',
                    label=f'median={ipi_pulses.median():.1f}')
    axes[0].set_title(f'irregular_ipi  (n={len(ipi_pulses):,})')
    axes[0].set_xlabel('Pulse count')
    axes[0].set_ylabel('File count')
    axes[0].legend(fontsize=9)

    axes[1].hist(retained_pulses.clip(upper=max_pulse), bins=bins_p,
                 color='steelblue', alpha=0.8, edgecolor='none')
    axes[1].axvline(retained_pulses.median(), color='navy', lw=1.5,
                    linestyle='--',
                    label=f'median={retained_pulses.median():.1f}')
    axes[1].set_title(f'retained  (n={len(retained_pulses):,})')
    axes[1].set_xlabel('Pulse count')
    axes[1].legend(fontsize=9)

    fig.suptitle('Pulse count: irregular_ipi vs retained', y=1.01)
    fig.tight_layout()
    out2 = args.out_dir / 'pulse_count_comparison.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out2}")

    print("\nDone.")


if __name__ == '__main__':
    main()
