#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-25
"""
Runs the full analysis pipeline for one or more chirp subpopulations.

For each requested subpopulation, this script repeats the following steps,
placing all outputs in a dedicated subdirectory named after the subpopulation:

  1. basic_charting.py      – bar chart of cluster first/last frequencies
  2. bat_measures_normality.py – Shapiro-Wilk normality test per cluster
  3. charting.py            – normality-in-clusters heatmap
  4. measures_in_clusters.py – measure discrimination analysis
  5. viz_measures_in_clusters.py – tendency and effect-size heatmaps

The histograms sub-command of viz_measures_in_clusters.py is NOT included
because it requires choosing a specific measure and cluster set interactively;
run it manually once you know which measures/clusters are interesting.

Usage examples
--------------
  # Single subpopulation:
  python run_analysis_per_subpopulation.py idiom-internal

  # Multiple subpopulations:
  python run_analysis_per_subpopulation.py idiom-internal idiom-starts idiom-ends

  # All subpopulations + force overwrite of existing outputs:
  python run_analysis_per_subpopulation.py \
      idiom-internal idiom-starts idiom-ends idiom-any non-idiom-random \
      --force

  # Override the root output directories:
  python run_analysis_per_subpopulation.py idiom-internal \
      --viz_root   ~/MyProject/viz \
      --barn_root  ~/MyProject/barn_results
"""

import argparse
import os
import subprocess
import sys
from enum import StrEnum
from pathlib import Path


# ─── path constants ───────────────────────────────────────────────────────────

HOME = Path.home()
WORKSPACE = HOME / "VSCodeWorkspaces" / "bats"

# Input data (the full population augmented file)
DATA_ROOT = (WORKSPACE / "src" / "result_analysis" / "data" /
             "andrewChen" / "analysis_results" /
             "2022_barn_2secs_myca_quantile_1_16")
FULL_POPULATION = DATA_ROOT / "all_chirp_measures_augmented.csv"
FULL_POPULATION_SCALED = DATA_ROOT / "all_chirp_measures_scaled.csv"

# Directory roots for outputs (can be overridden via CLI)
DEFAULT_VIZ_ROOT   = WORKSPACE / "src" / "result_analysis" / "data" / "AnalysisViz"
DEFAULT_BARN_ROOT  = WORKSPACE / "src" / "result_analysis" / "data" / "barn_results"

# Script locations (relative to the workspace root)
SCRIPTS = {
    'basic_charting':          WORKSPACE / "src" / "result_analysis" / "basic_charting.py",
    'bat_measures_normality':  WORKSPACE / "src" / "result_analysis" / "bat_measures_normality.py",
    'charting':                WORKSPACE / "src" / "result_analysis" / "charting.py",
    'measures_in_clusters':    WORKSPACE / "src" / "result_analysis" / "measures_in_clusters.py",
    'viz_measures_in_clusters':WORKSPACE / "src" / "result_analysis" / "viz_measures_in_clusters.py",
}

# Subpopulation CSV files produced by create_all_subpopulations.sh
SUBPOP_FILES = {
    'idiom-internal':   DATA_ROOT / "all_chirp_measures_idiom_internal.csv",
    'idiom-starts':     DATA_ROOT / "all_chirp_measures_idiom_starts.csv",
    'idiom-ends':       DATA_ROOT / "all_chirp_measures_idiom_ends.csv",
    # idiom-any: not produced by the shell script; use augmented file filtered live,
    # or add a separate generation step. For now map to the augmented file as fallback.
    'idiom-any':        DATA_ROOT / "all_chirp_measures_idiom_any.csv",
    'non-idiom-random': DATA_ROOT / "all_chirp_measures_match_idiom_start_pop.csv",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

class PopulationType(StrEnum):
    IDIOM_INTERNAL   = 'idiom-internal'
    IDIOM_STARTS     = 'idiom-starts'
    IDIOM_ENDS       = 'idiom-ends'
    IDIOM_ANY        = 'idiom-any'
    NON_IDIOM_RANDOM = 'non-idiom-random'


def run(cmd: list[str], description: str, dry_run: bool = False) -> int:
    """
    Run a subprocess command, printing what is being run.

    :param cmd: command as a list of strings
    :param description: human-readable description for log output
    :param dry_run: if True, only print the command without executing
    :return: process return code (0 on success; 0 for dry-run)
    """
    print(f"\n{'─'*60}")
    print(f"  STEP: {description}")
    print(f"  CMD : {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")
    if dry_run:
        print("  [DRY RUN – not executed]")
        return 0
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: command exited with code {result.returncode}", file=sys.stderr)
    return result.returncode


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─── per-subpopulation pipeline ───────────────────────────────────────────────

def run_pipeline_for_subpop(
        pop_type: str,
        subpop_file: Path,
        viz_root: Path,
        barn_root: Path,
        force: bool,
        dry_run: bool,
        normality_outfile_override: Path | None = None,
) -> None:
    """
    Execute the full analysis pipeline for one subpopulation.

    Output sets are placed in:
        <viz_root>/<pop_type>/          – visualizations
        <barn_root>/<pop_type>/         – analysis CSVs

    :param pop_type: subpopulation identifier string (e.g. 'idiom-internal')
    :param subpop_file: path to the subpopulation CSV
    :param viz_root: root directory for visualization outputs
    :param barn_root: root directory for analysis-result outputs
    :param force: whether to overwrite existing output files
    :param dry_run: if True, print commands but do not execute them
    :param normality_outfile_override: optional explicit path for the
        normality CSV produced in step 2; derived automatically when None
    """

    # ── output directories for this subpopulation ──
    subpop_viz_dir  = ensure_dir(viz_root  / pop_type)
    subpop_barn_dir = ensure_dir(barn_root / pop_type)

    force_flag = ["--force"] if force else []

    # ── derived intermediate filenames ──
    normality_csv = (normality_outfile_override
                     if normality_outfile_override is not None
                     else subpop_barn_dir / "bats_measures_normality_all.csv")

    cluster_profiles_csv = subpop_barn_dir / "meas_towards_clusters_cluster_profiles.csv"

    print(f"\n{'═'*60}")
    print(f"  SUBPOPULATION: {pop_type}")
    print(f"  Input file   : {subpop_file}")
    print(f"  Viz output   : {subpop_viz_dir}")
    print(f"  Barn output  : {subpop_barn_dir}")
    print(f"{'═'*60}")

    # ── Step 1: bar chart of cluster first/last ──────────────────────────────
    barchart_out = subpop_viz_dir / "cluster_firsts_and_lasts_barchart.png"
    run(
        [
            sys.executable, str(SCRIPTS['basic_charting']),
            str(subpop_file),
            "bar",
            "cluster", "is_first", "is_last",
            "--title", "Cluster first/last in sequence",
            "--xlabel", "Cluster ID",
            "--ylabel", "Frequency",
            "--outfile", str(barchart_out),
        ],
        description=f"[{pop_type}] Bar chart – cluster first/last in sequence",
        dry_run=dry_run,
    )

    # ── Step 2: Shapiro-Wilk normality test ──────────────────────────────────
    run(
        [
            sys.executable, str(SCRIPTS['bat_measures_normality']),
            str(subpop_file),
            "shapiro",
            "--clustered",
            "--numerics",
            "--outfile", str(normality_csv),
        ] + force_flag,
        description=f"[{pop_type}] Shapiro-Wilk normality test per cluster",
        dry_run=dry_run,
    )

    # ── Step 3: normality heatmap ─────────────────────────────────────────────
    run(
        [
            sys.executable, str(SCRIPTS['charting']),
            "--normality_in_clusters", str(normality_csv),
            "--outdir", str(subpop_viz_dir),
        ],
        description=f"[{pop_type}] Normality-in-clusters heatmap",
        dry_run=dry_run,
    )

    # ── Step 4: measure discrimination analysis ───────────────────────────────
    run(
        [
            sys.executable, str(SCRIPTS['measures_in_clusters']),
            "--autocols",
            "--outdir", str(subpop_barn_dir),
        ] + force_flag + [
            str(subpop_file),
        ],
        description=f"[{pop_type}] Measure discrimination analysis",
        dry_run=dry_run,
    )

    # ── Step 5a: tendency heatmap ─────────────────────────────────────────────
    tendency_viz_dir = ensure_dir(subpop_viz_dir / "MeasureImportance")
    run(
        [
            sys.executable, str(SCRIPTS['viz_measures_in_clusters']),
            str(subpop_file),
            str(cluster_profiles_csv),
            "--illustrations", "tendency-heat", "effect-size-heat",
            "--outdir", str(tendency_viz_dir),
        ],
        description=f"[{pop_type}] Tendency + effect-size heatmaps",
        dry_run=dry_run,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )

    parser.add_argument(
        "populations",
        metavar="POPULATION",
        type=PopulationType,
        nargs="+",
        choices=list(PopulationType),
        help=(
            "One or more subpopulation types to analyse.\n"
            "Choices: " + ", ".join(list(PopulationType))
        ),
    )

    parser.add_argument(
        "--viz_root",
        type=Path,
        default=DEFAULT_VIZ_ROOT,
        help=(
            "Root directory for visualization outputs.\n"
            f"Default: {DEFAULT_VIZ_ROOT}\n"
            "A subdirectory named after each subpopulation will be created here."
        ),
    )

    parser.add_argument(
        "--barn_root",
        type=Path,
        default=DEFAULT_BARN_ROOT,
        help=(
            "Root directory for analysis-result CSVs.\n"
            f"Default: {DEFAULT_BARN_ROOT}\n"
            "A subdirectory named after each subpopulation will be created here."
        ),
    )

    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output files without asking.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print commands that would be executed without running them.",
    )

    return parser.parse_args()


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    for pop_type in args.populations:
        subpop_file = SUBPOP_FILES.get(str(pop_type))

        if subpop_file is None:
            print(f"ERROR: No subpopulation file mapping found for '{pop_type}'", file=sys.stderr)
            sys.exit(1)

        if not subpop_file.exists():
            print(
                f"ERROR: Subpopulation file not found: {subpop_file}\n"
                f"       Run create_all_subpopulations.sh first (or check SUBPOP_FILES mapping).",
                file=sys.stderr,
            )
            sys.exit(1)

        run_pipeline_for_subpop(
            pop_type=str(pop_type),
            subpop_file=subpop_file,
            viz_root=args.viz_root,
            barn_root=args.barn_root,
            force=args.force,
            dry_run=args.dry_run,
        )

    print("\n✓ All requested subpopulations processed.")


if __name__ == "__main__":
    main()
