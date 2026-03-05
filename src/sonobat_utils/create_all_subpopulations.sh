#!/bin/bash
# @Author: Andreas Paepcke
# @Date:    2026-02-25 09:04:05
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-25 12:33:49

# Initialize force variable
FORCE=""

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE="--force"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Define base path to keep the script readable
BASE_DIR="$HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16"

# ---------------------------------------------------------
# Idiom-any chirps:
# ---------------------------------------------------------
OUT_INTERNAL="$BASE_DIR/all_chirp_measures_idiom_any.csv"

echo "Subpopulation: all chirps internal, or on start/stop:"
src/sonobat_utils/data_selection.py \
    "$BASE_DIR/all_chirp_measures_augmented.csv" \
    --population_type idiom-any \
    --rand_selector all \
    --outfile "$OUT_INTERNAL" \
    $FORCE

echo "Yielded $(($(wc -l < "$OUT_INTERNAL") - 1)) rows"

# ---------------------------------------------------------
# Idiom-internal chirps:
# ---------------------------------------------------------
OUT_INTERNAL="$BASE_DIR/all_chirp_measures_idiom_internal.csv"

echo "Subpopulation: all idiom-internal chirps:"
src/sonobat_utils/data_selection.py \
    "$BASE_DIR/all_chirp_measures_augmented.csv" \
    --population_type idiom-internal \
    --rand_selector all \
    --outfile "$OUT_INTERNAL" \
    $FORCE

echo "Yielded $(($(wc -l < "$OUT_INTERNAL") - 1)) rows"

# ---------------------------------------------------------
# Idiom-starts chirps:
# ---------------------------------------------------------
OUT_STARTS="$BASE_DIR/all_chirp_measures_idiom_starts.csv"

echo "Subpopulation: all idiom-start chirps:"
src/sonobat_utils/data_selection.py \
    "$BASE_DIR/all_chirp_measures_augmented.csv" \
    --population_type idiom-starts \
    --rand_selector all \
    --outfile "$OUT_STARTS" \
    $FORCE

echo "Yielded $(($(wc -l < "$OUT_STARTS") - 1)) rows"

# ---------------------------------------------------------
# Idiom-ends chirps:
# ---------------------------------------------------------
OUT_ENDS="$BASE_DIR/all_chirp_measures_idiom_ends.csv"

echo "Subpopulation: all idiom-end chirps:"
src/sonobat_utils/data_selection.py \
    "$BASE_DIR/all_chirp_measures_augmented.csv" \
    --population_type idiom-ends \
    --rand_selector all \
    --outfile "$OUT_ENDS" \
    $FORCE

echo "Yielded $(($(wc -l < "$OUT_ENDS") - 1)) rows"

# ---------------------------------------------------------
# Non-Idiom chirps (match starts):
# ---------------------------------------------------------
OUT_MATCH_START="$BASE_DIR/all_chirp_measures_match_idiom_start_pop.csv"

echo "Subpopulation: all non-idiom chirps limit num rows to number of idioms"
src/sonobat_utils/data_selection.py \
    "$BASE_DIR/all_chirp_measures_augmented.csv" \
    --population_type non-idiom-random \
    --rand_selector match-idiom-start \
    --outfile "$OUT_MATCH_START" \
    $FORCE

echo "Yielded $(($(wc -l < "$OUT_MATCH_START") - 1)) rows"

# ---------------------------------------------------------
# Non-Idiom chirps (match internal):
# ---------------------------------------------------------
OUT_MATCH_INTERNAL="$BASE_DIR/all_chirp_measures_match_in_idiom_pop.csv"

echo "Subpopulation: all non-idiom chirps limit num rows to number chirps inside idioms"
src/sonobat_utils/data_selection.py \
    "$BASE_DIR/all_chirp_measures_augmented.csv" \
    --population_type non-idiom-random \
    --rand_selector match-in-idiom \
    --outfile "$OUT_MATCH_INTERNAL" \
    $FORCE

echo "Yielded $(($(wc -l < "$OUT_MATCH_INTERNAL") - 1)) rows"
