#!/usr/bin/env bash
# *************************************
# @Author: Andreas Paepcke
# @Date:   2026-04-16 13:42:25
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-25 08:52:27
# *************************************

# Run three related random forest trainings in parallel:
#
#   1. An 'omnibus' RF for all species
#   2. A binary RF for disambiguating Coto and Tabr
#   3. Another binary RF for disambiguating Lano and Tabr
#
# Once these trainings are done, run a hierarchical inference
# that evaluates success. That inference script is also for
# use with unlabeled data.
#
# Usage: train_random_forests.sh <chirp_measures_parquet>

POLL_INTERVAL=10   # seconds between progress updates

# ---- Argument validation -------------------------------------------- #

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename $0) <chirp_measures_parquet>" >&2
    echo "  chirp_measures_parquet: bats_*.parquet file produced by" >&2
    echo "                          sb_measures_postprocessing.py" >&2
    exit 1
fi

CHIRP_MEASURES=$1

if [[ ! -f "$CHIRP_MEASURES" ]]; then
    echo "Error: parquet file not found: $CHIRP_MEASURES" >&2
    exit 1
fi

# ---- Output directories --------------------------------------------- #

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
BASE=/data/random_forest/results_${TIMESTAMP}

DEST_DIR_MAIN=$BASE/main_run
DEST_DIR_COTO_TABR=$BASE/binary_coto_tabr
DEST_DIR_LANO_TABR=$BASE/binary_lano_tabr
DEST_DIR_VALIDATION=$BASE/validation

# Create all output dirs upfront so log redirects succeed immediately.
mkdir -p $DEST_DIR_MAIN $DEST_DIR_COTO_TABR $DEST_DIR_LANO_TABR $DEST_DIR_VALIDATION

# ---- Helper: print last log line with a labeled prefix -------------- #
# Usage: show_progress <label> <logfile>
show_progress() {
    local label=$1
    local logfile=$2
    if [[ -f "$logfile" ]]; then
        local last
        last=$(tail -1 "$logfile" 2>/dev/null)
        if [[ -n "$last" ]]; then
            printf "  [%-14s] %s\n" "$label" "$last"
        fi
    fi
}

# ---- Launch three training runs in parallel ------------------------- #

echo "$(date '+%H:%M:%S')  Launching training runs..."
TRAIN_START=$(date +%s)

src/species_classification/training/species_pred_random_forest.py \
    --input "$CHIRP_MEASURES" \
    --out-dir $DEST_DIR_MAIN \
    --min-species-count 500 --n-estimators 300 --n-jobs -1 \
    --exclude-species HiF Anpa Lafr Myvo \
    > $DEST_DIR_MAIN/train.log 2>&1 &
PID_MAIN=$!

src/species_classification/inference/species_pred_random_forest.py \
    --input "$CHIRP_MEASURES" \
    --out-dir $DEST_DIR_COTO_TABR \
    --mode binary --species-pair Coto Tabr \
    --n-estimators 300 --n-jobs -1 \
    > $DEST_DIR_COTO_TABR/train.log 2>&1 &
PID_COTO_TABR=$!

src/species_classification/inference/species_pred_random_forest.py \
    --input "$CHIRP_MEASURES" \
    --out-dir $DEST_DIR_LANO_TABR \
    --mode binary --species-pair Lano Tabr \
    --n-estimators 300 --n-jobs -1 \
    > $DEST_DIR_LANO_TABR/train.log 2>&1 &
PID_LANO_TABR=$!

# ---- Poll progress until all three training runs finish ------------- #

while kill -0 $PID_MAIN 2>/dev/null || \
      kill -0 $PID_COTO_TABR 2>/dev/null || \
      kill -0 $PID_LANO_TABR 2>/dev/null; do
    sleep $POLL_INTERVAL
    echo "--- $(date '+%H:%M:%S') ---"
    show_progress "omnibus"    $DEST_DIR_MAIN/train.log
    show_progress "coto_tabr"  $DEST_DIR_COTO_TABR/train.log
    show_progress "lano_tabr"  $DEST_DIR_LANO_TABR/train.log
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
TRAIN_MINS=$(( TRAIN_ELAPSED / 60 ))
TRAIN_SECS=$(( TRAIN_ELAPSED % 60 ))
echo ""
echo "$(date '+%H:%M:%S')  All training runs complete.  Elapsed: ${TRAIN_MINS}m ${TRAIN_SECS}s"
echo ""

# ---- Hierarchical inference + validation ---------------------------- #

echo "$(date '+%H:%M:%S')  Launching hierarchical inference..."
INFER_START=$(date +%s)

src/species_classification/inference/hierarchical_rf_predict.py \
    --input "$CHIRP_MEASURES" \
    --main-rf      $DEST_DIR_MAIN \
    --coto-tabr-rf $DEST_DIR_COTO_TABR \
    --lano-tabr-rf $DEST_DIR_LANO_TABR \
    --out-dir      $DEST_DIR_VALIDATION \
    > $DEST_DIR_VALIDATION/predict.log 2>&1 &
PID_INFER=$!

while kill -0 $PID_INFER 2>/dev/null; do
    sleep $POLL_INTERVAL
    echo "--- $(date '+%H:%M:%S') ---"
    show_progress "inference" $DEST_DIR_VALIDATION/predict.log
done

INFER_END=$(date +%s)
INFER_ELAPSED=$(( INFER_END - INFER_START ))
INFER_MINS=$(( INFER_ELAPSED / 60 ))
INFER_SECS=$(( INFER_ELAPSED % 60 ))
echo ""
echo "$(date '+%H:%M:%S')  Inference complete.  Elapsed: ${INFER_MINS}m ${INFER_SECS}s"
echo ""
echo "Results under: $BASE"
