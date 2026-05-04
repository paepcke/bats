#!/bin/bash
# @Author: Andreas Paepcke
# @Date:   2026-04-14 13:11:23
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-04 12:47:37

# entrypoint.sh
# Launches train_cnn.py via torchrun (multi-GPU) or python (single-GPU).
# All CLI args passed to this script are forwarded to train_cnn.py.
#
# Environment variables (set via docker run --env or -e):
#   MANIFEST_CSV     path to manifest CSV inside the container
#   OUT_DIR          path to output directory inside the container
#   EPOCHS           number of training epochs (default: 40)
#   BATCH            per-GPU batch size (default: 64)
#   WORKERS          DataLoader workers per GPU (default: 8)
#   NPROC_PER_NODE   number of GPUs to use (default: 1)

set -euo pipefail

MANIFEST_CSV="${MANIFEST_CSV:-/data/manifest.csv}"
OUT_DIR="${OUT_DIR:-/output}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# Optional: path to holdout_split.csv inside the container.
# Set via -e SPLIT_FILE=/data/holdout_split.csv in docker run.
# When set, train_cnn.py uses the pre-assigned file_id partitions
# instead of computing its own random split.
SPLIT_FILE="${SPLIT_FILE:-}"

mkdir -p "${OUT_DIR}"

# Build the optional --split-file argument.
SPLIT_FILE_ARG=""
if [[ -n "${SPLIT_FILE}" ]]; then
    if [[ ! -f "${SPLIT_FILE}" ]]; then
        echo "ERROR: SPLIT_FILE set to '${SPLIT_FILE}' but file not found inside container."
        exit 1
    fi
    SPLIT_FILE_ARG="--split-file ${SPLIT_FILE}"
    echo "  SPLIT_FILE   : ${SPLIT_FILE}"
fi

echo "=========================================="
echo "  Bat CNN Training"
echo "  MANIFEST_CSV : ${MANIFEST_CSV}"
echo "  OUT_DIR      : ${OUT_DIR}"
echo "  EPOCHS       : ${EPOCHS}"
echo "  BATCH        : ${BATCH} (per GPU)"
echo "  WORKERS      : ${WORKERS}"
echo "  NPROC        : ${NPROC_PER_NODE} GPU(s)"
echo "  SPLIT_FILE   : ${SPLIT_FILE:-<none — internal random split>}"
echo "  EXTRA ARGS   : $*"
echo "=========================================="

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    echo "Launching with torchrun (DDP, ${NPROC_PER_NODE} GPUs)"
    exec torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --standalone \
        -m species_classification.training.train_cnn \
        --manifest  "${MANIFEST_CSV}" \
        --out-dir   "${OUT_DIR}" \
        --epochs    "${EPOCHS}" \
        --batch     "${BATCH}" \
        --workers   "${WORKERS}" \
        ${SPLIT_FILE_ARG} \
        "$@"
else
    echo "Launching with python (single GPU)"
    exec python -m species_classification.training.train_cnn \
        --manifest  "${MANIFEST_CSV}" \
        --out-dir   "${OUT_DIR}" \
        --epochs    "${EPOCHS}" \
        --batch     "${BATCH}" \
        --workers   "${WORKERS}" \
        ${SPLIT_FILE_ARG} \
        "$@"
fi
