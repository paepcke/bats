#!/bin/bash
# @Author: Andreas Paepcke
# @Date:   2026-04-14 13:11:23
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-15 13:10:09

# entrypoint.sh
# Launches train_cnn.py via torchrun (multi-GPU) or python (single-GPU).
# All CLI args passed to this script are forwarded to train_cnn.py.
#
# Environment variables (set via docker run --env or -e):
#   MANIFEST_CSV     path to manifest CSV inside the container
#   OUT_DIR          path to output directory inside the container
#   EPOCHS           number of training epochs (default: 30)
#   BATCH            per-GPU batch size (default: 128)
#   LR               initial learning rate (default: 2e-3)
#   WORKERS          DataLoader workers per GPU (default: 8)
#   NPROC_PER_NODE   number of GPUs to use (default: 1)
#   SPLIT_FILE       path to holdout_split.csv inside the container

set -euo pipefail

MANIFEST_CSV="${MANIFEST_CSV:-/data/manifest.csv}"
OUT_DIR="${OUT_DIR:-/output}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-128}"
LR="${LR:-2e-3}"
WORKERS="${WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
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
echo "  Bat CNN Training — EfficientNet-B3"
echo "  MANIFEST_CSV : ${MANIFEST_CSV}"
echo "  OUT_DIR      : ${OUT_DIR}"
echo "  EPOCHS       : ${EPOCHS}"
echo "  BATCH        : ${BATCH} (per GPU)"
echo "  LR           : ${LR}"
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
        --lr        "${LR}" \
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
        --lr        "${LR}" \
        --workers   "${WORKERS}" \
        ${SPLIT_FILE_ARG} \
        "$@"
fi
