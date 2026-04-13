#!/bin/bash
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

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "  Bat CNN Training"
echo "  MANIFEST_CSV : ${MANIFEST_CSV}"
echo "  OUT_DIR      : ${OUT_DIR}"
echo "  EPOCHS       : ${EPOCHS}"
echo "  BATCH        : ${BATCH} (per GPU)"
echo "  WORKERS      : ${WORKERS}"
echo "  NPROC        : ${NPROC_PER_NODE} GPU(s)"
echo "  EXTRA ARGS   : $*"
echo "=========================================="

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    echo "Launching with torchrun (DDP, ${NPROC_PER_NODE} GPUs)"
    exec torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --standalone \
        -m species_classification.train_cnn \
        --manifest  "${MANIFEST_CSV}" \
        --out-dir   "${OUT_DIR}" \
        --epochs    "${EPOCHS}" \
        --batch     "${BATCH}" \
        --workers   "${WORKERS}" \
        "$@"
else
    echo "Launching with python (single GPU)"
    exec python -m species_classification.train_cnn \
        --manifest  "${MANIFEST_CSV}" \
        --out-dir   "${OUT_DIR}" \
        --epochs    "${EPOCHS}" \
        --batch     "${BATCH}" \
        --workers   "${WORKERS}" \
        "$@"
fi
