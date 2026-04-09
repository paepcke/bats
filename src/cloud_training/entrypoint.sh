#!/bin/bash
# @Author: Andreas Paepcke
# @Date:   2026-04-09 11:20:30
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-09 11:23:41
#!/usr/bin/env bash
# entrypoint.sh
# ---------------------------------------------------------------------
# Container startup sequence for bat CNN training on Vultr.
#
# Required environment variables (set via `docker run -e` or the
# Vultr VM launch script):
#
#   VULTR_API_KEY          Vultr API key (used by vultr_trainer.py to
#                          self-destruct the VM when training finishes)
#   OBJECT_STORE_BUCKET    e.g. "bat-crops"
#   OBJECT_STORE_ENDPOINT  e.g. "https://sjc1.vultrobjects.com"
#   OBJECT_STORE_ACCESS    S3-compatible access key
#   OBJECT_STORE_SECRET    S3-compatible secret key
#   MANIFEST_KEY           Object key for the manifest CSV,
#                          e.g. "manifest.csv"
#   MODEL_OUT_KEY_PREFIX   Object key prefix for outputs,
#                          e.g. "models/efficientnet_b0_v2"
#   VULTR_INSTANCE_ID      Instance ID — used for self-destruct
#
# Optional:
#   NUM_GPUS               Defaults to $(nvidia-smi -L | wc -l)
#   LOCAL_DATA_DIR         Where to copy PNGs on local NVMe (default /data)
#   EXTRA_TRAIN_ARGS       Extra args forwarded to train_cnn.py
# ---------------------------------------------------------------------
set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────
log() { echo "[entrypoint] $(date -u +%H:%M:%S) $*"; }
die() { log "FATAL: $*"; exit 1; }

# ── defaults ─────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-/data}
OUT_DIR="${LOCAL_DATA_DIR}/model_out"
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

log "GPUs detected: ${NUM_GPUS}"

# ── install aws cli (s3-compatible) if not present ───────────────────
if ! command -v aws &>/dev/null; then
    log "Installing awscli for S3-compatible object storage access..."
    pip install --quiet awscli
fi

export AWS_ACCESS_KEY_ID="${OBJECT_STORE_ACCESS}"
export AWS_SECRET_ACCESS_KEY="${OBJECT_STORE_SECRET}"
AWS="aws --endpoint-url ${OBJECT_STORE_ENDPOINT}"

# ── copy PNG crops from object storage to local NVMe ─────────────────
log "Copying spectrogram crops to local disk: ${LOCAL_DATA_DIR}/crops ..."
mkdir -p "${LOCAL_DATA_DIR}/crops"

# Sync the entire crops prefix in parallel (up to 50 concurrent transfers)
${AWS} s3 sync \
    "s3://${OBJECT_STORE_BUCKET}/crops/" \
    "${LOCAL_DATA_DIR}/crops/" \
    --no-progress \
    --cli-connect-timeout 30 \
    --cli-read-timeout 60 \
    &

# ── copy manifest ─────────────────────────────────────────────────────
log "Copying manifest..."
${AWS} s3 cp \
    "s3://${OBJECT_STORE_BUCKET}/${MANIFEST_KEY}" \
    "${LOCAL_DATA_DIR}/manifest.csv"

# Update crop_path prefix in manifest so paths point to local disk.
# The manifest stores absolute paths from quintus (/qnap/bats/jr_pipeline/...);
# we rewrite the leading path component to the local data dir.
log "Rewriting manifest paths to local prefix..."
sed -i "s|/qnap/bats/jr_pipeline/data/bat_crops|${LOCAL_DATA_DIR}/crops|g" \
    "${LOCAL_DATA_DIR}/manifest.csv"

# Wait for the crop sync to finish
log "Waiting for crop sync to complete..."
wait
log "Crop sync done."

# ── create output dir ─────────────────────────────────────────────────
mkdir -p "${OUT_DIR}"

# ── launch training ───────────────────────────────────────────────────
log "Launching torchrun with ${NUM_GPUS} GPU(s)..."

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    /app/train_cnn.py \
        --manifest "${LOCAL_DATA_DIR}/manifest.csv" \
        --out-dir  "${OUT_DIR}" \
        --exclude-species Myvo Mylu \
        --batch  256 \
        --lr     2e-3 \
        --epochs 40 \
        --workers 8 \
        ${EXTRA_TRAIN_ARGS}

TRAIN_EXIT=$?

# ── upload outputs to object storage ─────────────────────────────────
log "Uploading model outputs to object storage..."
${AWS} s3 sync \
    "${OUT_DIR}/" \
    "s3://${OBJECT_STORE_BUCKET}/${MODEL_OUT_KEY_PREFIX}/" \
    --no-progress

if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    log "Training exited with code ${TRAIN_EXIT} — NOT destroying VM."
    exit ${TRAIN_EXIT}
fi

log "Training complete. Outputs uploaded."

# ── self-destruct VM (called via vultr_trainer.py helper) ────────────
if [[ -n "${VULTR_INSTANCE_ID:-}" && -n "${VULTR_API_KEY:-}" ]]; then
    log "Destroying Vultr instance ${VULTR_INSTANCE_ID}..."
    python3 /app/vultr_trainer.py --destroy --instance-id "${VULTR_INSTANCE_ID}"
else
    log "VULTR_INSTANCE_ID or VULTR_API_KEY not set — skipping VM self-destruct."
fi

exit 0
