#!/bin/bash
# @Author: Andreas Paepcke
# @Date:   2026-04-07 12:30:56
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-10 13:13:24

# shutdown.sh
# Vultr VM shutdown/preemption script.
# Syncs the output directory (checkpoints, logs) to Vultr Object Storage
# before the VM is terminated.
#
# Source /etc/bat_training_env for credentials, or export env vars directly.

set -euo pipefail

LOG=/var/log/bat_shutdown.log
exec > >(tee -a "${LOG}") 2>&1

echo "====== Shutdown: $(date) ======"

if [[ -f /etc/bat_training_env ]]; then
    # shellcheck disable=SC1091
    source /etc/bat_training_env
fi

VULTR_OUTPUT_BUCKET="${VULTR_OUTPUT_BUCKET:-}"

if [[ -z "${VULTR_OUTPUT_BUCKET}" ]]; then
    echo "VULTR_OUTPUT_BUCKET not set — skipping checkpoint sync."
    exit 0
fi

OUTPUT_DIR=/mnt/disks/output
if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Syncing checkpoints to vultr:${VULTR_OUTPUT_BUCKET}/checkpoints/ ..."
    # 20s timeout to stay within typical preemption window.
    timeout 20 rclone copy \
        "${OUTPUT_DIR}/" \
        "vultr:${VULTR_OUTPUT_BUCKET}/checkpoints/" \
        --transfers 8 \
        --log-level INFO || true
    echo "Sync finished (or timed out): $(date)"
else
    echo "Output directory ${OUTPUT_DIR} not found — nothing to sync."
fi

echo "====== Shutdown complete: $(date) ======"
