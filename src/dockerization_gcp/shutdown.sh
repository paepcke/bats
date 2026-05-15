#!/bin/bash
# *********************************************
# @Author: Andreas Paepcke
# @Date:   2026-04-13 12:50:02
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-15 13:11:10
# *********************************************

# shutdown.sh
# GCP Compute Engine shutdown script.
# Runs when the VM is preempted or stopped (30-second window).
# Syncs the output directory to GCS before the VM dies.

set -euo pipefail

LOG=/var/log/bat_shutdown.log
exec > >(tee -a "${LOG}") 2>&1

echo "====== Shutdown: $(date) ======"

META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
H="Metadata-Flavor: Google"

GCS_OUTPUT_BUCKET=$(curl -sf -H "${H}" "${META}/GCS_OUTPUT_BUCKET" || echo "")

if [[ -z "${GCS_OUTPUT_BUCKET}" ]]; then
    echo "GCS_OUTPUT_BUCKET not set — skipping checkpoint sync."
    exit 0
fi

OUTPUT_DIR=/mnt/disks/output
if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Syncing checkpoints to gs://${GCS_OUTPUT_BUCKET}/checkpoints/ ..."
    # 20s timeout to stay within the 30s preemption window.
    timeout 20 gcloud storage rsync \
        "${OUTPUT_DIR}/" \
        "gs://${GCS_OUTPUT_BUCKET}/checkpoints/" || true
    echo "Sync finished (or timed out): $(date)"
else
    echo "Output directory ${OUTPUT_DIR} not found — nothing to sync."
fi

echo "====== Shutdown complete: $(date) ======"
