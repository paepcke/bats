#!/bin/bash
# ***********************************************
# @Author: Andreas Paepcke
# @Date:   2026-04-13 12:49:09
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-05 17:09:19
# ***********************************************

# startup.sh
# GCP Compute Engine startup script for bat CNN training.
# VM: a2-ultragpu-2g (2x A100 80GB), us-central1
# Host image: deeplearning-platform-release/common-cu129-ubuntu-2204-nvidia-580
#   (includes NVIDIA driver 580, Docker, NVIDIA container toolkit pre-installed)
#
# Handles:
#   1. Installing NVIDIA Container Toolkit (if not already present)
#   2. Creating the output bucket if it does not exist
#   3. Copying tar files from GCS to local SSD
#   4. Untarring crops to local SSD
#   5. Downloading and rewriting manifest crop_path to container-local paths
#   6. Pulling the training Docker image from Artifact Registry
#   7. Launching training (2-GPU DDP via torchrun)
#      - Passes --gcs-output-bucket so train_cnn.py uploads checkpoint_latest.pt
#        and best_model.pt to GCS after every epoch, independently of this
#        script's final sync and the shutdown.sh preemption handler.
#   8. Syncing outputs back to GCS on completion
#
# Authentication: VM must be created with a service account that has
# roles/storage.admin on both buckets. No tokens needed — gcloud uses
# the VM metadata server automatically.
#
# Required instance metadata keys (set via --metadata at VM creation):
#   GCS_DATA_BUCKET     e.g. bat_png_tar_files
#   GCS_OUTPUT_BUCKET   e.g. bat-training-output  (created here if absent)
#   GCS_CROPS_PREFIX    e.g. crops-tar             (prefix inside data bucket)
#   IMAGE_URI           e.g. us-central1-docker.pkg.dev/dresl-bats-2026/bats/bat-cnn:latest
#   NPROC_PER_NODE      number of GPUs (2 for a2-ultragpu-2g)
#   EPOCHS              training epochs (default: 40)
#   EXTRA_ARGS          additional train_cnn.py flags

set -euo pipefail

LOG=/var/log/bat_training.log
exec > >(tee -a "${LOG}") 2>&1

echo "====== Startup: $(date) ======"

# ── Read instance metadata ────────────────────────────────────
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
H="Metadata-Flavor: Google"

GCS_DATA_BUCKET=$(curl -sf -H "${H}" "${META}/GCS_DATA_BUCKET")
GCS_OUTPUT_BUCKET=$(curl -sf -H "${H}" "${META}/GCS_OUTPUT_BUCKET" || echo "bat-training-output")
GCS_CROPS_PREFIX=$(curl -sf -H "${H}" "${META}/GCS_CROPS_PREFIX" || echo "")
IMAGE_URI=$(curl -sf -H "${H}" "${META}/IMAGE_URI")
NPROC_PER_NODE=$(curl -sf -H "${H}" "${META}/NPROC_PER_NODE" || echo "2")
EPOCHS=$(curl -sf -H "${H}" "${META}/EPOCHS" || echo "40")
EXTRA_ARGS=$(curl -sf -H "${H}" "${META}/EXTRA_ARGS" || echo "")
SPLIT_FILE_KEY=$(curl -sf -H "${H}" "${META}/SPLIT_FILE_KEY" || echo "holdout_split.csv")

echo "GCS_DATA_BUCKET   : gs://${GCS_DATA_BUCKET}"
echo "GCS_OUTPUT_BUCKET : gs://${GCS_OUTPUT_BUCKET}"
echo "GCS_CROPS_PREFIX  : ${GCS_CROPS_PREFIX}"
echo "IMAGE_URI         : ${IMAGE_URI}"
echo "NPROC_PER_NODE    : ${NPROC_PER_NODE}"
echo "EPOCHS            : ${EPOCHS}"
echo "EXTRA_ARGS        : ${EXTRA_ARGS}"
echo "SPLIT_FILE_KEY    : ${SPLIT_FILE_KEY}"

# ── Create output bucket if it does not exist ─────────────────
if ! gcloud storage buckets describe "gs://${GCS_OUTPUT_BUCKET}" &>/dev/null; then
    echo "Creating output bucket gs://${GCS_OUTPUT_BUCKET} in us-central1..."
    gcloud storage buckets create "gs://${GCS_OUTPUT_BUCKET}" \
        --location=us-central1 \
        --uniform-bucket-level-access
    echo "Output bucket created."
else
    echo "Output bucket gs://${GCS_OUTPUT_BUCKET} already exists."
fi

# ── Install Docker ───────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "Docker installed."
else
    echo "Docker already installed: $(docker --version)"
fi

# ── NVIDIA Container Toolkit ──────────────────────────────────
if ! command -v nvidia-container-toolkit &>/dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L \
        "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
fi

# ── Mount data disk ───────────────────────────────────────────
# The 600GB SSD data disk is attached but may not be formatted/mounted yet.
DATA_DISK=/dev/disk/by-id/google-persistent-disk-1
DATA_DIR=/mnt/disks/data
OUTPUT_DIR=/mnt/disks/output

if [ -b "${DATA_DISK}" ]; then
    if ! blkid "${DATA_DISK}" | grep -q ext4; then
        echo "Formatting data disk..."
        mkfs.ext4 -F "${DATA_DISK}"
    fi
    mkdir -p "${DATA_DIR}"
    mount -o discard,defaults "${DATA_DISK}" "${DATA_DIR}" || true
    echo "Data disk mounted at ${DATA_DIR}."
else
    echo "WARNING: data disk not found at ${DATA_DISK}, using boot disk."
fi

TAR_DIR=${DATA_DIR}/tars
CROPS_DIR=${DATA_DIR}/crops
mkdir -p "${TAR_DIR}" "${CROPS_DIR}" "${OUTPUT_DIR}"

# ── Copy tar files from GCS ───────────────────────────────────
# Tars may be at the bucket root or under a prefix.
if [[ -n "${GCS_CROPS_PREFIX}" ]]; then
    TAR_SOURCE="gs://${GCS_DATA_BUCKET}/${GCS_CROPS_PREFIX}/*.tar"
else
    TAR_SOURCE="gs://${GCS_DATA_BUCKET}/*.tar"
fi
echo "Copying tar files from ${TAR_SOURCE} ..."
START=$(date +%s)
gcloud storage cp \
    "${TAR_SOURCE}" \
    "${TAR_DIR}/"
END=$(date +%s)
TAR_COUNT=$(ls "${TAR_DIR}"/*.tar 2>/dev/null | wc -l)
echo "Downloaded ${TAR_COUNT} tar files in $((END - START))s."

# ── Untar crops to local SSD ──────────────────────────────────
echo "Untarring ${TAR_COUNT} archives to ${CROPS_DIR} ..."
START=$(date +%s)
for tarfile in "${TAR_DIR}"/*.tar; do
    tar -xf "${tarfile}" -C "${CROPS_DIR}"
done
END=$(date +%s)
echo "Untar complete in $((END - START))s."

# Free up tar disk space — crops are now on SSD.
rm -rf "${TAR_DIR}"
echo "Tar files removed. Disk usage: $(du -sh ${CROPS_DIR})"

# ── Download and rewrite manifest ────────────────────────────
# Manifest crop_path values are absolute paths from quintus:
#   /qnap/bats/jr_pipeline/data/bat_crops/20200515_barn/00000001.png
# Rewrite to container-visible path:
#   /data/crops/20200515_barn/00000001.png
echo "Downloading manifest..."
gcloud storage cp \
    "gs://${GCS_DATA_BUCKET}/manifest.csv" \
    "${DATA_DIR}/manifest.csv"

MANIFEST_RAW="${DATA_DIR}/manifest.csv"
MANIFEST_FIXED="${DATA_DIR}/manifest_fixed.csv"

# Extract the source prefix (everything before the first YYYYMMDD_ component).
SOURCE_PREFIX=$(awk -F',' 'NR==2 {
    n = split($1, parts, "/")
    prefix = ""
    for (i = 1; i <= n; i++) {
        if (parts[i] ~ /^[0-9]{8}_/) { break }
        if (parts[i] != "") {
            prefix = prefix (prefix == "" ? "" : "/") parts[i]
        }
    }
    print prefix
}' "${MANIFEST_RAW}")

echo "Rewriting manifest: '${SOURCE_PREFIX}' -> '/data/crops'"
sed "s|${SOURCE_PREFIX}|/data/crops|g" "${MANIFEST_RAW}" > "${MANIFEST_FIXED}"

# Spot-check: verify a rewritten path actually exists on disk.
SAMPLE_PATH=$(awk -F',' 'NR==2 {print $1}' "${MANIFEST_FIXED}")
if [[ -f "${SAMPLE_PATH}" ]]; then
    echo "Manifest path check OK: ${SAMPLE_PATH}"
else
    echo "WARNING: sample path not found after rewrite: ${SAMPLE_PATH}"
    echo "  Check SOURCE_PREFIX detection and tar directory structure."
fi

# ── Download holdout split file ───────────────────────────────
# holdout_split.csv is produced once by make_holdout_split.py on quintus
# and uploaded to GCS alongside the manifest.  It maps file_id → partition
# so CNN and RF share an identical held-out test set.
SPLIT_FILE_HOST="${DATA_DIR}/holdout_split.csv"
SPLIT_FILE_GCS="gs://${GCS_DATA_BUCKET}/${SPLIT_FILE_KEY}"

if gcloud storage ls "${SPLIT_FILE_GCS}" &>/dev/null; then
    echo "Downloading split file from ${SPLIT_FILE_GCS} ..."
    gcloud storage cp "${SPLIT_FILE_GCS}" "${SPLIT_FILE_HOST}"
    echo "Split file downloaded: $(wc -l < "${SPLIT_FILE_HOST}") rows"
else
    echo "WARNING: split file not found at ${SPLIT_FILE_GCS}."
    echo "  Training will use an internal random split instead."
    echo "  To use a shared holdout, upload holdout_split.csv to GCS:"
    echo "    gcloud storage cp /path/to/holdout_split.csv ${SPLIT_FILE_GCS}"
    SPLIT_FILE_HOST=""
fi

# ── Configure Docker auth for Artifact Registry ───────────────
echo "Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# ── Pull Docker image ─────────────────────────────────────────
echo "Pulling image: ${IMAGE_URI}"
docker pull "${IMAGE_URI}"

# ── Resume: pull latest checkpoint from GCS if --resume ───────
if echo "${EXTRA_ARGS}" | grep -q -- "--resume"; then
    echo "Resume flag detected — syncing checkpoint from gs://${GCS_OUTPUT_BUCKET} ..."
    gcloud storage rsync \
        "gs://${GCS_OUTPUT_BUCKET}/checkpoints/" \
        "${OUTPUT_DIR}/" || true
fi

# ── Run training ──────────────────────────────────────────────
# --gcs-output-bucket is passed explicitly so train_cnn.py uploads
# checkpoint_latest.pt and best_model.pt to GCS after every epoch.
# This ensures checkpoints are durably stored even if the VM is
# hard-killed before this script's final sync or shutdown.sh runs.
echo "Starting training container: $(date)"
# Build optional split-file volume mount and env var.
SPLIT_MOUNT_ARG=""
SPLIT_ENV_ARG=""
if [[ -n "${SPLIT_FILE_HOST}" && -f "${SPLIT_FILE_HOST}" ]]; then
    SPLIT_MOUNT_ARG="-v ${SPLIT_FILE_HOST}:/data/holdout_split.csv:ro"
    SPLIT_ENV_ARG="-e SPLIT_FILE=/data/holdout_split.csv"
fi

docker run --rm \
    --gpus all \
    --shm-size=32g \
    -v "${CROPS_DIR}:/data/crops:ro" \
    -v "${DATA_DIR}/manifest_fixed.csv:/data/manifest.csv:ro" \
    -v "${OUTPUT_DIR}:/output" \
    ${SPLIT_MOUNT_ARG} \
    -e MANIFEST_CSV=/data/manifest.csv \
    -e OUT_DIR=/output \
    -e EPOCHS="${EPOCHS}" \
    -e NPROC_PER_NODE="${NPROC_PER_NODE}" \
    ${SPLIT_ENV_ARG} \
    "${IMAGE_URI}" \
    --gcs-output-bucket "${GCS_OUTPUT_BUCKET}" \
    ${EXTRA_ARGS}

echo "Training complete: $(date)"

# ── Sync outputs back to GCS ──────────────────────────────────
# This final rsync picks up final_model.pt, train_log.csv,
# confusion_matrix.png, and classification_report.txt, which are only
# written at the very end of training.  checkpoint_latest.pt and
# best_model.pt were already uploaded per-epoch by train_cnn.py.
echo "Syncing outputs to gs://${GCS_OUTPUT_BUCKET}/checkpoints/ ..."
gcloud storage rsync \
    "${OUTPUT_DIR}/" \
    "gs://${GCS_OUTPUT_BUCKET}/checkpoints/"
echo "Output sync complete."

echo "====== Startup script finished: $(date) ======"

# ── Self-delete the VM ────────────────────────────────────────
# Fetch instance name and zone from the metadata server so this
# script works without hardcoding the VM name.
INSTANCE_NAME=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)
INSTANCE_ZONE=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone \
    | cut -d/ -f4)
echo "Deleting VM ${INSTANCE_NAME} in ${INSTANCE_ZONE} ..."
gcloud compute instances delete "${INSTANCE_NAME}" \
    --zone="${INSTANCE_ZONE}" \
    --quiet
