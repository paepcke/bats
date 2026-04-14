#!/bin/bash
# ***********************************************
# @Author: Andreas Paepcke
# @Date:   2026-04-13 12:49:09
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-13 15:14:22
# ***********************************************
#!/bin/bash
# startup.sh
# GCP Compute Engine startup script for bat CNN training.
# VM: a2-ultragpu-2g (2x A100 80GB), us-central1
#
# Handles:
#   1. Installing NVIDIA Container Toolkit (if not already present)
#   2. Creating the output bucket if it does not exist
#   3. Copying tar files from GCS to local SSD
#   4. Untarring crops to local SSD
#   5. Downloading and rewriting manifest crop_path to container-local paths
#   6. Pulling the training Docker image from Artifact Registry
#   7. Launching training (2-GPU DDP via torchrun)
#   8. Syncing outputs back to GCS on completion
#
# Authentication: VM must be created with a service account that has
# roles/storage.admin on both buckets. No tokens needed — gcloud uses
# the VM metadata server automatically.
#
# Required instance metadata keys (set via --metadata at VM creation):
#   GCS_DATA_BUCKET     e.g. bat_png_tar_files
#   GCS_OUTPUT_BUCKET   e.g. bat-training-output  (created here if absent)
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
IMAGE_URI=$(curl -sf -H "${H}" "${META}/IMAGE_URI")
NPROC_PER_NODE=$(curl -sf -H "${H}" "${META}/NPROC_PER_NODE" || echo "2")
EPOCHS=$(curl -sf -H "${H}" "${META}/EPOCHS" || echo "40")
EXTRA_ARGS=$(curl -sf -H "${H}" "${META}/EXTRA_ARGS" || echo "")

echo "GCS_DATA_BUCKET   : gs://${GCS_DATA_BUCKET}"
echo "GCS_OUTPUT_BUCKET : gs://${GCS_OUTPUT_BUCKET}"
echo "IMAGE_URI         : ${IMAGE_URI}"
echo "NPROC_PER_NODE    : ${NPROC_PER_NODE}"
echo "EPOCHS            : ${EPOCHS}"
echo "EXTRA_ARGS        : ${EXTRA_ARGS}"

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
DATA_DISK=/dev/disk/by-id/google-data-disk
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
echo "Copying tar files from gs://${GCS_DATA_BUCKET}/*.tar ..."
START=$(date +%s)
gcloud storage cp \
    "gs://${GCS_DATA_BUCKET}/*.tar" \
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
        prefix = prefix (prefix == "" ? "" : "/") parts[i]
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
echo "Starting training container: $(date)"
docker run --rm \
    --gpus all \
    --shm-size=32g \
    -v "${CROPS_DIR}:/data/crops:ro" \
    -v "${DATA_DIR}/manifest_fixed.csv:/data/manifest.csv:ro" \
    -v "${OUTPUT_DIR}:/output" \
    -e MANIFEST_CSV=/data/manifest.csv \
    -e OUT_DIR=/output \
    -e EPOCHS="${EPOCHS}" \
    -e NPROC_PER_NODE="${NPROC_PER_NODE}" \
    "${IMAGE_URI}" \
    ${EXTRA_ARGS}

echo "Training complete: $(date)"

# ── Sync outputs back to GCS ──────────────────────────────────
echo "Syncing outputs to gs://${GCS_OUTPUT_BUCKET}/checkpoints/ ..."
gcloud storage rsync \
    "${OUTPUT_DIR}/" \
    "gs://${GCS_OUTPUT_BUCKET}/checkpoints/"
echo "Output sync complete."

echo "====== Startup script finished: $(date) ======"
