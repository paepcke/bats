#!/bin/bash
# @Author: Andreas Paepcke
# @Date:   2026-04-07 12:30:56
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-10 13:12:35

# startup.sh
# Vultr Compute Engine startup script for bat CNN training.
#
# Handles:
#   1. Installing rclone (if not already present)
#   2. Writing the rclone Vultr config from instance metadata
#   3. Downloading tar files from Vultr Object Storage to local SSD
#   4. Untarring crops to local SSD
#   5. Rewriting manifest crop_path column to container-local paths
#   6. Pulling the training Docker image
#   7. Launching training
#   8. Syncing outputs back to Vultr on completion
#
# Required metadata variables (set via Vultr user-data or startup env):
#   VULTR_ACCESS_KEY      Vultr Object Storage access key
#   VULTR_SECRET_KEY      Vultr Object Storage secret key
#   VULTR_ENDPOINT        e.g. sjc1.vultrobjects.com
#   VULTR_DATA_BUCKET     e.g. bat-spectrograms         (crops tar bucket)
#   VULTR_OUTPUT_BUCKET   e.g. bat-training-output
#   VULTR_CROPS_PREFIX    e.g. crops-tar                (prefix inside data bucket)
#   IMAGE_URI             Docker image URI
#   NPROC_PER_NODE        number of GPUs (default: 1)
#   EPOCHS                training epochs (default: 40)
#   EXTRA_ARGS            additional train_cnn.py flags (e.g. "--resume --patience 10")
#
# Path layout on the VM:
#   /mnt/disks/data/crops/          untarred PNG subdirectories
#   /mnt/disks/data/manifest.csv    path-rewritten manifest
#   /mnt/disks/data/tars/           downloaded tar files (cleaned up after untar)
#   /mnt/disks/output/              training outputs (checkpoints, logs, etc.)

set -euo pipefail

LOG=/var/log/bat_training.log
exec > >(tee -a "${LOG}") 2>&1

echo "====== Startup: $(date) ======"

# ── Read configuration ────────────────────────────────────────
# Support both Vultr user-data (env vars already exported) and
# explicit variable file at /etc/bat_training_env if present.
if [[ -f /etc/bat_training_env ]]; then
    # shellcheck disable=SC1091
    source /etc/bat_training_env
fi

VULTR_ACCESS_KEY="${VULTR_ACCESS_KEY:?VULTR_ACCESS_KEY must be set}"
VULTR_SECRET_KEY="${VULTR_SECRET_KEY:?VULTR_SECRET_KEY must be set}"
VULTR_ENDPOINT="${VULTR_ENDPOINT:-sjc1.vultrobjects.com}"
VULTR_DATA_BUCKET="${VULTR_DATA_BUCKET:?VULTR_DATA_BUCKET must be set}"
VULTR_OUTPUT_BUCKET="${VULTR_OUTPUT_BUCKET:?VULTR_OUTPUT_BUCKET must be set}"
VULTR_CROPS_PREFIX="${VULTR_CROPS_PREFIX:-crops-tar}"
IMAGE_URI="${IMAGE_URI:?IMAGE_URI must be set}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EPOCHS="${EPOCHS:-40}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "VULTR_ENDPOINT    : ${VULTR_ENDPOINT}"
echo "VULTR_DATA_BUCKET : ${VULTR_DATA_BUCKET}/${VULTR_CROPS_PREFIX}"
echo "IMAGE_URI         : ${IMAGE_URI}"
echo "NPROC_PER_NODE    : ${NPROC_PER_NODE}"
echo "EPOCHS            : ${EPOCHS}"
echo "EXTRA_ARGS        : ${EXTRA_ARGS}"

# ── Install rclone if needed ──────────────────────────────────
if ! command -v rclone &>/dev/null; then
    echo "Installing rclone..."
    curl -fsSL https://rclone.org/install.sh | bash
    echo "rclone installed: $(rclone version | head -1)"
fi

# ── Write rclone config ───────────────────────────────────────
RCLONE_CFG=/root/.config/rclone/rclone.conf
mkdir -p "$(dirname "${RCLONE_CFG}")"
cat > "${RCLONE_CFG}" <<EOF
[vultr]
type = s3
provider = Ceph
access_key_id = ${VULTR_ACCESS_KEY}
secret_access_key = ${VULTR_SECRET_KEY}
endpoint = ${VULTR_ENDPOINT}
EOF
echo "rclone config written."

# ── Local directories ─────────────────────────────────────────
DATA_DIR=/mnt/disks/data
TAR_DIR=${DATA_DIR}/tars
CROPS_DIR=${DATA_DIR}/crops
OUTPUT_DIR=/mnt/disks/output
mkdir -p "${TAR_DIR}" "${CROPS_DIR}" "${OUTPUT_DIR}"

# ── Download tar files from Vultr ─────────────────────────────
echo "Downloading tar files from vultr:${VULTR_DATA_BUCKET}/${VULTR_CROPS_PREFIX} ..."
START=$(date +%s)
rclone copy \
    "vultr:${VULTR_DATA_BUCKET}/${VULTR_CROPS_PREFIX}" \
    "${TAR_DIR}/" \
    --transfers 16 \
    --checkers 16 \
    --size-only \
    --stats 60s \
    --log-level INFO
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
echo "Tar files removed; disk usage: $(du -sh ${CROPS_DIR})"

# ── Download and rewrite manifest ────────────────────────────
# The manifest crop_path values are absolute paths from the source
# machine (e.g. /qnap/bats/jr_pipeline/data/bat_crops/20200515_barn/00000001.png).
# Rewrite them to the container-visible path (/data/crops/...).
echo "Downloading manifest..."
rclone copy \
    "vultr:${VULTR_DATA_BUCKET}/manifest.csv" \
    "${DATA_DIR}/" \
    --log-level INFO

MANIFEST_RAW="${DATA_DIR}/manifest.csv"
MANIFEST_FIXED="${DATA_DIR}/manifest_fixed.csv"

# Extract the source prefix from the first data row's crop_path.
# Handles any absolute path prefix before the date-directory component.
SOURCE_PREFIX=$(awk -F',' 'NR==2 {
    # Find last occurrence of a path component matching YYYYMMDD_ pattern
    n = split($1, parts, "/")
    prefix = ""
    for (i = 1; i <= n; i++) {
        if (parts[i] ~ /^[0-9]{8}_/) { break }
        prefix = prefix (prefix == "" ? "" : "/") parts[i]
    }
    print prefix
}' "${MANIFEST_RAW}")

echo "Rewriting manifest crop_path: '${SOURCE_PREFIX}' -> '/data/crops'"
sed "s|${SOURCE_PREFIX}|/data/crops|g" "${MANIFEST_RAW}" > "${MANIFEST_FIXED}"

# Spot-check: verify a rewritten path actually exists on disk.
SAMPLE_PATH=$(awk -F',' 'NR==2 {print $1}' "${MANIFEST_FIXED}")
if [[ -f "${SAMPLE_PATH}" ]]; then
    echo "Manifest path check OK: ${SAMPLE_PATH}"
else
    echo "WARNING: sample path not found after rewrite: ${SAMPLE_PATH}"
    echo "  Check SOURCE_PREFIX detection and tar directory structure."
fi

# ── Install NVIDIA Container Toolkit if needed ────────────────
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

# ── Pull Docker image ─────────────────────────────────────────
echo "Pulling image: ${IMAGE_URI}"
docker pull "${IMAGE_URI}"

# ── Resume: pull latest checkpoint from Vultr if --resume ─────
if echo "${EXTRA_ARGS}" | grep -q -- "--resume"; then
    echo "Resume flag detected — syncing checkpoint from vultr:${VULTR_OUTPUT_BUCKET} ..."
    rclone copy \
        "vultr:${VULTR_OUTPUT_BUCKET}/checkpoints/" \
        "${OUTPUT_DIR}/" \
        --size-only \
        --log-level INFO || true
fi

# ── Run training ──────────────────────────────────────────────
echo "Starting training container: $(date)"
docker run --rm \
    --gpus all \
    --shm-size=16g \
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

# ── Sync outputs back to Vultr ────────────────────────────────
echo "Syncing outputs to vultr:${VULTR_OUTPUT_BUCKET}/checkpoints/ ..."
rclone copy \
    "${OUTPUT_DIR}/" \
    "vultr:${VULTR_OUTPUT_BUCKET}/checkpoints/" \
    --transfers 8 \
    --log-level INFO
echo "Output sync complete."

echo "====== Startup script finished: $(date) ======"
