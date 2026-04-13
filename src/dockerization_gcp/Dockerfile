# ============================================================
# Bat species CNN training image
# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# Compatible with PyTorch 2.x cu121 wheels and NVIDIA A100/L4
#
# Data is supplied at runtime via volume mounts:
#   /data/crops/        untarred PNG subdirectories (read-only)
#   /data/manifest.csv  path-rewritten manifest CSV (read-only)
#   /output/            training outputs (checkpoints, logs)
#
# rclone download/untar happens in startup.sh on the host VM,
# not inside the container.
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        git \
        wget \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3      1

# ── Python dependencies ───────────────────────────────────────
# Install PyTorch first (large wheel, cached separately from the rest).
RUN pip install --no-cache-dir \
        torch==2.3.0+cu121 \
        torchvision==0.18.0+cu121 \
        torchaudio==2.3.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Install everything else the training script needs.
# Versions are pinned to match the bats conda environment.
RUN pip install --no-cache-dir \
        numpy>=2.0.0 \
        pandas>=2.2.2 \
        scikit-learn>=1.5.0 \
        Pillow \
        matplotlib \
        seaborn \
        tqdm \
        logging_singleton>=1.0.1 \
        pyarrow>=22.0.0

# ── Copy and install the bats package ────────────────────────
# We do a regular install (not -e) since there is no live source
# tree in the container.  Rebuild the image to pick up code changes.
WORKDIR /workspace
COPY . /workspace/bats
RUN pip install --no-cache-dir /workspace/bats

# ── Runtime defaults ─────────────────────────────────────────
# Overridden at docker run time via --env or direct args to entrypoint.
ENV MANIFEST_CSV=/data/manifest.csv
ENV OUT_DIR=/output
ENV EPOCHS=40
ENV BATCH=64
ENV WORKERS=8
ENV NPROC_PER_NODE=1

# entrypoint.sh handles both single-GPU and multi-GPU launch.
COPY src/dockerization_gcp/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
