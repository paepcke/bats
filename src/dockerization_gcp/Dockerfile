# ============================================================
# Bat species CNN training image
# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# Compatible with PyTorch 2.x cu121 wheels and NVIDIA A100 80GB
#
# Python 3.12 installed via deadsnakes PPA (Ubuntu 22.04 ships 3.10).
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

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System packages + Python 3.12 via deadsnakes PPA ─────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3.12-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Bootstrap pip for 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

# ── Python dependencies ───────────────────────────────────────
# Install PyTorch first (large wheel, cached separately from the rest).
RUN pip install --no-cache-dir \
        torch==2.3.0+cu121 \
        torchvision==0.18.0+cu121 \
        torchaudio==2.3.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Install everything else the training script needs.
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
# Build context is the repo root; pyproject.toml uses src layout.
WORKDIR /workspace
COPY . /workspace/bats
RUN pip install --no-cache-dir /workspace/bats

# ── Runtime defaults ─────────────────────────────────────────
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
