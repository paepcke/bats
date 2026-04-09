# ============================================================
# Bat chirp CNN training image
# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# PyTorch ships its own CUDA runtime, so we only need
# a driver-compatible base — 12.1 is close enough to
# the 12.0 on quintus that all wheels work.
# ============================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── system packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3 / pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

# ── working directory ────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
# Copy only the requirements file first so Docker can cache this
# layer and avoid a full reinstall on every code change.
COPY src/cloud_training/requirements_train.txt .

RUN pip install --no-cache-dir --upgrade pip

# PyTorch must be installed from its own index, not PyPI.
# cu121 wheels are compatible with CUDA 12.x drivers.
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies (no torch here — already installed above).
RUN pip install --no-cache-dir -r requirements_train.txt

# ── source tree ──────────────────────────────────────────────
# Copy the src tree and add it to PYTHONPATH so train_cnn.py can
# import any sibling modules directly.  We skip the full package
# install to avoid setup.cfg path gymnastics in the container;
# logging-singleton is already handled via requirements_train.txt.
COPY src/ src/
ENV PYTHONPATH="/app/src"

# ── training script ──────────────────────────────────────────
COPY src/species_classification/train_cnn.py train_cnn.py

# ── entrypoint ───────────────────────────────────────────────
# The entrypoint script handles:
#   1. Copying data from Vultr object storage to local NVMe
#   2. Launching torchrun
#   3. Uploading checkpoints / final model back to object storage
COPY src/cloud_training/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
