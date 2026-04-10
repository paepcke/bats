# Bat CNN Cloud Training on Vultr

End-to-end workflow for training the bat species CNN on a Vultr VM
using Docker and Vultr Object Storage.

---

## Overview

Training data (spectrogram PNG crops) is stored in Vultr Object Storage
as per-recording-session tar archives (`crops-tar/` prefix), created by
the tarring pipeline on quintus.  At VM startup, `startup.sh` downloads
and untars these archives to a local NVMe SSD, rewrites the manifest
`crop_path` column to match container-local paths, then runs the Docker
training container.

---

## Prerequisites

On quintus (or sparky):
```bash
# rclone must be configured with a [vultr] remote pointing to sjc1.
rclone config show vultr
# Should show: endpoint = sjc1.vultrobjects.com
```

---

## Step 1 — Upload training data (one time, then incremental)

Training data is uploaded as tar archives.  See the tarring pipeline
documentation.  After tarring is complete on quintus:

```bash
# Tars should already be uploading via the rclone loop.
# Verify:
rclone size vultr:bat-spectrograms/crops-tar
# Should show ~982 objects totalling ~256 GB.
```

Upload the manifest separately (it is not included in the tars):
```bash
rclone copy \
    /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \
    vultr:bat-spectrograms/ \
    --log-level INFO
```

---

## Step 2 — Create the output bucket (one time)

In the Vultr console, create a second Object Storage bucket for outputs:
- Region: Silicon Valley (sjc1)
- Name: `bat-training-output`

---

## Step 3 — Build and push the Docker image

From the root of the bats repo (where Dockerfile lives):
```bash
# Build
docker build -t <your-registry>/bat-cnn:latest .

# Test locally if you have a GPU:
docker run --rm --gpus all <your-registry>/bat-cnn:latest --help

# Push
docker push <your-registry>/bat-cnn:latest
```

Rebuild and push whenever `train_cnn.py` or other training code changes.
Data changes never require a rebuild.

---

## Step 4 — Prepare startup/shutdown scripts

Upload scripts to Vultr so the VM can fetch them at boot:
```bash
rclone copy startup.sh  vultr:bat-spectrograms/scripts/startup.sh
rclone copy shutdown.sh vultr:bat-spectrograms/scripts/shutdown.sh
```

Or copy them directly to the VM after creation via scp.

---

## Step 5 — Create the Vultr VM

In the Vultr console:
- **Location:** Silicon Valley (closest to Stanford/sjc1 bucket)
- **Plan:** Choose GPU instance (e.g. A100 or L4 tier)
- **OS:** Ubuntu 22.04
- **Storage:** Add a 400GB+ NVMe block volume for `/mnt/disks`

Set the following environment variables in user-data or `/etc/bat_training_env`
on the VM:

```bash
VULTR_ACCESS_KEY=<your-vultr-object-storage-access-key>
VULTR_SECRET_KEY=<your-vultr-object-storage-secret-key>
VULTR_ENDPOINT=sjc1.vultrobjects.com
VULTR_DATA_BUCKET=bat-spectrograms
VULTR_OUTPUT_BUCKET=bat-training-output
VULTR_CROPS_PREFIX=crops-tar
IMAGE_URI=<your-registry>/bat-cnn:latest
NPROC_PER_NODE=1        # match GPU count on chosen VM
EPOCHS=40
EXTRA_ARGS="--patience 7 --cw-power 0.5"
```

Then run `startup.sh` manually or attach as a startup script.

---

## Step 6 — Monitor training

```bash
# SSH into the VM and tail the log
tail -f /var/log/bat_training.log
```

---

## Step 7 — Resume after preemption

Add `--resume` to `EXTRA_ARGS` and rerun `startup.sh`:
```bash
export EXTRA_ARGS="--resume --patience 7 --cw-power 0.5"
bash startup.sh
```

`startup.sh` detects `--resume` in `EXTRA_ARGS` and syncs
`checkpoint_latest.pt` from the output bucket before launching training.

---

## Step 8 — Retrieve results

```bash
# Download outputs to quintus
rclone copy \
    vultr:bat-training-output/checkpoints/ \
    /qnap/bats/jr_pipeline/models/efficientnet_b0_v2/ \
    --log-level INFO
```

---

## VM disk layout

```
/mnt/disks/data/tars/           Downloaded tar files (deleted after untar)
/mnt/disks/data/crops/          Untarred PNG subdirectories (read-only mount)
/mnt/disks/data/manifest.csv    Original manifest (from Vultr bucket)
/mnt/disks/data/manifest_fixed.csv  Path-rewritten manifest (used by container)
/mnt/disks/output/              Training outputs (checkpoints, logs, etc.)
```

---

## Container volume mounts

```
/data/crops/        ← /mnt/disks/data/crops/          (read-only)
/data/manifest.csv  ← /mnt/disks/data/manifest_fixed.csv (read-only)
/output/            ← /mnt/disks/output/
```

---

## Cost estimate (indicative, Vultr pricing)

| Scenario                        | Time est. | Cost est.  |
|---------------------------------|-----------|------------|
| Validation run (5 epochs, L4)   | ~3 hrs    | ~$1–2      |
| Full run (40 epochs, L4)        | ~25 hrs   | ~$10–15    |
| Full run (40 epochs, A100)      | ~8 hrs    | ~$25–35    |

Early stopping typically reduces epoch count by 20–40%.

---

## Rebuilding after code changes

Only rebuild the image when training code changes.
Data changes never require a rebuild.

```bash
docker build -t <your-registry>/bat-cnn:latest .
docker push <your-registry>/bat-cnn:latest
# Then rerun startup.sh on the VM — it will pull the new image.
```
