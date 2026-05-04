# Bat CNN Cloud Training on GCP

End-to-end workflow for training the bat species CNN on a GCP VM
(a2-ultragpu-2g, 2× A100 80GB) using Docker and Google Cloud Storage.

---

## Overview

Training data lives in GCS bucket `bat_png_tar_files` (us-central1) as
per-recording-session tar archives under the `crops-tar/` prefix, uploaded
from quintus.  At VM startup, `startup.sh`:
1. Downloads all tar files to local NVMe SSD
2. Untars them to `/mnt/disks/data/crops/`
3. Rewrites manifest `crop_path` values to container-local paths
4. Downloads `holdout_split.csv` from GCS (if present) and mounts it into
   the container so CNN and RF share the same held-out test set
5. Runs the Docker training container (2-GPU DDP)
6. Syncs outputs back to GCS

The training container (`train_cnn.py`, `entrypoint.sh`, `Dockerfile`) is
**identical** to the Vultr version — only the host-side scripts change.

---

## One-time setup

### 1 — Create a service account for the VM

The VM authenticates to GCS via a service account attached at creation time.
No OAuth tokens or credential files needed.

```bash
PROJECT=dresl-bats-2026

# Create the service account
gcloud iam service-accounts create bat-training-sa \
    --display-name="Bat CNN Training" \
    --project=${PROJECT}

SA=bat-training-sa@${PROJECT}.iam.gserviceaccount.com

# Grant Storage Admin on the data bucket
gcloud storage buckets add-iam-policy-binding gs://bat_png_tar_files \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.admin"

# Grant Storage Admin at project level (covers output bucket creation)
gcloud projects add-iam-policy-binding ${PROJECT} \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.admin"

# Allow the VM to pull from Artifact Registry
gcloud projects add-iam-policy-binding ${PROJECT} \
    --member="serviceAccount:${SA}" \
    --role="roles/artifactregistry.reader"
```

### 2 — Create Artifact Registry repository

```bash
gcloud artifacts repositories create bats \
    --repository-format=docker \
    --location=us-central1 \
    --project=${PROJECT} \
    --description="Bat CNN training images"
```

### 3 — Generate and upload the holdout split file

The holdout split assigns every `file_id` to `train`, `val`, or `test`
once.  Both the CNN and RF use this file so they are evaluated on the same
held-out test set.  Generate it on quintus from the manifest:

```bash
python src/species_classification/training/make_holdout_split.py \
    --manifest /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \
    --out-dir  /qnap/bats/jr_pipeline/data \
    --seed 42
```

Then upload to GCS:

```bash
gcloud storage cp \
    /qnap/bats/jr_pipeline/data/holdout_split.csv \
    gs://bat_png_tar_files/holdout_split.csv
```

> Generate the split file **once** and reuse it for every subsequent CNN
> and RF training run.  Never regenerate it — doing so would change the
> test partition and invalidate cross-model comparisons.

### 4 — Upload the manifest (not included in tars)

```bash
gcloud storage cp \
    /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \
    gs://bat_png_tar_files/
```

### 5 — Upload startup/shutdown scripts to GCS

```bash
gcloud storage cp startup.sh  gs://bat_png_tar_files/scripts/startup.sh
gcloud storage cp shutdown.sh gs://bat_png_tar_files/scripts/shutdown.sh
```

---

## Build and push the Docker image

From the root of the bats repo:

```bash
PROJECT=dresl-bats-2026
IMAGE=us-central1-docker.pkg.dev/${PROJECT}/bats/bat-cnn:latest

# Authenticate Docker to Artifact Registry (one time per machine)
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build
docker build -f src/dockerization_gcp/Dockerfile -t ${IMAGE} .

# Test locally if you have a GPU
docker run --rm --gpus all ${IMAGE} --help

# Push
docker push ${IMAGE}
```

Rebuild and push only when `train_cnn.py` or other training code changes.
Data changes (including a new `holdout_split.csv`) never require a rebuild —
the split file is fetched from GCS at VM startup.

---

## Pre-seed GCS with a partial checkpoint from quintus

If quintus has already trained for some epochs, upload its checkpoint to GCS
before creating the VM so the GCP run can resume where quintus left off.

```bash
# Verify checkpoint_latest.pt is present — this is what --resume loads.
# best_model.pt alone is not sufficient (it contains only weights, not
# optimizer/scheduler/early-stop state).
ls -lh /qnap/bats/jr_pipeline/models/efficientnet_b0_v3/

# Upload to GCS (gcloud storage rsync creates the destination prefix
# automatically — no mkdir needed on either end).
gcloud storage rsync \
    /qnap/bats/jr_pipeline/models/efficientnet_b0_v3/ \
    gs://bat-training-output/checkpoints/ \
    --recursive
```

Then add `--resume` to `EXTRA_ARGS` when creating the VM (see next section).
`startup.sh` will pull the checkpoint to `/mnt/disks/output/` before
launching the container, and `train_cnn.py` will continue from the next epoch.

---

## Create and launch the VM

### Fresh run

```bash
PROJECT=dresl-bats-2026
SA=bat-training-sa@${PROJECT}.iam.gserviceaccount.com
IMAGE=us-central1-docker.pkg.dev/${PROJECT}/bats/bat-cnn:latest
ZONE=us-central1-a

gcloud compute instances create bat-cnn-training \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --machine-type=a2-ultragpu-2g \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=${SA} \
    --scopes=cloud-platform \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --create-disk=auto-delete=yes,size=600,type=pd-ssd,name=data-disk \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata=\
GCS_DATA_BUCKET=bat_png_tar_files,\
GCS_OUTPUT_BUCKET=bat-training-output,\
GCS_CROPS_PREFIX=crops-tar,\
SPLIT_FILE_KEY=holdout_split.csv,\
IMAGE_URI=${IMAGE},\
NPROC_PER_NODE=2,\
EPOCHS=40,\
EXTRA_ARGS="--patience 10 --cw-power 0.75",\
startup-script-url=gs://bat_png_tar_files/scripts/startup.sh,\
shutdown-script-url=gs://bat_png_tar_files/scripts/shutdown.sh
```

### Resuming from a quintus checkpoint (or after interruption)

#### 1 — Upload the quintus checkpoint to GCS

First verify that `checkpoint_latest.pt` is present in the quintus model
directory.  This file contains the full training state (model weights,
optimizer, scheduler, early-stop counters).  `best_model.pt` alone is **not**
sufficient for resumption — it contains only the model weights.

```bash
ls -lh /qnap/bats/jr_pipeline/models/efficientnet_b0_v3/
```

Then upload to GCS.  `gcloud storage rsync` creates the destination prefix
automatically — no `mkdir` needed.

```bash
gcloud storage rsync \
    /qnap/bats/jr_pipeline/models/efficientnet_b0_v3/ \
    gs://bat-training-output/checkpoints/ \
    --recursive
```

Spot-check that the upload landed:

```bash
gcloud storage ls -l gs://bat-training-output/checkpoints/
```

`checkpoint_latest.pt` should be ~150 MB.  If it is missing or suspiciously
small, do not proceed — training would restart from scratch.

#### 2 — Create the VM with `--resume`

Add `--resume` to `EXTRA_ARGS`:

```bash
gcloud compute instances create bat-cnn-training \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --machine-type=a2-ultragpu-2g \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=${SA} \
    --scopes=cloud-platform \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --create-disk=auto-delete=yes,size=600,type=pd-ssd,name=data-disk \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata=\
GCS_DATA_BUCKET=bat_png_tar_files,\
GCS_OUTPUT_BUCKET=bat-training-output,\
GCS_CROPS_PREFIX=crops-tar,\
SPLIT_FILE_KEY=holdout_split.csv,\
IMAGE_URI=${IMAGE},\
NPROC_PER_NODE=2,\
EPOCHS=40,\
EXTRA_ARGS="--resume --patience 10 --cw-power 0.75",\
startup-script-url=gs://bat_png_tar_files/scripts/startup.sh,\
shutdown-script-url=gs://bat_png_tar_files/scripts/shutdown.sh
```

Notes:
- `--scopes=cloud-platform` combined with the service account grants full
  GCS access without any credentials files.
- The 600GB SSD data disk accommodates 256GB of tars + 256GB untarred +
  headroom for the output directory.
- `a2-ultragpu-2g` is only available in select zones; `us-central1-a` and
  `us-central1-c` typically have availability. If creation fails try `-c`.
- `SPLIT_FILE_KEY` defaults to `holdout_split.csv` (root of the data
  bucket) if omitted.  If the file is absent from GCS, `startup.sh` logs
  a warning and training falls back to an internal random split.

---

## Monitor training

```bash
ZONE=us-central1-a

# Tail the startup log (data copy progress, then training output)
gcloud compute ssh bat-cnn-training --zone=${ZONE} \
    -- "tail -f /var/log/bat_training.log"

# Watch for checkpoint files appearing in GCS
gcloud storage ls -l gs://bat-training-output/checkpoints/
```

---

## Resume after interruption

If the VM is interrupted mid-run, the checkpoint already in GCS (uploaded
per-epoch by `train_cnn.py`) is the recovery point.  Update the metadata
and restart:

```bash
# Update metadata to add --resume
gcloud compute instances add-metadata bat-cnn-training \
    --zone=${ZONE} \
    --metadata=EXTRA_ARGS="--resume --patience 10 --cw-power 0.75"

# Restart (startup script runs automatically)
gcloud compute instances start bat-cnn-training --zone=${ZONE}
```

`startup.sh` detects `--resume` in `EXTRA_ARGS`, syncs
`checkpoint_latest.pt` from GCS before launching, and `train_cnn.py`
resumes from the next epoch.

---

## Retrieve results

`gcloud storage rsync` creates the local destination directory automatically
— no `mkdir` needed beforehand.

```bash
gcloud storage rsync \
    gs://bat-training-output/checkpoints/ \
    /qnap/bats/jr_pipeline/models/efficientnet_b0_v3/ \
    --recursive
```

---

## VM disk layout

```
/mnt/disks/data/tars/              Downloaded tar files (deleted after untar)
/mnt/disks/data/crops/             Untarred PNG subdirectories
/mnt/disks/data/manifest.csv       Original manifest from GCS
/mnt/disks/data/manifest_fixed.csv Path-rewritten manifest (used by container)
/mnt/disks/data/holdout_split.csv  Shared file_id partition map (from GCS)
/mnt/disks/output/                 Training outputs (checkpoints, logs, etc.)
```

## Container volume mounts

```
/data/crops/              <- /mnt/disks/data/crops/            (read-only)
/data/manifest.csv        <- /mnt/disks/data/manifest_fixed.csv (read-only)
/data/holdout_split.csv   <- /mnt/disks/data/holdout_split.csv  (read-only, if present)
/output/                  <- /mnt/disks/output/
```

---

## Recommended hyperparameters for the GCP run

Based on the quintus training run (early-stopped epoch 19, best val_acc 0.761):

| Flag | Value | Rationale |
|------|-------|-----------|
| `--batch` | 64 | Per-GPU. Conservative; prevents minority species from being drowned out at large batch sizes |
| `--lr` | 1e-3 | Default; 2e-3 caused instability in prior run |
| `--cw-power` | 0.75 | Stronger minority weighting than the previous 0.5 (sqrt) |
| `--patience` | 10 | Prior run stopped at epoch 19 with best at 10; more patience allows further LR-reduction recovery |
| `--epochs` | 40 | Unchanged |

Full command reflected in `EXTRA_ARGS` above:
```
--patience 10 --cw-power 0.75
```

---

## Cost estimate

| Scenario                          | Time est. | Cost est.      |
|-----------------------------------|-----------|----------------|
| Validation run (5 epochs, 2×A100) | ~1.5 hrs  | ~$16           |
| Full run (40 epochs, 2×A100)      | ~12 hrs   | ~$126          |

At $10.50/hr, early stopping (typically saving 20-40% of epochs) matters.
Default `--patience 10` will halt training if val_loss plateaus, keeping
costs down. A validation run first at 5 epochs is recommended to confirm
the pipeline before committing to a full run.

---

## Shutdown / delete VM when done

```bash
# Delete VM (data disk is auto-deleted due to auto-delete=yes flag)
gcloud compute instances delete bat-cnn-training --zone=${ZONE}

# Outputs remain safely in gs://bat-training-output/
```