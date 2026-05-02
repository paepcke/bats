# RF Inference on New Recordings (From-Scratch Pipeline)

End-to-end workflow for classifying bat species in new recordings that have
**not** been processed by SonoBat.  The pipeline produces a
`measures_classified.csv` suitable for `from_scratch_postprocessing.py`,
which writes the standard `bats_<ts>.parquet` used by all downstream tools.

For the CNN inference step (once the CNN model is trained) see
`README_cnn_inference.md`.

---

## Pipeline overview

```
new .wav recordings
        │
        ▼
  Step 1  wav_file_scrubber.py      — drop obvious noise files
        │
        ▼
  Step 2  wav_chopper.py            — slice retained files into 2-sec chunks
        │
        ▼
  Step 3  chirp_measures_extraction.py  — extract SonoBat-style acoustic measures
        │
        ▼
  Step 4  hierarchical_rf_predict.py    — two-stage RF species prediction
        │
        ▼
  Step 5  ensemble_reconcile.py         — combine RF (+ CNN when available)
        │
        ▼
  Step 6  rf_confidence_join.py         — compute per-chop confidence score
        │
        ▼
  Step 7  from_scratch_postprocessing.py — write bats_*.parquet + SQLite DB
```

---

## Environment variables used throughout

Set these once at the top of your shell session (or tmux window) before
running any step:

```bash
# Input recordings directory
RAW=/data2/marsh

# Working directory for all intermediate and output files
OUT=/data2/marsh_stanford_processed

# Trained RF model directory (results_<TIMESTAMP> from train_random_forests.sh)
RF_MODEL=/data/random_forest/results_20260417T144601

# Existing main parquet and DB (for merging at the end)
MAIN=/qnap/bats/all_data/bats_2026-04-12T17_12_34.399803.parquet
DB=/qnap/bats/chirp_meta.db

mkdir -p "$OUT"
```

---

## Step 1 — Scrub noise files

Drops full-recording `.wav` files that are clearly non-bat noise before
chopping.  Outputs a retained-file list used by Step 2.

Expected runtime: ~90 min for 510K files (sextus, all cores).

```bash
LOG=${OUT}/scrub_$(date +%Y%m%dT%H%M%S).log

nohup time src/species_classification/from_scratch_preprocessing/wav_file_scrubber.py \
    "$RAW" \
    --recursive \
    --min-pulses    3 \
    --ipi-cv        2.5 \
    --out-csv       $OUT/scrub_report.csv \
    --retained-list $OUT/retained_wavs.txt \
    --checkpoint    $OUT/scrub_checkpoint.csv \
    --timeout       180 \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

Key options:

| Flag | Default | Notes |
|------|---------|-------|
| `--min-pulses N` | 3 | Files with fewer detected pulses are scrubbed |
| `--ipi-cv F` | — | Inter-pulse-interval coefficient of variation ceiling; 2.5 retains more files than the default by tolerating harmonic sequences |
| `--timeout SECS` | 120 | Per-file analysis timeout |
| `--checkpoint CSV` | — | Resume point if the scrubber is interrupted |
| `--recursive` | off | Walk subdirectories |

> **Checkpoint/resume**: if the run is interrupted, re-run the identical
> command — it will skip already-processed files recorded in
> `scrub_checkpoint.csv`.

---

## Step 2 — Chop retained files into 2-second chunks

Slices every retained recording into 2-second `.wav` fragments and writes a
`chop_report.csv` that records the source duration of each chunk (used by
Step 6 to compute confidence scores).

Expected runtime: ~5 min for 220K files (sextus).

```bash
LOG=${OUT}/chop_$(date +%Y%m%dT%H%M%S).log

nohup time src/species_classification/from_scratch_preprocessing/wav_chopper.py \
    --file-list   $OUT/retained_wavs.txt \
    --out-dir     $OUT/chunks \
    --chunk-dur   2.0 \
    --checkpoint  $OUT/chop_checkpoint.csv \
    --summary     $OUT/chop_report.csv \
    --timeout     180 \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

Key options:

| Flag | Default | Notes |
|------|---------|-------|
| `--chunk-dur SECS` | 2.0 | Chunk length in seconds |
| `--checkpoint CSV` | — | Resume point if interrupted |
| `--summary CSV` | — | Per-chunk provenance (required for Step 6 Path 1) |
| `--timeout SECS` | 180 | Per-file timeout |

---

## Step 3 — Extract chirp measures

Runs a SonoBat-compatible acoustic feature extractor on every chunk and
writes one row per detected chirp to `measures.csv`.

Expected runtime: ~45 min for 660K chunks (sextus, all cores).

```bash
LOG=${OUT}/measures_$(date +%Y%m%dT%H%M%S).log

nohup time src/species_classification/from_scratch_preprocessing/chirp_measures_extraction.py \
    $OUT/chunks \
    --out-csv   $OUT/measures.csv \
    --recursive \
    --timeout   120 \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

Key options:

| Flag | Default | Notes |
|------|---------|-------|
| `--recursive` | off | Walk subdirectories of the chunks dir |
| `--workers N` | all cores | Parallel worker processes |
| `--timeout SECS` | 120 | Per-file timeout |
| `--done-csv CSV…` | — | Skip files already listed in these CSVs (resume) |

---

## Step 4 — Hierarchical RF inference

Two-stage prediction: the main multiclass RF assigns a species to every
chirp, then the Coto/Tabr and Lano/Tabr binary RFs resolve the two most
common inter-species confusions.

Expected runtime: ~5–15 min depending on chirp count.

```bash
LOG=${OUT}/rf_$(date +%Y%m%dT%H%M%S).log

nohup time src/species_classification/inference/hierarchical_rf_predict.py \
    --input        $OUT/measures.csv \
    --main-rf      $RF_MODEL/main_run \
    --coto-tabr-rf $RF_MODEL/binary_coto_tabr \
    --lano-tabr-rf $RF_MODEL/binary_lano_tabr \
    --out-dir      $OUT/rf_predictions \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

Output: `$OUT/rf_predictions/predictions.csv` — one row per chirp with
`final_species` and `final_rf_agreement` columns.

If the measures file contains a `species` column (labeled data), the script
also writes a `confusion_matrix.png` and `classification_report.txt` to
`--out-dir` automatically.

---

## Step 5 — Ensemble reconciliation

Combines RF predictions with CNN predictions (once the CNN is available)
into a single `ensemble_predictions.csv`.  In the interim, run RF-only mode.

### RF-only (current)

```bash
LOG=${OUT}/ensemble_$(date +%Y%m%dT%H%M%S).log

time src/species_classification/inference/ensemble_reconcile.py \
    --rf-predictions-csv  $OUT/rf_predictions/predictions.csv \
    --out-csv             $OUT/ensemble_predictions.csv \
    > "$LOG" 2>&1

cat "$LOG"
```

### Full ensemble (once CNN predictions are available)

```bash
LOG=${OUT}/ensemble_$(date +%Y%m%dT%H%M%S).log

time src/species_classification/inference/ensemble_reconcile.py \
    --rf-predictions-csv  $OUT/rf_predictions/predictions.csv \
    --cnn-predictions-csv $OUT/cnn_predictions/predictions.csv \
    --weight-rf           0.5 \
    --weight-cnn          0.5 \
    --out-csv             $OUT/ensemble_predictions.csv \
    > "$LOG" 2>&1

cat "$LOG"
```

The output path `$OUT/ensemble_predictions.csv` is identical in both modes;
Step 6 does not change between them.

---

## Step 6 — Confidence score join

Joins ensemble predictions onto the measures CSV and computes a
SonoBat-analog per-chop confidence score.  Three paths depending on what
duration information is available:

### Path 1 — From-scratch pipeline (use this for marsh)

`chop_report.csv` from Step 2 provides exact source duration per chunk.

```bash
LOG=${OUT}/confidence_$(date +%Y%m%dT%H%M%S).log

time src/species_classification/rf_confidence_join.py \
    --measures-csv    $OUT/measures.csv \
    --predictions-csv $OUT/ensemble_predictions.csv \
    --chop-report     $OUT/chop_report.csv \
    --out-csv         $OUT/measures_classified.csv \
    > "$LOG" 2>&1

cat "$LOG"
```

### Path 2 — Legacy SonoBat pipeline (barn/lake2)

No `chop_report.csv` available; duration is reconstructed from the manifest.

```bash
LOG=${OUT}/confidence_$(date +%Y%m%dT%H%M%S).log

time src/species_classification/rf_confidence_join.py \
    --measures-csv    $OUT/measures.csv \
    --predictions-csv $OUT/ensemble_predictions.csv \
    --manifest        /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \
    --out-csv         $OUT/measures_classified.csv \
    > "$LOG" 2>&1

cat "$LOG"
```

### Path 3 — Fallback (no duration info)

Use only if neither `chop_report.csv` nor a manifest is available.

```bash
LOG=${OUT}/confidence_$(date +%Y%m%dT%H%M%S).log

time src/species_classification/rf_confidence_join.py \
    --measures-csv    $OUT/measures.csv \
    --predictions-csv $OUT/ensemble_predictions.csv \
    --chop-duration   2.0 \
    --out-csv         $OUT/measures_classified.csv \
    > "$LOG" 2>&1

cat "$LOG"
```

---

## Step 7 — Postprocessing: write parquet and update SQLite DB

Converts `measures_classified.csv` into the standard
`bats_<ts>.parquet` / `bats_noise_<ts>.parquet` format, optionally
extending an existing parquet and DB so file_ids do not collide.

```bash
LOG=${OUT}/postproc_$(date +%Y%m%dT%H%M%S).log

nohup time src/species_classification/from_scratch_preprocessing/from_scratch_postprocessing.py \
    --measures-csv      $OUT/measures_classified.csv \
    --rec-site          marsh \
    --dest-dir          $OUT \
    --existing-parquet  $MAIN \
    --build-db          $DB \
    --add-daytime-columns \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

Key options:

| Flag | Default | Notes |
|------|---------|-------|
| `--rec-site SITE` | required | Recording site label, e.g. `marsh` |
| `--conf-thresh F` | 0.50 | Minimum confidence to enter the clean parquet |
| `--existing-parquet PATH` | — | Extend an existing file_id space (required before merging) |
| `--build-db PATH` | — | SQLite DB to create or extend |
| `--add-daytime-columns` | off | Append `was_daytime` / `time_of_day_pactime`; requires `--build-db` |

Outputs written to `$OUT`:

```
bats_<timestamp>.parquet         Clean chirp rows (confidence ≥ threshold)
bats_noise_<timestamp>.parquet   Rejected / noise rows
manifest.csv                     crop_path manifest for CNN spectrogram extraction
```