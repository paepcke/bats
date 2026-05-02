# Random Forest Training

Trains three Random Forest classifiers for bat species identification from
SonoBat acoustic measures:

1. **Main (omnibus)** — multiclass RF covering all species above a minimum
   count threshold.
2. **Coto/Tabr binary** — resolves the Corynorhinus townsendii / Tadarida
   brasiliensis confusion that the omnibus RF handles poorly.
3. **Lano/Tabr binary** — resolves the Lasionycteris noctivagans / Tadarida
   brasiliensis confusion.

All three are launched in parallel by `train_random_forests.sh`, which polls
progress and then runs hierarchical inference for immediate validation.  The
resulting model directories are the inputs to `hierarchical_rf_predict.py`
for inference on new data.

---

## Input

The chirp-level measures parquet produced by `sb_measures_postprocessing.py`:

```
/qnap/bats/all_data/bats_<timestamp>.parquet
```

It must contain a `species` column (the SonoBat four-letter code) and the
standard SonoBat acoustic measure columns.  Rows with composite labels
(e.g. `Laci/Lano`) are dropped automatically by the species-code regex
filter inside each training script.

---

## Outputs

`train_random_forests.sh` creates a timestamped results tree:

```
/data/random_forest/results_<TIMESTAMP>/
├── main_run/
│   ├── rf_model.joblib          Main multiclass RF (required by hierarchical_rf_predict.py)
│   ├── label_encoder.joblib     Species ↔ index mapping
│   ├── feature_names.json       Feature column list (exact order RF was trained on)
│   ├── train_config.csv         All hyperparameters for reproducibility
│   ├── train_log.csv            Per-species counts and split sizes
│   ├── confusion_matrix.png     Test-set confusion matrix
│   └── classification_report.txt  Per-species precision/recall/F1
├── binary_coto_tabr/
│   └── <same structure, two classes only>
├── binary_lano_tabr/
│   └── <same structure, two classes only>
└── validation/
    ├── predictions.csv          Per-chirp hierarchical predictions on the measures input
    ├── confusion_matrix.png     Hierarchical predictions vs ground truth
    └── classification_report.txt
```

---

## Run — orchestrated (recommended)

`train_random_forests.sh` launches all three training jobs in parallel and
runs the validation inference automatically when they finish.  Expected
runtime on sextus (Intel Xeon w7-3455, all cores): **20–40 minutes**.

```bash
PARQUET=/qnap/bats/all_data/bats_2026-04-12T17_12_34.399803.parquet
LOG=/data/random_forest/train_$(date +%Y%m%dT%H%M%S).log
mkdir -p /data/random_forest

nohup time src/species_classification/training/train_random_forests.sh \
    "$PARQUET" \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

> **tmux recommended** — the full run takes 20–40 min.  Either use `nohup`
> as above (so the job survives terminal disconnect) or run inside a tmux
> session.

The script prints the `results_<TIMESTAMP>/` path on completion.  Note that
path for the inference README steps.

---

## Run — individual scripts (if re-running a single model)

### Main multiclass RF

```bash
PARQUET=/qnap/bats/all_data/bats_2026-04-12T17_12_34.399803.parquet
OUT=/data/random_forest/results_<TIMESTAMP>/main_run
LOG=${OUT}/train_$(date +%Y%m%dT%H%M%S).log
mkdir -p "$OUT"

nohup time src/species_classification/training/species_pred_random_forest.py \
    --input              "$PARQUET" \
    --out-dir            "$OUT" \
    --min-species-count  500 \
    --n-estimators       300 \
    --n-jobs             -1 \
    --exclude-species    HiF Anpa Lafr Myvo \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

### Coto/Tabr binary RF

```bash
PARQUET=/qnap/bats/all_data/bats_2026-04-12T17_12_34.399803.parquet
OUT=/data/random_forest/results_<TIMESTAMP>/binary_coto_tabr
LOG=${OUT}/train_$(date +%Y%m%dT%H%M%S).log
mkdir -p "$OUT"

nohup time src/species_classification/training/species_pred_random_forest.py \
    --input          "$PARQUET" \
    --out-dir        "$OUT" \
    --mode           binary \
    --species-pair   Coto Tabr \
    --n-estimators   300 \
    --n-jobs         -1 \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

### Lano/Tabr binary RF

```bash
PARQUET=/qnap/bats/all_data/bats_2026-04-12T17_12_34.399803.parquet
OUT=/data/random_forest/results_<TIMESTAMP>/binary_lano_tabr
LOG=${OUT}/train_$(date +%Y%m%dT%H%M%S).log
mkdir -p "$OUT"

nohup time src/species_classification/training/species_pred_random_forest.py \
    --input          "$PARQUET" \
    --out-dir        "$OUT" \
    --mode           binary \
    --species-pair   Lano Tabr \
    --n-estimators   300 \
    --n-jobs         -1 \
    > "$LOG" 2>&1 &

sleep 1 && tail -f "$LOG"
```

---

## Key options — species_pred_random_forest.py

| Flag | Default | Notes |
|------|---------|-------|
| `--min-species-count N` | 500 | Drop species with fewer chirp fragments |
| `--n-estimators N` | 300 | More trees = better accuracy, slower training |
| `--n-jobs N` | -1 | -1 = all cores |
| `--exclude-species SP…` | — | Space-separated codes, multiclass mode only |
| `--mode` | `multiclass` | `binary` requires `--species-pair` |
| `--species-pair SP SP` | — | Two codes for binary mode |
| `--val-frac F` | 0.15 | Validation fraction of file_ids |
| `--test-frac F` | 0.15 | Test fraction of file_ids |
| `--max-features STR` | `sqrt` | `sqrt`, `log2`, or a float fraction |
| `--random-state N` | 42 | Seed for reproducibility |

The train/val/test split is at the **`file_id` level** (stratified by modal
species), preventing chirp-level data leakage between splits.

---

## Typical results (April 2026 run)

Training on the clean parquet
(`bats_2026-04-12T17_12_34.399803.parquet`, 18M rows, 11 species):

```
Omnibus RF   — test accuracy ~0.84, runtime ~25 min (all cores, 300 trees)
Coto/Tabr RF — AUC ~0.97, runtime ~3 min
Lano/Tabr RF — AUC ~0.96, runtime ~3 min
```