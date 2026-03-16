#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-03-16 15:41:14
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/species_classification/train_cnn.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-16 15:49:49
# **********************************************************
#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-16
# @File:   src/species_classification/train_cnn.py
#
# **********************************************************

"""
train_cnn.py
============
Fine-tune EfficientNet-B0 on per-chirp spectrogram crops produced by
``chirps_to_spectros.py`` for bat species classification.

Overview
--------
Input is the ``manifest.csv`` written by ``chirps_to_spectros.py``, which
contains one row per PNG crop with columns including ``crop_path``,
``species``, ``species_prob``, and ``file_id``.

The train/validation/test split is made at the **``file_id`` level**
(stratified by species modal label), so that all chirps from the same
2-second fragment land in the same split.  This prevents any information
leakage between splits that would arise from splitting at the chirp level.

Architecture
------------
EfficientNet-B0 pretrained on ImageNet.  The classifier head is replaced
with a linear layer sized to ``n_classes``.  Grayscale crops are replicated
to 3 channels before passing to the network (ImageNet weights expect RGB).

Training strategy
-----------------
* Phase 1 (head only, ``--freeze-epochs`` epochs): only the new classifier
  head is trained; the EfficientNet backbone is frozen.  Allows the head
  to reach a reasonable starting point before the backbone weights are
  perturbed.
* Phase 2 (full fine-tune, remaining epochs): entire network is trained
  with a lower learning rate (``--lr`` × ``--backbone-lr-factor``).

Outputs (all written to ``--out-dir``)
---------------------------------------
``best_model.pt``
    State dict of the epoch with the highest validation accuracy.
``final_model.pt``
    State dict after the last epoch.
``label_encoder.json``
    Maps integer class index ↔ species string.
``train_config.csv``
    All run hyperparameters for reproducibility.
``train_log.csv``
    Per-epoch: loss, accuracy, val_loss, val_accuracy.
``confusion_matrix.png``
    Confusion matrix on the held-out test set using ``best_model.pt``.
``classification_report.txt``
    Per-class precision/recall/F1 on the test set.

Typical usage
-------------
::

    python train_cnn.py \\
        --manifest  /raid/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --out-dir   /raid/bats/models/efficientnet_b0_v1 \\
        --epochs    40 \\
        --batch     64 \\
        --workers   8

To restrict to a subset of species::

    python train_cnn.py \\
        --manifest  /raid/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --out-dir   /raid/bats/models/top5_v1 \\
        --species   Myca Myyu Lano Tabr Laci \\
        --epochs    40
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EPOCHS:            int   = 40
_DEFAULT_BATCH:             int   = 64
_DEFAULT_LR:                float = 1e-3
_DEFAULT_BACKBONE_LR_FACTOR:float = 0.1    # backbone LR = LR × factor
_DEFAULT_FREEZE_EPOCHS:     int   = 5      # epochs to train head only
_DEFAULT_WEIGHT_DECAY:      float = 1e-4
_DEFAULT_WORKERS:           int   = 4
_DEFAULT_VAL_FRAC:          float = 0.15
_DEFAULT_TEST_FRAC:         float = 0.15
_DEFAULT_MIN_PROB:          float = 0.80
_DEFAULT_MIN_CROPS:         int   = 50     # drop species with fewer crops
_IMG_SIZE:                  int   = 224


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChirpCropDataset(Dataset):
    """
    PyTorch Dataset for per-chirp spectrogram PNG crops.

    Loads grayscale PNGs and replicates to 3 channels for EfficientNet.
    Applies data augmentation during training (horizontal flip, brightness
    jitter) and only normalisation during validation/test.

    :param df:         DataFrame with columns ``crop_path`` and ``label``
                       (integer class index).
    :param augment:    If ``True``, apply training augmentations.
    :param img_size:   Resize target (square).
    """

    # ImageNet normalisation statistics — applied after grayscale → RGB copy.
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        df:       pd.DataFrame,
        augment:  bool = False,
        img_size: int  = _IMG_SIZE,
    ) -> None:
        self.paths  = df['crop_path'].tolist()
        self.labels = df['label'].tolist()

        base = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # L → RGB
            transforms.Normalize(self._MEAN, self._STD),
        ]
        aug = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(self._MEAN, self._STD),
        ]
        self.transform = transforms.Compose(aug if augment else base)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img), self.labels[idx]


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def make_splits(
    df:         pd.DataFrame,
    val_frac:   float = _DEFAULT_VAL_FRAC,
    test_frac:  float = _DEFAULT_TEST_FRAC,
    seed:       int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train/val/test at the ``file_id`` level, stratified by
    each fragment's modal species label.

    Splitting at ``file_id`` level ensures all chirps from the same 2-second
    fragment land in the same split, preventing leakage.

    :param df:        Full crop DataFrame with ``file_id`` and ``species``
                      columns.
    :param val_frac:  Fraction of file_ids for validation.
    :param test_frac: Fraction of file_ids for test.
    :param seed:      Random seed for reproducibility.
    :return:          ``(train_df, val_df, test_df)``
    """
    rng = np.random.default_rng(seed)

    # One row per file_id: modal species determines stratum.
    fid_species = (
        df.groupby('file_id')['species']
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={'species': 'modal_species'})
    )

    train_fids, val_fids, test_fids = [], [], []

    for sp, grp in fid_species.groupby('modal_species'):
        fids = grp['file_id'].values.copy()
        rng.shuffle(fids)
        n        = len(fids)
        n_test   = max(1, int(round(n * test_frac)))
        n_val    = max(1, int(round(n * val_frac)))
        n_train  = n - n_test - n_val
        if n_train < 1:
            # Too few fragments for this species — put everything in train.
            train_fids.extend(fids.tolist())
            continue
        test_fids .extend(fids[:n_test].tolist())
        val_fids  .extend(fids[n_test:n_test + n_val].tolist())
        train_fids.extend(fids[n_test + n_val:].tolist())

    train_set = set(train_fids)
    val_set   = set(val_fids)
    test_set  = set(test_fids)

    train_df = df[df['file_id'].isin(train_set)].copy()
    val_df   = df[df['file_id'].isin(val_set)].copy()
    test_df  = df[df['file_id'].isin(test_set)].copy()

    log.info(
        f'Split: {len(train_df):,} train / {len(val_df):,} val / '
        f'{len(test_df):,} test crops  '
        f'({len(train_fids):,} / {len(val_fids):,} / {len(test_fids):,} file_ids)'
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(n_classes: int, device: torch.device) -> nn.Module:
    """
    Build EfficientNet-B0 with a fresh classifier head sized to *n_classes*.

    Pretrained ImageNet weights are loaded for the backbone.  The original
    classifier (1000-class) is replaced with ``Linear(1280, n_classes)``.

    :param n_classes: Number of bat species classes.
    :param device:    Target device.
    :return:          Model moved to *device*.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, n_classes),
    )
    return model.to(device)


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters except the classifier head.

    :param model: EfficientNet-B0 model.
    """
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreeze all parameters.

    :param model: EfficientNet-B0 model.
    """
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    train:     bool,
) -> tuple[float, float]:
    """
    Run one epoch of training or evaluation.

    :param model:     The model.
    :param loader:    DataLoader for this split.
    :param criterion: Loss function.
    :param optimizer: Optimiser (``None`` during eval).
    :param device:    Compute device.
    :param train:     If ``True``, update weights; else eval mode.
    :return:          ``(mean_loss, accuracy)``
    """
    model.train(train)
    total_loss = 0.0
    n_correct  = 0
    n_total    = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    pbar = tqdm(loader, leave=False) if _TQDM else loader

    with ctx:
        for imgs, labels in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds       = logits.argmax(dim=1)
            n_correct  += (preds == labels).sum().item()
            n_total    += len(labels)

    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_test(
    model:        nn.Module,
    test_df:      pd.DataFrame,
    label_to_idx: dict[str, int],
    device:       torch.device,
    batch_size:   int,
    n_workers:    int,
    out_dir:      Path,
) -> None:
    """
    Run model on the test set, write confusion matrix PNG and
    classification report TXT.

    :param model:        Trained model in eval mode.
    :param test_df:      Test split DataFrame.
    :param label_to_idx: Species → integer index mapping.
    :param device:       Compute device.
    :param batch_size:   Inference batch size.
    :param n_workers:    DataLoader worker count.
    :param out_dir:      Directory for output files.
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names  = [idx_to_label[i] for i in range(len(label_to_idx))]

    loader = DataLoader(
        ChirpCropDataset(test_df, augment=False),
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = n_workers,
        pin_memory  = True,
    )

    all_preds  = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            preds  = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds .extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Test set confusion matrix — EfficientNet-B0')
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    log.info(f'Saved confusion matrix to {out_dir / "confusion_matrix.png"}')

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=3, zero_division=0,
    )
    (out_dir / 'classification_report.txt').write_text(report)
    log.info(f'Test classification report:\n{report}')


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

class CnnTrainer:
    """
    Fine-tune EfficientNet-B0 on bat species spectrogram crops.

    :param manifest_csv:        Path to manifest CSV from
                                ``chirps_to_spectros.py``.
    :param out_dir:             Output directory for checkpoints and logs.
    :param species:             If non-empty, restrict to these species codes.
    :param min_prob:            Minimum ``species_prob`` to include a crop.
    :param min_crops_per_class: Drop species with fewer crops than this.
    :param epochs:              Total training epochs.
    :param freeze_epochs:       Epochs to train classifier head only before
                                unfreezing the backbone.
    :param batch_size:          Training batch size.
    :param lr:                  Initial learning rate (head phase).
    :param backbone_lr_factor:  Backbone LR = lr × factor (full fine-tune phase).
    :param weight_decay:        AdamW weight decay.
    :param val_frac:            Fraction of file_ids for validation.
    :param test_frac:           Fraction of file_ids for test.
    :param n_workers:           DataLoader worker processes.
    :param device_str:          ``'cuda'``, ``'cpu'``, or ``'auto'``.
    :param seed:                Random seed.
    """

    def __init__(
        self,
        manifest_csv:        str | Path,
        out_dir:             str | Path,
        species:             Sequence[str]  = (),
        min_prob:            float          = _DEFAULT_MIN_PROB,
        min_crops_per_class: int            = _DEFAULT_MIN_CROPS,
        epochs:              int            = _DEFAULT_EPOCHS,
        freeze_epochs:       int            = _DEFAULT_FREEZE_EPOCHS,
        batch_size:          int            = _DEFAULT_BATCH,
        lr:                  float          = _DEFAULT_LR,
        backbone_lr_factor:  float          = _DEFAULT_BACKBONE_LR_FACTOR,
        weight_decay:        float          = _DEFAULT_WEIGHT_DECAY,
        val_frac:            float          = _DEFAULT_VAL_FRAC,
        test_frac:           float          = _DEFAULT_TEST_FRAC,
        n_workers:           int            = _DEFAULT_WORKERS,
        device_str:          str            = 'auto',
        seed:                int            = 42,
    ) -> None:
        self.manifest_csv        = Path(manifest_csv)
        self.out_dir             = Path(out_dir)
        self.species             = list(species)
        self.min_prob            = min_prob
        self.min_crops_per_class = min_crops_per_class
        self.epochs              = epochs
        self.freeze_epochs       = freeze_epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.backbone_lr_factor  = backbone_lr_factor
        self.weight_decay        = weight_decay
        self.val_frac            = val_frac
        self.test_frac           = test_frac
        self.n_workers           = n_workers
        self.seed                = seed

        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> pd.DataFrame:
        """
        Load and filter the manifest CSV.

        Filters applied in order:
        1. ``species`` not NaN and matches ``[A-Z][a-z]{3}`` pattern.
        2. ``species_prob`` >= ``self.min_prob`` (or NaN — kept with warning).
        3. ``crop_path`` file must exist on disk.
        4. If ``self.species`` non-empty, restrict to those species.
        5. Drop species with fewer than ``self.min_crops_per_class`` crops.

        :return: Filtered DataFrame ready for splitting.
        """
        import re
        _sp_re = re.compile(r'^[A-Z][a-z]{3}$')

        log.info(f'Loading manifest: {self.manifest_csv}')
        df = pd.read_csv(self.manifest_csv)
        log.info(f'  {len(df):,} total rows')

        # Species filter
        df = df[df['species'].notna()]
        df = df[df['species'].apply(lambda s: bool(_sp_re.match(str(s))))]
        log.info(f'  {len(df):,} rows with valid species code')

        # Confidence filter
        prob_col = pd.to_numeric(df['species_prob'], errors='coerce')
        df = df[prob_col.isna() | (prob_col >= self.min_prob)]
        log.info(f'  {len(df):,} rows after confidence filter (min_prob={self.min_prob})')

        # Species subset filter
        if self.species:
            df = df[df['species'].isin(self.species)]
            log.info(f'  {len(df):,} rows after species filter {self.species}')

        # Drop species below minimum crop count
        counts = df['species'].value_counts()
        valid_species = counts[counts >= self.min_crops_per_class].index
        dropped = counts[counts < self.min_crops_per_class]
        if len(dropped):
            log.warn(
                f'Dropping {len(dropped)} species with < {self.min_crops_per_class} '
                f'crops: {dropped.to_dict()}'
            )
        df = df[df['species'].isin(valid_species)]
        log.info(f'  {len(df):,} rows after min-crops filter')

        # Verify crop files exist (sample check on first 100)
        missing = [p for p in df['crop_path'].iloc[:100] if not Path(p).exists()]
        if missing:
            log.warn(f'{len(missing)} sample crop paths not found on disk '
                     f'(first: {missing[0]}). Check --manifest path.')

        log.info(f'Species distribution:\n{df["species"].value_counts().to_string()}')
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Execute the full training pipeline:
        load → split → build model → train → evaluate → save artefacts.
        """
        _t0 = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ── Load and split ─────────────────────────────────────────────
        df = self._load_manifest()

        # Encode species labels as integers
        species_list  = sorted(df['species'].unique().tolist())
        label_to_idx  = {sp: i for i, sp in enumerate(species_list)}
        idx_to_label  = {i: sp for sp, i in label_to_idx.items()}
        n_classes     = len(species_list)
        df['label']   = df['species'].map(label_to_idx)
        log.info(f'{n_classes} classes: {species_list}')

        # Save label encoder
        (self.out_dir / 'label_encoder.json').write_text(
            json.dumps({'label_to_idx': label_to_idx,
                        'idx_to_label': {str(k): v for k, v in idx_to_label.items()}},
                       indent=2)
        )

        train_df, val_df, test_df = make_splits(
            df, self.val_frac, self.test_frac, self.seed
        )

        # Class weights for imbalanced dataset (inverse frequency)
        train_counts = train_df['label'].value_counts().sort_index()
        class_weights = torch.tensor(
            [1.0 / max(train_counts.get(i, 1), 1) for i in range(n_classes)],
            dtype=torch.float32,
        ).to(self.device)
        class_weights = class_weights / class_weights.sum() * n_classes

        # ── DataLoaders ────────────────────────────────────────────────
        train_loader = DataLoader(
            ChirpCropDataset(train_df, augment=True),
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.n_workers,
            pin_memory  = True,
            drop_last   = True,
        )
        val_loader = DataLoader(
            ChirpCropDataset(val_df, augment=False),
            batch_size  = self.batch_size * 2,
            shuffle     = False,
            num_workers = self.n_workers,
            pin_memory  = True,
        )

        # ── Model ──────────────────────────────────────────────────────
        log.info(f'Building EfficientNet-B0 ({n_classes} classes) on {self.device}')
        model     = build_model(n_classes, self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ── Phase 1: head only ─────────────────────────────────────────
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.freeze_epochs
        )

        log.info(f'Phase 1: training head only for {self.freeze_epochs} epochs')

        best_val_acc  = 0.0
        best_epoch    = 0
        log_rows: list[dict] = []

        for epoch in range(1, self.epochs + 1):
            # Switch to full fine-tune after freeze_epochs
            if epoch == self.freeze_epochs + 1:
                log.info('Phase 2: unfreezing backbone, reducing LR')
                unfreeze_all(model)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr           = self.lr * self.backbone_lr_factor,
                    weight_decay = self.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.epochs - self.freeze_epochs
                )

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, self.device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None, self.device, train=False
            )
            scheduler.step()

            log.info(
                f'Epoch {epoch:3d}/{self.epochs}  '
                f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
                f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}'
            )
            log_rows.append({
                'epoch':      epoch,
                'train_loss': round(train_loss, 6),
                'train_acc':  round(train_acc,  6),
                'val_loss':   round(val_loss,   6),
                'val_acc':    round(val_acc,    6),
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                torch.save(model.state_dict(), self.out_dir / 'best_model.pt')
                log.info(f'  ✓ New best val_acc={best_val_acc:.4f} — saved best_model.pt')

        torch.save(model.state_dict(), self.out_dir / 'final_model.pt')
        log.info(f'Saved final_model.pt  (best was epoch {best_epoch})')

        # ── Training log ───────────────────────────────────────────────
        pd.DataFrame(log_rows).to_csv(self.out_dir / 'train_log.csv', index=False)

        # ── Test evaluation ────────────────────────────────────────────
        log.info('Loading best_model.pt for test evaluation ...')
        model.load_state_dict(
            torch.load(self.out_dir / 'best_model.pt', map_location=self.device)
        )
        evaluate_test(
            model, test_df, label_to_idx,
            self.device, self.batch_size * 2, self.n_workers, self.out_dir,
        )

        # ── Config CSV ─────────────────────────────────────────────────
        elapsed = time.perf_counter() - _t0
        pd.DataFrame([
            {'parameter': 'manifest_csv',        'value': str(self.manifest_csv)},
            {'parameter': 'out_dir',             'value': str(self.out_dir)},
            {'parameter': 'n_classes',           'value': n_classes},
            {'parameter': 'species',             'value': str(species_list)},
            {'parameter': 'epochs',              'value': self.epochs},
            {'parameter': 'freeze_epochs',       'value': self.freeze_epochs},
            {'parameter': 'batch_size',          'value': self.batch_size},
            {'parameter': 'lr',                  'value': self.lr},
            {'parameter': 'backbone_lr_factor',  'value': self.backbone_lr_factor},
            {'parameter': 'weight_decay',        'value': self.weight_decay},
            {'parameter': 'val_frac',            'value': self.val_frac},
            {'parameter': 'test_frac',           'value': self.test_frac},
            {'parameter': 'min_prob',            'value': self.min_prob},
            {'parameter': 'min_crops_per_class', 'value': self.min_crops_per_class},
            {'parameter': 'device',              'value': str(self.device)},
            {'parameter': 'seed',                'value': self.seed},
            {'parameter': 'best_epoch',          'value': best_epoch},
            {'parameter': 'best_val_acc',        'value': round(best_val_acc, 6)},
            {'parameter': 'elapsed_secs',        'value': round(elapsed, 1)},
        ]).to_csv(self.out_dir / 'train_config.csv', index=False)

        log.info(
            f'Training complete in {elapsed/60:.1f} min  '
            f'best val_acc={best_val_acc:.4f} at epoch {best_epoch}'
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog='train_cnn',
        description=(
            'Fine-tune EfficientNet-B0 on bat species spectrogram crops.\n\n'
            'Input: manifest.csv from chirps_to_spectros.py\n'
            'Output: model checkpoints, training log, confusion matrix'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--manifest', required=True, metavar='CSV',
        help='manifest.csv from chirps_to_spectros.py',
    )
    parser.add_argument(
        '--out-dir', required=True, metavar='DIR',
        help='output directory for checkpoints and logs',
    )
    parser.add_argument(
        '--species', nargs='+', default=[], metavar='SP',
        help='restrict to these species codes (default: all)',
    )
    parser.add_argument(
        '--epochs', type=int, default=_DEFAULT_EPOCHS,
        help=f'total training epochs (default: {_DEFAULT_EPOCHS})',
    )
    parser.add_argument(
        '--freeze-epochs', type=int, default=_DEFAULT_FREEZE_EPOCHS,
        help=f'epochs to train head only (default: {_DEFAULT_FREEZE_EPOCHS})',
    )
    parser.add_argument(
        '--batch', type=int, default=_DEFAULT_BATCH,
        help=f'batch size (default: {_DEFAULT_BATCH})',
    )
    parser.add_argument(
        '--lr', type=float, default=_DEFAULT_LR,
        help=f'initial learning rate (default: {_DEFAULT_LR})',
    )
    parser.add_argument(
        '--backbone-lr-factor', type=float, default=_DEFAULT_BACKBONE_LR_FACTOR,
        help=f'backbone LR = lr × factor (default: {_DEFAULT_BACKBONE_LR_FACTOR})',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=_DEFAULT_WEIGHT_DECAY,
        help=f'AdamW weight decay (default: {_DEFAULT_WEIGHT_DECAY})',
    )
    parser.add_argument(
        '--val-frac', type=float, default=_DEFAULT_VAL_FRAC,
        help=f'validation fraction of file_ids (default: {_DEFAULT_VAL_FRAC})',
    )
    parser.add_argument(
        '--test-frac', type=float, default=_DEFAULT_TEST_FRAC,
        help=f'test fraction of file_ids (default: {_DEFAULT_TEST_FRAC})',
    )
    parser.add_argument(
        '--min-prob', type=float, default=_DEFAULT_MIN_PROB,
        help=f'minimum species_prob (default: {_DEFAULT_MIN_PROB})',
    )
    parser.add_argument(
        '--min-crops', type=int, default=_DEFAULT_MIN_CROPS,
        help=f'minimum crops per species (default: {_DEFAULT_MIN_CROPS})',
    )
    parser.add_argument(
        '--workers', type=int, default=_DEFAULT_WORKERS,
        help=f'DataLoader worker processes (default: {_DEFAULT_WORKERS})',
    )
    parser.add_argument(
        '--device', default='auto',
        help='cuda / cpu / auto (default: auto)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random seed (default: 42)',
    )

    args = parser.parse_args()

    if not Path(args.manifest).exists():
        parser.error(f'manifest not found: {args.manifest}')

    return args


def main() -> None:
    args = _parse_args()

    trainer = CnnTrainer(
        manifest_csv        = args.manifest,
        out_dir             = args.out_dir,
        species             = args.species,
        min_prob            = args.min_prob,
        min_crops_per_class = args.min_crops,
        epochs              = args.epochs,
        freeze_epochs       = args.freeze_epochs,
        batch_size          = args.batch,
        lr                  = args.lr,
        backbone_lr_factor  = args.backbone_lr_factor,
        weight_decay        = args.weight_decay,
        val_frac            = args.val_frac,
        test_frac           = args.test_frac,
        n_workers           = args.workers,
        device_str          = args.device,
        seed                = args.seed,
    )
    trainer.run()


if __name__ == '__main__':
    main()