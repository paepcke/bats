#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-03-16 15:41:14
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/species_classification/train_cnn.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-30 09:20:35
# **********************************************************

"""
train_cnn.py
============
Fine-tune EfficientNet-B0 on per-chirp spectrogram crops produced by
``chirps_to_spectros.py`` for bat species classification.

Supports single-GPU and multi-GPU training via PyTorch
DistributedDataParallel (DDP).  Launch with ``torchrun`` for multi-GPU:

    torchrun --nproc_per_node=2 train_cnn.py \\
        --manifest  /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --out-dir   /qnap/bats/jr_pipeline/models/efficientnet_b0_v1 \\
        --epochs    25 \\
        --batch     64 \\
        --workers   8

Single-GPU (unchanged from before):

    python train_cnn.py \\
        --manifest  /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --out-dir   /qnap/bats/jr_pipeline/models/efficientnet_b0_v1 \\
        --epochs    25 \\
        --batch     64 \\
        --workers   8

Overview
--------
Input is the ``manifest.csv`` written by ``chirps_to_spectros.py``, which
contains one row per PNG crop with columns including ``crop_path``,
``species``, ``species_prob``, and ``file_id``.

The train/validation/test split is made at the **``file_id`` level**
(stratified by species modal label), so that all chirps from the same
2-second fragment land in the same split.  This prevents any information
leakage between splits that would arise from splitting at the chirp level.

DDP strategy
------------
* One process per GPU, launched via ``torchrun --nproc_per_node=N``.
* Each process owns one GPU (``local_rank``).
* ``DistributedSampler`` ensures each GPU sees a non-overlapping shard of
  the training data each epoch, with shuffling coordinated across ranks.
* Gradients are all-reduced automatically by DDP after each backward pass.
* Validation, test evaluation, checkpoint saving, and logging are performed
  only on rank 0 to avoid duplicate writes.
* Class weights are broadcast from rank 0 to all ranks after construction.
* Batch size in ``--batch`` is **per GPU**; effective batch size =
  ``--batch × n_gpus``.

Architecture
------------
EfficientNet-B0 pretrained on ImageNet.  The classifier head is replaced
with a linear layer sized to ``n_classes``.  Grayscale crops are replicated
to 3 channels before passing to the network (ImageNet weights expect RGB).

Training strategy
-----------------
* Phase 1 (head only, ``--freeze-epochs`` epochs): only the new classifier
  head is trained; the EfficientNet backbone is frozen.
* Phase 2 (full fine-tune, remaining epochs): entire network is trained
  with a lower learning rate (``--lr`` × ``--backbone-lr-factor``).

Outputs (all written to ``--out-dir`` by rank 0)
-------------------------------------------------
``best_model.pt``       State dict of the epoch with highest val accuracy.
``final_model.pt``      State dict after the last epoch.
``label_encoder.json``  Maps integer class index ↔ species string.
``train_config.csv``    All run hyperparameters for reproducibility.
``train_log.csv``       Per-epoch: loss, accuracy, val_loss, val_accuracy.
``confusion_matrix.png``  Confusion matrix on held-out test set.
``classification_report.txt``  Per-class precision/recall/F1 on test set.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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
_DEFAULT_BACKBONE_LR_FACTOR:float = 0.1
_DEFAULT_FREEZE_EPOCHS:     int   = 5
_DEFAULT_WEIGHT_DECAY:      float = 1e-4
_DEFAULT_WORKERS:           int   = 4
_DEFAULT_VAL_FRAC:          float = 0.15
_DEFAULT_TEST_FRAC:         float = 0.15
_DEFAULT_MIN_PROB:          float = 0.80
_DEFAULT_MIN_CROPS:         int   = 50
_IMG_SIZE:                  int   = 224


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_ddp() -> bool:
    """Return True if we were launched under torchrun (DDP mode)."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def _setup_ddp() -> tuple[int, int, torch.device]:
    """
    Initialise the DDP process group and return (rank, world_size, device).

    :return: ``(rank, world_size, device)``
    """
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return rank, world_size, device


def _teardown_ddp() -> None:
    """Clean up the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_rank0(rank: int) -> bool:
    return rank == 0


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
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
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

    :param df:        Full crop DataFrame with ``file_id`` and ``species``.
    :param val_frac:  Fraction of file_ids for validation.
    :param test_frac: Fraction of file_ids for test.
    :param seed:      Random seed for reproducibility.
    :return:          ``(train_df, val_df, test_df)``
    """
    rng = np.random.default_rng(seed)

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
            train_fids.extend(fids.tolist())
            continue
        test_fids .extend(fids[:n_test].tolist())
        val_fids  .extend(fids[n_test:n_test + n_val].tolist())
        train_fids.extend(fids[n_test + n_val:].tolist())

    train_df = df[df['file_id'].isin(set(train_fids))].copy()
    val_df   = df[df['file_id'].isin(set(val_fids))].copy()
    test_df  = df[df['file_id'].isin(set(test_fids))].copy()

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

    :param n_classes: Number of bat species classes.
    :param device:    Target device.
    :return:          Model moved to *device*.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, n_classes),
    )
    return model.to(device)


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters except the classifier head.

    :param model: EfficientNet-B0 model (may be DDP-wrapped).
    """
    # Unwrap DDP to access named parameters.
    base = model.module if isinstance(model, DDP) else model
    for name, param in base.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreeze all parameters.

    :param model: EfficientNet-B0 model (may be DDP-wrapped).
    """
    base = model.module if isinstance(model, DDP) else model
    for param in base.parameters():
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
    rank:      int = 0,
    world_size:int = 1,
) -> tuple[float, float]:
    """
    Run one epoch of training or evaluation.

    In DDP mode, loss and accuracy are all-reduced across ranks so the
    returned values are the global average (rank 0 receives the result).

    :param model:      The model (may be DDP-wrapped).
    :param loader:     DataLoader for this split.
    :param criterion:  Loss function.
    :param optimizer:  Optimiser (``None`` during eval).
    :param device:     Compute device.
    :param train:      If ``True``, update weights; else eval mode.
    :param rank:       This process's rank.
    :param world_size: Total number of processes.
    :return:           ``(mean_loss, accuracy)`` — global average in DDP mode.
    """
    model.train(train)
    total_loss = 0.0
    n_correct  = 0
    n_total    = 0

    ctx  = torch.enable_grad() if train else torch.no_grad()
    # Show progress bar only on rank 0 to avoid interleaved output.
    pbar = tqdm(loader, leave=False) if (_TQDM and rank == 0) else loader

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

    # All-reduce across DDP ranks so rank 0 gets global metrics.
    if world_size > 1:
        t = torch.tensor([total_loss, float(n_correct), float(n_total)],
                         device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, n_correct, n_total = t[0].item(), t[1].item(), t[2].item()

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
    classification report TXT.  Called on rank 0 only.

    :param model:        Trained model (may be DDP-wrapped; unwrapped here).
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

    # Unwrap DDP for inference.
    base_model = model.module if isinstance(model, DDP) else model

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
    base_model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs  = imgs.to(device, non_blocking=True)
            preds = base_model(imgs).argmax(dim=1).cpu().numpy()
            all_preds .extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(len(class_names))))
    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names)), max(6, len(class_names) - 2))
    )
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

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=3, zero_division=0,
    )
    (out_dir / 'classification_report.txt').write_text(report)
    log.info(f'Test classification report:\n{report}')


# ---------------------------------------------------------------------------
# Main training class
# ---------------------------------------------------------------------------

class CnnTrainer:
    """
    Fine-tune EfficientNet-B0 on bat species spectrogram crops.

    Supports single-GPU and multi-GPU (DDP) training.  In DDP mode, launch
    with ``torchrun --nproc_per_node=N``; batch size is **per GPU**.

    :param manifest_csv:        Path to manifest CSV from chirps_to_spectros.
    :param out_dir:             Output directory for checkpoints and logs.
    :param species:             If non-empty, restrict to these species codes.
    :param min_prob:            Minimum ``species_prob`` to include a crop.
    :param min_crops_per_class: Drop species with fewer crops than this.
    :param epochs:              Total training epochs.
    :param freeze_epochs:       Epochs to train classifier head only.
    :param batch_size:          Training batch size **per GPU**.
    :param lr:                  Initial learning rate (head phase).
    :param backbone_lr_factor:  Backbone LR = lr × factor (full fine-tune).
    :param weight_decay:        AdamW weight decay.
    :param val_frac:            Fraction of file_ids for validation.
    :param test_frac:           Fraction of file_ids for test.
    :param n_workers:           DataLoader worker processes per GPU.
    :param device_str:          ``'cuda'``, ``'cpu'``, or ``'auto'``.
                                Ignored in DDP mode (device set by local rank).
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
        self.device_str          = device_str

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> pd.DataFrame:
        """
        Load and filter the manifest CSV.

        :return: Filtered DataFrame ready for splitting.
        """
        import re
        _sp_re = re.compile(r'^[A-Z][a-z]{3}$')

        log.info(f'Loading manifest: {self.manifest_csv}')
        df = pd.read_csv(self.manifest_csv)
        log.info(f'  {len(df):,} total rows')

        df = df[df['species'].notna()]
        df = df[df['species'].apply(lambda s: bool(_sp_re.match(str(s))))]
        log.info(f'  {len(df):,} rows with valid species code')

        prob_col = pd.to_numeric(df['species_prob'], errors='coerce')
        df = df[prob_col.isna() | (prob_col >= self.min_prob)]
        log.info(f'  {len(df):,} rows after confidence filter (min_prob={self.min_prob})')

        if self.species:
            df = df[df['species'].isin(self.species)]
            log.info(f'  {len(df):,} rows after species filter {self.species}')

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
        Execute the full training pipeline.

        Automatically detects DDP mode (torchrun) vs single-GPU/CPU mode.
        In DDP mode heavy setup (data loading, model build) runs on all
        ranks, but logging, checkpointing, and evaluation only on rank 0.
        """
        _t0 = time.perf_counter()

        # ── DDP / device setup ─────────────────────────────────────────
        if _is_ddp():
            rank, world_size, device = _setup_ddp()
        else:
            rank, world_size = 0, 1
            if self.device_str == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(self.device_str)

        is_main = _is_rank0(rank)

        if is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(self.seed + rank)   # different seed per rank for augmentation
        np.random.seed(self.seed + rank)

        # ── Load and split (all ranks, same data) ─────────────────────
        # All ranks load the full manifest so splits are identical.
        df = self._load_manifest()

        species_list  = sorted(df['species'].unique().tolist())
        label_to_idx  = {sp: i for i, sp in enumerate(species_list)}
        idx_to_label  = {i: sp for sp, i in label_to_idx.items()}
        n_classes     = len(species_list)
        df['label']   = df['species'].map(label_to_idx)

        if is_main:
            log.info(f'{n_classes} classes: {species_list}')
            (self.out_dir / 'label_encoder.json').write_text(
                json.dumps({'label_to_idx': label_to_idx,
                            'idx_to_label': {str(k): v
                                             for k, v in idx_to_label.items()}},
                           indent=2)
            )

        train_df, val_df, test_df = make_splits(
            df, self.val_frac, self.test_frac, self.seed
        )

        # ── Class weights (rank 0 computes, broadcasts to all) ─────────
        train_counts  = train_df['label'].value_counts().sort_index()
        class_weights = torch.tensor(
            [1.0 / max(train_counts.get(i, 1), 1) for i in range(n_classes)],
            dtype=torch.float32,
        ).to(device)
        class_weights = class_weights / class_weights.sum() * n_classes
        if world_size > 1:
            dist.broadcast(class_weights, src=0)

        # ── DataLoaders ────────────────────────────────────────────────
        train_sampler = DistributedSampler(
            ChirpCropDataset(train_df, augment=True),
            num_replicas=world_size, rank=rank, shuffle=True,
            seed=self.seed,
        ) if world_size > 1 else None

        train_loader = DataLoader(
            ChirpCropDataset(train_df, augment=True),
            batch_size  = self.batch_size,
            shuffle     = (train_sampler is None),
            sampler     = train_sampler,
            num_workers = self.n_workers,
            pin_memory  = True,
            drop_last   = True,
        )
        # Val/test only on rank 0 — no sampler needed.
        val_loader = DataLoader(
            ChirpCropDataset(val_df, augment=False),
            batch_size  = self.batch_size * 2,
            shuffle     = False,
            num_workers = self.n_workers,
            pin_memory  = True,
        )

        # ── Model ──────────────────────────────────────────────────────
        if is_main:
            log.info(
                f'Building EfficientNet-B0 ({n_classes} classes) on {device}'
                + (f' × {world_size} GPUs (DDP)' if world_size > 1 else '')
            )
        model     = build_model(n_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        if world_size > 1:
            # find_unused_parameters=True is required during Phase 1
            # when the backbone is frozen and its parameters receive no
            # gradients.  The overhead in Phase 2 is negligible.
            model = DDP(model, device_ids=[device.index],
                        find_unused_parameters=True)

        # ── Phase 1: head only ─────────────────────────────────────────
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.freeze_epochs
        )

        if is_main:
            log.info(f'Phase 1: training head only for {self.freeze_epochs} epochs')

        best_val_acc = 0.0
        best_epoch   = 0
        log_rows: list[dict] = []

        for epoch in range(1, self.epochs + 1):

            # Advance DistributedSampler epoch so each GPU gets a fresh shard.
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Switch to full fine-tune after freeze_epochs.
            if epoch == self.freeze_epochs + 1:
                if is_main:
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
                model, train_loader, criterion, optimizer,
                device, train=True, rank=rank, world_size=world_size,
            )

            # Validation on rank 0 only (no DistributedSampler on val_loader).
            if is_main:
                val_loss, val_acc = run_epoch(
                    model, val_loader, criterion, None,
                    device, train=False, rank=0, world_size=1,
                )
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
                    # Save unwrapped state dict.
                    base = model.module if isinstance(model, DDP) else model
                    torch.save(base.state_dict(),
                               self.out_dir / 'best_model.pt')
                    log.info(
                        f'  ✓ New best val_acc={best_val_acc:.4f}'
                        ' — saved best_model.pt'
                    )

            # Barrier: all ranks wait before next epoch.
            if world_size > 1:
                dist.barrier()

            scheduler.step()

        # ── Save final model (rank 0) ───────────────────────────────────
        if is_main:
            base = model.module if isinstance(model, DDP) else model
            torch.save(base.state_dict(), self.out_dir / 'final_model.pt')
            log.info(f'Saved final_model.pt  (best was epoch {best_epoch})')

            pd.DataFrame(log_rows).to_csv(
                self.out_dir / 'train_log.csv', index=False
            )

            # ── Test evaluation ────────────────────────────────────────
            log.info('Loading best_model.pt for test evaluation ...')
            base = model.module if isinstance(model, DDP) else model
            base.load_state_dict(
                torch.load(self.out_dir / 'best_model.pt',
                           map_location=device)
            )
            evaluate_test(
                model, test_df, label_to_idx,
                device, self.batch_size * 2, self.n_workers, self.out_dir,
            )

            # ── Config CSV ─────────────────────────────────────────────
            elapsed = time.perf_counter() - _t0
            pd.DataFrame([
                {'parameter': 'manifest_csv',        'value': str(self.manifest_csv)},
                {'parameter': 'out_dir',             'value': str(self.out_dir)},
                {'parameter': 'n_classes',           'value': n_classes},
                {'parameter': 'species',             'value': str(species_list)},
                {'parameter': 'epochs',              'value': self.epochs},
                {'parameter': 'freeze_epochs',       'value': self.freeze_epochs},
                {'parameter': 'batch_size_per_gpu',  'value': self.batch_size},
                {'parameter': 'effective_batch_size','value': self.batch_size * world_size},
                {'parameter': 'world_size',          'value': world_size},
                {'parameter': 'lr',                  'value': self.lr},
                {'parameter': 'backbone_lr_factor',  'value': self.backbone_lr_factor},
                {'parameter': 'weight_decay',        'value': self.weight_decay},
                {'parameter': 'val_frac',            'value': self.val_frac},
                {'parameter': 'test_frac',           'value': self.test_frac},
                {'parameter': 'min_prob',            'value': self.min_prob},
                {'parameter': 'min_crops_per_class', 'value': self.min_crops_per_class},
                {'parameter': 'device',              'value': str(device)},
                {'parameter': 'seed',                'value': self.seed},
                {'parameter': 'best_epoch',          'value': best_epoch},
                {'parameter': 'best_val_acc',        'value': round(best_val_acc, 6)},
                {'parameter': 'elapsed_secs',        'value': round(elapsed, 1)},
            ]).to_csv(self.out_dir / 'train_config.csv', index=False)

            log.info(
                f'Training complete in {elapsed/60:.1f} min  '
                f'best val_acc={best_val_acc:.4f} at epoch {best_epoch}'
            )

        _teardown_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog='train_cnn',
        description=(
            'Fine-tune EfficientNet-B0 on bat species spectrogram crops.\n\n'
            'Single GPU:\n'
            '  python train_cnn.py --manifest ... --out-dir ...\n\n'
            'Multi-GPU (DDP):\n'
            '  torchrun --nproc_per_node=2 train_cnn.py --manifest ... --out-dir ...\n\n'
            'Batch size is per GPU in both modes.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--manifest', required=True, metavar='CSV')
    parser.add_argument('--out-dir',  required=True, metavar='DIR')
    parser.add_argument('--species',  nargs='+', default=[], metavar='SP')
    parser.add_argument('--epochs',   type=int,   default=_DEFAULT_EPOCHS)
    parser.add_argument('--freeze-epochs', type=int, default=_DEFAULT_FREEZE_EPOCHS)
    parser.add_argument('--batch',    type=int,   default=_DEFAULT_BATCH)
    parser.add_argument('--lr',       type=float, default=_DEFAULT_LR)
    parser.add_argument('--backbone-lr-factor', type=float,
                        default=_DEFAULT_BACKBONE_LR_FACTOR)
    parser.add_argument('--weight-decay', type=float, default=_DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--val-frac',  type=float, default=_DEFAULT_VAL_FRAC)
    parser.add_argument('--test-frac', type=float, default=_DEFAULT_TEST_FRAC)
    parser.add_argument('--min-prob',  type=float, default=_DEFAULT_MIN_PROB)
    parser.add_argument('--min-crops', type=int,   default=_DEFAULT_MIN_CROPS)
    parser.add_argument('--workers',   type=int,   default=_DEFAULT_WORKERS)
    parser.add_argument('--device',    default='auto')
    parser.add_argument('--seed',      type=int,   default=42)

    args = parser.parse_args()
    if not Path(args.manifest).exists():
        parser.error(f'manifest not found: {args.manifest}')
    return args


def main() -> None:
    """CLI entry point."""
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
