#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-18 08:57:33
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-18 08:57:53
# **********************************************************

"""
validate_cnn.py
===============
Evaluate a trained bat-species CNN checkpoint on the held-out test set
and produce per-species metrics, a confusion matrix, and a summary CSV.

Designed as a companion to ``train_cnn.py`` but entirely standalone —
it does not import from that module.  Uses the same preprocessing
pipeline (grayscale → 3-channel → ImageNet normalisation) and the same
``holdout_split.csv`` / manifest loading logic.

Outputs written to ``--out-dir``:
    classification_report.txt   sklearn per-species precision/recall/F1
    confusion_matrix.png        normalised heatmap
    predictions.csv             per-crop true label, predicted label,
                                top-1 probability, correct (bool)
    val_summary.txt             overall accuracy + macro/weighted F1

Usage
-----
::

    python src/species_classification/validation/validate_cnn.py \\
        --checkpoint /qnap/bats/jr_pipeline/models/efficientnet_b3_v1/best_model.pt \\
        --encoder    /qnap/bats/jr_pipeline/models/efficientnet_b3_v1/label_encoder.json \\
        --manifest   /qnap/bats/jr_pipeline/data/bat_crops_v2/manifest.csv \\
        --split-file /qnap/bats/jr_pipeline/data/holdout_split.csv \\
        --out-dir    /qnap/bats/jr_pipeline/models/efficientnet_b3_v1/validation \\
        --batch      256 \\
        --workers    8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants — must match train_cnn.py / CropPreprocessor
# ---------------------------------------------------------------------------

_IMG_SIZE = 224
_MEAN     = [0.485, 0.456, 0.406]
_STD      = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Load a trained EfficientNet-B3 checkpoint and label encoder.

    Handles DDP ``module.`` prefix stripping and supports both B0
    and B3 checkpoint sizes (auto-detects from in_features).

    :param checkpoint_path: Path to ``best_model.pt`` or any checkpoint.
    :param encoder_path:    Path to ``label_encoder.json``.
    :param device:          Torch device.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        encoder_path:    Path,
        device:          torch.device,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.encoder_path    = encoder_path
        self.device          = device

    def load(self) -> Tuple[nn.Module, Dict[str, int], Dict[int, str]]:
        """
        Load and return the model and label mappings.

        :return: ``(model, label_to_idx, idx_to_label)``
        """
        with open(self.encoder_path) as fh:
            enc = json.load(fh)
        label_to_idx: Dict[str, int] = enc['label_to_idx']
        idx_to_label: Dict[int, str] = {
            int(k): v for k, v in enc['idx_to_label'].items()
        }
        n_classes = len(label_to_idx)
        log.info(f'Label encoder: {n_classes} classes: {sorted(label_to_idx)}')

        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, n_classes),
        )

        state = torch.load(self.checkpoint_path, map_location=self.device)
        # Strip DDP 'module.' prefix if present.
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        log.info(f'Loaded checkpoint: {self.checkpoint_path}')
        return model, label_to_idx, idx_to_label


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """
    Minimal inference dataset for bat spectrogram crops.

    Applies the same transform pipeline as ``ChirpCropDataset`` in
    ``train_cnn.py``: grayscale → replicate to 3 channels → resize →
    ImageNet normalise.  No augmentation.

    :param df:       DataFrame with ``crop_path`` and ``species`` columns.
    :param label_to_idx: Species → class index mapping.
    """

    _transform = transforms.Compose([
        transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(_MEAN, _STD),
    ])

    def __init__(
        self,
        df:           pd.DataFrame,
        label_to_idx: Dict[str, int],
    ) -> None:
        self.paths   = df['crop_path'].tolist()
        self.labels  = [label_to_idx[s] for s in df['species']]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path  = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('L')
            tensor = self._transform(img)
        except Exception as exc:
            log.warn(f'Failed to load {path}: {exc} — using zeros')
            tensor = torch.zeros(3, _IMG_SIZE, _IMG_SIZE)
        return tensor, label, path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestSetLoader:
    """
    Load the test partition from manifest + holdout split.

    :param manifest_path:  Path to ``manifest.csv``.
    :param split_path:     Path to ``holdout_split.csv``.
    :param label_to_idx:   Species → class index mapping.
    :param exclude_species: Species codes to exclude (e.g. Myvo, Mylu).
    :param min_conf:        Minimum SonoBat confidence (default 0.0 — use
                            all crops that passed training filter).
    """

    def __init__(
        self,
        manifest_path:   Path,
        split_path:      Optional[Path],
        label_to_idx:    Dict[str, int],
        exclude_species: List[str] = (),
        min_conf:        float     = 0.0,
    ) -> None:
        self.manifest_path   = manifest_path
        self.split_path      = split_path
        self.label_to_idx    = label_to_idx
        self.exclude_species = set(exclude_species)
        self.min_conf        = min_conf

    def load(self) -> pd.DataFrame:
        """
        Return the test-partition DataFrame ready for inference.

        :return: DataFrame with ``crop_path``, ``species``, ``file_id``,
                 ``chirp_idx`` columns, restricted to known species and
                 the test partition.
        """
        log.info(f'Loading manifest: {self.manifest_path}')
        df = pd.read_csv(self.manifest_path, low_memory=False)
        log.info(f'  {len(df):,} total rows')

        # Keep only known species.
        df = df[df['species'].isin(self.label_to_idx)].copy()
        log.info(f'  {len(df):,} rows with known species')

        # Exclude blacklisted species.
        if self.exclude_species:
            df = df[~df['species'].isin(self.exclude_species)].copy()
            log.info(
                f'  {len(df):,} rows after excluding '
                f'{sorted(self.exclude_species)}'
            )

        # Confidence filter.
        if self.min_conf > 0 and 'confidence' in df.columns:
            df = df[df['confidence'] >= self.min_conf].copy()
            log.info(f'  {len(df):,} rows after confidence >= {self.min_conf}')

        # Split filter.
        if self.split_path is not None and self.split_path.exists():
            split_df = pd.read_csv(self.split_path)
            test_ids = set(
                split_df.loc[split_df['partition'] == 'test', 'file_id']
                .astype(int)
            )
            df = df[df['file_id'].astype(int).isin(test_ids)].copy()
            log.info(f'  {len(df):,} rows in test partition')
        else:
            log.warn(
                'No split file supplied or found — '
                'evaluating on ALL crops (not recommended).'
            )

        if df.empty:
            log.err('Test set is empty after filtering. Check paths and split file.')
            sys.exit(1)

        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CnnEvaluator:
    """
    Run inference on the test set and compute per-species metrics.

    :param model:        Eval-mode EfficientNet-B3.
    :param idx_to_label: Class index → species mapping.
    :param device:       Torch device.
    :param batch_size:   Inference batch size.
    :param n_workers:    DataLoader worker count.
    :param out_dir:      Directory to write output files.
    """

    def __init__(
        self,
        model:        nn.Module,
        idx_to_label: Dict[int, str],
        device:       torch.device,
        batch_size:   int  = 256,
        n_workers:    int  = 8,
        out_dir:      Path = Path('.'),
    ) -> None:
        self.model        = model
        self.idx_to_label = idx_to_label
        self.device       = device
        self.batch_size   = batch_size
        self.n_workers    = n_workers
        self.out_dir      = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #

    def run(self, df: pd.DataFrame, label_to_idx: Dict[str, int]) -> None:
        """
        Run inference on *df* and write all output files.

        :param df:           Test-set DataFrame from :class:`TestSetLoader`.
        :param label_to_idx: Species → class index mapping.
        """
        dataset = CropDataset(df, label_to_idx)
        loader  = DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.n_workers,
            pin_memory  = True,
        )

        all_true:     List[int]   = []
        all_pred:     List[int]   = []
        all_prob:     List[float] = []
        all_paths:    List[str]   = []

        log.info(f'Running inference on {len(dataset):,} crops ...')
        with torch.no_grad():
            for batch_idx, (tensors, labels, paths) in enumerate(loader):
                tensors = tensors.to(self.device)
                logits  = self.model(tensors)
                probs   = torch.softmax(logits, dim=1)
                top_prob, top_idx = probs.max(dim=1)

                all_true.extend(labels.tolist())
                all_pred.extend(top_idx.cpu().tolist())
                all_prob.extend(top_prob.cpu().tolist())
                all_paths.extend(paths)

                if batch_idx % 100 == 0:
                    done = (batch_idx + 1) * self.batch_size
                    log.info(f'  {min(done, len(dataset)):,}/{len(dataset):,}')

        # Map indices back to species labels.
        true_labels = [self.idx_to_label[i] for i in all_true]
        pred_labels = [self.idx_to_label[i] for i in all_pred]
        species     = sorted(self.idx_to_label.values())

        self._write_predictions(all_paths, true_labels, pred_labels, all_prob)
        self._write_classification_report(true_labels, pred_labels, species)
        self._write_confusion_matrix(true_labels, pred_labels, species)
        self._write_summary(true_labels, pred_labels)

    # ----------------------------------------------------------------------- #

    def _write_predictions(
        self,
        paths:       List[str],
        true_labels: List[str],
        pred_labels: List[str],
        probs:       List[float],
    ) -> None:
        """
        Write per-crop prediction CSV.

        :param paths:       Crop file paths.
        :param true_labels: True species labels.
        :param pred_labels: Predicted species labels.
        :param probs:       Top-1 softmax probabilities.
        """
        pred_df = pd.DataFrame({
            'crop_path' : paths,
            'true'      : true_labels,
            'predicted' : pred_labels,
            'top1_prob' : [round(p, 4) for p in probs],
            'correct'   : [t == p for t, p in zip(true_labels, pred_labels)],
        })
        out = self.out_dir / 'predictions.csv'
        pred_df.to_csv(out, index=False)
        log.info(f'Predictions written: {out}')

    # ----------------------------------------------------------------------- #

    def _write_classification_report(
        self,
        true_labels: List[str],
        pred_labels: List[str],
        species:     List[str],
    ) -> None:
        """
        Write sklearn classification report (per-species P/R/F1/support).

        :param true_labels: True species labels.
        :param pred_labels: Predicted species labels.
        :param species:     Ordered list of species codes.
        """
        report = classification_report(
            true_labels,
            pred_labels,
            labels       = species,
            target_names = species,
            digits       = 4,
            zero_division= 0,
        )
        out = self.out_dir / 'classification_report.txt'
        out.write_text(report)
        log.info(f'Classification report written: {out}')
        log.info(f'\n{report}')

    # ----------------------------------------------------------------------- #

    def _write_confusion_matrix(
        self,
        true_labels: List[str],
        pred_labels: List[str],
        species:     List[str],
    ) -> None:
        """
        Write normalised confusion matrix PNG.

        :param true_labels: True species labels.
        :param pred_labels: Predicted species labels.
        :param species:     Ordered list of species codes for axis labels.
        """
        cm = confusion_matrix(true_labels, pred_labels, labels=species)
        # Row-normalise (recall per true class).
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.where(
                cm.sum(axis=1, keepdims=True) > 0,
                cm / cm.sum(axis=1, keepdims=True),
                0.0,
            )

        n = len(species)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(species, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(species, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title('Test set confusion matrix — EfficientNet-B3\n(row-normalised)',
                     fontsize=12)

        # Annotate cells with normalised value.
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                if val > 0:
                    ax.text(
                        j, i, f'{val:.2f}',
                        ha='center', va='center',
                        fontsize=7,
                        color='white' if val > 0.6 else 'black',
                    )

        plt.tight_layout()
        out = self.out_dir / 'confusion_matrix.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info(f'Confusion matrix written: {out}')

    # ----------------------------------------------------------------------- #

    def _write_summary(
        self,
        true_labels: List[str],
        pred_labels: List[str],
    ) -> None:
        """
        Write overall accuracy and macro/weighted F1 to summary text file.

        :param true_labels: True species labels.
        :param pred_labels: Predicted species labels.
        """
        acc        = accuracy_score(true_labels, pred_labels)
        macro_f1   = f1_score(true_labels, pred_labels,
                              average='macro',    zero_division=0)
        weighted_f1= f1_score(true_labels, pred_labels,
                              average='weighted', zero_division=0)

        lines = [
            'CNN Validation Summary',
            '=' * 40,
            f'Overall accuracy : {acc:.4f}  ({acc*100:.2f}%)',
            f'Macro F1         : {macro_f1:.4f}',
            f'Weighted F1      : {weighted_f1:.4f}',
            f'Test crops       : {len(true_labels):,}',
        ]
        summary = '\n'.join(lines)
        out = self.out_dir / 'val_summary.txt'
        out.write_text(summary + '\n')
        log.info(f'Summary written: {out}')
        log.info(f'\n{summary}')


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ValidationRunner:
    """
    Top-level orchestrator: loads model, data, runs evaluation.

    :param checkpoint_path:  Path to ``best_model.pt``.
    :param encoder_path:     Path to ``label_encoder.json``.
    :param manifest_path:    Path to ``manifest.csv``.
    :param split_path:       Path to ``holdout_split.csv``.
    :param out_dir:          Directory for output files.
    :param exclude_species:  Species to exclude from evaluation.
    :param batch_size:       Inference batch size.
    :param n_workers:        DataLoader workers.
    :param device_str:       ``'auto'``, ``'cuda'``, ``'cpu'``, etc.
    """

    def __init__(
        self,
        checkpoint_path:  Path,
        encoder_path:     Path,
        manifest_path:    Path,
        split_path:       Optional[Path],
        out_dir:          Path,
        exclude_species:  List[str]  = (),
        batch_size:       int        = 256,
        n_workers:        int        = 8,
        device_str:       str        = 'auto',
    ) -> None:
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)
        log.info(f'Device: {device}')

        loader = ModelLoader(checkpoint_path, encoder_path, device)
        model, label_to_idx, idx_to_label = loader.load()

        test_loader = TestSetLoader(
            manifest_path   = manifest_path,
            split_path      = split_path,
            label_to_idx    = label_to_idx,
            exclude_species = list(exclude_species),
        )
        test_df = test_loader.load()

        self.evaluator    = CnnEvaluator(
            model        = model,
            idx_to_label = idx_to_label,
            device       = device,
            batch_size   = batch_size,
            n_workers    = n_workers,
            out_dir      = out_dir,
        )
        self.test_df      = test_df
        self.label_to_idx = label_to_idx

    def run(self) -> None:
        """Execute inference and write all output files."""
        self.evaluator.run(self.test_df, self.label_to_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='validate_cnn',
        description=(
            'Evaluate a trained bat-species CNN on the held-out test set.\n'
            'Produces per-species classification report, confusion matrix,\n'
            'per-crop predictions CSV, and overall summary.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', required=True, metavar='PT',
        help='Path to model checkpoint (best_model.pt or checkpoint_epoch_NNNN.pt).',
    )
    parser.add_argument(
        '--encoder', default=None, metavar='JSON',
        help=(
            'Path to label_encoder.json.  '
            'Defaults to label_encoder.json in the same directory as --checkpoint.'
        ),
    )
    parser.add_argument(
        '--manifest', required=True, metavar='CSV',
        help='Path to manifest.csv from chirps_to_spectros.py.',
    )
    parser.add_argument(
        '--split-file', default=None, metavar='CSV',
        help=(
            'Path to holdout_split.csv.  When supplied, only test-partition\n'
            'crops are evaluated.  Omit to evaluate all crops (not recommended).'
        ),
    )
    parser.add_argument(
        '--out-dir', required=True, metavar='DIR',
        help='Directory for output files (created if absent).',
    )
    parser.add_argument(
        '--exclude-species', nargs='+', default=[], metavar='SP',
        help='Species codes to exclude (e.g. --exclude-species Myvo Mylu Lafr).',
    )
    parser.add_argument(
        '--batch', type=int, default=256, metavar='N',
        help='Inference batch size (default: 256).',
    )
    parser.add_argument(
        '--workers', type=int, default=8, metavar='N',
        help='DataLoader worker count (default: 8).',
    )
    parser.add_argument(
        '--device', default='auto',
        help='cuda | cpu | cuda:0 | auto (default: auto).',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        log.err(f'Checkpoint not found: {ckpt}')
        sys.exit(1)

    encoder = Path(args.encoder) if args.encoder else ckpt.parent / 'label_encoder.json'
    if not encoder.exists():
        log.err(f'Label encoder not found: {encoder}')
        sys.exit(1)

    manifest = Path(args.manifest)
    if not manifest.exists():
        log.err(f'Manifest not found: {manifest}')
        sys.exit(1)

    split = Path(args.split_file) if args.split_file else None

    runner = ValidationRunner(
        checkpoint_path = ckpt,
        encoder_path    = encoder,
        manifest_path   = manifest,
        split_path      = split,
        out_dir         = Path(args.out_dir),
        exclude_species = args.exclude_species,
        batch_size      = args.batch,
        n_workers       = args.workers,
        device_str      = args.device,
    )
    runner.run()


if __name__ == '__main__':
    main()
