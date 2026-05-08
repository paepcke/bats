#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-08 16:11:10
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-08 16:43:23
# **********************************************************

"""
gradcam_bats.py
==============
Produce Grad-CAM activation-map visualisations for the Coto, Lano, and Tabr
species classes using the trained EfficientNet-B0 bat classifier.

For each of the three species the script samples:
  * N correctly classified crops   (true label == predicted label == species)
  * N misclassified crops          (true label == species, predicted != species)

It then renders a figure per sample that shows the original spectrogram and
its Grad-CAM overlay side-by-side, saved as a PNG under ``--out-dir``.

The hook is placed on ``model.features[-1]``, the last Conv2dNormActivation
block in EfficientNet-B0, which is the canonical Grad-CAM target for this
architecture.

Usage
-----
Single GPU / CPU::

    python gradcam_bat.py \\
        --model   /qnap/bats/jr_pipeline/models/efficientnet_b0_gcp_v1/best_model.pt \\
        --encoder /qnap/bats/jr_pipeline/models/efficientnet_b0_gcp_v1/label_encoder.json \\
        --manifest /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --crops-dir /qnap/bats/jr_pipeline/data/bat_crops/ \\
        --out-dir  /qnap/bats/jr_pipeline/gradcam_results \\
        --n-samples 8 \\
        --partition test

Outputs
-------
``<out-dir>/<species>_correct_<i>.png``    — correctly classified examples
``<out-dir>/<species>_misclassified_<i>.png`` — misclassified examples
``<out-dir>/summary.txt``                  — per-species hit/miss counts
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants matching train_cnn.py
# ---------------------------------------------------------------------------

_IMG_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TARGET_SPECIES = ('Coto', 'Lano', 'Tabr')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Load a trained EfficientNet-B0 checkpoint produced by ``train_cnn.py``.

    Handles the DDP ``module.`` prefix that may be present in older checkpoints
    saved before rank-0 state-dict stripping was added.

    :param model_path:   Path to ``best_model.pt``.
    :param encoder_path: Path to ``label_encoder.json`` in the same model dir.
    :param device:       Torch device string or object.
    """

    def __init__(
        self,
        model_path:   Path,
        encoder_path: Path,
        device:       torch.device,
    ) -> None:
        self.model_path   = model_path
        self.encoder_path = encoder_path
        self.device       = device

    def load(self) -> Tuple[nn.Module, Dict[str, int], Dict[int, str]]:
        """
        Load model and label encoder.

        :return: ``(model, label_to_idx, idx_to_label)``
        """
        with open(self.encoder_path) as fh:
            enc = json.load(fh)
        label_to_idx: Dict[str, int] = enc['label_to_idx']
        idx_to_label: Dict[int, str] = {int(k): v for k, v in enc['idx_to_label'].items()}

        n_classes = len(label_to_idx)
        log.info(f'Label encoder: {n_classes} classes: {sorted(label_to_idx.keys())}')

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, n_classes),
        )

        state = torch.load(self.model_path, map_location=self.device)
        # Strip DDP 'module.' prefix if present.
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        log.info(f'Loaded model from {self.model_path}')
        return model, label_to_idx, idx_to_label


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

class CropPreprocessor:
    """
    Reproduce the inference-time transform used in ``train_cnn.py``.

    Grayscale PNG → 3-channel tensor normalised with ImageNet statistics.
    """

    def __init__(self, img_size: int = _IMG_SIZE) -> None:
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __call__(self, path: str) -> torch.Tensor:
        """
        Load and preprocess a single crop PNG.

        :param path: Absolute path to the PNG file.
        :return:     Normalised tensor of shape ``(3, H, W)``.
        """
        img = Image.open(path).convert('L')
        return self.transform(img)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCam:
    """
    Grad-CAM for EfficientNet-B0 hooked on ``model.features[-1]``.

    Forward and backward hooks capture the feature maps and gradients from
    the last convolutional block.  The CAM is upsampled to the input image
    size and ReLU-clamped.

    :param model:  Eval-mode EfficientNet-B0 (not DDP-wrapped).
    :param device: Torch device.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device

        self._fmaps: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        # EfficientNet-B0: hook the last Conv block.
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    # -- hooks -------------------------------------------------------------- #

    def _fwd_hook(self, module, inp, out) -> None:
        self._fmaps = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out) -> None:
        self._grads = grad_out[0].detach()

    # -- public API --------------------------------------------------------- #

    def compute(
        self,
        tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for *class_idx*.

        :param tensor:    Pre-processed image tensor of shape ``(3, H, W)``
                          (no batch dim).
        :param class_idx: Index of the class to explain.
        :return:          Float32 array of shape ``(H, W)`` in ``[0, 1]``.
        """
        self.model.zero_grad()
        inp = tensor.unsqueeze(0).to(self.device).requires_grad_(True)

        logits = self.model(inp)                      # (1, n_classes)
        score  = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients → channel weights.
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._fmaps).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(_IMG_SIZE, _IMG_SIZE),
            mode='bilinear',
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        vmax = cam.max()
        if vmax > 0:
            cam /= vmax
        return cam.astype(np.float32)


# ---------------------------------------------------------------------------
# Manifest sampling
# ---------------------------------------------------------------------------

class SampleSelector:
    """
    Sample correct and misclassified crops from the manifest for a set of
    species, running batched inference to obtain predictions.

    :param manifest_path:    Path to ``manifest.csv``.
    :param crops_dir:        Root directory under which crop PNGs live.
    :param model:            Eval-mode model.
    :param preprocessor:     ``CropPreprocessor`` instance.
    :param label_to_idx:     Species → class index mapping.
    :param idx_to_label:     Class index → species mapping.
    :param device:           Torch device.
    :param split_file:       Path to ``holdout_split.csv`` (columns: ``file_id``,
                             ``partition``).  When supplied, only rows whose
                             ``file_id`` maps to ``split_partition`` are kept.
                             This is the same split used during training so
                             Grad-CAM samples are drawn from the true held-out
                             test set.  Pass ``None`` to use all rows.
    :param split_partition:  Which partition to keep when *split_file* is given
                             (``'test'``, ``'val'``, or ``'train'``).
                             Default: ``'test'``.
    :param primary_harmonic: If ``True`` keep only ``harmonic_idx == 0`` rows
                             (fundamental chirps, skipping harmonic copies).
    :param batch_size:       Inference batch size.
    """

    def __init__(
        self,
        manifest_path:   Path,
        crops_dir:       Path,
        model:           nn.Module,
        preprocessor:    CropPreprocessor,
        label_to_idx:    Dict[str, int],
        idx_to_label:    Dict[int, str],
        device:          torch.device,
        split_file:      Optional[Path] = None,
        split_partition: str            = 'test',
        primary_harmonic: bool          = True,
        batch_size:      int            = 64,
    ) -> None:
        self.model        = model
        self.preprocessor = preprocessor
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.device       = device
        self.batch_size   = batch_size

        log.info(f'Reading manifest: {manifest_path}')
        df = pd.read_csv(manifest_path, low_memory=False)

        # Keep only species the model knows.
        df = df[df['species'].isin(label_to_idx)].copy()

        # Split-file filter: join on file_id to restrict to train/val/test.
        # holdout_split.csv has columns file_id, partition with values
        # 'train' / 'val' / 'test' — this is the authoritative split used
        # during training (make_splits() in train_cnn.py).
        if split_file is not None:
            log.info(f'Loading split file: {split_file}')
            split_df  = pd.read_csv(split_file)
            split_map = dict(zip(split_df['file_id'].astype(int),
                                 split_df['partition']))
            before = len(df)
            df['_split'] = df['file_id'].astype(int).map(split_map)
            df = df[df['_split'] == split_partition].drop(columns=['_split']).copy()
            log.info(
                f'Split filter "{split_partition}": {before:,} → {len(df):,} rows'
            )
            if len(df) == 0:
                log.warn(
                    f'Split filter produced 0 rows.  Check that holdout_split.csv '
                    f'file_ids overlap with the manifest.'
                )
        else:
            log.info('No split file supplied — using all manifest rows.')

        # Primary-harmonic filter: keep only fundamental chirps (harmonic_idx == 0).
        # Harmonic copies share acoustic content with their fundamental and would
        # produce near-identical Grad-CAM maps, so excluding them avoids redundancy.
        if primary_harmonic and 'harmonic_idx' in df.columns:
            before = len(df)
            df = df[df['harmonic_idx'] == 0].copy()
            log.info(f'Primary-harmonic filter: {before:,} → {len(df):,} rows')

        # Keep only target species.
        df = df[df['species'].isin(_TARGET_SPECIES)].copy()
        log.info(f'Target-species rows: {len(df):,}')

        # Normalise crop_path to absolute if needed.
        if crops_dir is not None:
            df['crop_path'] = df['crop_path'].apply(
                lambda p: str(p) if Path(p).is_absolute() else str(crops_dir / p)
            )

        self.df = df

    # ----------------------------------------------------------------------- #

    def run_inference(self) -> pd.DataFrame:
        """
        Run batched forward passes over all target-species crops.

        :return: DataFrame with added ``predicted`` (species str) column.
        """
        paths  = self.df['crop_path'].tolist()
        preds:  List[int] = []

        log.info(f'Running inference on {len(paths):,} crops …')
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start:start + self.batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    tensors.append(self.preprocessor(p))
                except Exception as exc:
                    log.warn(f'Could not load {p}: {exc}; using zeros.')
                    tensors.append(torch.zeros(3, _IMG_SIZE, _IMG_SIZE))

            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                pred_idxs = logits.argmax(dim=1).cpu().tolist()
            preds.extend(pred_idxs)

            if (start // self.batch_size) % 20 == 0:
                log.info(f'  … {start + len(batch_paths):,}/{len(paths):,}')

        df = self.df.copy()
        df['pred_idx']  = preds
        df['predicted'] = [self.idx_to_label[i] for i in preds]
        return df

    # ----------------------------------------------------------------------- #

    def select_samples(
        self,
        df_pred:   pd.DataFrame,
        species:   str,
        n_samples: int,
        seed:      int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (correct_df, misclassified_df) for *species*.

        :param df_pred:   DataFrame with ``predicted`` column from
                          :meth:`run_inference`.
        :param species:   Species code (``'Coto'``, ``'Lano'``, or ``'Tabr'``).
        :param n_samples: Max samples per category.
        :param seed:      Random seed for sampling.
        :return:          ``(correct_df, misclassified_df)``
        """
        sp_df   = df_pred[df_pred['species'] == species]
        correct = sp_df[sp_df['predicted'] == species]
        wrong   = sp_df[sp_df['predicted'] != species]

        rng = np.random.default_rng(seed)

        def _sample(sub: pd.DataFrame) -> pd.DataFrame:
            if len(sub) == 0:
                return sub
            idx = rng.choice(len(sub), size=min(n_samples, len(sub)), replace=False)
            return sub.iloc[sorted(idx)]

        return _sample(correct), _sample(wrong)


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class GradCamVisualiser:
    """
    Render and save Grad-CAM figures.

    Each figure shows the original (de-normalised) spectrogram on the left and
    the Grad-CAM overlay on the right.

    :param gradcam:     :class:`GradCam` instance.
    :param preprocessor: :class:`CropPreprocessor` instance.
    :param idx_to_label: Class index → species mapping.
    :param out_dir:     Directory to write PNG figures.
    """

    def __init__(
        self,
        gradcam:      GradCam,
        preprocessor: CropPreprocessor,
        idx_to_label: Dict[int, str],
        out_dir:      Path,
    ) -> None:
        self.gradcam      = gradcam
        self.preprocessor = preprocessor
        self.idx_to_label = idx_to_label
        self.out_dir      = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #

    @staticmethod
    def _denorm(tensor: torch.Tensor) -> np.ndarray:
        """
        Reverse ImageNet normalisation for display.

        :param tensor: ``(3, H, W)`` normalised tensor.
        :return:       ``(H, W, 3)`` uint8 array.
        """
        mean = torch.tensor(_MEAN).view(3, 1, 1)
        std  = torch.tensor(_STD).view(3, 1, 1)
        img  = (tensor * std + mean).clamp(0, 1)
        # Spectrogram is grayscale replicated — use single channel for display.
        gray = img[0].numpy()
        return (gray * 255).astype(np.uint8)

    # ----------------------------------------------------------------------- #

    def _render_one(
        self,
        path:       str,
        true_sp:    str,
        pred_sp:    str,
        save_path:  Path,
    ) -> None:
        """
        Produce and save a two-panel Grad-CAM figure.

        :param path:      Absolute path to the crop PNG.
        :param true_sp:   True species label.
        :param pred_sp:   Predicted species label.
        :param save_path: Output PNG path.
        """
        tensor = self.preprocessor(path)

        # Grad-CAM w.r.t. the TRUE class (what the model actually activated for
        # the correct answer) and the PREDICTED class (what it fired on instead).
        true_idx = {v: k for k, v in self.idx_to_label.items()}[true_sp]
        pred_idx = {v: k for k, v in self.idx_to_label.items()}[pred_sp]

        cam_true = self.gradcam.compute(tensor, true_idx)
        cam_pred = self.gradcam.compute(tensor, pred_idx) if pred_sp != true_sp else None

        gray = self._denorm(tensor)
        n_panels = 3 if cam_pred is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

        # Panel 0: raw spectrogram.
        axes[0].imshow(gray, cmap='gray', origin='upper')
        axes[0].set_title(f'True: {true_sp}  |  Pred: {pred_sp}', fontsize=10)
        axes[0].axis('off')

        # Panel 1: CAM for the true class.
        axes[1].imshow(gray, cmap='gray', origin='upper')
        axes[1].imshow(cam_true, cmap='jet', alpha=0.45, origin='upper',
                       vmin=0, vmax=1)
        axes[1].set_title(f'Grad-CAM → {true_sp} (true class)', fontsize=10)
        axes[1].axis('off')

        # Panel 2 (misclassified only): CAM for the predicted class.
        if cam_pred is not None:
            axes[2].imshow(gray, cmap='gray', origin='upper')
            axes[2].imshow(cam_pred, cmap='jet', alpha=0.45, origin='upper',
                           vmin=0, vmax=1)
            axes[2].set_title(f'Grad-CAM → {pred_sp} (predicted class)', fontsize=10)
            axes[2].axis('off')

        crop_name = Path(path).name
        fig.suptitle(crop_name, fontsize=8, y=0.02)
        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ----------------------------------------------------------------------- #

    def render_batch(
        self,
        df:       pd.DataFrame,
        species:  str,
        tag:      str,
    ) -> int:
        """
        Render Grad-CAM figures for a batch of crops.

        :param df:      DataFrame rows with ``crop_path``, ``species``,
                        ``predicted`` columns.
        :param species: Species label (used in filenames).
        :param tag:     ``'correct'`` or ``'misclassified'``.
        :return:        Number of figures saved.
        """
        saved = 0
        for i, row in enumerate(df.itertuples()):
            save_path = self.out_dir / f'{species}_{tag}_{i:03d}.png'
            try:
                self._render_one(
                    path      = row.crop_path,
                    true_sp   = row.species,
                    pred_sp   = row.predicted,
                    save_path = save_path,
                )
                saved += 1
                log.info(f'  saved {save_path.name}')
            except Exception as exc:
                log.warn(f'  failed on {row.crop_path}: {exc}')
        return saved


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class GradCamRunner:
    """
    Top-level orchestrator: loads model, runs inference, and renders figures.

    :param model_path:    Path to ``best_model.pt``.
    :param encoder_path:  Path to ``label_encoder.json``.
    :param manifest_path: Path to ``manifest.csv``.
    :param crops_dir:     Root directory of PNG crops.
    :param out_dir:       Output directory for figures.
    :param n_samples:     Number of correct and misclassified samples per species.
    :param split_file:       Path to ``holdout_split.csv`` (file_id → partition).
                             When supplied only the *split_partition* subset is used,
                             matching the held-out test set from training.
                             Pass ``None`` to use all manifest rows.
    :param split_partition:  Which partition to keep: ``'test'``, ``'val'``, or
                             ``'train'``.  Ignored when *split_file* is ``None``.
    :param primary_harmonic: If ``True`` keep only ``harmonic_idx == 0`` rows.
    :param batch_size:       Inference batch size.
    :param device_str:       ``'auto'``, ``'cpu'``, ``'cuda'``, ``'cuda:0'``, etc.
    :param seed:             Random seed for sample selection.
    """

    def __init__(
        self,
        model_path:      Path,
        encoder_path:    Path,
        manifest_path:   Path,
        crops_dir:       Path,
        out_dir:         Path,
        n_samples:       int            = 8,
        split_file:      Optional[Path] = None,
        split_partition: str            = 'test',
        primary_harmonic: bool          = True,
        batch_size:      int            = 64,
        device_str:      str            = 'auto',
        seed:            int            = 42,
    ) -> None:
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)
        log.info(f'Device: {device}')

        loader = ModelLoader(model_path, encoder_path, device)
        model, label_to_idx, idx_to_label = loader.load()

        preprocessor = CropPreprocessor()
        gradcam      = GradCam(model, device)
        selector     = SampleSelector(
            manifest_path    = manifest_path,
            crops_dir        = crops_dir,
            model            = model,
            preprocessor     = preprocessor,
            label_to_idx     = label_to_idx,
            idx_to_label     = idx_to_label,
            device           = device,
            split_file       = split_file,
            split_partition  = split_partition,
            primary_harmonic = primary_harmonic,
            batch_size       = batch_size,
        )
        visualiser = GradCamVisualiser(gradcam, preprocessor, idx_to_label, out_dir)

        self.model           = model
        self.label_to_idx    = label_to_idx
        self.idx_to_label    = idx_to_label
        self.selector        = selector
        self.visualiser      = visualiser
        self.n_samples       = n_samples
        self.split_file      = split_file
        self.split_partition = split_partition
        self.primary_harmonic = primary_harmonic
        self.seed            = seed
        self.out_dir         = out_dir

    # ----------------------------------------------------------------------- #

    def run(self) -> None:
        """Execute inference, sample selection, and figure generation."""
        df_pred = self.selector.run_inference()

        summary_lines: List[str] = []

        for sp in _TARGET_SPECIES:
            if sp not in self.label_to_idx:
                log.warn(f'{sp} not in label encoder — skipping.')
                continue

            correct_df, wrong_df = self.selector.select_samples(
                df_pred, sp, self.n_samples, self.seed
            )
            log.info(
                f'{sp}: {len(correct_df)} correct, {len(wrong_df)} misclassified '
                f'(from {len(df_pred[df_pred["species"] == sp]):,} total)'
            )

            n_c = self.visualiser.render_batch(correct_df, sp, 'correct')
            n_m = self.visualiser.render_batch(wrong_df,   sp, 'misclassified')

            summary_lines.append(
                f'{sp}: {n_c} correct figures, {n_m} misclassified figures'
            )

            # Log the confusion breakdown for misclassified crops.
            if len(wrong_df) > 0:
                breakdown = wrong_df['predicted'].value_counts().to_dict()
                log.info(f'  {sp} confused as: {breakdown}')
                summary_lines.append(f'  confused as: {breakdown}')

        summary_path = self.out_dir / 'summary.txt'
        summary_path.write_text('\n'.join(summary_lines) + '\n')
        log.info(f'Summary written to {summary_path}')
        log.info(f'All figures saved under {self.out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='gradcam_bat',
        description=(
            'Grad-CAM visualisation for Coto, Lano, Tabr bat species.\n'
            'Produces side-by-side spectrogram + activation map figures\n'
            'for correct and misclassified crops.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--model', required=True, metavar='PT',
        help='Path to best_model.pt',
    )
    parser.add_argument(
        '--encoder', default=None, metavar='JSON',
        help=(
            'Path to label_encoder.json. '
            'Defaults to label_encoder.json in the same directory as --model.'
        ),
    )
    parser.add_argument(
        '--manifest', required=True, metavar='CSV',
        help='Path to manifest.csv',
    )
    parser.add_argument(
        '--crops-dir', default=None, metavar='DIR',
        help=(
            'Root directory of PNG crops. '
            'Used only when crop_path in the manifest is relative. '
            'Absolute paths in the manifest are used as-is.'
        ),
    )
    parser.add_argument(
        '--out-dir', required=True, metavar='DIR',
        help='Directory where output PNG figures are written.',
    )
    parser.add_argument(
        '--n-samples', type=int, default=8, metavar='N',
        help='Number of correct and misclassified crops to visualise per species (default: 8).',
    )
    parser.add_argument(
        '--split-file', default=None, metavar='CSV',
        help=(
            'Path to holdout_split.csv (columns: file_id, partition). '
            'When supplied, only crops whose file_id maps to --split-partition '
            'are used, matching the held-out set from training. '
            'Omit to sample from all manifest rows.'
        ),
    )
    parser.add_argument(
        '--split-partition', default='test', metavar='PART',
        help='Partition to draw samples from: test (default), val, or train.',
    )
    parser.add_argument(
        '--primary-harmonic', action='store_true', default=True,
        help='Keep only harmonic_idx==0 rows (fundamental chirps). Default: on.',
    )
    parser.add_argument(
        '--all-harmonics', dest='primary_harmonic', action='store_false',
        help='Include harmonic copies (harmonic_idx > 0) as well.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N',
        help='Inference batch size (default: 64).',
    )
    parser.add_argument(
        '--device', default='auto',
        help='cuda | cpu | cuda:0 | auto (default: auto).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sample selection (default: 42).',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        log.err(f'Model not found: {model_path}')
        sys.exit(1)

    encoder_path = (
        Path(args.encoder) if args.encoder
        else model_path.parent / 'label_encoder.json'
    )
    if not encoder_path.exists():
        log.err(f'Label encoder not found: {encoder_path}')
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.err(f'Manifest not found: {manifest_path}')
        sys.exit(1)

    crops_dir  = Path(args.crops_dir) if args.crops_dir else None
    split_file = Path(args.split_file) if args.split_file else None

    runner = GradCamRunner(
        model_path       = model_path,
        encoder_path     = encoder_path,
        manifest_path    = manifest_path,
        crops_dir        = crops_dir,
        out_dir          = Path(args.out_dir),
        n_samples        = args.n_samples,
        split_file       = split_file,
        split_partition  = args.split_partition,
        primary_harmonic = args.primary_harmonic,
        batch_size       = args.batch_size,
        device_str       = args.device,
        seed             = args.seed,
    )
    runner.run()


if __name__ == '__main__':
    main()
