#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-08 16:11:10
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-18 09:42:19
# **********************************************************

"""
gradcam_bat.py
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
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

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

        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features  # 1536 for B3
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
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
    Reproduce the inference-time transform used in ``train_cnn.py``, with an
    optional low-percentile row background subtraction stage.

    Pipeline (bg_subtract=True)::

        grayscale PIL → numpy → subtract row percentile floor → clip → PIL
        → Resize(224) → ToTensor → repeat 3ch → ImageNet normalise

    For each frequency bin (row) the Nth percentile intensity across all time
    columns is used as the noise-floor estimate.  Using a low percentile
    (default: 10th) rather than the median (50th) is critical for species
    with persistent narrow-band or shallow-sweep calls (Lano, Laci, Epfu):
    the median is pulled up by call energy when the call occupies >50% of
    time columns, causing self-cancellation.  The 10th percentile stays close
    to the true silence floor even for long-duration calls.

    A per-row silence guard skips subtraction on rows whose estimated floor
    is below ``min_floor_dn`` digital numbers — these rows are already near
    black and subtraction would only amplify quantisation noise.

    The subtraction is applied *before* resizing so the percentile is
    computed on the native-resolution spectrogram, avoiding blur artefacts.

    :param img_size:     Resize target (default: 224).
    :param bg_subtract:  Enable background subtraction.
    :param bg_percentile: Row percentile used as noise-floor estimate.
                          10 works well for most bat species; lower values
                          are more conservative (less subtraction).
    :param min_floor_dn: Rows whose estimated floor is below this value
                         (0-255) are left untouched.  Default: 4.
    """

    def __init__(
        self,
        img_size:      int   = _IMG_SIZE,
        bg_subtract:   bool  = False,
        bg_percentile: float = 10.0,
        min_floor_dn:  int   = 4,
    ) -> None:
        self.bg_subtract   = bg_subtract
        self.bg_percentile = bg_percentile
        self.min_floor_dn  = min_floor_dn
        self.transform     = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(_MEAN, _STD),
        ])

    # ----------------------------------------------------------------------- #

    def subtract_bg(self, img: Image.Image) -> Image.Image:
        """
        Subtract the per-row low-percentile noise floor from a grayscale image.

        Public so that ``test_bg_subtract.py`` can call it directly for
        visual inspection without running the full preprocessing pipeline.

        :param img: Grayscale PIL image (mode ``'L'``).
        :return:    Background-suppressed grayscale PIL image.
        """
        arr   = np.array(img, dtype=np.int16)                          # (H, W)
        floor = np.percentile(arr, self.bg_percentile,
                              axis=1, keepdims=True)                   # (H, 1)
        # Silence guard: skip rows that are already near-black.
        floor = np.where(floor < self.min_floor_dn, 0, floor)
        arr   = np.clip(arr - floor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode='L')

    # ----------------------------------------------------------------------- #

    def __call__(self, path: str) -> torch.Tensor:
        """
        Load and preprocess a single crop PNG.

        :param path: Absolute path to the PNG file.
        :return:     Normalised tensor of shape ``(3, H, W)``.
        """
        img = Image.open(path).convert('L')
        if self.bg_subtract:
            img = self.subtract_bg(img)
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
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Compute the Grad-CAM heatmap for *class_idx* and return raw logits.

        :param tensor:    Pre-processed image tensor of shape ``(3, H, W)``
                          (no batch dim).
        :param class_idx: Index of the class to explain.
        :return:          Tuple of:
                          * Float32 CAM array of shape ``(H, W)`` in ``[0, 1]``.
                          * 1-D logits tensor of shape ``(n_classes,)`` (detached,
                            on CPU).  Identical for both calls on the same image,
                            so callers may reuse the value from the first call.
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
        return cam.astype(np.float32), logits.detach().cpu().squeeze()


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

    @staticmethod
    def _focus_score(cam: np.ndarray, gray: np.ndarray,
                     bright_pct: float = 0.20) -> float:
        """
        Fraction of total CAM activation that falls on bright spectrogram pixels.

        A high score means the model is attending to actual call energy; a low
        score means it is firing on background/silence.

        :param cam:        Normalised CAM array ``(H, W)`` in ``[0, 1]``.
        :param gray:       De-normalised grayscale image ``(H, W)`` uint8.
        :param bright_pct: Top fraction of pixel intensities considered
                           "call energy".  Default: top 20 %.
        :return:           Scalar in ``[0, 1]``.
        """
        threshold   = np.percentile(gray, 100 * (1 - bright_pct))
        call_mask   = gray >= threshold          # True where call energy is
        total_act   = cam.sum()
        if total_act == 0:
            return 0.0
        return float(cam[call_mask].sum() / total_act)

    # ----------------------------------------------------------------------- #

    def _render_one(
        self,
        path:         str,
        true_sp:      str,
        pred_sp:      str,
        save_path:    Path,
        idx_to_label: Dict[int, str],
    ) -> dict:
        """
        Produce and save a Grad-CAM figure and return per-image metrics.

        :param path:         Absolute path to the crop PNG.
        :param true_sp:      True species label.
        :param pred_sp:      Predicted species label.
        :param save_path:    Output PNG path.
        :param idx_to_label: Class index → species mapping (for top-3 labels).
        :return:             Dict with keys ``focus_true``, ``focus_pred``
                             (focus scores, pred==None when correctly classified),
                             ``top3`` (list of ``(species, prob)`` tuples),
                             ``pred_prob`` (softmax probability of predicted class),
                             ``true_prob`` (softmax probability of true class).
        """
        tensor = self.preprocessor(path)

        label_to_idx = {v: k for k, v in idx_to_label.items()}
        true_idx = label_to_idx[true_sp]
        pred_idx = label_to_idx[pred_sp]

        cam_true, logits = self.gradcam.compute(tensor, true_idx)
        cam_pred, _      = self.gradcam.compute(tensor, pred_idx) if pred_sp != true_sp                            else (None, None)

        # Softmax probabilities for top-3 reporting.
        probs    = torch.softmax(logits, dim=0).numpy()
        top3_idx = probs.argsort()[::-1][:3]
        top3     = [(idx_to_label[int(i)], float(probs[i])) for i in top3_idx]
        pred_prob = float(probs[pred_idx])
        true_prob = float(probs[true_idx])

        gray = self._denorm(tensor)

        focus_true = self._focus_score(cam_true, gray)
        focus_pred = self._focus_score(cam_pred, gray) if cam_pred is not None else None

        # ── figure ──────────────────────────────────────────────────────────
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
        axes[1].set_title(
            f'Grad-CAM → {true_sp} (true)  focus={focus_true:.2f}', fontsize=10
        )
        axes[1].axis('off')

        # Panel 2 (misclassified only): CAM for the predicted class.
        if cam_pred is not None:
            axes[2].imshow(gray, cmap='gray', origin='upper')
            axes[2].imshow(cam_pred, cmap='jet', alpha=0.45, origin='upper',
                           vmin=0, vmax=1)
            axes[2].set_title(
                f'Grad-CAM → {pred_sp} (pred)  focus={focus_pred:.2f}', fontsize=10
            )
            axes[2].axis('off')

        top3_str = '  '.join(f'{sp}:{p:.2f}' for sp, p in top3)
        crop_name = Path(path).name
        fig.suptitle(f'{crop_name}   [{top3_str}]', fontsize=8, y=0.02)
        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        return {
            'focus_true' : round(focus_true, 4),
            'focus_pred' : round(focus_pred, 4) if focus_pred is not None else None,
            'top3'       : top3,
            'pred_prob'  : round(pred_prob, 4),
            'true_prob'  : round(true_prob, 4),
        }

    # ----------------------------------------------------------------------- #

    def render_batch(
        self,
        df:       pd.DataFrame,
        species:  str,
        tag:      str,
    ) -> List[dict]:
        """
        Render Grad-CAM figures for a batch of crops.

        :param df:      DataFrame rows with ``crop_path``, ``species``,
                        ``predicted``, ``file_id``, ``chirp_idx``,
                        ``Filename`` columns.
        :param species: Species label (used in filenames).
        :param tag:     ``'correct'`` or ``'misclassified'``.
        :return:        List of record dicts for each successfully saved figure,
                        with keys ``figure``, ``file_id``, ``chirp_idx``,
                        ``Filename``, ``true``, ``predicted``.
        """
        records: List[dict] = []
        for i, row in enumerate(df.itertuples()):
            save_path = self.out_dir / f'{species}_{tag}_{i:03d}.png'
            try:
                metrics = self._render_one(
                    path         = row.crop_path,
                    true_sp      = row.species,
                    pred_sp      = row.predicted,
                    save_path    = save_path,
                    idx_to_label = self.idx_to_label,
                )
                log.info(f'  saved {save_path.name}  '
                         f'focus={metrics["focus_true"]:.2f}  '
                         f'pred_prob={metrics["pred_prob"]:.2f}')
                records.append({
                    'figure'     : save_path.name,
                    'file_id'    : int(row.file_id),
                    'chirp_idx'  : int(row.chirp_idx),
                    'Filename'   : getattr(row, 'Filename', ''),
                    'true'       : row.species,
                    'predicted'  : row.predicted,
                    'confidence' : round(float(getattr(row, 'confidence', float('nan'))), 3),
                    **metrics,
                })
            except Exception as exc:
                log.warn(f'  failed on {row.crop_path}: {exc}')
        return records


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
        measures_path:   Optional[Path] = None,
        bg_subtract:     bool           = False,
        bg_percentile:   float          = 10.0,
        min_floor_dn:    int            = 4,
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

        preprocessor = CropPreprocessor(
            bg_subtract   = bg_subtract,
            bg_percentile = bg_percentile,
            min_floor_dn  = min_floor_dn,
        )
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
        self.measures_path   = measures_path
        self.bg_subtract     = bg_subtract
        self.bg_percentile   = bg_percentile
        self.min_floor_dn    = min_floor_dn
        self.seed            = seed
        self.out_dir         = out_dir

    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #

    @staticmethod
    def _format_table(records: List[dict], aux_map: dict) -> str:
        """
        Format a list of figure records as a fixed-width text table.

        :param records: List of dicts from :meth:`GradCamVisualiser.render_batch`.
        :param aux_map: Mapping of ``(file_id, chirp_idx)`` →
                        ``{'rec_site': str, 'was_daytime': str}``,
                        or empty dict when no measures file was supplied.
        :return:        Multi-line string with header + one row per figure.
        """
        has_aux = bool(aux_map)

        # Core columns always present.
        cols = [
            'figure', 'file_id', 'chirp_idx',
            'true', 'predicted',
            'confidence', 'pred_prob', 'true_prob',
            'focus_true', 'focus_pred',
            'top3',
            'Filename',
        ]
        if has_aux:
            cols += ['rec_site', 'was_daytime']

        rows = []
        for r in records:
            top3_str = '|'.join(f'{sp}:{p:.2f}' for sp, p in r.get('top3', []))
            fp = r.get('focus_pred')
            row = [
                r['figure'],
                str(r['file_id']),
                str(r['chirp_idx']),
                r['true'],
                r['predicted'],
                f'{r.get("confidence", float("nan")):.3f}',
                f'{r.get("pred_prob", float("nan")):.3f}',
                f'{r.get("true_prob",  float("nan")):.3f}',
                f'{r.get("focus_true", float("nan")):.3f}',
                f'{fp:.3f}' if fp is not None else '—',
                top3_str,
                str(r.get('Filename', '')),
            ]
            if has_aux:
                key = (r['file_id'], r['chirp_idx'])
                aux = aux_map.get(key, {})
                row.append(str(aux.get('rec_site',    '?')))
                row.append(str(aux.get('was_daytime', '?')))
            rows.append(row)

        widths = [max(len(c), max((len(r[i]) for r in rows), default=0))
                  for i, c in enumerate(cols)]
        sep  = '  '.join('-' * w for w in widths)
        hdr  = '  '.join(c.ljust(w) for c, w in zip(cols, widths))
        body = '\n'.join(
            '  '.join(cell.ljust(w) for cell, w in zip(row, widths))
            for row in rows
        )
        return '\n'.join([hdr, sep, body])

    # ----------------------------------------------------------------------- #

    def _load_aux_map(self, records: List[dict]) -> dict:
        """
        Look up ``rec_site`` and ``was_daytime`` for each figure record from
        the measures parquet.

        Reads only the rows needed via a file_id pre-filter.
        Uses ``Utils.read_df_file()`` to respect thrift size limits.

        :param records: Figure records containing ``file_id`` and ``chirp_idx``.
        :return:        Dict mapping ``(file_id, chirp_idx)`` →
                        ``{'rec_site': str, 'was_daytime': str}``.
                        Returns empty dict when ``measures_path`` is ``None``
                        or the parquet lacks a ``rec_site`` column.
                        Note: ``was_daytime`` will be True for barn recordings
                        near roost-exit at dusk — treat as descriptive only.
        """
        if self.measures_path is None:
            return {}
        try:
            from sonobat_utils.utils import Utils
            log.info(f'Loading measures for aux lookup: {self.measures_path}')
            mdf = Utils.read_df_file(self.measures_path)
            if 'rec_site' not in mdf.columns:
                log.warn('measures parquet has no rec_site column; skipping aux lookup')
                return {}
            needed_fids = {r['file_id'] for r in records}
            keep_cols = ['file_id', 'chirp_idx', 'rec_site']
            if 'was_daytime' in mdf.columns:
                keep_cols.append('was_daytime')
            sub = mdf[mdf['file_id'].astype(int).isin(needed_fids)][keep_cols].copy()
            result = {}
            for row in sub.itertuples():
                key = (int(row.file_id), int(row.chirp_idx))
                result[key] = {
                    'rec_site'   : str(row.rec_site),
                    'was_daytime': str(getattr(row, 'was_daytime', '?')),
                }
            return result
        except Exception as exc:
            log.warn(f'Could not load aux map from measures parquet: {exc}')
            return {}

    # ----------------------------------------------------------------------- #

    def run(self) -> None:
        """Execute inference, sample selection, and figure generation."""
        df_pred = self.selector.run_inference()

        summary_lines: List[str] = []
        all_records:   List[dict] = []

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

            rec_c = self.visualiser.render_batch(correct_df,  sp, 'correct')
            rec_m = self.visualiser.render_batch(wrong_df,    sp, 'misclassified')

            all_records.extend(rec_c)
            all_records.extend(rec_m)

            summary_lines.append(
                f'\n=== {sp}: {len(rec_c)} correct, {len(rec_m)} misclassified ==='
            )

            # Confusion breakdown for misclassified crops.
            if len(wrong_df) > 0:
                breakdown = wrong_df['predicted'].value_counts().to_dict()
                log.info(f'  {sp} confused as: {breakdown}')
                summary_lines.append(f'  confused as: {breakdown}')

            # Confidence stats: compare SonoBat confidence for correct vs misclassified.
            def _conf_stats(recs: List[dict]) -> str:
                vals = [r['confidence'] for r in recs
                        if r['confidence'] == r['confidence']]  # drop NaN
                if not vals:
                    return 'n/a'
                arr = np.array(vals)
                return (f'mean={arr.mean():.3f}  std={arr.std():.3f}  '
                        f'min={arr.min():.3f}  max={arr.max():.3f}')

            summary_lines.append(
                f'  confidence correct:       {_conf_stats(rec_c)}'
            )
            summary_lines.append(
                f'  confidence misclassified: {_conf_stats(rec_m)}'
            )

            # Focus score stats: are correct crops better focused on call energy?
            def _focus_stats(recs: List[dict], key: str) -> str:
                vals = [r[key] for r in recs if r.get(key) is not None]
                if not vals:
                    return 'n/a'
                arr = np.array(vals)
                return f'mean={arr.mean():.3f}  std={arr.std():.3f}'

            summary_lines.append(
                f'  focus_true correct:       {_focus_stats(rec_c, "focus_true")}'
            )
            summary_lines.append(
                f'  focus_true misclassified: {_focus_stats(rec_m, "focus_true")}'
            )

            # Per-figure table for this species.
            aux_map = self._load_aux_map(rec_c + rec_m)
            if rec_c:
                summary_lines.append('\n--- correct ---')
                summary_lines.append(self._format_table(rec_c, aux_map))
            if rec_m:
                summary_lines.append('\n--- misclassified ---')
                summary_lines.append(self._format_table(rec_m, aux_map))

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
        '--measures', default=None, metavar='PARQUET',
        help=(
            'Path to the measures parquet (e.g. bats_2026-04-23T....parquet). '
            'When supplied, rec_site is looked up per figure and added to the '
            'summary table. The (file_id, chirp_idx) foreign key is used for the join.'
        ),
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
        '--bg-subtract', action='store_true', default=False,
        help=(
            'Apply per-row low-percentile background subtraction before '
            'inference and Grad-CAM.  For each frequency bin (row) the Nth '
            'percentile intensity across time columns is used as the noise-floor '
            'estimate and subtracted.  Suppresses stationary background hum and '
            'vertical streaks while preserving call energy.'
        ),
    )
    parser.add_argument(
        '--bg-percentile', type=float, default=10.0, metavar='PCT',
        help=(
            'Row percentile (0-100) used as the per-row noise-floor estimate '
            'when --bg-subtract is active.  Lower values are more conservative '
            '(less subtraction, safer for persistent narrow-band calls like '
            'Lano, Laci, Epfu).  Default: 10.',
        ),
    )
    parser.add_argument(
        '--min-floor-dn', type=int, default=4, metavar='DN',
        help=(
            'Rows whose estimated noise floor is below this digital-number '
            'value (0-255) are left untouched — they are already near-black '
            'and subtraction would only amplify quantisation noise.  Default: 4.'
        ),
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

    measures_path = Path(args.measures) if args.measures else None

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
        measures_path    = measures_path,
        bg_subtract      = args.bg_subtract,
        bg_percentile    = args.bg_percentile,
        min_floor_dn     = args.min_floor_dn,
        batch_size       = args.batch_size,
        device_str       = args.device,
        seed             = args.seed,
    )
    runner.run()


if __name__ == '__main__':
    main()
