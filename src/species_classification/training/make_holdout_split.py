#!/usr/bin/env python
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-05-04 10:49:18
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-04 10:49:41
# **********************************************************

"""
make_holdout_split.py
=====================
Create a shared train/val/test partition of ``file_id`` values that is
consumed by both ``train_cnn.py`` and ``species_pred_random_forest.py``,
guaranteeing that CNN and RF models are evaluated on the same held-out
test set and never trained on it.

The split is performed at the **file_id level**, stratified by each
fragment's modal species label, so every partition receives a proportional
share of every species.  All chirps from a given file_id land in exactly
one partition.

Input
-----
The manifest CSV written by ``chirps_to_spectros.py``::

    columns: crop_path, partition, species, confidence, file_id, ...

The parquet file produced by ``sb_measures_postprocessing.py`` is also
accepted (must contain ``file_id`` and ``species`` columns).

Output
------
``holdout_split.csv``
    Two-column CSV with ``file_id`` (int) and ``partition``
    (``train`` | ``val`` | ``test``).  This is the single source of truth
    consumed by both classifiers via their ``--split-file`` argument.

``holdout_split_summary.txt``
    Human-readable breakdown: species × partition counts.

Usage
-----
::

    python make_holdout_split.py \\
        --manifest /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --out-dir  /qnap/bats/jr_pipeline/data \\
        --val-frac  0.15 \\
        --test-frac 0.15 \\
        --seed      42

The resulting ``holdout_split.csv`` is then passed to both classifiers::

    torchrun ... train_cnn.py --split-file /path/to/holdout_split.csv ...
    species_pred_random_forest.py --split-file /path/to/holdout_split.csv ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from logging_service import LoggingService

log = LoggingService()

_DEFAULT_VAL_FRAC  = 0.15
_DEFAULT_TEST_FRAC = 0.15
_DEFAULT_SEED      = 42


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class HoldoutSplitMaker:
    """
    Produce a stratified file_id-level train/val/test split from a manifest
    or measures parquet and write it to ``holdout_split.csv``.

    :param input_path: Path to manifest CSV or measures parquet.
    :param out_dir:    Directory for output files (created if absent).
    :param val_frac:   Fraction of file_ids reserved for validation.
    :param test_frac:  Fraction of file_ids reserved for test.
    :param seed:       Random seed for reproducibility.
    """

    def __init__(
        self,
        input_path: str | Path,
        out_dir:    str | Path,
        val_frac:   float = _DEFAULT_VAL_FRAC,
        test_frac:  float = _DEFAULT_TEST_FRAC,
        seed:       int   = _DEFAULT_SEED,
    ) -> None:
        self.input_path = Path(input_path)
        self.out_dir    = Path(out_dir)
        self.val_frac   = val_frac
        self.test_frac  = test_frac
        self.seed       = seed

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def _load(self) -> pd.DataFrame:
        """
        Load the input file and return a DataFrame with at least
        ``file_id`` and ``species`` columns.

        Accepts manifest CSV or measures parquet/feather.

        :return: DataFrame with ``file_id`` (int) and ``species`` (str).
        """
        suffix = self.input_path.suffix.lower()
        log.info(f'Loading {self.input_path}')

        try:
            if suffix == '.csv':
                df = pd.read_csv(self.input_path, low_memory=False)
            elif suffix == '.parquet':
                df = pd.read_parquet(self.input_path)
            elif suffix in ('.feather', '.ftr'):
                df = pd.read_feather(self.input_path)
            else:
                log.warn(f'Unknown suffix {suffix!r} — attempting CSV read')
                df = pd.read_csv(self.input_path, low_memory=False)
        except Exception as exc:
            log.err(f'Cannot read {self.input_path}: {exc}')
            sys.exit(1)

        log.info(f'Loaded {len(df):,} rows, {len(df.columns)} columns')

        for col in ('file_id', 'species'):
            if col not in df.columns:
                log.err(f'Required column {col!r} not found in input file')
                sys.exit(1)

        return df[['file_id', 'species']].copy()

    # ------------------------------------------------------------------ #
    #  Split                                                               #
    # ------------------------------------------------------------------ #

    def _make_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each unique ``file_id`` to train, val, or test, stratified
        by each fragment's modal species.

        :param df: DataFrame with ``file_id`` and ``species`` columns.
        :return:   DataFrame with columns ``file_id`` and ``partition``.
        """
        # Drop rows with missing species — only labeled fragments used
        # to determine the partition assignment.
        labeled = df[df['species'].notna()].copy()
        # Drop composite labels (e.g. 'Myca/Myyu') — ambiguous.
        labeled = labeled[~labeled['species'].str.contains('/', na=False)]

        log.info(f'{labeled["file_id"].nunique():,} unique file_ids with clean labels')

        # Modal species per file_id.
        modal = (
            labeled.groupby('file_id')['species']
            .agg(lambda s: s.mode().iloc[0])
            .rename('modal_species')
            .reset_index()
        )

        rng = np.random.default_rng(self.seed)
        records = []

        for sp, grp in modal.groupby('modal_species'):
            fids = grp['file_id'].values.copy()
            rng.shuffle(fids)
            n       = len(fids)
            n_test  = max(1, round(n * self.test_frac))
            n_val   = max(1, round(n * self.val_frac))
            n_train = n - n_test - n_val
            if n_train < 1:
                # Too few file_ids for this species — put all in train
                # so test/val aren't left with nothing to evaluate on.
                log.warn(
                    f'Species {sp!r} has only {n} file_id(s) — '
                    f'assigning all to train'
                )
                for fid in fids:
                    records.append({'file_id': fid, 'partition': 'train'})
                continue
            for fid in fids[:n_test]:
                records.append({'file_id': fid, 'partition': 'test'})
            for fid in fids[n_test:n_test + n_val]:
                records.append({'file_id': fid, 'partition': 'val'})
            for fid in fids[n_test + n_val:]:
                records.append({'file_id': fid, 'partition': 'train'})

        split_df = pd.DataFrame(records)
        split_df['file_id'] = split_df['file_id'].astype(int)

        counts = split_df['partition'].value_counts()
        log.info(
            f'file_id split: '
            f'train {counts.get("train", 0):,} | '
            f'val   {counts.get("val",   0):,} | '
            f'test  {counts.get("test",  0):,}'
        )
        return split_df

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def _write_summary(
        self,
        df:       pd.DataFrame,
        split_df: pd.DataFrame,
    ) -> None:
        """
        Write a human-readable species × partition breakdown to
        ``holdout_split_summary.txt``.

        :param df:       Original DataFrame with ``file_id`` and ``species``.
        :param split_df: Split DataFrame with ``file_id`` and ``partition``.
        """
        labeled = df[df['species'].notna()].copy()
        labeled = labeled[~labeled['species'].str.contains('/', na=False)]

        # Modal species per file_id.
        modal = (
            labeled.groupby('file_id')['species']
            .agg(lambda s: s.mode().iloc[0])
            .rename('modal_species')
            .reset_index()
        )
        merged = modal.merge(split_df, on='file_id', how='inner')

        pivot = (
            merged.groupby(['modal_species', 'partition'])
            .size()
            .unstack(fill_value=0)
        )
        # Ensure all three columns present.
        for col in ('train', 'val', 'test'):
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[['train', 'val', 'test']]
        pivot['total'] = pivot.sum(axis=1)
        pivot.loc['TOTAL'] = pivot.sum()

        lines = [
            f'Holdout split summary',
            f'  seed={self.seed}  val_frac={self.val_frac}  '
            f'test_frac={self.test_frac}',
            f'  input: {self.input_path}',
            '',
            'file_id counts per species × partition:',
            pivot.to_string(),
        ]
        summary_path = self.out_dir / 'holdout_split_summary.txt'
        summary_path.write_text('\n'.join(lines) + '\n')
        log.info(f'Saved {summary_path}')
        log.info(f'\n{pivot.to_string()}')

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """
        Execute the split and write output files.

        :return: Path to the written ``holdout_split.csv``.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        df       = self._load()
        split_df = self._make_split(df)

        out_path = self.out_dir / 'holdout_split.csv'
        split_df.to_csv(out_path, index=False)
        log.info(f'Saved {out_path}  ({len(split_df):,} file_id rows)')

        self._write_summary(df, split_df)
        return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: ``argparse.Namespace`` with validated attributes.
    """
    parser = argparse.ArgumentParser(
        prog='make_holdout_split',
        description=(
            'Create a shared train/val/test file_id split for use by\n'
            'both train_cnn.py and species_pred_random_forest.py.\n\n'
            'Output: holdout_split.csv  (file_id, partition)'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--manifest', '--input', '-i',
        required=True,
        metavar='PATH',
        dest='input',
        help=(
            'Manifest CSV (from chirps_to_spectros.py) or measures parquet\n'
            '(from sb_measures_postprocessing.py).  Must contain file_id\n'
            'and species columns.'
        ),
    )
    parser.add_argument(
        '--out-dir', '-o',
        required=True,
        metavar='DIR',
        help='Directory for holdout_split.csv and summary (created if absent).',
    )
    parser.add_argument(
        '--val-frac',
        type=float,
        default=_DEFAULT_VAL_FRAC,
        metavar='F',
        help=f'Fraction of file_ids for validation (default: {_DEFAULT_VAL_FRAC}).',
    )
    parser.add_argument(
        '--test-frac',
        type=float,
        default=_DEFAULT_TEST_FRAC,
        metavar='F',
        help=f'Fraction of file_ids for test (default: {_DEFAULT_TEST_FRAC}).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=_DEFAULT_SEED,
        metavar='N',
        help=f'Random seed for reproducibility (default: {_DEFAULT_SEED}).',
    )

    args = parser.parse_args()
    if not Path(args.input).exists():
        parser.error(f'Input file not found: {args.input}')
    if args.val_frac + args.test_frac >= 1.0:
        parser.error('--val-frac + --test-frac must be < 1.0')
    return args


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    maker = HoldoutSplitMaker(
        input_path = args.input,
        out_dir    = args.out_dir,
        val_frac   = args.val_frac,
        test_frac  = args.test_frac,
        seed       = args.seed,
    )
    out = maker.run()
    log.info(f'Done.  Split file: {out}')


if __name__ == '__main__':
    main()
