#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-03-15 09:46:12
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/species_classification/chirps_to_spectros.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-15 10:06:06
# **********************************************************

"""
Extract per-chirp spectrogram crops from original full-length bat recordings
for CNN species classification training.

Background
----------
Rather than training on 2-second SonoBat fragments or full 50-second
recordings, we crop a short window around each individually detected chirp.
This gives the CNN the full call shape — FM sweep curvature, CF tail length,
harmonic structure — at a resolution where acoustically similar species
(e.g. *Lano* vs *Tabr*) are distinguishable.

Inputs
------
``sonobat3_2_species_ids.feather``
    Chirp-level measures file produced by ``sono_batch_processing.py``.
    Must contain: ``Filename``, ``TimeInFile``, ``species``, ``species_prob``.

``match_report.csv``
    Output of ``wav_path_resolver.py``.  Maps each ``Filename`` stem to the
    resolved full-recording ``.wav`` path.

Pipeline per chirp
------------------
1. Look up the full-recording ``.wav`` path via ``Filename``.
2. Seek to ``TimeInFile`` ms within that recording using ``soundfile``
   (no full-file load).
3. Extract a ``--window-ms`` window centered on the chirp onset.
4. Compute a linear-scale power spectrogram restricted to
   ``--freq-lo`` – ``--freq-hi`` kHz.
5. Normalise to [0, 255] (log-power, clipped at ``--dynamic-range`` dB).
6. Resize to ``--img-size`` × ``--img-size`` pixels.
7. Save as PNG under ``<out-dir>/<species>/``.
8. Append a row to the master manifest CSV.

Output layout
-------------
Chirp PNGs are stored under ``<out-dir>/<YYYYMMDD>_<site>/`` where the
date and site are derived from the ``Filename`` stem.  This partitioning
is stable and trustworthy regardless of species label quality; species
labels live only in ``manifest.csv`` and can be revised without moving
files.

::

    <out-dir>/
        20220427_lake2/
            00000001.png
            00000002.png
            ...
        20220706_barn/
            00000001.png
            ...
        manifest.csv          # master index: crop_path, species, prob, ...
        extractor_config.csv  # run parameters and statistics

``manifest.csv`` columns
------------------------
``crop_path``, ``species``, ``species_prob``, ``file_id``, ``Filename``,
``time_in_file_ms``, ``match_quality``

Typical usage
-------------
::

    python chirps_to_spectros.py \\
        --feather  sonobat3_2_species_ids.feather \\
        --matches  recording_file_locations/match_report.csv \\
        --out-dir  /data/bat_crops \\
        --min-prob 0.80 \\
        --window-ms 50 \\
        --freq-lo 15 \\
        --freq-hi 80 \\
        --img-size 224 \\
        --workers 16
"""

import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.signal as signal
from PIL import Image

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False
    import scipy.io.wavfile as wavfile

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

_DEFAULT_WINDOW_MS:     float = 50.0
_DEFAULT_FREQ_LO_KHZ:  float = 15.0
_DEFAULT_FREQ_HI_KHZ:  float = 80.0
_DEFAULT_IMG_SIZE:      int   = 224
_DEFAULT_MIN_PROB:      float = 0.80
_DEFAULT_DYNAMIC_RANGE: float = 60.0   # dB — log-power normalisation range
_DEFAULT_WORKERS:       int   = max(1, (os.cpu_count() or 4) - 2)
_STFT_WINDOW_MS:        float = 0.25   # same as chirp_measures_extraction.py


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpectroExtractionResult:
    """
    Summary returned by :meth:`ChirpSpectroExtractor.run`.

    :param n_chirps_input:   Chirp rows in feather after confidence filter.
    :param n_crops_written:  PNG files successfully written.
    :param n_failed:         Chirps that could not be cropped (missing wav,
                             seek error, degenerate spectrogram).
    :param n_skipped:        Chirps skipped (already in manifest from a
                             prior incremental run).
    :param species_counts:   Dict mapping species code to crop count.
    :param elapsed_secs:     Wall-clock seconds.
    :param out_dir:          Output directory.
    """
    n_chirps_input:  int
    n_crops_written: int
    n_failed:        int
    n_skipped:       int
    species_counts:  dict
    elapsed_secs:    float
    out_dir:         Path

    def summary(self) -> str:
        """
        Return a human-readable multi-line run summary.

        :return: Formatted string with crop statistics.
        """
        mins, secs = divmod(self.elapsed_secs, 60)
        elapsed_str = f'{int(mins)}m {secs:.1f}s' if mins else f'{secs:.1f}s'
        lines = [
            'ChirpSpectroExtractor complete:',
            f'  * {self.n_chirps_input:,} chirps passed confidence filter',
            f'  * {self.n_crops_written:,} crops written',
            f'  * {self.n_failed:,} failed (missing wav / seek error)',
            f'  * {self.n_skipped:,} skipped (already done)',
            '  * Crops per species:',
        ]
        for sp, count in sorted(self.species_counts.items()):
            lines.append(f'      {sp:6s}  {count:,}')
        lines += [f'  * Elapsed: {elapsed_str}', f'  * Output:  {self.out_dir}']
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _chirp_to_spectro(
    wav_path:      str,
    time_ms:       float,
    window_ms:     float,
    freq_lo_hz:    float,
    freq_hi_hz:    float,
    img_size:      int,
    dynamic_range: float,
) -> Optional[np.ndarray]:
    """
    Load a short audio window from a full recording and return a normalised
    spectrogram image as a uint8 numpy array of shape
    ``(img_size, img_size)``.

    Uses ``soundfile`` for seek-based partial reads when available,
    falling back to ``scipy.io.wavfile`` (full file load) otherwise.

    :param wav_path:      Path to the full-recording ``.wav`` file.
    :param time_ms:       Chirp onset within the recording (ms).
    :param window_ms:     Total window duration centred on onset (ms).
    :param freq_lo_hz:    Lower frequency bound for spectrogram (Hz).
    :param freq_hi_hz:    Upper frequency bound for spectrogram (Hz).
    :param img_size:      Output image size in pixels (square).
    :param dynamic_range: Log-power normalisation range (dB).
    :return:              uint8 array of shape ``(img_size, img_size)``,
                          or ``None`` on any error.
    """
    try:
        half_ms = window_ms / 2.0

        if _SF_AVAILABLE:
            with sf.SoundFile(wav_path) as f:
                sr = f.samplerate
                total_frames = len(f)
                # Centre window on onset; clamp to file boundaries.
                start_s  = max(0.0, (time_ms - half_ms) / 1000.0)
                end_s    = min(total_frames / sr, (time_ms + half_ms) / 1000.0)
                start_fr = int(start_s * sr)
                n_frames = int((end_s - start_s) * sr)
                if n_frames < 4:
                    return None
                f.seek(start_fr)
                data = f.read(n_frames, dtype='float32', always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                audio = data
        else:
            # Full file load — slow for large files but always available.
            sr_raw, raw = wavfile.read(wav_path)
            if raw.ndim > 1:
                raw = raw.mean(axis=1)
            if raw.dtype == np.int16:
                raw = raw.astype(np.float32) / 32768.0
            elif raw.dtype == np.int32:
                raw = raw.astype(np.float32) / 2_147_483_648.0
            else:
                raw = raw.astype(np.float32)
            sr = sr_raw
            start_fr = max(0, int((time_ms - half_ms) / 1000.0 * sr))
            end_fr   = min(len(raw), int((time_ms + half_ms) / 1000.0 * sr))
            audio    = raw[start_fr:end_fr]

        # Apply TE correction if needed (same threshold as MeasureExtractor).
        if sr < 80_000:
            sr *= 10

        if len(audio) < 4:
            return None

        # ── Spectrogram ────────────────────────────────────────────────
        nperseg  = max(64, int(round(_STFT_WINDOW_MS / 1000.0 * sr)))
        noverlap = nperseg * 3 // 4
        freqs, _, Sxx = signal.spectrogram(
            audio, fs=sr,
            window='hann', nperseg=nperseg, noverlap=noverlap,
            scaling='spectrum',
        )

        # Restrict to target frequency band.
        band = (freqs >= freq_lo_hz) & (freqs <= freq_hi_hz)
        if band.sum() < 2:
            return None
        Sxx_band = Sxx[band, :]

        # ── Log-power normalisation → [0, 255] uint8 ──────────────────
        # Convert to dB, clip to dynamic range, scale to 0-255.
        # Small epsilon avoids log(0).
        eps      = 1e-12
        Sxx_db   = 10.0 * np.log10(Sxx_band + eps)
        db_max   = Sxx_db.max()
        db_floor = db_max - dynamic_range
        Sxx_clip = np.clip(Sxx_db, db_floor, db_max)
        Sxx_norm = ((Sxx_clip - db_floor) / dynamic_range * 255.0).astype(np.uint8)

        # Frequency axis: low freq at bottom → flip so low freq is at bottom
        # of image (PIL origin is top-left, so we flip vertically).
        Sxx_norm = np.flipud(Sxx_norm)

        # ── Resize to img_size × img_size ─────────────────────────────
        img = Image.fromarray(Sxx_norm, mode='L')
        img = img.resize((img_size, img_size), Image.LANCZOS)
        return np.array(img)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ChirpSpectroExtractor:
    """
    Extract per-chirp spectrogram crops from full bat recordings.

    Reads chirp metadata from a feather file, resolves each chirp's source
    ``.wav`` via a match-report CSV, extracts a short audio window around
    the chirp onset, computes a linear-scale spectrogram, and saves the
    result as a PNG under ``<out_dir>/<YYYYMMDD>_<site>/``.

    Output is incremental: a manifest CSV tracks which chirps have already
    been processed so re-runs skip completed work.

    :param feather_path:    Path to the chirp-level feather file.
    :param match_csv:       Path to ``match_report.csv`` from
                            :class:`~wav_path_resolver.WavPathResolver`.
    :param out_dir:         Root output directory for PNG crops.
    :param min_prob:        Minimum ``species_prob`` to include a chirp.
    :param window_ms:       Crop window duration in ms, centred on chirp onset.
    :param freq_lo_khz:     Lower frequency bound for spectrogram (kHz).
    :param freq_hi_khz:     Upper frequency bound for spectrogram (kHz).
    :param img_size:        Output PNG size in pixels (square).
    :param dynamic_range:   Log-power normalisation range in dB.
    :param n_workers:       Parallel worker processes.
    :param match_quality:   Accepted match quality values from match report.
                            Defaults to ``['window', 'nearest']``; pass
                            ``['window']`` to exclude fallback matches.
    """

    def __init__(
        self,
        feather_path:   str | Path,
        match_csv:      str | Path,
        out_dir:        str | Path,
        min_prob:       float          = _DEFAULT_MIN_PROB,
        window_ms:      float          = _DEFAULT_WINDOW_MS,
        freq_lo_khz:    float          = _DEFAULT_FREQ_LO_KHZ,
        freq_hi_khz:    float          = _DEFAULT_FREQ_HI_KHZ,
        img_size:       int            = _DEFAULT_IMG_SIZE,
        dynamic_range:  float          = _DEFAULT_DYNAMIC_RANGE,
        n_workers:      int            = _DEFAULT_WORKERS,
        match_quality:  Sequence[str]  = ('window', 'nearest'),
    ) -> None:
        self.feather_path  = Path(feather_path)
        self.match_csv     = Path(match_csv)
        self.out_dir       = Path(out_dir)
        self.min_prob      = min_prob
        self.window_ms     = window_ms
        self.freq_lo_hz    = freq_lo_khz * 1000.0
        self.freq_hi_hz    = freq_hi_khz * 1000.0
        self.img_size      = img_size
        self.dynamic_range = dynamic_range
        self.n_workers     = n_workers
        self.match_quality = set(match_quality)

    # ------------------------------------------------------------------ #
    #  Data loading and preparation                                       #
    # ------------------------------------------------------------------ #

    def _load_work_items(self) -> pd.DataFrame:
        """
        Load the feather file and match report, apply filters, and return
        a DataFrame of chirps ready for crop extraction.

        Filters applied:
        * ``species`` is not NaN and is a clean 4-char code
        * ``species_prob`` >= ``self.min_prob`` (or prob is NaN but species
          is present — some SonoBat rows have no prob)
        * ``matched_wav`` is non-empty and the file exists
        * ``match_quality`` is in ``self.match_quality``

        :return: DataFrame with columns ``Filename``, ``TimeInFile``,
                 ``species``, ``species_prob``, ``file_id``, ``matched_wav``,
                 ``match_quality``.
        """
        log.info(f'Loading feather: {self.feather_path} ...')
        df = pd.read_feather(self.feather_path)
        log.info(f'  {len(df):,} total chirp rows')

        log.info(f'Loading match report: {self.match_csv} ...')
        matches = pd.read_csv(self.match_csv)
        # Keep only columns we need and those with a resolved wav path.
        matches = matches[matches['matched_wav'].notna() &
                          (matches['matched_wav'] != '')]
        matches = matches[matches['match_quality'].isin(self.match_quality)]
        matches = matches[['Filename', 'matched_wav', 'match_quality']]
        log.info(
            f'  {len(matches):,} match-report rows with resolved paths '
            f'(quality filter: {self.match_quality})'
        )

        # Join on Filename.
        merged = df.merge(matches, on='Filename', how='inner')
        log.info(f'  {len(merged):,} rows after join on Filename')

        # Species filter: clean 4-char code only.
        import re
        _sp_re = re.compile(r'^[A-Z][a-z]{3}$')
        merged = merged[merged['species'].notna()]
        merged = merged[merged['species'].apply(
            lambda s: bool(_sp_re.match(str(s)))
        )]
        log.info(f'  {len(merged):,} rows with valid species code')

        # Confidence filter — allow NaN prob through if species is present.
        prob_ok = merged['species_prob'].isna() | \
                  (merged['species_prob'] >= self.min_prob)
        merged = merged[prob_ok]
        log.info(
            f'  {len(merged):,} rows after confidence filter '
            f'(min_prob={self.min_prob})'
        )

        # Verify wav files exist (sample check — full check per-worker).
        needed_cols = ['Filename', 'TimeInFile', 'species',
                       'species_prob', 'file_id', 'matched_wav',
                       'match_quality']
        missing = [c for c in needed_cols if c not in merged.columns]
        if missing:
            log.warn(f'Missing columns in merged DataFrame: {missing}')
            sys.exit(1)

        return merged[needed_cols].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Manifest helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_done_set(manifest_path: Path) -> set[str]:
        """
        Return the set of ``Filename`` stems already in the manifest CSV,
        enabling incremental re-runs.

        :param manifest_path: Path to an existing ``manifest.csv``.
        :return:              Set of ``Filename`` strings already processed.
        """
        if not manifest_path.exists():
            return set()
        try:
            mdf = pd.read_csv(manifest_path, usecols=['Filename'])
            done = set(mdf['Filename'].dropna().unique().tolist())
            log.info(f'Loaded {len(done):,} already-done Filenames from manifest')
            return done
        except Exception as exc:
            log.warn(f'Could not read manifest: {exc}')
            return set()

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> ExtractionResult:
        """
        Extract spectrogram crops for all qualifying chirps and write PNGs
        plus a manifest CSV.

        Processing is parallelised across ``self.n_workers`` processes.
        Each worker calls :func:`_chirp_to_spectro` for one chirp and returns the
        image array; the main process writes the PNG and appends the
        manifest row.

        :return: :class:`SpectroExtractionResult` with summary statistics.
        """
        _t0 = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if not _SF_AVAILABLE:
            log.warn(
                'soundfile not installed — falling back to scipy.io.wavfile '
                '(full-file loads; slower). Install with: pip install soundfile'
            )

        # ── Load and filter work items ─────────────────────────────────
        work_df = self._load_work_items()
        n_input = len(work_df)

        # ── Incremental skip ──────────────────────────────────────────
        manifest_path = self.out_dir / 'manifest.csv'
        done_set      = self._load_done_set(manifest_path)
        if done_set:
            work_df = work_df[~work_df['Filename'].isin(done_set)]
            log.info(
                f'Skipping {len(done_set):,} already-done; '
                f'{len(work_df):,} remaining'
            )
        n_skipped = n_input - len(work_df)

        if work_df.empty:
            log.info('Nothing to do — all chirps already processed')
            return SpectroExtractionResult(
                n_chirps_input  = n_input,
                n_crops_written = 0,
                n_failed        = 0,
                n_skipped       = n_skipped,
                species_counts  = {},
                elapsed_secs    = time.perf_counter() - _t0,
                out_dir         = self.out_dir.resolve(),
            )

        # ── Create per-(date+site) subdirectories ────────────────────
        # Partition key is derived from the Filename stem:
        #   lake2_-20220427_192220_2secs  ->  20220427_lake2
        # This is stable regardless of species label quality.
        import re as _re
        _date_re = _re.compile(r'(\d{8})')
        _site_keywords = ('barn', 'lake2', 'lake', 'jasper', 'bats')

        def _partition_key(filename: str) -> str:
            """
            Return a stable ``YYYYMMDD_<site>`` partition key from a
            fragment Filename stem.

            :param filename: Fragment stem, e.g. ``lake2_-20220427_192220_2secs``.
            :return:         Partition key, e.g. ``20220427_lake2``.
            """
            m = _date_re.search(filename)
            date_part = m.group(1) if m else 'unknown'
            fn_lower  = filename.lower()
            site_part = next(
                (kw for kw in _site_keywords if kw in fn_lower), 'unknown'
            )
            return f'{date_part}_{site_part}'

        work_df = work_df.copy()
        work_df['_partition'] = work_df['Filename'].apply(_partition_key)
        for part in work_df['_partition'].unique():
            (self.out_dir / part).mkdir(exist_ok=True)

        # Per-partition counters for sequential PNG naming.
        # Load existing counts from manifest if incremental.
        partition_counters: dict[str, int] = {}
        if manifest_path.exists():
            try:
                mdf = pd.read_csv(manifest_path, usecols=['partition'])
                for pt, cnt in mdf['partition'].value_counts().items():
                    partition_counters[pt] = int(cnt)
            except Exception:
                pass

        # ── Open manifest for appending ───────────────────────────────
        manifest_exists = manifest_path.exists()
        manifest_fh = open(manifest_path, 'a', newline='')
        manifest_writer = csv.DictWriter(manifest_fh, fieldnames=[
            'crop_path', 'partition', 'species', 'species_prob',
            'file_id', 'Filename', 'time_in_file_ms', 'match_quality',
        ])
        if not manifest_exists:
            manifest_writer.writeheader()

        # ── Parallel extraction ───────────────────────────────────────
        n_written = 0
        n_failed  = 0
        partition_counts: dict[str, int] = dict(partition_counters)

        rows = work_df.to_dict('records')
        total = len(rows)
        log.info(
            f'Extracting {total:,} chirps using {self.n_workers} workers ...'
        )

        pbar = tqdm(total=total, unit='chirp') if _TQDM else None

        # Submit in batches to avoid holding all futures in memory.
        batch_size = self.n_workers * 8

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            for batch_start in range(0, total, batch_size):
                batch = rows[batch_start: batch_start + batch_size]
                futures = {
                    pool.submit(
                        _chirp_to_spectro,
                        row['matched_wav'],
                        float(row['TimeInFile']),
                        self.window_ms,
                        self.freq_lo_hz,
                        self.freq_hi_hz,
                        self.img_size,
                        self.dynamic_range,
                    ): row
                    for row in batch
                }

                for fut in as_completed(futures):
                    row = futures[fut]
                    sp  = row['species']
                    try:
                        img_array = fut.result()
                    except Exception as exc:
                        log.warn(f'Worker error for {row["Filename"]}: {exc}')
                        img_array = None

                    if img_array is None:
                        n_failed += 1
                    else:
                        part = row['_partition']
                        idx  = partition_counts.get(part, 0) + 1
                        partition_counts[part] = idx
                        fname    = f'{idx:08d}.png'
                        out_path = self.out_dir / part / fname
                        try:
                            Image.fromarray(img_array, mode='L').save(out_path)
                            manifest_writer.writerow({
                                'crop_path'      : str(out_path),
                                'partition'      : part,
                                'species'        : sp,
                                'species_prob'   : row.get('species_prob', ''),
                                'file_id'        : row.get('file_id', ''),
                                'Filename'       : row['Filename'],
                                'time_in_file_ms': row['TimeInFile'],
                                'match_quality'  : row.get('match_quality', ''),
                            })
                            n_written += 1
                        except Exception as exc:
                            log.warn(f'Could not save {out_path}: {exc}')
                            n_failed += 1

                    if pbar:
                        pbar.update(1)

                manifest_fh.flush()

        if pbar:
            pbar.close()
        manifest_fh.close()

        # ── Write config ──────────────────────────────────────────────
        elapsed = time.perf_counter() - _t0
        pd.DataFrame([
            {'parameter': 'feather_path',   'value': str(self.feather_path)},
            {'parameter': 'match_csv',      'value': str(self.match_csv)},
            {'parameter': 'min_prob',       'value': self.min_prob},
            {'parameter': 'window_ms',      'value': self.window_ms},
            {'parameter': 'freq_lo_khz',    'value': self.freq_lo_hz / 1000},
            {'parameter': 'freq_hi_khz',    'value': self.freq_hi_hz / 1000},
            {'parameter': 'img_size',       'value': self.img_size},
            {'parameter': 'dynamic_range',  'value': self.dynamic_range},
            {'parameter': 'n_workers',      'value': self.n_workers},
            {'parameter': 'match_quality',  'value': str(list(self.match_quality))},
            {'parameter': 'n_chirps_input', 'value': n_input},
            {'parameter': 'n_crops_written','value': n_written},
            {'parameter': 'n_failed',       'value': n_failed},
            {'parameter': 'n_skipped',      'value': n_skipped},
            {'parameter': 'elapsed_secs',   'value': round(elapsed, 1)},
        ]).to_csv(self.out_dir / 'extractor_config.csv', index=False)

        # Final species counts from manifest (label lives in manifest only).
        try:
            mdf_final = pd.read_csv(manifest_path, usecols=['species'])
            final_counts = mdf_final['species'].value_counts().to_dict()
        except Exception:
            final_counts = {}

        return SpectroExtractionResult(
            n_chirps_input  = n_input,
            n_crops_written = n_written,
            n_failed        = n_failed,
            n_skipped       = n_skipped,
            species_counts  = final_counts,
            elapsed_secs    = elapsed,
            out_dir         = self.out_dir.resolve(),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments for :class:`ChirpSpectroExtractor`.

    :return: ``argparse.Namespace`` with validated attributes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='chirps_to_spectros',
        description=(
            'Extract per-chirp spectrogram crops from full bat recordings\n'
            'for CNN species classification training.\n\n'
            'Requires the feather file from sono_batch_processing.py and\n'
            'the match_report.csv from wav_path_resolver.py.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--feather',
        required=True,
        metavar='PATH',
        help='Chirp-level feather file (sonobat3_2_species_ids.feather).',
    )
    parser.add_argument(
        '--matches',
        required=True,
        metavar='PATH',
        help='match_report.csv from wav_path_resolver.py.',
    )
    parser.add_argument(
        '-o', '--out-dir',
        required=True,
        metavar='DIR',
        help='Output directory for PNG crops and manifest.',
    )
    parser.add_argument(
        '--min-prob',
        type=float,
        default=_DEFAULT_MIN_PROB,
        metavar='F',
        help=f'Minimum species_prob to include (default: {_DEFAULT_MIN_PROB}).',
    )
    parser.add_argument(
        '--window-ms',
        type=float,
        default=_DEFAULT_WINDOW_MS,
        metavar='MS',
        help=f'Crop window in ms, centred on chirp onset (default: {_DEFAULT_WINDOW_MS}).',
    )
    parser.add_argument(
        '--freq-lo',
        type=float,
        default=_DEFAULT_FREQ_LO_KHZ,
        metavar='KHZ',
        help=f'Lower frequency bound in kHz (default: {_DEFAULT_FREQ_LO_KHZ}).',
    )
    parser.add_argument(
        '--freq-hi',
        type=float,
        default=_DEFAULT_FREQ_HI_KHZ,
        metavar='KHZ',
        help=f'Upper frequency bound in kHz (default: {_DEFAULT_FREQ_HI_KHZ}).',
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=_DEFAULT_IMG_SIZE,
        metavar='PX',
        help=f'Output PNG size in pixels, square (default: {_DEFAULT_IMG_SIZE}).',
    )
    parser.add_argument(
        '--dynamic-range',
        type=float,
        default=_DEFAULT_DYNAMIC_RANGE,
        metavar='DB',
        help=f'Log-power normalisation range in dB (default: {_DEFAULT_DYNAMIC_RANGE}).',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=_DEFAULT_WORKERS,
        metavar='N',
        help=f'Parallel worker processes (default: {_DEFAULT_WORKERS}).',
    )
    parser.add_argument(
        '--window-only',
        action='store_true',
        help=(
            'Only use chirps with match_quality="window".\n'
            'Default: accept both "window" and "nearest" matches.'
        ),
    )

    args = parser.parse_args()

    for attr, label in [('feather', 'feather'), ('matches', 'match CSV')]:
        p = Path(getattr(args, attr))
        if not p.exists():
            parser.error(f'{label} not found: {p}')

    args.feather  = Path(args.feather)
    args.matches  = Path(args.matches)
    args.out_dir  = Path(args.out_dir)
    args.match_quality = ['window'] if args.window_only else ['window', 'nearest']
    return args


def main() -> None:
    """
    CLI entry point for :class:`ChirpSpectroExtractor`.
    """
    args = _parse_args()

    extractor = ChirpSpectroExtractor(
        feather_path  = args.feather,
        match_csv     = args.matches,
        out_dir       = args.out_dir,
        min_prob      = args.min_prob,
        window_ms     = args.window_ms,
        freq_lo_khz   = args.freq_lo,
        freq_hi_khz   = args.freq_hi,
        img_size      = args.img_size,
        dynamic_range = args.dynamic_range,
        n_workers     = args.n_workers,
        match_quality = args.match_quality,
    )

    result = extractor.run()
    log.info(result.summary())
    sys.exit(0 if result.n_crops_written > 0 else 1)


# ------------------- Main Section --------------
if __name__ == '__main__':
    main() 