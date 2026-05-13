#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-03-15 09:46:12
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/species_classification/chirps_to_spectros.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-05-13 11:55:00
# **********************************************************

# NOTE: we made some changes to this code after running it
#       to produce 12M .png files, which took a long time. 
#       So, when running again down the line, there *might*
#       be issues.

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
``bats_<timestamp>.parquet``
    Clean chirp-level dataset produced by ``sb_measures_postprocessing.py``.
    Must contain: ``file_id``, ``TimeInFile``, ``species``, ``confidence``.
    The ``Filename`` column (fragment stem) is read directly from the parquet;
    feather files from the older ``sono_batch_processing.py`` pipeline are
    also accepted (column ``species_prob`` is mapped to ``confidence``).

Fragment ``.wav`` directories
    Directories containing the 2-second fragment ``.wav`` files produced
    by SonoBat's Long File Parser (passed via ``--fragment-dirs``).
    These are at true 250 kHz (TE=1) and provide adequate sample density
    for high-resolution spectrograms.  ``TimeInFile`` from the parquet
    gives the chirp onset within each fragment.

Pipeline per chirp
------------------
1. Resolve the ``Filename`` stem to its 2-second fragment ``.wav`` path
   by walking ``--fragment-dirs``.
2. Seek to ``TimeInFile`` ms within that fragment using ``soundfile``.
   Fragment files are at true 250 kHz (TE=1), giving adequate sample
   density for high-resolution spectrograms.
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
``crop_path``, ``species``, ``confidence``, ``file_id``, ``Filename``,
``time_in_file_ms``, ``match_quality``

Typical usage
-------------
::

    python chirps_to_spectros.py \\
        --data    /qnap/bats/all_data/bats_<timestamp>.parquet \\
        --out-dir /data2/bat_crops \\
        --min-conf 0.80 \\
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
from sonobat_utils.wav_file_info import WavInfo, RecordingType
from sonobat_utils.utils import Utils

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PRE_MS:        float = 3.0    # ms before chirp onset
_DEFAULT_POST_MS:       float = 17.0   # ms after chirp onset (captures FM sweep tail)
_DEFAULT_FREQ_LO_KHZ:  float = 15.0
_DEFAULT_FREQ_HI_KHZ:  float = 80.0
_DEFAULT_IMG_SIZE:      int   = 224
_DEFAULT_MIN_CONF:      float = 0.80
_DEFAULT_DYNAMIC_RANGE: float = 40.0   # dB — tighter range suppresses noise floor
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
    wav_path:           str,
    time_ms:            float,
    pre_ms:             float,
    post_ms:            float,
    freq_lo_hz:         float,
    freq_hi_hz:         float,
    img_size:           int,
    dynamic_range:      float,
    pcen:                  bool  = False,
    pcen_time_constant:    float = 0.1,
    pcen_snr_threshold_db: float = 18.0,
) -> Optional[np.ndarray]:
    """
    Load a short audio window from a full recording and return a normalised
    spectrogram image as a uint8 numpy array of shape
    ``(img_size, img_size)``.

    The window runs from ``time_ms - pre_ms`` to ``time_ms + post_ms``.
    Asymmetric defaults (3ms pre, 17ms post) capture the full FM sweep
    descent that follows chirp onset without admitting excess pre-call noise.

    Two normalisation modes are available:

    **Log-power (default, pcen=False)**
        Anchors on the peak-power frame in the bat band.  Prevents noise
        spikes elsewhere in the window from compressing the call signal
        into the gray midrange.

    **PCEN (pcen=True)**
        Per-Channel Energy Normalization via ``librosa.pcen``.  Applies an
        adaptive gain control and non-linear compression per frequency bin,
        using a running average to estimate and suppress the stationary
        background noise floor.  Particularly effective for recordings with
        heavy background hum or broadband environmental noise.  The
        ``pcen_time_constant`` controls the time scale (seconds) of the
        adaptive filter — shorter values track faster-changing noise floors.
        Bat calls are short transients so 0.1 s works well; the librosa
        default (0.395 s) is tuned for bird calls and is too slow here.

    Uses ``soundfile`` for seek-based partial reads when available,
    falling back to ``scipy.io.wavfile`` (full file load) otherwise.

    :param wav_path:           Path to the full-recording ``.wav`` file.
    :param time_ms:            Chirp onset within the recording (ms).
    :param pre_ms:             Ms of audio before chirp onset.
    :param post_ms:            Ms of audio after chirp onset.
    :param freq_lo_hz:         Lower frequency bound for spectrogram (Hz).
    :param freq_hi_hz:         Upper frequency bound for spectrogram (Hz).
    :param img_size:           Output image size in pixels (square).
    :param dynamic_range:      Log-power normalisation range (dB).  Used
                               only when ``pcen=False``.
    :param pcen:                  If ``True`` enable adaptive PCEN: PCEN is
                               applied only when the recording SNR is below
                               ``pcen_snr_threshold_db``; cleaner recordings
                               use log-power normalisation.  Requires
                               ``librosa``.
    :param pcen_time_constant: Time constant (s) for the PCEN adaptive
                               filter.  Default: 0.1 s.
    :param pcen_snr_threshold_db: SNR threshold (dB) below which PCEN is
                               applied when ``pcen=True``.  SNR is computed
                               as peak / median power across the bat-band
                               spectrogram before normalisation.  Recordings
                               with SNR >= this value are considered clean
                               and use log-power normalisation instead.
                               Default: 18.0 dB.
    :return:                   uint8 array of shape ``(img_size, img_size)``,
                               or ``None`` on any error.
    """
    try:
        # ── Determine recording type (TE/DR) via WavInfo ──────────────
        info     = WavInfo.from_path(wav_path)
        te       = info.te
        sr_true  = info.sr_true

        if not info.is_ultrasonic:
            return None   # FD or unknown — cannot produce valid spectrogram

        # ── Seek and load in file-domain frames ───────────────────────
        # time_ms is in true milliseconds; convert to file-domain frames.
        start_fr = info.true_ms_to_file_frame(time_ms - pre_ms)
        n_frames = info.true_ms_to_file_frames_count(pre_ms + post_ms)

        if _SF_AVAILABLE:
            with sf.SoundFile(wav_path) as f:
                n_frames = min(n_frames, info.file_frames - start_fr)
                if n_frames < 4:
                    return None
                f.seek(start_fr)
                data = f.read(n_frames, dtype='float32', always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                audio = data
        else:
            sr_raw, raw = wavfile.read(wav_path)
            if raw.ndim > 1:
                raw = raw.mean(axis=1)
            if raw.dtype == np.int16:
                raw = raw.astype(np.float32) / 32768.0
            elif raw.dtype == np.int32:
                raw = raw.astype(np.float32) / 2_147_483_648.0
            else:
                raw = raw.astype(np.float32)
            end_fr = min(len(raw), start_fr + n_frames)
            audio  = raw[start_fr:end_fr]

        if len(audio) < 4:
            return None

        # ── Spectrogram in file domain, scaled to true domain ─────────
        # Run spectrogram with sr_header so scipy's frame/freq counts
        # match the loaded samples, then scale freq axis by te.
        sr_header  = info.sr_header
        nperseg    = min(len(audio) // 2,
                         max(64, int(round(_STFT_WINDOW_MS / 1000.0 * sr_true))))
        nperseg    = max(8, nperseg)
        noverlap   = nperseg * 3 // 4
        freqs_file, _, Sxx = signal.spectrogram(
            audio, fs=sr_header,
            window='hann', nperseg=nperseg, noverlap=noverlap,
            scaling='spectrum',
        )
        # Scale file-domain frequencies to true frequencies.
        freqs = freqs_file * te

        # Restrict to target frequency band.
        band = (freqs >= freq_lo_hz) & (freqs <= freq_hi_hz)
        if band.sum() < 2:
            return None
        Sxx_band = Sxx[band, :]

        # ── SNR estimate for adaptive PCEN switching ─────────────────
        # Computed on the raw linear power spectrogram before any
        # normalisation, giving a physics-based measure of recording quality
        # that is independent of normalisation choices.
        #
        # SNR = peak_power / median_power across all (freq, time) bins.
        # A high ratio means a strong call above a quiet floor (clean).
        # A low ratio means background competes with the call (noisy).
        # Converting to dB: SNR_db = 10 * log10(peak / median).
        #
        # Empirical threshold from sample comparison:
        #   clean recordings (ref_mean_DN < 40):  SNR typically > 20 dB
        #   noisy recordings (ref_mean_DN > 80):  SNR typically < 15 dB
        # A threshold of 18 dB gives a clean gap between the two populations.
        peak_power  = float(Sxx_band.max())
        median_power = float(np.median(Sxx_band))
        if peak_power <= 0:
            return None
        snr_db = 10.0 * np.log10(
            peak_power / (median_power + 1e-12)
        )
        # Decide whether to use PCEN: apply it when requested AND the
        # recording is noisy enough to benefit from adaptive gain control.
        # On clean recordings PCEN can boost background streaks relative to
        # the call (confirmed empirically on Laci crops); log-power is safer.
        use_pcen = pcen and (snr_db < pcen_snr_threshold_db)

        # ── Normalisation ─────────────────────────────────────────────
        if use_pcen:
            # PCEN: Per-Channel Energy Normalization.
            # Applies adaptive gain control and non-linear compression per
            # frequency bin using a running average to estimate and suppress
            # the stationary background noise floor.  Applied only when the
            # recording SNR is below pcen_snr_threshold_db, i.e. the
            # background is loud enough that PCEN's adaptive suppression
            # genuinely helps rather than hurting clean calls.
            #
            # librosa.pcen expects power spectrogram values in a range
            # comparable to integer-PCM amplitudes (~2^31).  Sxx_band from
            # scipy.signal.spectrogram is in units of (amplitude)^2 / Hz
            # when scaling='spectrum'; scale up before passing to PCEN.
            import librosa
            hop_length = nperseg - noverlap
            Sxx_pcen = librosa.pcen(
                Sxx_band * (2 ** 31),
                sr            = sr_true,
                hop_length    = hop_length,
                time_constant = pcen_time_constant,
                gain          = 0.98,
                bias          = 2.0,
                power         = 0.5,
                b             = None,   # auto-compute from time_constant + sr
                eps           = 1e-6,
            )
            # PCEN output is in [0, ~bias^power] ≈ [0, ~1.4]; normalise to
            # [0, 255].  Use the 99th percentile as ceiling to avoid a single
            # noise spike compressing the whole image.
            ceil = float(np.percentile(Sxx_pcen, 99))
            if ceil <= 0:
                return None
            Sxx_norm = np.clip(Sxx_pcen / ceil * 255.0, 0, 255).astype(np.uint8)
        else:
            # Log-power normalisation anchored on call peak.
            # Used for clean recordings (SNR >= pcen_snr_threshold_db) and
            # whenever pcen=False.  Anchoring on the peak-power frame
            # prevents noise spikes elsewhere in the window from compressing
            # the call signal into the gray midrange.
            peak_frame = int(np.argmax(Sxx_band.sum(axis=0)))
            peak_power_frame = float(Sxx_band[:, peak_frame].max())
            if peak_power_frame <= 0:
                return None
            db_ref   = 10.0 * np.log10(peak_power_frame + 1e-12)
            eps      = 1e-12
            Sxx_db   = 10.0 * np.log10(Sxx_band + eps)
            db_floor = db_ref - dynamic_range
            Sxx_clip = np.clip(Sxx_db, db_floor, db_ref)
            Sxx_norm = ((Sxx_clip - db_floor) / dynamic_range * 255.0).astype(np.uint8)

        # Frequency axis: flip so low frequencies are at the image bottom
        # (PIL origin is top-left).
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

    Reads chirp metadata from a ``.parquet`` file produced by
    ``sb_measures_postprocessing.py`` (or a legacy ``.feather`` file),
    resolves each chirp's source ``.wav`` fragment by stem, extracts a
    short audio window around the chirp onset, computes a linear-scale
    spectrogram, and saves the result as a PNG under
    ``<out_dir>/<YYYYMMDD>_<site>/``.

    Output is incremental: a manifest CSV tracks which chirps have already
    been processed so re-runs skip completed work.

    :param data_path:       Path to the parquet (or feather) file.
    :param out_dir:         Root output directory for PNG crops.
    :param min_conf:        Minimum ``confidence`` score to include a chirp.
    :param pre_ms:          Ms of audio before chirp onset (default 3ms).
    :param post_ms:         Ms of audio after chirp onset (default 17ms).
    :param freq_lo_khz:     Lower frequency bound for spectrogram (kHz).
    :param freq_hi_khz:     Upper frequency bound for spectrogram (kHz).
    :param img_size:        Output PNG size in pixels (square).
    :param dynamic_range:   Log-power normalisation range in dB.
    :param n_workers:       Parallel worker processes.
    :param match_quality:   Accepted match quality values (legacy feather
                            only; ignored when reading parquet).
    :param sample:          If > 0, stop after writing this many crops.
                            Useful for quick visual inspection before a
                            full run.  Combines with ``sample_species``
                            and ``sample_partition``.
    :param sample_species:  Restrict sampling to these species codes
                            (e.g. ``['Myca', 'Tabr']``).  Ignored when
                            ``sample == 0``.
    :param sample_partition: Restrict sampling to this partition key
                            (e.g. ``'20220706_bats'``).  Ignored when
                            ``sample == 0``.
    """

    def __init__(
        self,
        data_path:      str | Path,
        out_dir:        str | Path,
        fragment_dirs:  Sequence[str | Path] = (),
        min_conf:       float          = _DEFAULT_MIN_CONF,
        pre_ms:         float          = _DEFAULT_PRE_MS,
        post_ms:        float          = _DEFAULT_POST_MS,
        freq_lo_khz:    float          = _DEFAULT_FREQ_LO_KHZ,
        freq_hi_khz:    float          = _DEFAULT_FREQ_HI_KHZ,
        img_size:       int            = _DEFAULT_IMG_SIZE,
        dynamic_range:      float          = _DEFAULT_DYNAMIC_RANGE,
        n_workers:          int            = _DEFAULT_WORKERS,
        pcen:                  bool           = False,
        pcen_time_constant:    float          = 0.1,
        pcen_snr_threshold_db: float          = 18.0,
        filename_map_path:     Optional[Path] = None,
        match_quality:      Sequence[str]  = ('window',),
        sample:             int            = 0,
        sample_species:     Sequence[str]  = (),
        sample_partition:   str            = '',
    ) -> None:
        self.data_path     = Path(data_path)
        self.out_dir       = Path(out_dir)
        self.fragment_dirs = [Path(d) for d in fragment_dirs]
        self.min_conf      = min_conf
        self.pre_ms        = pre_ms
        self.post_ms       = post_ms
        self.freq_lo_hz    = freq_lo_khz * 1000.0
        self.freq_hi_hz    = freq_hi_khz * 1000.0
        self.img_size      = img_size
        self.dynamic_range = dynamic_range
        self.n_workers     = n_workers
        self.pcen                  = pcen
        self.pcen_time_constant    = pcen_time_constant
        self.pcen_snr_threshold_db = pcen_snr_threshold_db
        self.filename_map_path     = Path(filename_map_path) if filename_map_path else None
        self.match_quality      = set(match_quality)
        self.sample             = sample
        self.sample_species     = list(sample_species)
        self.sample_partition   = sample_partition

    # ------------------------------------------------------------------ #
    #  Fragment index                                                     #
    # ------------------------------------------------------------------ #

    def _build_fragment_index(self) -> dict[str, str]:
        """
        Walk ``self.fragment_dirs`` and build a mapping from fragment stem
        to absolute ``.wav`` path.

        Fragment stems take the form ``<prefix>-<YYYYMMDD>_<HHMMSS>_2secs``
        and are identical to the ``Filename`` values in the feather file.

        :return: Dict mapping stem string → absolute path string.
        """
        index: dict[str, str] = {}
        for base in self.fragment_dirs:
            if not base.exists():
                log.warn(f'Fragment dir not found: {base}')
                continue
            for p in base.rglob('*.wav'):
                index[p.stem] = str(p.resolve())
        log.info(
            f'Fragment index: {len(index):,} stems from '
            f'{len(self.fragment_dirs)} director(ies)'
        )
        return index

    # ------------------------------------------------------------------ #
    #  Data loading and preparation                                       #
    # ------------------------------------------------------------------ #

    def _load_work_items(self) -> pd.DataFrame:
        """
        Load the parquet (or legacy feather) file, apply filters, and return
        a DataFrame of chirps ready for crop extraction.

        Accepts both the new parquet format from ``sb_measures_postprocessing.py``
        (confidence column) and the legacy feather format from
        ``sono_batch_processing.py`` (species_prob column, mapped to confidence).

        Filters applied:

        * ``species`` is not NaN and is a clean 4-char or slash-composite code
        * ``confidence`` >= ``self.min_conf``
        * fragment wav resolved from ``Filename`` stem via ``--fragment-dirs``
        * ``match_quality`` filter applied only when the column is present
          (legacy feather only)

        :return: DataFrame with columns ``Filename``, ``TimeInFile``,
                 ``species``, ``confidence``, ``file_id``, ``fragment_wav``,
                 and optional legacy columns ``matched_wav``, ``match_quality``,
                 ``TimeInOrigRecording``.
        """
        log.info(f'Loading data: {self.data_path} ...')

        # Parquet files from sb_measures_postprocessing.py can exceed PyArrow's
        # default thrift size limits due to the large file_map in schema metadata.
        # Read via ParquetFile directly with raised limits; feather uses Utils.
        import pyarrow.parquet as _pq
        import json as _json
        _THRIFT_LIMIT = 1_000_000_000

        if self.data_path.suffix in ('.parquet', '.pq'):
            _pf  = _pq.ParquetFile(
                self.data_path,
                thrift_string_size_limit    = _THRIFT_LIMIT,
                thrift_container_size_limit = _THRIFT_LIMIT,
            )
            df = _pf.read().to_pandas()
        else:
            df = Utils.read_df_file(self.data_path)

        # Legacy feather compatibility: map species_prob → confidence.
        if 'species_prob' in df.columns and 'confidence' not in df.columns:
            df = df.rename(columns={'species_prob': 'confidence'})
            log.info('  Legacy feather: renamed species_prob → confidence')

        # Parquet format: derive Filename from the file_map embedded in the
        # parquet metadata.  The file_map values are fragment stems (set by
        # sb_measures_postprocessing._collect_all_raw path normalization).
        # Legacy feather already carries a Filename column directly.
        #
        # Fallback: if the parquet has no bats_metadata (e.g. the measures
        # parquet bats_<timestamp>.parquet which was not written via
        # BatsData.to_parquet()), use the --filename-map CSV instead.
        # That CSV must have columns file_id and Filename (available from
        # the manifest at <out-dir>/manifest.csv of any prior run).
        if 'Filename' not in df.columns:
            # Try embedded file_map first.
            _schema   = _pq.read_schema(self.data_path, memory_map=True)
            _meta_raw = _schema.metadata or {}
            _meta_key = b'bats_metadata'

            if _meta_key in _meta_raw:
                _file_map = {
                    int(k): v
                    for k, v in _json.loads(
                        _meta_raw[_meta_key].decode()
                    )['file_map'].items()
                }
                log.info('  Filename derived from embedded parquet file_map')
            elif self.filename_map_path is not None:
                # Fallback: load file_id → Filename from the supplied CSV.
                log.info(
                    f'  No bats_metadata in parquet; loading Filename map ' 
                    f'from {self.filename_map_path}'
                )
                _fmap_df  = pd.read_csv(
                    self.filename_map_path,
                    usecols   = ['file_id', 'Filename'],
                    low_memory= False,
                ).dropna(subset=['Filename']).drop_duplicates('file_id')
                _file_map = dict(
                    zip(_fmap_df['file_id'].astype(int),
                        _fmap_df['Filename'].astype(str))
                )
                log.info(f'  Filename map loaded: {len(_file_map):,} file_ids')
            else:
                raise KeyError(
                    f'{self.data_path} has no bats_metadata and no '
                    f'--filename-map was supplied.  Either use a parquet '
                    f'written by BatsData.to_parquet(), or pass '
                    f'--filename-map <manifest.csv> to provide the '
                    f'file_id → Filename mapping.'
                )

            df['Filename'] = df['file_id'].map(_file_map)
            n_unmapped = df['Filename'].isna().sum()
            if n_unmapped:
                log.warn(
                    f'  {n_unmapped:,} rows have a file_id absent from '
                    f'the file_map and will be dropped.'
                )
                df = df[df['Filename'].notna()].copy()
            log.info(f'  Filename mapped for {len(df):,} rows')

        # Strip Windows CRLF artefacts from string columns.
        for _col in ('Filename', 'species', 'confidence'):
            if _col in df.columns and df[_col].dtype == object:
                df[_col] = df[_col].astype(str).str.strip()
        log.info(f'  {len(df):,} total chirp rows')

        # ── Build fragment index and join fragment paths ──────────────
        if self.fragment_dirs:
            frag_index = self._build_fragment_index()
            if not frag_index:
                log.err(
                    'Fragment index is empty — none of the --fragment-dirs '
                    'exist or contain .wav files.  Check the paths and retry.'
                )
                sys.exit(1)
            df['fragment_wav'] = df['Filename'].map(
                lambda s: frag_index.get(str(s))
            )
            merged = df[df['fragment_wav'].notna()].copy()
            log.info(
                f'  {len(merged):,} rows with resolved fragment wav'
            )
            if merged.empty:
                log.err(
                    'No chirps could be matched to fragment WAV files. '
                    'Check that --fragment-dirs contains the 2-second '
                    'fragment .wav files whose stems match the Filename '
                    'column (e.g. lake2_-20220706_000013_2secs.wav).'
                )
                sys.exit(1)
        else:
            log.warn(
                'No --fragment-dirs supplied.  '
                'Pass fragment .wav directories for crop extraction.'
            )
            sys.exit(1)

        # Species filter: clean 4-char code or slash-composite (e.g. Laci/Lano).
        import re
        _sp_re = re.compile(r'^[A-Z][a-z]{3}(/[A-Z][a-z]{3})*$')
        merged = merged[merged['species'].notna()]
        merged = merged[merged['species'].apply(
            lambda s: bool(_sp_re.match(str(s)))
        )]
        log.info(f'  {len(merged):,} rows with valid species code')

        # Confidence filter.
        if 'confidence' not in merged.columns:
            log.warn(
                'No confidence column found — skipping confidence filter. '
                'All chirps will be included regardless of confidence score.'
            )
        else:
            conf_ok = merged['confidence'].notna() & \
                      (merged['confidence'] >= self.min_conf)
            merged = merged[conf_ok]
        log.info(
            f'  {len(merged):,} rows after confidence filter '
            f'(min_conf={self.min_conf})'
        )

        # match_quality filter: present in legacy feather only; parquet rows
        # have NaN here and pass through unchanged.
        if 'match_quality' in merged.columns and self.match_quality:
            mq = merged['match_quality']
            merged = merged[mq.isna() | mq.isin(self.match_quality)]
            log.info(
                f'  {len(merged):,} rows after match_quality filter '
                f'(accepted: {sorted(self.match_quality)} + NaN)'
            )

        needed_cols = ['Filename', 'TimeInFile', 'TimeInOrigRecording',
                       'species', 'confidence', 'file_id', 'chirp_idx',
                       'fragment_wav', 'matched_wav', 'match_quality']
        # Optional columns absent from parquet — fill with NA so downstream
        # manifest writing works without branching.
        for opt in ('matched_wav', 'match_quality', 'TimeInOrigRecording', 'chirp_idx'):
            if opt not in merged.columns:
                merged[opt] = pd.NA
        available = [c for c in needed_cols if c in merged.columns]
        return merged[available].reset_index(drop=True)

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

    def run(self) -> SpectroExtractionResult:
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

        # ── Sample-mode filters ───────────────────────────────────────
        # Applied before the incremental skip so the sample reflects the
        # requested subset, not whatever happens to be unprocessed.
        if self.sample > 0:
            if self.sample_partition:
                # Assign partition keys temporarily for filtering.
                import re as _re2
                _dr2 = _re2.compile(r'(\d{8})')
                _sk2 = ('barn', 'lake2', 'lake', 'jasper', 'bats')
                def _pk2(fn):
                    m = _dr2.search(fn)
                    d = m.group(1) if m else 'unknown'
                    s = next((k for k in _sk2 if k in fn.lower()), 'unknown')
                    return f'{d}_{s}'
                work_df = work_df[
                    work_df['Filename'].apply(_pk2) == self.sample_partition
                ].copy()
                log.info(
                    f'Sample partition filter: {len(work_df):,} rows '
                    f'match partition "{self.sample_partition}"'
                )
            if self.sample_species:
                work_df = work_df[
                    work_df['species'].isin(self.sample_species)
                ].copy()
                log.info(
                    f'Sample species filter: {len(work_df):,} rows '
                    f'match species {self.sample_species}'
                )
            if len(work_df) > self.sample:
                work_df = work_df.sample(
                    n=self.sample, random_state=42
                ).copy()
            log.info(
                f'Sample mode: will extract {len(work_df):,} crops then stop'
            )
            n_input = len(work_df)   # recount after sample filter

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
        # For incremental runs load existing counts from manifest so PNG
        # numbering continues correctly.  For fresh runs start at zero.
        partition_counters: dict[str, int] = {}
        if done_set and manifest_path.exists():
            # Incremental run: resume numbering from existing manifest.
            try:
                mdf = pd.read_csv(manifest_path, usecols=['partition'])
                for pt, cnt in mdf['partition'].value_counts().items():
                    partition_counters[pt] = int(cnt)
            except Exception:
                pass

        # ── Open manifest for writing (temp → atomic rename on success) ──
        # Writing to a temp file and renaming on completion prevents a
        # stale or corrupt manifest from a prior run being appended to,
        # which would mix PNG references across two different runs.
        _MANIFEST_FIELDS = [
            'crop_path', 'partition', 'species', 'confidence',
            'file_id', 'chirp_idx', 'harmonic_idx', 'Filename',
            'time_in_orig_rec_ms', 'time_in_file_ms', 'match_quality',
            'matched_wav',
        ]
        is_incremental = bool(done_set)
        if is_incremental:
            # Append mode: continue an existing manifest.
            manifest_fh = open(manifest_path, 'a', newline='')
            manifest_writer = csv.DictWriter(manifest_fh, fieldnames=_MANIFEST_FIELDS)
            manifest_tmp_path = None  # no rename needed
        else:
            # Fresh run: write to a temp file, rename atomically at the end.
            manifest_tmp_path = manifest_path.with_suffix('.tmp')
            manifest_fh = open(manifest_tmp_path, 'w', newline='')
            manifest_writer = csv.DictWriter(manifest_fh, fieldnames=_MANIFEST_FIELDS)
            manifest_writer.writeheader()

        # ── Parallel extraction ───────────────────────────────────────
        n_written = 0
        n_failed  = 0
        partition_counts: dict[str, int] = dict(partition_counters)
        # harmonic_idx: counts how many crops have already been written for
        # each (file_id, chirp_idx) pair.  First write → 0, second → 1, etc.
        # Populated from the existing manifest on incremental runs so that
        # new harmonics continue the correct index rather than restarting at 0.
        harmonic_counts: dict[tuple, int] = {}
        if done_set and manifest_path.exists():
            try:
                _hdf = pd.read_csv(
                    manifest_path,
                    usecols=['file_id', 'chirp_idx', 'harmonic_idx'],
                    low_memory=False,
                )
                for _, r in _hdf[
                    (_hdf['file_id'] >= 0) & (_hdf['chirp_idx'] >= 0)
                ].iterrows():
                    key = (int(r['file_id']), int(r['chirp_idx']))
                    harmonic_counts[key] = max(
                        harmonic_counts.get(key, -1),
                        int(r.get('harmonic_idx', 0))
                    )
                # Convert max-seen to next-to-assign (max + 1)
                harmonic_counts = {k: v + 1 for k, v in harmonic_counts.items()}
            except Exception:
                pass

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
                        row['fragment_wav'],
                        float(row['TimeInFile']),
                        self.pre_ms,
                        self.post_ms,
                        self.freq_lo_hz,
                        self.freq_hi_hz,
                        self.img_size,
                        self.dynamic_range,
                        self.pcen,
                        self.pcen_time_constant,
                        self.pcen_snr_threshold_db,
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
                            fid      = row.get('file_id', -1)
                            cidx     = row.get('chirp_idx', -1)
                            hkey     = (int(fid), int(cidx)) if fid != -1 and cidx != -1 else None
                            hidx     = harmonic_counts.get(hkey, 0) if hkey else -1
                            if hkey:
                                harmonic_counts[hkey] = hidx + 1
                            manifest_writer.writerow({
                                'crop_path'          : str(out_path),
                                'partition'          : part,
                                'species'            : sp,
                                'confidence'         : row.get('confidence', ''),
                                'file_id'            : fid,
                                'chirp_idx'          : cidx,
                                'harmonic_idx'       : hidx,
                                'Filename'           : row['Filename'],
                                'time_in_orig_rec_ms': row.get('TimeInOrigRecording', ''),
                                'time_in_file_ms'    : row['TimeInFile'],
                                'match_quality'      : row.get('match_quality', ''),
                                'matched_wav'        : row.get('matched_wav', ''),
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

        # Atomically promote temp manifest to final path (fresh runs only).
        if manifest_tmp_path is not None:
            manifest_tmp_path.replace(manifest_path)

        # ── Write config ──────────────────────────────────────────────
        elapsed = time.perf_counter() - _t0
        pd.DataFrame([
            {'parameter': 'data_path',      'value': str(self.data_path)},
            {'parameter': 'min_conf',       'value': self.min_conf},
            {'parameter': 'pre_ms',         'value': self.pre_ms},
            {'parameter': 'post_ms',        'value': self.post_ms},
            {'parameter': 'freq_lo_khz',    'value': self.freq_lo_hz / 1000},
            {'parameter': 'freq_hi_khz',    'value': self.freq_hi_hz / 1000},
            {'parameter': 'img_size',       'value': self.img_size},
            {'parameter': 'dynamic_range',  'value': self.dynamic_range},
            {'parameter': 'pcen',                  'value': self.pcen},
            {'parameter': 'pcen_time_constant',    'value': self.pcen_time_constant},
            {'parameter': 'pcen_snr_threshold_db', 'value': self.pcen_snr_threshold_db},
            {'parameter': 'n_workers',       'value': self.n_workers},
            {'parameter': 'match_quality',   'value': str(list(self.match_quality))},
            {'parameter': 'sample',          'value': self.sample},
            {'parameter': 'sample_species',  'value': str(self.sample_species)},
            {'parameter': 'sample_partition','value': self.sample_partition},
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
            'Primary input is the bats_<timestamp>.parquet produced by\n'
            'sb_measures_postprocessing.py.  Legacy .feather files from\n'
            'sono_batch_processing.py are also accepted.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--data',
        required=True,
        metavar='PATH',
        help=(
            'Parquet or feather file of chirp-level data.\n'
            'Parquet: bats_<timestamp>.parquet from sb_measures_postprocessing.py\n'
            'Feather: legacy sonobat3_2_species_ids.feather'
        ),
    )
    parser.add_argument(
        '--fragment-dirs',
        nargs='+',
        default=[],
        metavar='DIR',
        help=(
            'Directories containing 2-second fragment .wav files\n'
            '(output of SonoBat Long File Parser).  These are walked\n'
            'recursively to build a Filename → path index used for\n'
            'spectrogram extraction via TimeInFile.'
        ),
    )
    parser.add_argument(
        '-o', '--out-dir',
        required=True,
        metavar='DIR',
        help='Output directory for PNG crops and manifest.',
    )
    parser.add_argument(
        '--min-conf',
        type=float,
        default=_DEFAULT_MIN_CONF,
        metavar='F',
        help=f'Minimum confidence score to include (default: {_DEFAULT_MIN_CONF}).',
    )
    parser.add_argument(
        '--pre-ms',
        type=float,
        default=_DEFAULT_PRE_MS,
        metavar='MS',
        help=f'Ms before chirp onset (default: {_DEFAULT_PRE_MS}).',
    )
    parser.add_argument(
        '--post-ms',
        type=float,
        default=_DEFAULT_POST_MS,
        metavar='MS',
        help=f'Ms after chirp onset (default: {_DEFAULT_POST_MS}).',
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
        '--sample',
        type=int,
        default=0,
        metavar='N',
        help=(
            'Stop after writing N crops (default: 0 = no limit).\n'
            'Combine with --sample-species and --sample-partition\n'
            'for fast visual inspection before a full run.'
        ),
    )
    parser.add_argument(
        '--sample-species',
        nargs='+',
        default=[],
        metavar='SP',
        help='Restrict sampling to these species codes, e.g. Myca Tabr.',
    )
    parser.add_argument(
        '--sample-partition',
        default='',
        metavar='KEY',
        help='Restrict sampling to this partition, e.g. 20220706_bats.',
    )
    parser.add_argument(
        '--filename-map',
        default=None,
        metavar='CSV',
        help=(
            'CSV with columns file_id and Filename providing the\n'
            'file_id → fragment-stem mapping.  Required when the input\n'
            'parquet has no embedded bats_metadata (e.g. the measures\n'
            'parquet bats_<timestamp>.parquet).  The manifest.csv from\n'
            'any prior chirps_to_spectros.py run is a suitable source:\n'
            '  --filename-map <out-dir>/manifest.csv'
        ),
    )
    parser.add_argument(
        '--pcen',
        action='store_true',
        default=False,
        help=(
            'Use PCEN (Per-Channel Energy Normalization) instead of\n'
            'log-power normalisation.  PCEN applies adaptive gain control\n'
            'and non-linear compression per frequency bin, estimating and\n'
            'suppressing the stationary background noise floor via a running\n'
            'average.  Recommended for recordings with heavy background hum\n'
            'or broadband environmental noise (barn, outdoor).  Requires\n'
            'librosa (pip install librosa).'
        ),
    )
    parser.add_argument(
        '--pcen-time-constant',
        type=float,
        default=0.1,
        metavar='SEC',
        help=(
            'Time constant (s) for the PCEN adaptive noise-floor filter.\n'
            'Shorter values track faster-changing backgrounds.\n'
            'Default: 0.1 s (suitable for short bat chirps).\n'
            'The librosa default (0.395 s) is tuned for bird calls.'
        ),
    )
    parser.add_argument(
        '--pcen-snr-threshold',
        type=float,
        default=18.0,
        metavar='DB',
        help=(
            'SNR threshold (dB) for adaptive PCEN switching (used with --pcen).\n'
            'SNR = 10*log10(peak / median) of the raw bat-band spectrogram.\n'
            'Recordings with SNR below this value are noisy and get PCEN;\n'
            'cleaner recordings (SNR >= threshold) use log-power normalisation.\n'
            'Default: 18.0 dB.  Lower values apply PCEN more aggressively;\n'
            'higher values restrict it to only the noisiest recordings.'
        ),
    )
    parser.add_argument(
        '--include-nearest',
        action='store_true',
        help=(
            'Also accept chirps with match_quality="nearest" (legacy feather only).\n'
            'Default: window matches only.  Has no effect on parquet input.'
        ),
    )

    args = parser.parse_args()

    data_p = Path(args.data)
    if not data_p.exists():
        parser.error(f'Data file not found: {data_p}')

    args.data      = data_p
    args.out_dir   = Path(args.out_dir)
    args.match_quality = ['window', 'nearest'] if args.include_nearest else ['window']
    return args


def main() -> None:
    """
    CLI entry point for :class:`ChirpSpectroExtractor`.
    """
    args = _parse_args()

    extractor = ChirpSpectroExtractor(
        data_path     = args.data,
        out_dir       = args.out_dir,
        fragment_dirs = args.fragment_dirs,
        min_conf      = args.min_conf,
        pre_ms        = args.pre_ms,
        post_ms       = args.post_ms,
        freq_lo_khz   = args.freq_lo,
        freq_hi_khz   = args.freq_hi,
        img_size      = args.img_size,
        dynamic_range       = args.dynamic_range,
        n_workers           = args.workers,
        pcen                   = args.pcen,
        pcen_time_constant     = args.pcen_time_constant,
        pcen_snr_threshold_db  = args.pcen_snr_threshold,
        filename_map_path      = Path(args.filename_map) if args.filename_map else None,
        match_quality       = args.match_quality,
        sample            = args.sample,
        sample_species    = args.sample_species,
        sample_partition  = args.sample_partition,
    )

    result = extractor.run()
    log.info(result.summary())
    sys.exit(0 if result.n_crops_written > 0 else 1)


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()
