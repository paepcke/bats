#!/usr/bin/env python3
# **********************************************************
# @Author: Andreas Paepcke
# @File:   compare_measures.py
# @Description: Compare from-scratch acoustic measures against SonoBat
#               CumulativeParameters measures for the Marsh recording site.
#
# **********************************************************
"""
measures_extraction_verification.py
====================================
Compare acoustic measures produced by the from-scratch pipeline
(``chirp_measures_extraction.py`` output) against those produced by SonoBat
(``*_CumulativeParameters_*.txt`` files) for the same recordings.

Both pipelines apply identical column names (``HiFreq``, ``FreqCtr``, etc.)
but differ in two ways that require alignment before comparison:

1. **Absolute time frame**:
   - From-scratch ``TimeInFile``: chirp onset in ms from the **start of the
     original full recording** (chunk_offset already added by
     ``MeasureExtractor``).
   - SonoBat ``TimeInFile``: chirp onset in ms from the **start of the
     2-second snippet**.  The snippet's own offset within the original
     recording is encoded in its ``Filename`` timestamp:
     ``fragment_HHMMSS - recording_start_HHMMSS``.
     
     For Marsh we have no ``match_report.csv`` (no ``wav_path_resolver``
     run), so we cannot look up ``recording_start_HHMMSS`` from an external
     table.  Instead we recover it directly from the original WAV filename
     embedded in the SonoBat Filename stem (see ``--marsh-wav-dir``).

2. **Recording stem format**:
   - From-scratch ``file_id``: ``marsh1_D20130617T211645m968-Myca``
     (original WAV stem; may carry a species suffix after the last ``-``).
   - SonoBat ``Filename``:     ``marsh1_PST_D20150726T202036m894_2secs``
     (carries a ``PST_`` infix and ``_2secs`` suffix; both stripped for
     matching).

Alignment strategy
------------------
For each SonoBat chirp row the absolute onset is computed as::

    abs_ms_sb = snippet_offset_ms + sb_TimeInFile_ms

where ``snippet_offset_ms`` = (snippet_HHMMSS − rec_start_HHMMSS) × 1000.

A from-scratch chirp is considered a *match* when its ``TimeInFile`` lies
within ``--tol-ms`` milliseconds of ``abs_ms_sb`` **and** its ``file_id``
stem matches the SonoBat recording stem.

Output
------
Three files are written to ``--out-dir``:

``comparison_summary.csv``
    One row per acoustic measure:
    mean bias (FS − SB), SD of differences, R², Pearson r, n_matched pairs.

``comparison_pairs.csv``
    Every matched chirp pair with both sets of raw measures side-by-side.
    Useful for scatter plots and Bland-Altman plots in a notebook.

``bland_altman.png``
    Grid of Bland-Altman plots (mean vs. difference) for each measure.

``scatter.png``
    Grid of scatter plots (FS vs. SB) for each measure with R² annotation.

CLI usage
---------
::

    python compare_measures.py \\
        --fs-csv        /data2/marsh_stanford_processed/measures.csv \\
        --sb-cumul-dir  /path/to/marsh_sonobat_output \\
        --marsh-wav-dir /data2/marsh \\
        --out-dir       /data2/marsh_stanford_processed/measure_comparison \\
        --tol-ms        10 \\
        --max-pairs     200000

Notes
-----
* Only measures present in BOTH outputs are compared (see ``COMPARE_COLS``).
* ``PrecedingIntrvl`` is excluded: SonoBat measures it within the 2-second
  snippet, from-scratch measures it within the chunk *and* the first chirp
  in each chunk gets NaN.  The definitions differ in boundary cases.
* The script is read-only: it never modifies your existing pipeline files.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Columns that exist in both pipelines with the same definition
# PrecedingIntrvl excluded (boundary-condition differences; see docstring).
# ---------------------------------------------------------------------------

COMPARE_COLS: list[str] = [
    'HiFreq',
    'Bndwdth',
    'FreqMaxPwr',
    'PrcntMaxAmpDur',
    'FreqKnee',
    'PrcntKneeDur',
    'StartF',
    'UpprKnFreq',
    'HiFtoUpprKnAmp',
    'HiFtoKnAmp',
    'HiFtoFcAmp',
    'UpprKnToKnAmp',
    'KnToFcAmp',
    'LdgToFcAmp',
    'FreqCtr',
    'FFwd32dB',
    'FFwd20dB',
    'FFwd15dB',
    'FBak5dB',
    'FFwd5dB',
    'Bndw32dB',
    'Amp1stQrtl',
    'Amp2ndQrtl',
    'Amp3rdQrtl',
    'Amp4thQrtl',
    '1st10kHzSlp',
    '1st5to15kHzSlp',
    '1st10kHzExp',
    '1st5to15kHzExp',
    'AmpK@start',
]

# Regex to extract the YYYYMMDD_HHMMSS component from a filename.
# Handles both SonoBat stems (marsh1_PST_D20150726T202036m894_2secs)
# and original WAV names (marsh1_PST_D20150726T202036m894.wav).
_TS_RE = re.compile(r'D(\d{8})T(\d{6})')

# SonoBat Filename normalisation: strip _2secs suffix and PST_ infix.
_STRIP_2SECS = re.compile(r'_2secs$', re.IGNORECASE)
_STRIP_PST   = re.compile(r'_?PST_', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hhmmss_to_ms(hhmmss: str) -> Optional[int]:
    """
    Convert a 6-digit HHMMSS string to milliseconds from midnight.

    :param hhmmss: Six-digit time string, e.g. ``'202036'``.
    :return: Milliseconds, or ``None`` if unparseable.
    """
    if not isinstance(hhmmss, str) or len(hhmmss) != 6:
        return None
    try:
        hh, mm, ss = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:])
        return (hh * 3600 + mm * 60 + ss) * 1000
    except ValueError:
        return None


def _recording_stem(s: str) -> str:
    """
    Normalise a string to a bare recording stem for matching.

    Strips:
    * trailing ``_2secs`` (SonoBat convention)
    * ``_PST_`` or ``PST_`` infix (Marsh naming)
    * species suffix after the last ``-`` if it looks like a 4-char code
      (from-scratch ``file_id`` may carry ``-Myca`` etc.)

    :param s: Raw filename stem or file_id string.
    :return: Normalised stem.
    """
    s = _STRIP_2SECS.sub('', s)
    s = _STRIP_PST.sub('_', s).strip('_')
    # Strip trailing species code: -Myca / -Tabr etc.
    s = re.sub(r'-[A-Z][a-z]{3}$', '', s)
    return s


def _snippet_offset_ms(sb_filename: str, rec_start_hhmmss: str) -> Optional[float]:
    """
    Compute the 2-second snippet's start offset within the original recording.

    :param sb_filename:     SonoBat Filename stem, e.g.
                            ``'marsh1_PST_D20150726T202036m894_2secs'``.
    :param rec_start_hhmmss: Recording start time as HHMMSS, e.g. ``'202012'``.
    :return: Offset in milliseconds, or ``None`` if timestamps can't be parsed.
    """
    m = _TS_RE.search(sb_filename)
    if not m:
        return None
    frag_ms  = _hhmmss_to_ms(m.group(2))
    start_ms = _hhmmss_to_ms(rec_start_hhmmss)
    if frag_ms is None or start_ms is None:
        return None
    diff = frag_ms - start_ms
    if diff < 0:
        diff += 86_400_000   # midnight-spanning recording
    # Sanity cap: snippets beyond 55 s imply a timestamp parsing error
    # (same guard used in sono_batch_processing.py, converted to ms here).
    if diff > 55_000:
        diff = 0
    return float(diff)


# ---------------------------------------------------------------------------
# SonoBat CumulativeParameters loader
# ---------------------------------------------------------------------------

class SonoBatLoader:
    """
    Load and prepare SonoBat ``*_CumulativeParameters_*.txt`` data.

    Accepts **either** an explicit list of files (``params_files``) **or** a
    root directory to glob recursively (``root_dir``).  Exactly one must be
    supplied.

    Concatenates all files and computes ``abs_time_ms`` = snippet_offset_ms
    + TimeInFile for each chirp row.

    The recording start time for each snippet is recovered from the original
    WAV filename in ``wav_dir`` (Marsh naming:
    ``marsh1_PST_D20150726T202036m894.wav``).  The recording is identified by
    its date-time stem after normalisation, so ``PST_`` prefixes and
    ``_2secs`` suffixes are stripped before lookup.

    :param wav_dir:      Directory containing original Marsh ``.wav`` files,
                         searched recursively.  Used to look up recording
                         start times from filenames.
    :param params_files: Explicit list of ``*_CumulativeParameters_*.txt``
                         paths.  Takes priority over ``root_dir`` when both
                         are supplied.
    :param root_dir:     Root directory to search recursively for
                         ``*_CumulativeParameters_*.txt`` files.  Used only
                         when ``params_files`` is not supplied.
    :raises ValueError:  If neither ``params_files`` nor ``root_dir`` is given.
    """

    _PARAMS_GLOB = '**/*_CumulativeParameters_*.txt'

    def __init__(
        self,
        wav_dir:      Path,
        params_files: Optional[list[Path]] = None,
        root_dir:     Optional[Path]       = None,
    ) -> None:
        self.wav_dir      = wav_dir
        self.params_files = params_files
        self.root_dir     = root_dir
        if not params_files and root_dir is None:
            raise ValueError('Provide either params_files or root_dir')

    # ------------------------------------------------------------------

    def _build_rec_start_map(self) -> dict[str, str]:
        """
        Scan ``wav_dir`` for ``.wav`` files and build a mapping from the
        normalised recording stem → HHMMSS recording-start time.

        The recording start time is the HHMMSS embedded in the WAV filename
        itself (e.g. ``D20150726T202036m894`` → ``'202036'``).  For Marsh
        files the first snippet of a recording has the same timestamp as the
        original WAV, so the snippet offset for that snippet is 0 ms and all
        later snippets carry incremented timestamps.

        :return: Dict mapping normalised stem → HHMMSS string.
        """
        mapping: dict[str, str] = {}
        for wav in sorted(self.wav_dir.rglob('*.wav')):
            m = _TS_RE.search(wav.stem)
            if not m:
                continue
            norm = _recording_stem(wav.stem)
            # The WAV *itself* is the recording start, so its HHMMSS is
            # the recording_start_time for all snippets chopped from it.
            mapping[norm] = m.group(2)    # HHMMSS of the original recording
        return mapping

    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load all CumulativeParameters files, add ``abs_time_ms`` and
        normalised ``rec_stem``, and return the combined DataFrame.

        Uses ``self.params_files`` when supplied; otherwise globs
        ``self.root_dir`` recursively for ``*_CumulativeParameters_*.txt``.

        :return: DataFrame with one row per SonoBat chirp and added columns
                 ``abs_time_ms`` (float) and ``rec_stem`` (str).
        :raises FileNotFoundError: If no Parameters files are found.
        """
        if self.params_files:
            params_files = self.params_files
            missing = [p for p in params_files if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f'{len(missing)} supplied file(s) not found:\n'
                    + '\n'.join(f'  {p}' for p in missing)
                )
        else:
            params_files = sorted(self.root_dir.glob(self._PARAMS_GLOB))
            if not params_files:
                raise FileNotFoundError(
                    f'No *_CumulativeParameters_*.txt files found under {self.root_dir}'
                )
        print(f'[SB]  Loading {len(params_files):,} CumulativeParameters file(s):')
        for p in params_files:
            print(f'        {p}')

        chunks: list[pd.DataFrame] = []
        for p in params_files:
            try:
                df = pd.read_csv(p, sep='\t', low_memory=False)
                chunks.append(df)
            except Exception as exc:
                print(f'[SB]  WARNING: could not read {p.name}: {exc}', file=sys.stderr)

        if not chunks:
            raise ValueError('All CumulativeParameters files failed to load')

        df = pd.concat(chunks, ignore_index=True)
        print(f'[SB]  Loaded {len(df):,} chirp rows from SonoBat')

        # Strip .wav extension from Filename if present (defensive)
        if 'Filename' in df.columns:
            df['Filename'] = df['Filename'].astype(str).str.replace(
                r'\.wav$', '', regex=True, case=False
            )
        else:
            raise KeyError('CumulativeParameters files have no "Filename" column')

        df['rec_stem'] = df['Filename'].apply(_recording_stem)

        # Build recording-start lookup from original WAVs
        rec_start_map = self._build_rec_start_map()
        print(f'[SB]  Built {len(rec_start_map):,} recording-start entries from {self.wav_dir}')

        # Compute absolute onset time per chirp.
        #
        # Two SonoBat output styles exist:
        #
        # 1. Snippet-based (lake2/barn): Filename encodes the 2-second fragment
        #    timestamp; TimeInFile is ms within that snippet.  The snippet's own
        #    offset within the original recording must be added.
        #
        # 2. Full-recording (Marsh): Filename IS the original recording stem;
        #    TimeInFile is already absolute ms within the recording.  Snippet
        #    offset is always 0; WAV lookup is unnecessary.
        #
        # We try the WAV-lookup path first.  If it fails (rec_start not found,
        # or snippet offset unparseable) we fall back to treating TimeInFile as
        # already absolute — correct for Marsh-style output and harmless for
        # snippet-style output whose offset happens to be 0.
        def _abs_ms(row) -> float:
            tif = pd.to_numeric(row.get('TimeInFile', float('nan')), errors='coerce')
            if pd.isna(tif):
                return float('nan')
            # Try snippet-offset path (lake2/barn style)
            rec_start = rec_start_map.get(row['rec_stem'])
            if rec_start is not None:
                offset = _snippet_offset_ms(str(row['Filename']), rec_start)
                if offset is not None:
                    return offset + tif
            # Fall back: TimeInFile is already absolute (Marsh full-recording style)
            return float(tif)

        df['abs_time_ms'] = df.apply(_abs_ms, axis=1)

        n_resolved = df['abs_time_ms'].notna().sum()
        print(
            f'[SB]  Resolved absolute onset time for {n_resolved:,} / {len(df):,} chirps '
            f'({100*n_resolved/max(len(df),1):.1f}%)'
        )
        return df


# ---------------------------------------------------------------------------
# From-scratch loader
# ---------------------------------------------------------------------------

def load_from_scratch(csv_path: Path) -> pd.DataFrame:
    """
    Load the from-scratch ``measures.csv`` and normalise ``rec_stem``.

    ``TimeInFile`` is already the absolute onset ms in the original recording
    (chunk offset already added by ``MeasureExtractor``).

    :param csv_path: Path to the ``measures.csv`` produced by
                     ``chirp_measures_extraction.py``.
    :return: DataFrame with one row per from-scratch chirp and added column
             ``rec_stem`` (str).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    print(f'[FS]  Loaded {len(df):,} chirp rows from from-scratch measures.csv')
    df['rec_stem'] = df['file_id'].astype(str).apply(_recording_stem)
    return df


# ---------------------------------------------------------------------------
# Chirp matching
# ---------------------------------------------------------------------------

def match_chirps(
    df_fs: pd.DataFrame,
    df_sb: pd.DataFrame,
    tol_ms: float = 10.0,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Match from-scratch chirps to SonoBat chirps by recording stem and onset time.

    For each SonoBat chirp, find the from-scratch chirp in the same recording
    with the closest ``TimeInFile`` within ``tol_ms`` tolerance.  One-to-one
    matching: once a from-scratch chirp is claimed it cannot match again.

    Strategy: sort both sides by ``(rec_stem, abs_time)``, then use a merge
    on ``rec_stem`` + ``asof`` join on time.  Falls back to a loop for
    recordings with few chirps.

    :param df_fs:     From-scratch DataFrame (must have ``rec_stem``, ``TimeInFile``).
    :param df_sb:     SonoBat DataFrame (must have ``rec_stem``, ``abs_time_ms``).
    :param tol_ms:    Maximum onset-time difference to accept as a match (ms).
    :param max_pairs: If set, randomly sample this many matched pairs for
                      large datasets (keeps plot/CSV sizes manageable).
    :return: DataFrame of matched pairs with columns suffixed ``_fs`` / ``_sb``.
    """
    # Work on copies with only the columns we need; keep full measure cols
    measure_cols_fs = [c for c in COMPARE_COLS if c in df_fs.columns]
    measure_cols_sb = [c for c in COMPARE_COLS if c in df_sb.columns]

    key_fs = df_fs[['rec_stem', 'TimeInFile'] + measure_cols_fs].copy()
    key_fs['TimeInFile'] = pd.to_numeric(key_fs['TimeInFile'], errors='coerce')
    key_fs = key_fs.dropna(subset=['TimeInFile']).sort_values(['rec_stem', 'TimeInFile'])
    key_fs = key_fs.reset_index(drop=True)

    key_sb = df_sb[['rec_stem', 'abs_time_ms'] + measure_cols_sb].copy()
    key_sb = key_sb.dropna(subset=['abs_time_ms']).sort_values(['rec_stem', 'abs_time_ms'])
    key_sb = key_sb.reset_index(drop=True)

    stems_fs = set(key_fs['rec_stem'].unique())
    stems_sb = set(key_sb['rec_stem'].unique())
    common   = stems_fs & stems_sb
    print(
        f'[Match] Stems in FS: {len(stems_fs):,}  SB: {len(stems_sb):,}  '
        f'Common: {len(common):,}'
    )
    if not common:
        raise ValueError(
            'No recording stems matched between from-scratch and SonoBat data.\n'
            'Check --marsh-wav-dir and that the same recordings were processed.\n'
            'Sample FS stems: ' + str(list(stems_fs)[:5]) + '\n'
            'Sample SB stems: ' + str(list(stems_sb)[:5])
        )

    matched_rows: list[dict] = []
    used_fs_indices: set[int] = set()

    # Process stem by stem for memory efficiency
    fs_by_stem = key_fs.groupby('rec_stem', sort=False)
    sb_by_stem = key_sb.groupby('rec_stem', sort=False)

    for stem in common:
        try:
            grp_fs = fs_by_stem.get_group(stem).reset_index(drop=True)
            grp_sb = sb_by_stem.get_group(stem).reset_index(drop=True)
        except KeyError:
            continue

        t_fs = grp_fs['TimeInFile'].values
        t_sb = grp_sb['abs_time_ms'].values

        # For each SB chirp, find nearest FS chirp within tolerance.
        # np.searchsorted + bracket check is O(n log n) total.
        claimed_fs: set[int] = set()
        for i_sb in range(len(t_sb)):
            t = t_sb[i_sb]
            idx = np.searchsorted(t_fs, t)
            best_i = None
            best_d = tol_ms + 1.0
            for candidate in [idx - 1, idx, idx + 1]:
                if 0 <= candidate < len(t_fs) and candidate not in claimed_fs:
                    d = abs(t_fs[candidate] - t)
                    if d <= tol_ms and d < best_d:
                        best_d = d
                        best_i = candidate
            if best_i is None:
                continue
            claimed_fs.add(best_i)

            row_fs = grp_fs.iloc[best_i]
            row_sb = grp_sb.iloc[i_sb]
            pair: dict = {
                'rec_stem': stem,
                'time_fs':  t_fs[best_i],
                'time_sb':  t,
                'dt_ms':    best_d,
            }
            for col in COMPARE_COLS:
                if col in grp_fs.columns:
                    pair[f'{col}_fs'] = row_fs[col]
                if col in grp_sb.columns:
                    pair[f'{col}_sb'] = row_sb[col]
            matched_rows.append(pair)

    pairs = pd.DataFrame(matched_rows)
    print(f'[Match] Matched {len(pairs):,} chirp pairs (tol={tol_ms} ms)')

    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs.sample(n=max_pairs, random_state=42).reset_index(drop=True)
        print(f'[Match] Sampled down to {max_pairs:,} pairs')

    return pairs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-measure comparison statistics from matched pairs.

    For each measure in ``COMPARE_COLS``:

    * ``n``        : number of pairs with both values non-NaN
    * ``mean_bias``: mean(FS − SB)  — positive = FS higher
    * ``sd_diff``  : std(FS − SB)
    * ``loa_lo``   : lower limit of agreement (Bland-Altman: bias − 1.96 × SD)
    * ``loa_hi``   : upper limit of agreement (bias + 1.96 × SD)
    * ``r``        : Pearson correlation coefficient
    * ``r2``       : R²

    :param pairs: DataFrame of matched pairs (columns ``{col}_fs``, ``{col}_sb``).
    :return: Summary DataFrame indexed by measure name.
    """
    records: list[dict] = []
    for col in COMPARE_COLS:
        fs_col = f'{col}_fs'
        sb_col = f'{col}_sb'
        if fs_col not in pairs.columns or sb_col not in pairs.columns:
            continue
        v_fs = pd.to_numeric(pairs[fs_col], errors='coerce')
        v_sb = pd.to_numeric(pairs[sb_col], errors='coerce')
        mask = v_fs.notna() & v_sb.notna()
        n    = mask.sum()
        if n < 2:
            records.append({'measure': col, 'n': n,
                            'mean_bias': np.nan, 'sd_diff': np.nan,
                            'loa_lo': np.nan, 'loa_hi': np.nan,
                            'r': np.nan, 'r2': np.nan})
            continue
        diff   = v_fs[mask] - v_sb[mask]
        bias   = diff.mean()
        sd     = diff.std(ddof=1)
        r      = float(np.corrcoef(v_fs[mask], v_sb[mask])[0, 1])
        records.append({
            'measure'  : col,
            'n'        : n,
            'mean_bias': round(bias, 6),
            'sd_diff'  : round(sd, 6),
            'loa_lo'   : round(bias - 1.96 * sd, 6),
            'loa_hi'   : round(bias + 1.96 * sd, 6),
            'r'        : round(r, 4),
            'r2'       : round(r ** 2, 4),
        })
    return pd.DataFrame(records).set_index('measure')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_grid(
    pairs: pd.DataFrame,
    out_path: Path,
    mode: str,
) -> None:
    """
    Render a grid of either Bland-Altman or scatter plots and save to disk.

    :param pairs:    Matched pairs DataFrame.
    :param out_path: Destination PNG path.
    :param mode:     ``'bland_altman'`` or ``'scatter'``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    available = [
        c for c in COMPARE_COLS
        if f'{c}_fs' in pairs.columns and f'{c}_sb' in pairs.columns
    ]
    n = len(available)
    if n == 0:
        print(f'[Plot] No shared measure columns — skipping {out_path.name}')
        return

    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, col in zip(axes_flat, available):
        v_fs = pd.to_numeric(pairs[f'{col}_fs'], errors='coerce')
        v_sb = pd.to_numeric(pairs[f'{col}_sb'], errors='coerce')
        mask = v_fs.notna() & v_sb.notna()
        x, y = v_fs[mask].values, v_sb[mask].values
        if len(x) < 2:
            ax.set_visible(False)
            continue

        if mode == 'bland_altman':
            mean = (x + y) / 2.0
            diff = x - y
            bias = diff.mean()
            sd   = diff.std(ddof=1)
            ax.scatter(mean, diff, s=3, alpha=0.3, color='steelblue', rasterized=True)
            ax.axhline(bias,            color='red',    lw=1.5, label=f'bias={bias:.3g}')
            ax.axhline(bias + 1.96 * sd, color='orange', lw=1, ls='--',
                       label=f'+1.96SD={bias+1.96*sd:.3g}')
            ax.axhline(bias - 1.96 * sd, color='orange', lw=1, ls='--',
                       label=f'-1.96SD={bias-1.96*sd:.3g}')
            ax.set_xlabel('Mean (FS, SB)')
            ax.set_ylabel('FS − SB')
            ax.legend(fontsize=5, loc='upper right')
        else:
            r  = float(np.corrcoef(x, y)[0, 1])
            ax.scatter(x, y, s=3, alpha=0.3, color='steelblue', rasterized=True)
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
            ax.set_xlabel('FS')
            ax.set_ylabel('SB')
            ax.text(0.05, 0.93, f'R²={r**2:.3f}  n={mask.sum():,}',
                    transform=ax.transAxes, fontsize=7, va='top')

        ax.set_title(col, fontsize=8)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    title = 'Bland-Altman: FS vs SB' if mode == 'bland_altman' else 'Scatter: FS vs SB'
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class MeasureComparer:
    """
    Orchestrate loading, alignment, comparison, and reporting.

    :param fs_csv:        Path to ``measures.csv`` from the from-scratch pipeline.
    :param marsh_wav_dir: Directory containing the original Marsh ``.wav`` files,
                          used to recover recording-start timestamps.
    :param out_dir:       Directory for all output files.
    :param tol_ms:        Chirp-matching tolerance in milliseconds.
    :param max_pairs:     Cap on matched pairs for downstream analysis (None = all).
    :param sb_files:      Explicit list of ``*_CumulativeParameters_*.txt`` paths.
                          Takes priority over ``sb_cumul_dir``.
    :param sb_cumul_dir:  Root directory searched recursively for
                          ``*_CumulativeParameters_*.txt`` files.  Used only
                          when ``sb_files`` is not supplied.
    """

    def __init__(
        self,
        fs_csv:        Path,
        marsh_wav_dir: Path,
        out_dir:       Path,
        tol_ms:        float           = 10.0,
        max_pairs:     Optional[int]   = None,
        sb_files:      Optional[list[Path]] = None,
        sb_cumul_dir:  Optional[Path]  = None,
    ) -> None:
        if not sb_files and sb_cumul_dir is None:
            raise ValueError('Provide either sb_files or sb_cumul_dir')
        self.fs_csv        = fs_csv
        self.sb_files      = sb_files
        self.sb_cumul_dir  = sb_cumul_dir
        self.marsh_wav_dir = marsh_wav_dir
        self.out_dir       = out_dir
        self.tol_ms        = tol_ms
        self.max_pairs     = max_pairs

    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the full comparison pipeline and write all output files.

        :return: None
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load data
        df_fs = load_from_scratch(self.fs_csv)
        loader = SonoBatLoader(
            wav_dir      = self.marsh_wav_dir,
            params_files = self.sb_files,
            root_dir     = self.sb_cumul_dir,
        )
        df_sb = loader.load()

        # 2. Match chirps
        pairs = match_chirps(df_fs, df_sb, tol_ms=self.tol_ms, max_pairs=self.max_pairs)
        if pairs.empty:
            print('[!] No matched pairs — check stem normalisation and --tol-ms')
            return

        # 3. Write pairs CSV (full, for notebook use)
        pairs_path = self.out_dir / 'comparison_pairs.csv'
        pairs.to_csv(pairs_path, index=False)
        print(f'[Out] Wrote {pairs_path}  ({len(pairs):,} rows)')

        # 4. Summary statistics
        summary = compute_summary(pairs)
        summary_path = self.out_dir / 'comparison_summary.csv'
        summary.to_csv(summary_path)
        print(f'\n[Summary] Per-measure statistics:')
        print(summary.to_string())
        print(f'\n[Out] Wrote {summary_path}')

        # 5. Plots
        _plot_grid(pairs, self.out_dir / 'bland_altman.png', mode='bland_altman')
        _plot_grid(pairs, self.out_dir / 'scatter.png',      mode='scatter')

        # 6. Quick match-rate diagnostic
        n_sb_total = df_sb['abs_time_ms'].notna().sum()
        n_fs_total = df_fs['TimeInFile'].notna().sum()
        print(
            f'\n[Diagnostic]\n'
            f'  SB chirps with resolved abs_time  : {n_sb_total:,}\n'
            f'  FS chirps                          : {n_fs_total:,}\n'
            f'  Matched pairs                      : {len(pairs):,}\n'
            f'  Match rate vs SB                   : '
            f'{100*len(pairs)/max(n_sb_total,1):.1f}%\n'
            f'  Match rate vs FS                   : '
            f'{100*len(pairs)/max(n_fs_total,1):.1f}%'
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: ``argparse.Namespace`` with validated attributes.
    """
    parser = argparse.ArgumentParser(
        prog='compare_measures',
        description=textwrap.dedent('''\
            Compare from-scratch acoustic measures against SonoBat measures
            for the Marsh recording site.
        '''),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--fs-csv',
        required=True,
        metavar='CSV',
        help='measures.csv from chirp_measures_extraction.py',
    )
    parser.add_argument(
        '--sb-files',
        nargs='+',
        default=None,
        metavar='TXT',
        help=(
            'Explicit paths to one or more *_CumulativeParameters_*.txt files.\n'
            'Example (Marsh four-VM split):\n'
            '  --sb-files /data/win_share/batch1_CumulativeParameters_v30.2.20250912.txt \\\n'
            '             /data/win_share/batch2_CumulativeParameters_v30.2.20250912.txt \\\n'
            '             /data/win_share/batch3_CumulativeParameters_v30.2.20250912.txt \\\n'
            '             /data/win_share/batch4_CumulativeParameters_v30.2.20250912.txt\n'
            'Use --sb-cumul-dir instead to glob a directory recursively.'
        ),
    )
    parser.add_argument(
        '--sb-cumul-dir',
        default=None,
        metavar='DIR',
        help=(
            'Root directory searched recursively for *_CumulativeParameters_*.txt files.\n'
            'Ignored when --sb-files is supplied.'
        ),
    )
    parser.add_argument(
        '--marsh-wav-dir',
        required=True,
        metavar='DIR',
        help='Directory containing original Marsh .wav files (searched recursively).\n'
             'Used to recover recording-start timestamps from filenames.',
    )
    parser.add_argument(
        '--out-dir',
        required=True,
        metavar='DIR',
        help='Output directory for comparison CSV and plots',
    )
    parser.add_argument(
        '--tol-ms',
        type=float,
        default=10.0,
        metavar='MS',
        help='Chirp-matching tolerance in milliseconds (default: 10)',
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=None,
        metavar='N',
        help='Cap matched pairs to this many (random sample; default: all)',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    if not args.sb_files and not args.sb_cumul_dir:
        print('ERROR: provide --sb-files or --sb-cumul-dir', file=sys.stderr)
        sys.exit(1)

    sb_files = [Path(p) for p in args.sb_files] if args.sb_files else None

    MeasureComparer(
        fs_csv        = Path(args.fs_csv),
        marsh_wav_dir = Path(args.marsh_wav_dir),
        out_dir       = Path(args.out_dir),
        tol_ms        = args.tol_ms,
        max_pairs     = args.max_pairs,
        sb_files      = sb_files,
        sb_cumul_dir  = Path(args.sb_cumul_dir) if args.sb_cumul_dir else None,
    ).run()


if __name__ == '__main__':
    main()
