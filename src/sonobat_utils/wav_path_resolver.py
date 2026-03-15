#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-03-14 19:02:24
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/wav_path_resolver.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-14 19:02:59
# **********************************************************

"""
Resolve each ``Filename`` stem in the SonoBat species feather file to the
corresponding full-recording ``.wav`` path on disk.

Background
----------
The SonoBat pipeline was run on Windows VMs, causing the ``Filename`` column
in ``sonobat3_2_species_ids.feather`` to carry location prefixes
(``barn-``, ``lake2-``, ``bats-``) that do not reliably match the actual
directory structure on disk.  The only stable identifier shared between the
feather file and the original recordings is the **date + time** embedded in
both filenames:

* Fragment stem:    ``lake2_-20220427_192220_2secs``  → date ``20220427``, time ``192220``
* Full recording:  ``barn1_D20220427T194240m165.wav`` → date ``20220427``, time ``194240``

The fragment timestamp is an *offset within* the original recording, not the
recording start time.  A fragment timestamped ``192220`` belongs to a
recording that started at or before ``192220`` and ended at or after it —
i.e. within the window ``[recording_start, recording_start + 55s]``.

Match strategy
--------------
1. Index all ``.wav`` files found under the configured search roots, parsing
   each filename for a ``YYYYMMDD`` date and ``HHMMSS`` time.  Two filename
   patterns are recognised:

   * Full recordings: ``*_D<YYYYMMDD>T<HHMMSS>*.wav``
     (e.g. ``barn1_D20220327T194240m165.wav``)
   * Fragment files:  ``*<YYYYMMDD>_<HHMMSS>*.wav``
     (e.g. ``lake2_-20220427_192220_2secs.wav``)

2. For each unique ``Filename`` stem in the feather file, extract its date
   and fragment timestamp.

3. Search the index for full recordings on the same date whose time window
   ``[rec_start, rec_start + 55s]`` contains the fragment timestamp.

4. If no window match is found, fall back to the nearest full recording on
   the same date within +/- ``--fallback-window`` seconds of the fragment
   timestamp.

5. Record match quality:

   ``window``   — fragment timestamp falls inside the recording window
   ``nearest``  — fallback nearest-timestamp match
   ``none``     — no match found

Output files (all written to ``--out-dir``)
-------------------------------------------
``match_report.csv``
    One row per unique ``Filename`` stem with columns:
    ``Filename``, ``fragment_date``, ``fragment_time``, ``matched_wav``,
    ``match_quality``, ``recording_start_time``, ``site_guess``

``unmatched.csv``
    Subset of ``match_report.csv`` where ``match_quality == 'none'``.

``wav_index.csv``
    All ``.wav`` files found under the search roots, with parsed date/time
    and a ``wav_type`` column (``full_recording`` or ``fragment``).

``resolver_config.csv``
    Run parameters and summary statistics.

Typical usage
-------------
::

    cd /path/to/data/root
    python wav_path_resolver.py \\
        --root . \\
        --feather sonobat3_2_species_ids.feather \\
        --out-dir ./wav_resolver_results \\
        --full-recording-dirs barn/SMB_BARN_BATS jasperridge jasperride \\
        --fragment-dirs barn_sonobat3_2_processed lake2_sonobat3_2_processed
"""

import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from logging_service import LoggingService

log = LoggingService()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches full-recording filenames: *_D20220327T194240*.wav
_FULL_REC_RE = re.compile(r'_D(\d{8})T(\d{6})')

# Matches fragment filenames: *20220427_192220*.wav  (with optional prefix chars)
_FRAGMENT_RE = re.compile(r'(\d{8})_(\d{6})')

# Assumed maximum recording duration — fragment timestamps must fall within
# [rec_start, rec_start + _MAX_REC_DUR_S] to count as a window match.
_MAX_REC_DUR_S: int = 55

# Default fallback search window in seconds when no window match is found.
_DEFAULT_FALLBACK_S: int = 30

# Subdirectories to search for full recordings (relative to root).
_DEFAULT_FULL_REC_DIRS: list[str] = [
    'barn/SMB_BARN_BATS',
    'jasperridge',
    'jasperride',
]

# Subdirectories to search for fragment .wav files (relative to root).
_DEFAULT_FRAGMENT_DIRS: list[str] = [
    'barn_sonobat3_2_processed',
    'lake2_sonobat3_2_processed',
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WavEntry:
    """
    One ``.wav`` file found during the index walk.

    :param path:      Absolute path.
    :param wav_type:  ``'full_recording'`` or ``'fragment'``.
    :param date_str:  ``YYYYMMDD`` string parsed from the filename.
    :param time_str:  ``HHMMSS`` string parsed from the filename.
    :param dt:        Parsed :class:`datetime` object.
    """
    path:     Path
    wav_type: str
    date_str: str
    time_str: str
    dt:       datetime


@dataclass
class MatchRecord:
    """
    Result of attempting to match one feather ``Filename`` to a full recording.

    :param filename:             Original feather ``Filename`` stem.
    :param fragment_date:        ``YYYYMMDD`` from the fragment stem.
    :param fragment_time:        ``HHMMSS`` from the fragment stem.
    :param matched_wav:          Resolved path string, or empty string.
    :param match_quality:        ``'window'``, ``'nearest'``, or ``'none'``.
    :param recording_start_time: ``HHMMSS`` of the matched recording, or empty.
    :param site_guess:           Inferred site from matched path, or empty.
    :param n_candidates:         Number of same-date full recordings considered.
    """
    filename:             str
    fragment_date:        str
    fragment_time:        str
    matched_wav:          str        = ''
    match_quality:        str        = 'none'
    recording_start_time: str        = ''
    site_guess:           str        = ''
    n_candidates:         int        = 0


@dataclass
class ResolverResult:
    """
    Aggregate result returned by :meth:`WavPathResolver.run`.

    :param n_unique_filenames: Unique ``Filename`` stems in the feather file.
    :param n_window_matches:   Matched via recording-window strategy.
    :param n_nearest_matches:  Matched via nearest-timestamp fallback.
    :param n_unmatched:        No match found.
    :param n_wav_indexed:      Total ``.wav`` files in the index.
    :param n_full_recordings:  Full recordings in the index.
    :param n_fragments:        Fragment files in the index.
    :param elapsed_secs:       Wall-clock seconds.
    :param out_dir:            Output directory.
    """
    n_unique_filenames: int
    n_window_matches:   int
    n_nearest_matches:  int
    n_unmatched:        int
    n_wav_indexed:      int
    n_full_recordings:  int
    n_fragments:        int
    elapsed_secs:       float
    out_dir:            Path

    def summary(self) -> str:
        """
        Return a human-readable result summary.

        :return: Formatted multi-line string.
        """
        mins, secs = divmod(self.elapsed_secs, 60)
        elapsed_str = f'{int(mins)}m {secs:.1f}s' if mins else f'{secs:.1f}s'
        hit_rate = (
            100.0 * (self.n_window_matches + self.n_nearest_matches)
            / self.n_unique_filenames
            if self.n_unique_filenames else 0.0
        )
        return (
            f"WavPathResolver complete:\n"
            f"  * {self.n_wav_indexed:,} .wav files indexed "
            f"({self.n_full_recordings:,} full recordings, "
            f"{self.n_fragments:,} fragments)\n"
            f"  * {self.n_unique_filenames:,} unique Filenames in feather\n"
            f"  * {self.n_window_matches:,} window matches\n"
            f"  * {self.n_nearest_matches:,} nearest-timestamp fallback matches\n"
            f"  * {self.n_unmatched:,} unmatched  "
            f"(hit rate: {hit_rate:.1f}%)\n"
            f"  * Elapsed: {elapsed_str}\n"
            f"  * Output:  {self.out_dir}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WavPathResolver:
    """
    Resolve SonoBat feather ``Filename`` stems to original ``.wav`` paths.

    :param root:              Root directory containing all data.
    :param feather_path:      Path to ``sonobat3_2_species_ids.feather``.
    :param out_dir:           Directory for output CSVs.
    :param full_rec_dirs:     Subdirectory names (relative to root) to search
                              for full recordings.
    :param fragment_dirs:     Subdirectory names (relative to root) to search
                              for fragment ``.wav`` files.
    :param fallback_window_s: Maximum seconds from fragment timestamp to
                              recording start time for a fallback match.
    :param recursive:         If ``True``, descend into all subdirectories
                              rather than only the named ones.
    """

    def __init__(
        self,
        root:              str | Path,
        feather_path:      str | Path,
        out_dir:           str | Path,
        full_rec_dirs:     Sequence[str] = _DEFAULT_FULL_REC_DIRS,
        fragment_dirs:     Sequence[str] = _DEFAULT_FRAGMENT_DIRS,
        fallback_window_s: int           = _DEFAULT_FALLBACK_S,
        recursive:         bool          = True,
    ) -> None:
        self.root              = Path(root).resolve()
        self.feather_path      = Path(feather_path)
        self.out_dir           = Path(out_dir)
        self.full_rec_dirs     = [self.root / d for d in full_rec_dirs]
        self.fragment_dirs     = [self.root / d for d in fragment_dirs]
        self.fallback_window_s = fallback_window_s
        self.recursive         = recursive

    # ------------------------------------------------------------------ #
    #  Filename parsing                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_full_recording(path: Path) -> Optional[WavEntry]:
        """
        Parse a full-recording ``.wav`` filename for date and time.

        Expects the pattern ``_D<YYYYMMDD>T<HHMMSS>`` anywhere in the stem,
        e.g. ``barn1_D20220327T194240m165.wav``.

        :param path: Path to candidate ``.wav`` file.
        :return:     :class:`WavEntry` or ``None`` if pattern not found.
        """
        m = _FULL_REC_RE.search(path.stem)
        if not m:
            return None
        date_str, time_str = m.group(1), m.group(2)
        try:
            dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except ValueError:
            return None
        return WavEntry(
            path     = path.resolve(),
            wav_type = 'full_recording',
            date_str = date_str,
            time_str = time_str,
            dt       = dt,
        )

    @staticmethod
    def _parse_fragment(path: Path) -> Optional[WavEntry]:
        """
        Parse a fragment ``.wav`` filename for date and time.

        Expects ``<YYYYMMDD>_<HHMMSS>`` anywhere in the stem,
        e.g. ``lake2_-20220427_192220_2secs.wav``.

        :param path: Path to candidate ``.wav`` file.
        :return:     :class:`WavEntry` or ``None`` if pattern not found.
        """
        m = _FRAGMENT_RE.search(path.stem)
        if not m:
            return None
        date_str, time_str = m.group(1), m.group(2)
        try:
            dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except ValueError:
            return None
        return WavEntry(
            path     = path.resolve(),
            wav_type = 'fragment',
            date_str = date_str,
            time_str = time_str,
            dt       = dt,
        )

    @staticmethod
    def _parse_feather_filename(stem: str) -> tuple[str, str]:
        """
        Extract ``(YYYYMMDD, HHMMSS)`` from a feather ``Filename`` stem.

        The stem may carry an arbitrary location prefix before the date,
        e.g. ``lake2_-20220427_192220_2secs`` or ``barn-20220427_192220_2secs``.
        The date+time are identified via :data:`_FRAGMENT_RE`.

        :param stem: ``Filename`` value from the feather file (no extension).
        :return:     ``(date_str, time_str)`` or ``('', '')`` if unparseable.
        """
        m = _FRAGMENT_RE.search(stem)
        if not m:
            return '', ''
        return m.group(1), m.group(2)

    # ------------------------------------------------------------------ #
    #  .wav index                                                         #
    # ------------------------------------------------------------------ #

    def _build_index(self) -> tuple[list[WavEntry], list[WavEntry]]:
        """
        Walk the configured directories and build separate lists of full
        recordings and fragment ``.wav`` files.

        :return: ``(full_recordings, fragments)`` — each a list of
                 :class:`WavEntry` objects with parsed timestamps.
        """
        full_recordings: list[WavEntry] = []
        fragments:       list[WavEntry] = []
        seen: set[Path] = set()

        def _walk(dirs: list[Path], parser, target: list[WavEntry]) -> None:
            for base in dirs:
                if not base.exists():
                    log.warn(f'Directory not found, skipping: {base}')
                    continue
                glob_fn = base.rglob if self.recursive else base.glob
                for p in glob_fn('*.wav'):
                    rp = p.resolve()
                    if rp in seen:
                        continue
                    seen.add(rp)
                    entry = parser(p)
                    if entry is not None:
                        target.append(entry)

        log.info('Indexing full recordings ...')
        _walk(self.full_rec_dirs, self._parse_full_recording, full_recordings)
        log.info(f'  {len(full_recordings):,} full recordings indexed')

        log.info('Indexing fragment .wav files ...')
        _walk(self.fragment_dirs, self._parse_fragment, fragments)
        log.info(f'  {len(fragments):,} fragment files indexed')

        return full_recordings, fragments

    # ------------------------------------------------------------------ #
    #  Matching                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _site_from_path(path: Path) -> str:
        """
        Guess the site name from a ``.wav`` file path by looking for known
        site keywords in the path components.

        :param path: Absolute path.
        :return:     Site string (e.g. ``'barn'``, ``'lake2'``) or ``'unknown'``.
        """
        parts_lower = [p.lower() for p in path.parts]
        for keyword in ('barn', 'lake2', 'lake', 'jasper'):
            if any(keyword in part for part in parts_lower):
                return keyword
        return 'unknown'

    def _match_one(
        self,
        stem:         str,
        date_str:     str,
        time_str:     str,
        by_date:      dict[str, list[WavEntry]],
    ) -> MatchRecord:
        """
        Attempt to match one feather ``Filename`` stem to a full recording.

        Strategy:
        1. Look up all full recordings on ``date_str``.
        2. For each candidate, check whether the fragment's absolute
           datetime falls within ``[rec_start, rec_start + 55s]``.
        3. If multiple window matches exist, choose the one whose start
           time is closest to the fragment time (i.e. the recording that
           started most recently before the fragment).
        4. If no window match, fall back to the nearest-start-time
           recording within ``self.fallback_window_s``.

        :param stem:      Feather ``Filename`` stem.
        :param date_str:  ``YYYYMMDD`` parsed from *stem*.
        :param time_str:  ``HHMMSS`` parsed from *stem*.
        :param by_date:   Full-recording index keyed by ``YYYYMMDD``.
        :return:          :class:`MatchRecord`.
        """
        rec = MatchRecord(
            filename      = stem,
            fragment_date = date_str,
            fragment_time = time_str,
        )

        candidates = by_date.get(date_str, [])
        rec.n_candidates = len(candidates)

        if not candidates:
            return rec

        try:
            frag_dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except ValueError:
            return rec

        # ── Window match ───────────────────────────────────────────────
        window_matches = [
            e for e in candidates
            if e.dt <= frag_dt <= e.dt + timedelta(seconds=_MAX_REC_DUR_S)
        ]
        if window_matches:
            # Prefer the recording whose start is closest (most recent before frag)
            best = max(window_matches, key=lambda e: e.dt)
            rec.matched_wav          = str(best.path)
            rec.match_quality        = 'window'
            rec.recording_start_time = best.time_str
            rec.site_guess           = self._site_from_path(best.path)
            return rec

        # ── Nearest-timestamp fallback ─────────────────────────────────
        nearest = min(candidates, key=lambda e: abs((e.dt - frag_dt).total_seconds()))
        gap_s = abs((nearest.dt - frag_dt).total_seconds())
        if gap_s <= self.fallback_window_s:
            rec.matched_wav          = str(nearest.path)
            rec.match_quality        = 'nearest'
            rec.recording_start_time = nearest.time_str
            rec.site_guess           = self._site_from_path(nearest.path)

        return rec

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> ResolverResult:
        """
        Build the ``.wav`` index, resolve all feather ``Filename`` stems,
        and write output CSVs.

        :return: :class:`ResolverResult` with summary statistics.
        """
        _t0 = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Load feather ──────────────────────────────────────────────
        log.info(f'Loading feather: {self.feather_path} ...')
        try:
            df = pd.read_feather(self.feather_path)
        except Exception as exc:
            log.warn(f'Cannot read feather file: {exc}')
            sys.exit(1)

        if 'Filename' not in df.columns:
            log.warn('Feather file has no Filename column — aborting')
            sys.exit(1)

        unique_stems = df['Filename'].dropna().unique().tolist()
        log.info(f'{len(unique_stems):,} unique Filename stems to resolve')

        # ── Build .wav index ──────────────────────────────────────────
        full_recordings, fragments = self._build_index()

        # Index full recordings by date for fast lookup
        by_date: dict[str, list[WavEntry]] = {}
        for entry in full_recordings:
            by_date.setdefault(entry.date_str, []).append(entry)

        # ── Match ─────────────────────────────────────────────────────
        log.info('Matching Filename stems to full recordings ...')
        match_records: list[MatchRecord] = []

        for i, stem in enumerate(unique_stems):
            if i % 10_000 == 0 and i > 0:
                log.info(f'  Processed {i:,} / {len(unique_stems):,} stems ...')
            date_str, time_str = self._parse_feather_filename(stem)
            if not date_str:
                match_records.append(MatchRecord(
                    filename      = stem,
                    fragment_date = '',
                    fragment_time = '',
                    match_quality = 'none',
                    n_candidates  = 0,
                ))
                continue
            rec = self._match_one(stem, date_str, time_str, by_date)
            match_records.append(rec)

        # ── Tally ─────────────────────────────────────────────────────
        n_window  = sum(1 for r in match_records if r.match_quality == 'window')
        n_nearest = sum(1 for r in match_records if r.match_quality == 'nearest')
        n_none    = sum(1 for r in match_records if r.match_quality == 'none')

        log.info(
            f'Match results: {n_window:,} window, '
            f'{n_nearest:,} nearest, {n_none:,} unmatched'
        )

        # ── Write outputs ─────────────────────────────────────────────
        match_df = pd.DataFrame([
            {
                'Filename':             r.filename,
                'fragment_date':        r.fragment_date,
                'fragment_time':        r.fragment_time,
                'matched_wav':          r.matched_wav,
                'match_quality':        r.match_quality,
                'recording_start_time': r.recording_start_time,
                'site_guess':           r.site_guess,
                'n_candidates':         r.n_candidates,
            }
            for r in match_records
        ])
        match_path = self.out_dir / 'match_report.csv'
        match_df.to_csv(match_path, index=False)
        log.info(f'Wrote {len(match_df):,} rows to {match_path}')

        unmatched_df = match_df[match_df['match_quality'] == 'none']
        unmatched_path = self.out_dir / 'unmatched.csv'
        unmatched_df.to_csv(unmatched_path, index=False)
        log.info(f'Wrote {len(unmatched_df):,} unmatched rows to {unmatched_path}')

        # Full .wav index
        wav_index_rows = [
            {
                'path':     str(e.path),
                'wav_type': e.wav_type,
                'date':     e.date_str,
                'time':     e.time_str,
                'site':     self._site_from_path(e.path),
            }
            for e in (full_recordings + fragments)
        ]
        wav_index_df = pd.DataFrame(wav_index_rows)
        wav_index_path = self.out_dir / 'wav_index.csv'
        wav_index_df.to_csv(wav_index_path, index=False)
        log.info(f'Wrote {len(wav_index_df):,} rows to {wav_index_path}')

        # Site breakdown in match report
        if 'site_guess' in match_df.columns:
            site_counts = (
                match_df[match_df['match_quality'] != 'none']
                .groupby('site_guess')
                .size()
            )
            log.info(f'Matched by site:\n{site_counts.to_string()}')

        elapsed = time.perf_counter() - _t0

        pd.DataFrame([
            {'parameter': 'root',               'value': str(self.root)},
            {'parameter': 'feather_path',       'value': str(self.feather_path)},
            {'parameter': 'full_rec_dirs',      'value': str([str(d) for d in self.full_rec_dirs])},
            {'parameter': 'fragment_dirs',      'value': str([str(d) for d in self.fragment_dirs])},
            {'parameter': 'fallback_window_s',  'value': self.fallback_window_s},
            {'parameter': 'max_rec_dur_s',      'value': _MAX_REC_DUR_S},
            {'parameter': 'n_unique_filenames', 'value': len(unique_stems)},
            {'parameter': 'n_window_matches',   'value': n_window},
            {'parameter': 'n_nearest_matches',  'value': n_nearest},
            {'parameter': 'n_unmatched',        'value': n_none},
            {'parameter': 'n_full_recordings',  'value': len(full_recordings)},
            {'parameter': 'n_fragments',        'value': len(fragments)},
            {'parameter': 'elapsed_secs',       'value': round(elapsed, 1)},
        ]).to_csv(self.out_dir / 'resolver_config.csv', index=False)

        return ResolverResult(
            n_unique_filenames = len(unique_stems),
            n_window_matches   = n_window,
            n_nearest_matches  = n_nearest,
            n_unmatched        = n_none,
            n_wav_indexed      = len(full_recordings) + len(fragments),
            n_full_recordings  = len(full_recordings),
            n_fragments        = len(fragments),
            elapsed_secs       = elapsed,
            out_dir            = self.out_dir.resolve(),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments for :class:`WavPathResolver`.

    :return: ``argparse.Namespace`` with validated attributes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='wav_path_resolver',
        description=(
            'Resolve SonoBat feather Filename stems to original .wav paths\n'
            'using timestamp-based matching.\n\n'
            'Run from the data root directory, or pass --root explicitly.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--root',
        default='.',
        metavar='DIR',
        help='Root directory containing all data (default: current directory).',
    )
    parser.add_argument(
        '--feather',
        default='sonobat3_2_species_ids.feather',
        metavar='PATH',
        help=(
            'Path to species feather file, relative to --root\n'
            '(default: sonobat3_2_species_ids.feather).'
        ),
    )
    parser.add_argument(
        '-o', '--out-dir',
        default='wav_resolver_results',
        metavar='DIR',
        help='Output directory for CSVs (default: wav_resolver_results).',
    )
    parser.add_argument(
        '--full-recording-dirs',
        nargs='+',
        default=_DEFAULT_FULL_REC_DIRS,
        metavar='DIR',
        help=(
            'Subdirectories (relative to --root) to search for full\n'
            f'recordings (default: {_DEFAULT_FULL_REC_DIRS}).'
        ),
    )
    parser.add_argument(
        '--fragment-dirs',
        nargs='+',
        default=_DEFAULT_FRAGMENT_DIRS,
        metavar='DIR',
        help=(
            'Subdirectories (relative to --root) to search for 2-sec\n'
            f'fragment .wav files (default: {_DEFAULT_FRAGMENT_DIRS}).'
        ),
    )
    parser.add_argument(
        '--fallback-window',
        type=int,
        default=_DEFAULT_FALLBACK_S,
        metavar='SECS',
        help=(
            f'Maximum seconds gap for fallback nearest-timestamp match\n'
            f'(default: {_DEFAULT_FALLBACK_S}).'
        ),
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not descend into subdirectories (default: recursive).',
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    feather = Path(args.feather)
    if not feather.is_absolute():
        feather = root / feather
    if not feather.exists():
        parser.error(f'Feather file not found: {feather}')

    args.root      = root
    args.feather   = feather
    args.out_dir   = Path(args.out_dir)
    args.recursive = not args.no_recursive
    return args


def main() -> None:
    """
    CLI entry point for :class:`WavPathResolver`.
    """
    args = _parse_args()

    resolver = WavPathResolver(
        root              = args.root,
        feather_path      = args.feather,
        out_dir           = args.out_dir,
        full_rec_dirs     = args.full_recording_dirs,
        fragment_dirs     = args.fragment_dirs,
        fallback_window_s = args.fallback_window,
        recursive         = args.recursive,
    )

    result = resolver.run()
    log.info(result.summary())
    sys.exit(0 if result.n_window_matches + result.n_nearest_matches > 0 else 1)


# ------------------- Main Section --------------
if __name__ == '__main__':
    main()
