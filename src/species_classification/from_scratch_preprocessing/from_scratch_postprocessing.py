#!/usr/bin/env python3
# **********************************************************
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-04-24 18:09:44
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-24 18:10:06
# **********************************************************

"""
from_scratch_postprocessing.py
================================
Adapter that converts the CSV produced by
:class:`~chirp_detection.chirp_measures_extraction.MeasureExtractor` (our
from-scratch pipeline) into the same ``bats_<ts>.parquet`` /
``bats_noise_<ts>.parquet`` / SQLite format that
:class:`~sonobat_utils.sb_measures_postprocessing.SonoBatPostProcessor`
produces from SonoBat's ``*_CumulativeParameters_*.txt`` files.

All downstream consumers — :mod:`bat_db_builder`,
:mod:`sb_measures_add_daytime_columns`, the RF and CNN training pipelines —
work completely unchanged.

Background: five translation problems
---------------------------------------
When the from-scratch pipeline (wav_file_scrubber → wav_chopper →
chirp_measures_extraction → RF/CNN classifier) is run on new recordings,
the output CSV differs from what SonoBatPostProcessor expects in five ways:

1. ``file_id`` column (string stem, e.g. ``marsh1_D20220723T215745m074``)
   instead of a ``Path`` column processed by ``PathEncoder``.
2. ``is_last`` column present but not a feature; would pollute the
   ``MeasureNormalizer``'s feature set.
3. ``species`` column already populated by the RF/CNN (SB workflow derives
   species from a separate ``*_CumulativeSonoBatch_*.txt`` stream and merges
   it in a later step).
4. ``confidence`` column already computed by the RF/CNN (SB workflow computes
   it from ``Prob``, ``#Maj``, ``#Accp`` via the composite formula in
   ``_finalize_species``).  Because ``confidence`` is numeric it would be
   absorbed into the normalizer's feature set and scaled, destroying its
   meaning.
5. No ``rec_site`` column — must be injected from a CLI argument.

This module resolves all five problems and is the ONLY file that needs to
change as the from-scratch and SB-based pipelines diverge.  No edits are
required to ``sb_measures_postprocessing.py``, ``bat_db_builder.py``, or
``sb_measures_add_daytime_columns.py``.

Workflow position
-----------------
::

    wav_file_scrubber.py          # filter non-bat full recordings
         ↓
    wav_chopper.py                # chop into 2-sec chunks
         ↓
    chirp_measures_extraction.py  # extract per-chirp acoustic measures
         ↓  (measures.csv — no species yet)
    species_pred_random_forest.py # (or CNN) → appends species + confidence
         ↓  (measures_classified.csv)
    from_scratch_postprocessing.py  ← YOU ARE HERE
         ↓
    bats_<ts>.parquet  /  bats_noise_<ts>.parquet  /  chirp_meta.db
         ↓
    bat_db_builder.py  (adds chirp_info + chirp_spectrograms)
         ↓
    sb_measures_add_daytime_columns.py  (adds was_daytime, time_of_day_pactime)

CLI usage
---------
::

    python from_scratch_postprocessing.py \\
        --measures-csv  /qnap/src/marsh_stanford_processed/measures_classified.csv \\
        --rec-site      marsh \\
        --dest-dir      /qnap/src/marsh_stanford_processed \\
        --conf-thresh   0.50 \\
        --build-db      /qnap/bats/chirp_meta.db \\
        --add-daytime-columns

If the classifier has not yet run and ``species``/``confidence`` columns are
absent, all rows land in the noise parquet with ``species='noise'``.  You can
re-run this script once classification is complete.
"""

import argparse
import sqlite3
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from logging_service import LoggingService
from sonobat_utils.utils import Utils

# Re-use shared machinery from the SB-based pipeline unchanged.
from sonobat_utils.sb_measures_postprocessing import (
    BatsData,
    MeasureNormalizer,
    PathEncoder,
    SonoBatPostProcessor,   # borrowed for CONF_ACCEPT_THRESH_DEFAULT only
    _PST,
)

# Schema DDL is identical to sb_measures_postprocessing._seed_recordings_db;
# duplicated here so this module has no circular import dependency.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS png_dirs (
    dir_id    INTEGER PRIMARY KEY,
    dir_path  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS recordings (
    file_id    INTEGER PRIMARY KEY,
    filename   TEXT,
    rec_site   TEXT,
    rec_period TEXT
);

CREATE TABLE IF NOT EXISTS chirp_info (
    file_id      INTEGER NOT NULL REFERENCES recordings(file_id),
    chirp_idx    INTEGER NOT NULL,
    measures_row INTEGER NOT NULL,
    species      TEXT,
    confidence   REAL,
    PRIMARY KEY (file_id, chirp_idx)
);

CREATE TABLE IF NOT EXISTS chirp_spectrograms (
    file_id      INTEGER NOT NULL,
    chirp_idx    INTEGER NOT NULL,
    harmonic_idx INTEGER NOT NULL,
    dir_id       INTEGER NOT NULL REFERENCES png_dirs(dir_id),
    png_filename TEXT NOT NULL,
    PRIMARY KEY (file_id, chirp_idx, harmonic_idx),
    FOREIGN KEY (file_id, chirp_idx) REFERENCES chirp_info(file_id, chirp_idx),
    UNIQUE (dir_id, png_filename)
);

CREATE INDEX IF NOT EXISTS idx_chirp_species ON chirp_info(species);
CREATE INDEX IF NOT EXISTS idx_spec_dir      ON chirp_spectrograms(dir_id);
CREATE INDEX IF NOT EXISTS idx_spec_harmonic ON chirp_spectrograms(harmonic_idx);
"""


class FromScratchPostProcessor:
    """
    Parallel to :class:`~sb_measures_postprocessing.SonoBatPostProcessor`
    but reads the CSV produced by
    :class:`~chirp_detection.chirp_measures_extraction.MeasureExtractor`
    (optionally with ``species`` and ``confidence`` columns appended by
    the RF/CNN classifier) instead of SonoBat's ``CumulativeParameters``
    and ``CumulativeSonoBatch`` text files.

    Output parquets are schema-identical to those produced by
    ``SonoBatPostProcessor``, so ``bat_db_builder.py``,
    ``sb_measures_add_daytime_columns.py``, and all downstream consumers
    work without modification.

    Translation problems handled internally
    ----------------------------------------
    1. **file_id column**: The CSV carries a string ``file_id`` (original
       recording stem, e.g. ``marsh1_D20220723T215745m074``).  This class
       passes it to ``PathEncoder`` as the "path" string, producing the same
       integer ``file_id`` representation used throughout the pipeline.  The
       stem string ends up in ``recordings.filename``, which
       ``sb_measures_add_daytime_columns.py`` parses to recover the recording
       timestamp — all four timestamp patterns in ``_FNAME_PATTERNS`` handle
       the Jasper Ridge naming conventions.

    2. **is_last column**: Dropped before normalization; not a feature and
       not needed downstream.

    3. **species already present**: Used directly for the clean/noise split
       instead of being merged from a separate species stream.

    4. **confidence already computed**: Stashed before :class:`MeasureNormalizer`
       runs so it is not treated as an acoustic feature and scaled; re-attached
       after normalization using index alignment (which correctly handles outlier
       rows dropped during filtering).

    5. **rec_site absent**: Injected from the ``--rec-site`` argument as a
       pandas Categorical column, matching the dtype used by
       ``SonoBatPostProcessor``.

    :param measures_csv:    Path to the classified measures CSV (output of
                            ``chirp_measures_extraction.py``, optionally with
                            ``species`` and ``confidence`` columns appended by
                            the RF/CNN classifier).
    :param rec_site:        Recording site label (e.g. ``'marsh'``).
    :param dest_dir:        Directory where the output Parquet files are written.
    :param conf_thresh:     Minimum confidence for a chirp to enter the clean
                            parquet.  Rows below this threshold go to the noise
                            parquet as ``'unkn'``.  Default 0.50.
    :param db_path:         If given, create (or open) a SQLite database at
                            this path and seed the ``recordings`` table with
                            the ``file_id → filename`` mapping produced here.
    :param add_daytime_cols: If True (requires ``db_path``), invoke
                            :class:`DaytimeColumnAdder` to append
                            ``was_daytime`` and ``time_of_day_pactime`` columns
                            to the written parquet files.
    """

    CONF_ACCEPT_THRESH_DEFAULT: float = SonoBatPostProcessor.CONF_ACCEPT_THRESH_DEFAULT

    def __init__(
        self,
        measures_csv:      str | Path,
        rec_site:          str,
        dest_dir:          str | Path,
        conf_thresh:       float = CONF_ACCEPT_THRESH_DEFAULT,
        db_path:           str | Path | None = None,
        add_daytime_cols:  bool = False,
        existing_parquet:  str | Path | None = None,
    ) -> None:
        self.log = LoggingService()
        self.timestamp = datetime.now(_PST).isoformat().replace(':', '_')

        self.measures_csv     = Path(measures_csv)
        self.rec_site         = rec_site
        self.dest_dir         = Path(dest_dir)
        self.conf_thresh      = conf_thresh
        self.db_path          = Path(db_path) if db_path is not None else None
        self.add_daytime_cols  = add_daytime_cols
        self.existing_parquet  = (Path(existing_parquet)
                                  if existing_parquet is not None else None)

        # Build site categorical dtype before loading so the column is
        # created with a consistent category set from the start.
        self.site_dtype = pd.CategoricalDtype(
            categories=sorted({rec_site}), ordered=False
        )

        # ── 1. Load and validate the measures CSV ─────────────────────────
        df_raw = self._load_measures_csv()

        # ── 2. Build PathEncoder — extending existing if provided ────────
        # PathEncoder was designed for file path strings; using the recording
        # stem directly is equivalent because stems are already globally unique.
        #
        # CRITICAL: when --existing-parquet is supplied we MUST extend the
        # existing PathEncoder rather than creating a fresh one starting at 0.
        # A fresh encoder would collide with every file_id already in the
        # main parquet.  The append-only extension preserves all existing IDs
        # and assigns new IDs starting at max(existing) + 1.
        new_stems = list(df_raw['file_id'].unique())
        self.path_encoder = self._build_path_encoder(new_stems)

        # PathEncoder.encode_df() expects a column named 'Path'.
        df_for_encode = df_raw.rename(columns={'file_id': 'Path'})
        df_encoded    = self.path_encoder.encode_df(df_for_encode)
        # file_id is now an integer; original stem is in path_encoder.id_to_path.

        # ── 3. Stash species + confidence before normalization ─────────────
        # 'species' is a string column and would survive _extract_numeric_features
        # unchanged, but stashing both together keeps the logic explicit.
        # 'confidence' is numeric and NOT in NON_FEATURE_COLS; without stashing
        # it would be scaled by RobustScaler, destroying its [0, 1] semantics.
        stash_cols = [c for c in ('species', 'confidence') if c in df_encoded.columns]
        df_stash    = df_encoded[stash_cols].copy() if stash_cols else None
        df_for_norm = df_encoded.drop(columns=stash_cols)

        # ── 4. Normalize acoustic measures ────────────────────────────────
        # When extending an existing parquet we MUST re-use the existing
        # normalizer so that new rows are on the same scale as old rows.
        # Fitting a fresh normalizer on new-site data alone would produce
        # incompatible scaled values that cannot be mixed in training.
        if self.existing_parquet is not None:
            existing_bats   = Utils.read_df_file(str(self.existing_parquet))
            meas_normalizer = existing_bats.normalizer
            df_normalized   = meas_normalizer.transform(df_for_norm)
            self.log.info(
                f"Normalization: re-using existing normalizer from "
                f"{self.existing_parquet.name}; "
                f"{len(df_normalized):,} rows transformed "
                f"(no outlier filtering — existing scaler applied directly)"
            )
        else:
            meas_normalizer = MeasureNormalizer()
            df_normalized   = meas_normalizer.fit_transform(df_for_norm)
            self.log.info(
                f"Normalization: fresh fit; "
                f"{meas_normalizer.n_rows_before_:,} rows in, "
                f"{meas_normalizer.n_rows_after_:,} rows out "
                f"({meas_normalizer.n_rows_before_ - meas_normalizer.n_rows_after_:,} "
                f"outlier rows dropped)"
            )
        self.normalizer = meas_normalizer

        # ── 5. Re-attach species + confidence (index-aligned) ─────────────
        # fit_transform may drop outlier rows; .loc[df_normalized.index] aligns
        # the stash to the survivor set automatically.
        if df_stash is not None:
            df_normalized = df_normalized.join(
                df_stash.loc[df_normalized.index]
            )

        # ── 6. Capture file_id → site mapping for DB seeding ──────────────
        fid_to_site: dict[int, str] = {
            int(fid): rec_site
            for fid in df_normalized['file_id'].unique()
        }

        # ── 7. Split into clean and noise parquets ─────────────────────────
        df_clean, df_noise = self._split_clean_noise(df_normalized)

        # ── 8. Write bats_<ts>.parquet ────────────────────────────────────
        self.bats_data = BatsData(
            df        = df_clean,
            file_map  = self.path_encoder.id_to_path,
            normalizer= self.normalizer,
            timestamp = self.timestamp,
        )
        out_path = self.dest_dir / f"bats_{self.timestamp}.parquet"
        self.bats_data.to_parquet(out_path)
        self.log.info(f"Wrote clean parquet  → {out_path}  ({len(df_clean):,} rows)")

        # ── 9. Write bats_noise_<ts>.parquet ──────────────────────────────
        self.bats_noise = BatsData(
            df        = df_noise,
            file_map  = self.path_encoder.id_to_path,
            normalizer= self.normalizer,
            timestamp = self.timestamp,
        )
        noise_path = self.dest_dir / f"bats_noise_{self.timestamp}.parquet"
        self.bats_noise.to_parquet(noise_path)
        self.log.info(f"Wrote noise parquet  → {noise_path}  ({len(df_noise):,} rows)")

        # ── 10. Optional: seed SQLite recordings table ────────────────────
        if self.db_path is not None:
            self._seed_recordings_db(fid_to_site)

        # ── 11. Optional: add daytime columns ─────────────────────────────
        if self.add_daytime_cols:
            if self.db_path is None:
                self.log.warn(
                    "--add-daytime-columns requires --build-db; skipping daytime step."
                )
            else:
                from sonobat_utils.sb_measures_add_daytime_columns import DaytimeColumnAdder
                self.log.info("Adding daytime columns to measures parquet …")
                DaytimeColumnAdder(
                    measures_path=str(out_path),
                    db_path=str(self.db_path),
                ).run()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_path_encoder(self, new_stems: list[str]) -> PathEncoder:
        """
        Build a :class:`PathEncoder` for the new recording stems.

        If ``existing_parquet`` was supplied, the existing file_map is
        loaded from it and the new stems are appended starting at
        ``max(existing_ids) + 1``, preserving all existing IDs.

        If no existing parquet was supplied, a fresh encoder is created
        with IDs starting at 0.

        Collision detection is performed in both cases: if any new stem
        is already present in the existing file_map, the run is aborted.

        :param new_stems: List of recording stem strings from the
                          measures CSV ``file_id`` column.
        :return:          Configured :class:`PathEncoder`.
        :raises SystemExit: On stem collisions.
        """
        if self.existing_parquet is None:
            encoder = PathEncoder.from_paths(new_stems, sort_paths=False)
            self.log.info(
                f"PathEncoder: fresh; {len(new_stems):,} stems "
                f"→ ids 0…{len(new_stems) - 1}"
            )
            return encoder

        # Load existing file_map from the main parquet.
        existing_bats     = Utils.read_df_file(str(self.existing_parquet))
        existing_file_map = existing_bats.file_map   # dict: int id → stem str

        existing_stems = set(existing_file_map.values())
        new_stems_set  = set(new_stems)
        collisions     = existing_stems & new_stems_set
        if collisions:
            sample = sorted(collisions)[:5]
            self.log.err(
                f"{len(collisions):,} recording stem(s) in the measures CSV "
                f"already exist in {self.existing_parquet.name}. "
                f"Sample: {sample}. "
                f"Re-running from_scratch_postprocessing.py on data that is "
                f"already in the main parquet is not permitted."
            )
            sys.exit(1)

        # Reconstruct encoder: existing stems in their original ID order,
        # then new stems appended in sorted order for reproducibility.
        # PathEncoder.from_paths(..., sort_paths=False) assigns IDs 0, 1, 2…
        # in the order of the list, so preserving existing ID order is
        # equivalent to preserving existing IDs.
        existing_ordered = [existing_file_map[i]
                            for i in sorted(existing_file_map.keys())]
        new_stems_sorted  = sorted(new_stems)
        all_stems_ordered = existing_ordered + new_stems_sorted

        encoder         = PathEncoder.from_paths(all_stems_ordered,
                                                  sort_paths=False)
        n_existing      = len(existing_ordered)
        n_new           = len(new_stems_sorted)
        new_id_start    = n_existing
        new_id_end      = n_existing + n_new - 1
        self.log.info(
            f"PathEncoder: extended {n_existing:,} existing stems with "
            f"{n_new:,} new stems → ids {new_id_start}…{new_id_end}"
        )
        return encoder

    def _load_measures_csv(self) -> pd.DataFrame:
        """
        Read the measures CSV, validate required columns, inject ``rec_site``,
        and handle the five translation problems described in the class docstring.

        Absent ``species`` / ``confidence`` columns are filled with ``None``
        / ``0.0`` respectively and a warning is emitted.  The ``is_last``
        column is silently dropped if present (it is used internally by
        :class:`~chirp_detection.chirp_measures_extraction.MeasureExtractor`
        to mark the last chirp in each recording but is not needed downstream).

        :return: Cleaned DataFrame ready for PathEncoder and MeasureNormalizer.
        :raises SystemExit: If required columns are missing.
        """
        if not self.measures_csv.exists():
            self.log.err(f"Measures CSV not found: {self.measures_csv}")
            sys.exit(1)

        self.log.info(f"Loading measures CSV: {self.measures_csv}")
        df = pd.read_csv(self.measures_csv, low_memory=False)
        self.log.info(f"  {len(df):,} rows, {len(df.columns)} columns")

        # --- Validate required columns ---
        required = {'file_id', 'chirp_idx', 'TimeInFile'}
        missing  = required - set(df.columns)
        if missing:
            self.log.err(
                f"Measures CSV missing required columns: {missing}. "
                f"Was it produced by chirp_measures_extraction.py?"
            )
            sys.exit(1)

        # --- Translation 2: drop is_last (not a feature, not needed downstream) ---
        if 'is_last' in df.columns:
            df.drop(columns=['is_last'], inplace=True)

        # --- Translation 3 & 4: require species + confidence columns ---
        # Both columns must have been appended by rf_confidence_join.py.
        # A silent fallback is not permitted: a missing 'confidence' would
        # silently move every row into the noise parquet, and a missing
        # 'species' would do the same — both without any diagnostic signal.
        if 'species' not in df.columns:
            self.log.err(
                "No 'species' column in measures CSV. "
                "Run rf_confidence_join.py to join hierarchical_rf_predict.py "
                "output onto the measures CSV before calling this script."
            )
            sys.exit(1)
        if 'confidence' not in df.columns:
            self.log.err(
                "No 'confidence' column in measures CSV. "
                "Run rf_confidence_join.py to compute the SB-analog confidence "
                "score from the RF predictions before calling this script."
            )
            sys.exit(1)

        # --- Translation 5: inject rec_site ---
        df['rec_site'] = pd.Categorical(
            [self.rec_site] * len(df), dtype=self.site_dtype
        )

        return df

    def _split_clean_noise(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the normalized DataFrame into a clean parquet (confident
        species IDs) and a noise parquet (low-confidence or unclassified
        rows), matching the split logic of
        :meth:`~sb_measures_postprocessing.SonoBatPostProcessor._merge`.

        **Clean DataFrame**: rows where ``confidence >= conf_thresh`` and
        ``species`` is not NaN.

        **Noise DataFrame**: two kinds of rows, same schema as clean:

        - ``species = 'unkn'``: ``0 < confidence < conf_thresh``
          (classified but below threshold).  Measures are real.
          Suitable for RF/CNN noise/reject class.
        - ``species = 'noise'``: unclassified rows (``confidence == 0`` or
          ``species`` is NaN).  Suitable for CNN noise class.

        :param df: Fully normalized DataFrame with ``species`` and
                   ``confidence`` columns re-attached.
        :return: ``(df_clean, df_noise)`` tuple.
        """
        # --- Clean: confident, real species ---
        clean_mask = (
            df['confidence'].notna()
            & (df['confidence'] >= self.conf_thresh)
            & df['species'].notna()
        )
        df_clean = df[clean_mask].copy()

        # --- Noise part 1: classified but below threshold ---
        unkn_mask = (
            df['confidence'].notna()
            & (df['confidence'] > 0)
            & (df['confidence'] < self.conf_thresh)
        )
        df_unkn = df[unkn_mask].copy()
        df_unkn['species'] = 'unkn'

        # --- Noise part 2: unclassified (no species or zero/NaN confidence) ---
        noclass_mask = ~(clean_mask | unkn_mask)
        df_noclass   = df[noclass_mask].copy()
        df_noclass['species']    = 'noise'
        df_noclass['confidence'] = 0.0

        df_noise = pd.concat([df_unkn, df_noclass], ignore_index=True)

        self.log.info(
            f"Split: {len(df_clean):,} clean, "
            f"{len(df_unkn):,} 'unkn', "
            f"{len(df_noclass):,} 'noise'"
        )
        return df_clean, df_noise

    def _seed_recordings_db(self, fid_to_site: dict[int, str]) -> None:
        """
        Create (or open) the SQLite database at ``self.db_path``, apply the
        full schema DDL, and insert one row per ``file_id`` into the
        ``recordings`` table.

        ``filename`` is the original recording stem (the value stored in
        ``path_encoder.id_to_path``), e.g. ``marsh1_D20220723T215745m074``.
        This is the key that ``sb_measures_add_daytime_columns.py`` uses to
        parse recording timestamps via its ``_FNAME_PATTERNS`` registry —
        all Jasper Ridge naming conventions are handled by those patterns.

        ``rec_period`` (train/val/test partition) is left ``NULL`` here; it
        is filled later by :mod:`bat_db_builder` when the spectrogram
        manifest is available.

        ``INSERT OR IGNORE`` makes re-runs safe.

        :param fid_to_site: Mapping of integer ``file_id`` → ``rec_site``
                            string, built from the post-normalization survivor
                            set.
        """
        rows = [
            (
                fid,
                self.path_encoder.id_to_path[fid],   # original recording stem
                site,
                None,                                 # rec_period: bat_db_builder fills this
            )
            for fid, site in fid_to_site.items()
        ]

        self.log.info(
            f"Seeding recordings table in {self.db_path} "
            f"({len(rows):,} file_ids) …"
        )
        with sqlite3.connect(str(self.db_path)) as con:
            con.executescript(_SCHEMA_SQL)
            con.executemany(
                "INSERT OR IGNORE INTO recordings "
                "(file_id, filename, rec_site, rec_period) VALUES (?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.log.info("  recordings table seeded.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for :class:`FromScratchPostProcessor`.

    :return: Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog='from_scratch_postprocessing.py',
        description=(
            "Convert a classified measures CSV (output of "
            "chirp_measures_extraction.py + RF/CNN classifier) into the "
            "same bats_<ts>.parquet / bats_noise_<ts>.parquet / SQLite "
            "format produced by sb_measures_postprocessing.py.\n\n"
            "All downstream consumers (bat_db_builder, "
            "sb_measures_add_daytime_columns, RF/CNN training pipelines) "
            "work without modification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Typical usage after classification:
              python from_scratch_postprocessing.py \\
                  --measures-csv  /qnap/src/marsh_stanford_processed/measures_classified.csv \\
                  --rec-site      marsh \\
                  --dest-dir      /qnap/src/marsh_stanford_processed \\
                  --conf-thresh   0.50 \\
                  --build-db      /qnap/bats/chirp_meta.db \\
                  --add-daytime-columns

            Run before classification (all rows → noise parquet):
              python from_scratch_postprocessing.py \\
                  --measures-csv  /qnap/src/marsh_stanford_processed/measures.csv \\
                  --rec-site      marsh \\
                  --dest-dir      /qnap/src/marsh_stanford_processed
        """),
    )
    parser.add_argument(
        '--measures-csv',
        required=True, metavar='CSV', type=Path,
        help=(
            'Path to the measures CSV produced by chirp_measures_extraction.py, '
            'optionally with species and confidence columns appended by the '
            'RF/CNN classifier.'
        ),
    )
    parser.add_argument(
        '--rec-site',
        required=True, metavar='SITE',
        help="Recording site label (e.g. 'marsh').",
    )
    parser.add_argument(
        '--dest-dir',
        required=True, metavar='DIR', type=Path,
        help='Directory where the output .parquet files are written.',
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        default=FromScratchPostProcessor.CONF_ACCEPT_THRESH_DEFAULT,
        metavar='FLOAT',
        help=(
            'Minimum confidence score [0–1] for a chirp row to enter the '
            'clean parquet.  '
            f'Default: {FromScratchPostProcessor.CONF_ACCEPT_THRESH_DEFAULT}.'
        ),
    )
    parser.add_argument(
        '--build-db',
        default=None, dest='db_path', metavar='DB_PATH',
        help=(
            'Create (or open) a SQLite database at DB_PATH and seed its '
            'recordings table with the file_id → filename mapping produced '
            'during this run.  Required if --add-daytime-columns is used.'
        ),
    )
    parser.add_argument(
        '--add-daytime-columns',
        action='store_true', default=False, dest='add_daytime_cols',
        help=(
            'After writing the measures parquet, invoke '
            'sb_measures_add_daytime_columns.py to append was_daytime and '
            'time_of_day_pactime columns.  Requires --build-db.'
        ),
    )
    parser.add_argument(
        '--existing-parquet',
        default=None, dest='existing_parquet', metavar='PARQUET',
        type=Path,
        help=(
            'Path to the existing main bats_*.parquet (the SB-originated '
            'or previously merged one).  When supplied, the PathEncoder '
            'is extended from the existing file_map so new file_ids do '
            'not collide with existing ones, and the existing '
            'MeasureNormalizer is re-used so all rows share the same '
            'scale.  Required whenever the output of this run will be '
            'merged into the main parquet via merge_into_main.py. '
            'Omit only for fully standalone datasets.'
        ),
    )

    args = parser.parse_args()

    if not args.measures_csv.exists():
        parser.error(f'--measures-csv path does not exist: {args.measures_csv}')
    if args.add_daytime_cols and args.db_path is None:
        parser.error('--add-daytime-columns requires --build-db.')

    return args


def main() -> None:
    """
    CLI entry point for :class:`FromScratchPostProcessor`.

    :return: None
    """
    args = _parse_args()
    FromScratchPostProcessor(
        measures_csv      = args.measures_csv,
        rec_site          = args.rec_site,
        dest_dir          = args.dest_dir,
        conf_thresh       = args.conf_thresh,
        db_path           = args.db_path,
        add_daytime_cols  = args.add_daytime_cols,
        existing_parquet  = args.existing_parquet,
    )


if __name__ == '__main__':
    main()
