#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-31 11:29:40
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sb_measures_postprocessing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-14 15:36:51
#
# **********************************************************

"""
Combine SonoBat output from one or more recording sites into a single,
normalized dataset suitable for clustering and ML training.

Inputs are the *_CumulativeSonoBatch_*.txt and *_CumulativeParameters_*.txt
files produced by SonoBat's Long File Parser.  Multiple recording sites
(e.g. 'barn', 'lake2') are each associated with a root directory that is
searched recursively for those files.

Phase I — Collection and merging:

1. For each (root_dir, rec_site) pair, collect all
   *_CumulativeParameters_*.txt files (tab-separated; physical chirp
   measures) and *_CumulativeSonoBatch_*.txt files (species determinations).
2. Add a 'rec_site' column (pandas Categorical) to every row sourced from
   that site.
3. Add a 'chirp_idx' column: 0-based integer position of each chirp within
   its recording, derived by sorting on TimeInFile within each file_id.
4. Assign a single unified 'file_id' (int32) across all sites, built from
   the union of all file paths so that IDs are globally unique.
5. Merge measures and species estimates on file_id; log a warning for every
   species row that finds no matching measures row.

Phase II — Normalization:

1. Chirp measurements with amplitude-ratio or energy values exceeding
   Q3 + 5·IQR are excluded as likely recording artifacts.
2. Amplitude-ratio and acoustic energy features (HiFtoKnAmp,
   HiFtoUpprKnAmp, LdgToFcAmp, HiFtoFcAmp, KnToFcAmp, 1st5to15kHzExp,
   1st10kHzExp) are log-transformed (log1p) to address right-skewed
   distributions.
3. All remaining features are normalized with RobustScaler (median/IQR).
4. Identifier columns (file_id, chirp_idx, rec_site, cluster, TimeInFile)
   are excluded from normalization.
5. Confidence is computed as:

       confidence = Prob × (0.7 × (#Maj/#Accp) + 0.3 × log1p(#Accp)/log1p(30))

   ``#Maj/#Accp`` is the consensus fraction (naturally in (0,1]).
   ``log1p(#Accp)/log1p(30)`` is the log-normalized evidence weight ([0,1]).
   No fitted scaler is needed; all values are bounded by construction.

Phase III — Optional incremental update:

1. New SonoBat data can be merged with a previously saved dataset by
   inverse-transforming the existing data, appending the new raw data, and
   re-fitting the normalizer from scratch.

Output: two .parquet files written via BatsData.to_parquet():

  bats_<timestamp>.parquet       — clean dataset, confidence >= threshold,
                                   real species labels only.  Use for
                                   clustering, analysis, RF and CNN species
                                   training.

  bats_noise_<timestamp>.parquet — noise dataset, same schema.  Two row types:
                                   'unkn'  : low-confidence chirp rows with
                                             real measures.  Use for RF and
                                             CNN noise/reject class.
                                   'noise' : no-detection file rows, measures
                                             all NaN.  Use for CNN noise class
                                             only; find .wav via file_map.

Both files carry the same ancillary metadata (file_id↔path map, normalizer
state) in their Parquet schema metadata, so no sidecar files are needed.

CLI usage:
    python sb_measures_postprocessing.py \
        --dest-dir /path/to/output \
        --root-dirs /data2/barn /data2/lake2 \
        --rec-sites barn lake2 \
        --conf-thresh 0.50
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import textwrap

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from logging_service import LoggingService
from sklearn.preprocessing import RobustScaler

# ---------------------------- Class BatsData -------------

class BatsData:
    """
    Envelope that bundles the final merged DataFrame with the ancillary
    metadata needed to interpret and inverse-transform it.

    The main DataFrame (``self.df``) is a plain pandas DataFrame and should
    be used as such for all filtering, analysis, and ML work.  The envelope
    is only needed at the boundary — when reading from or writing to disk.

    The 'rec_site' column in ``df`` is a pandas Categorical whose categories
    are exactly the site names passed to the pipeline.  No integer site codes
    are ever exposed to the caller.

    Parquet is the sole serialization format.  All metadata (file_map,
    normalizer state) is stored as JSON in the Parquet schema metadata, so
    no sidecar files are required.

    Typical usage::

        # --- Produced by the pipeline ---
        bats = BatsData(df=df_final, file_map=encoder.id_to_path,
                        normalizer=normalizer, timestamp=ts)
        bats.to_parquet(dest_dir / 'measures.parquet')

        # --- Consumed downstream ---
        bats = BatsData.read_parquet('measures.parquet')
        df = bats.df                          # plain DataFrame from here on
        barn = df[df.rec_site == 'barn']
        bats.to_parquet(barn, 'barn_only.parquet')  # save subset

    :param df: Final merged, normalized DataFrame.
    :param file_map: Mapping from integer file_id to original file path string.
    :param normalizer: Fitted MeasureNormalizer instance.
    :param timestamp: ISO timestamp string from the processing run.
    """

    _META_KEY = b'bats_metadata'

    def __init__(
        self,
        df: pd.DataFrame,
        file_map: dict[int, str],
        normalizer: 'MeasureNormalizer',
        timestamp: str,
    ):
        self.df         = df
        self.file_map   = file_map
        self.normalizer = normalizer
        self.timestamp  = timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_parquet(self, df_or_path: 'pd.DataFrame | str | Path',
                   path: 'str | Path | None' = None) -> None:
        """
        Write a DataFrame (plus this envelope's metadata) to a Parquet file.

        Two call signatures are supported::

            bats.to_parquet('out.parquet')           # saves bats.df
            bats.to_parquet(modified_df, 'out.parquet')  # saves modified_df

        The 'rec_site' Categorical is preserved natively by PyArrow.
        All other metadata (file_map, normalizer state) is serialized as
        JSON in the Parquet schema metadata.

        :param df_or_path: Either a DataFrame to save, or the output path
                           (in which case ``self.df`` is saved).
        :param path: Output path; required when ``df_or_path`` is a DataFrame.
        :return: None
        """
        if isinstance(df_or_path, pd.DataFrame):
            df   = df_or_path
            dest = Path(path)
        else:
            df   = self.df
            dest = Path(df_or_path)

        meta_dict = {
            'timestamp' : self.timestamp,
            'file_map'  : {str(k): v for k, v in self.file_map.items()},
            'normalizer': self.normalizer.to_dict(),
        }
        meta_json = json.dumps(meta_dict)

        table    = pa.Table.from_pandas(df)
        existing = table.schema.metadata or {}
        new_meta = {**existing, self._META_KEY: meta_json.encode()}
        table    = table.replace_schema_metadata(new_meta)

        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)

    @classmethod
    def read_parquet(cls, path: 'str | Path') -> 'BatsData':
        """
        Load a Parquet file written by :meth:`to_parquet` and return a
        fully reconstructed BatsData envelope.

        The 'rec_site' column is restored as a pandas Categorical
        automatically by PyArrow.

        :param path: Path to the .parquet file.
        :return: BatsData with ``df``, ``file_map``, ``normalizer``, and
                 ``timestamp`` populated.
        :raises KeyError: If the file is missing the expected metadata key.
        """
        table    = pq.read_table(path)
        raw_meta = table.schema.metadata or {}

        if cls._META_KEY not in raw_meta:
            raise KeyError(
                f"Parquet file {path!r} has no '{cls._META_KEY.decode()}' "
                f"metadata key — was it written by BatsData.to_parquet()?"
            )

        meta_dict  = json.loads(raw_meta[cls._META_KEY].decode())
        file_map   = {int(k): v for k, v in meta_dict['file_map'].items()}
        normalizer = MeasureNormalizer.from_dict(meta_dict['normalizer'])
        timestamp  = meta_dict['timestamp']
        df         = table.to_pandas()

        return cls(df=df, file_map=file_map,
                   normalizer=normalizer, timestamp=timestamp)


# ---------------------------- Class CompositeSpecies -------------

class CompositeSpecies:
    '''
    Holds a set of bat species names whose order
    is alphabetical, no matter how the names were added.
    Importantly, one can add two instances to a set, and
    if the member species are the same, only one of those
    instances will enter the set.

    Two instances are equal if their member sets are equal.
    '''
    def __init__(self, member_species: list[str] | str):
        '''
        Add members as list of species names, or as a 
        string of species separated by '/'

        :param member_species: species names
        :raises ValueError: if species strings are neither a list,
            not a string with slash as separator
        '''
        if isinstance(member_species, str):
            self.members = set(member_species.split('/'))
            if len(self.members) < 2:
                raise ValueError(f"CompositeSpecies consist of at least two species")
        else:
            self.members = set(member_species)
    def __eq__(self, other): 
        other_members: set[str] = other.members
        return other_members == self.members
    def __str__(self):
        # Species group is the alpha-sorted member species
        # Separated with slashes:
        return '/'.join(sorted(self.members))
    def __repr__(self):
        info = f"<CompositeSpecies {str(self)} at {hex(id(self))}>"
        return info
    def __hash__(self):
        return hash(str(self))

# --------------------------- Class SonoBatPostProcessor -------------

class SonoBatPostProcessor:

    MIN_ACCEPT_PROB       = 0.9
    WEIGHT_ON_CONSENSUS   = 0.7
    WEIGHT_ON_EVIDENCE    = 0.3
    # Fixed ceiling for log-normalizing #Accp.  Values above this are clipped
    # to 1.0 by the evidence formula.  30 comfortably covers the observed
    # maximum (~28) while leaving a small margin.
    ACCP_LOG_CEIL         = 30

    RELEVANT_MEASURES_COLS = [
        'TimeInFile', 'PrecedingIntrvl', 'HiFreq', 'Bndwdth', 'FreqMaxPwr',
        'PrcntMaxAmpDur', 'FreqKnee', 'PrcntKneeDur', 'StartF', 'UpprKnFreq',
        'HiFtoUpprKnAmp', 'HiFtoKnAmp', 'HiFtoFcAmp', 'UpprKnToKnAmp',
        'KnToFcAmp', 'LdgToFcAmp', 'FreqCtr', 'FFwd32dB', 'FFwd20dB',
        'FFwd15dB', 'FBak5dB', 'FFwd5dB', 'Bndw32dB', 'Amp1stQrtl',
        'Amp2ndQrtl', 'Amp3rdQrtl', 'Amp4thQrtl', '1st10kHzSlp',
        '1st5to15kHzSlp', '1st10kHzExp', '1st5to15kHzExp', 'AmpK@start'
    ]    

    #------------------------------------
    # Constructor
    #-------------------
    
    CONF_ACCEPT_THRESH_DEFAULT = 0.50

    def __init__(self,
                 root_dirs:   list[str | Path],
                 rec_sites:   list[str],
                 dest_dir:    str | Path,
                 conf_thresh: float = CONF_ACCEPT_THRESH_DEFAULT,
                 ):
        """
        :param root_dirs:   One root directory per recording site, searched
                            recursively for SonoBat output files.
        :param rec_sites:   Site label for each root_dir (must be same length).
        :param dest_dir:    Directory where output Parquet file is written.
        :param conf_thresh: Minimum confidence score for a chirp row to be
                            retained in the final dataset.  Rows with NaN
                            confidence are also dropped.  Default 0.50.
        :raises ValueError: If ``root_dirs`` and ``rec_sites`` differ in length.
        """
        if len(root_dirs) != len(rec_sites):
            raise ValueError(
                f"root_dirs and rec_sites must be the same length; "
                f"got {len(root_dirs)} root_dirs and {len(rec_sites)} rec_sites"
            )

        self.log = LoggingService()
        self.timestamp = datetime.now().isoformat().replace(':', '_')

        self.root_dirs   = [Path(r) for r in root_dirs]
        self.rec_sites   = rec_sites
        self.dest_dir    = Path(dest_dir)
        self.conf_thresh = conf_thresh

        self.rejected_entries = []
        self.composite_species: set[CompositeSpecies] = set()

        # Build categorical dtype once from the known site list so that
        # every batch gets the same category set regardless of which sites
        # are actually present in that batch.
        self.site_dtype = pd.CategoricalDtype(
            categories=sorted(set(rec_sites)), ordered=False
        )

        # Collect raw measures and species across all sites.
        # A single PathEncoder is built over the union of both path sets
        # to guarantee globally unique file_ids.
        df_measures_raw, df_species_raw = self._collect_all_raw(
            self.root_dirs, self.rec_sites
        )

        # Unified path encoding
        all_paths = pd.concat(
            [df_measures_raw['Path'], df_species_raw['Path']]
        ).unique()
        self.path_encoder = PathEncoder.from_paths(all_paths, sort_paths=True)

        df_measures_encoded = self.path_encoder.encode_df(df_measures_raw)
        df_species_encoded  = self.path_encoder.encode_df(df_species_raw)

        # Normalize measures
        meas_normalizer = MeasureNormalizer()
        df_normalized   = meas_normalizer.fit_transform(df_measures_encoded)
        self.normalizer = meas_normalizer

        # Species post-processing
        df_species_final = self._finalize_species(df_species_encoded)

        # Merge: produces clean df and noise df
        df_clean, df_noise = self._merge(
            df_normalized, df_species_final, self.conf_thresh
        )

        # --- Main (clean) dataset ---
        self.bats_data = BatsData(
            df        = df_clean,
            file_map  = self.path_encoder.id_to_path,
            normalizer= self.normalizer,
            timestamp = self.timestamp,
        )
        out_path = self.dest_dir / f"bats_{self.timestamp}.parquet"
        self.bats_data.to_parquet(out_path)
        self.log.info(f"Wrote {out_path}")

        # --- Noise dataset ---
        self.bats_noise = BatsData(
            df        = df_noise,
            file_map  = self.path_encoder.id_to_path,
            normalizer= self.normalizer,
            timestamp = self.timestamp,
        )
        noise_path = self.dest_dir / f"bats_noise_{self.timestamp}.parquet"
        self.bats_noise.to_parquet(noise_path)
        self.log.info(f"Wrote {noise_path}")
            
    #------------------------------------
    # _collect_all_raw
    #-------------------

    def _collect_all_raw(
        self,
        root_dirs: list[Path],
        rec_sites: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Iterate over (root_dir, rec_site) pairs, loading raw measures and
        species DataFrames from each.  Both DataFrames retain their 'Path'
        column (not yet encoded) and gain a 'rec_site' Categorical column.

        The 'Path' column in both DataFrames is normalized to the bare
        filename stem (no directory, no extension) so that Windows-style
        paths from the SonoBat VM (e.g. ``Y:\\batch4\\...\\lake2_-..._2secs.wav``)
        match the Linux paths in the CumulativeParameters files.  The stem
        is the natural unique key used as 'Filename' throughout the pipeline.

        'chirp_idx' is added to the measures DataFrame: a 0-based integer
        giving the position of each chirp within its recording, derived by
        sorting on TimeInFile within each unique Path stem.

        :param root_dirs: Directories to search, one per site.
        :param rec_sites: Site label for each directory.
        :return: Tuple of (df_measures_raw, df_species_raw), both with
                 'Path' and 'rec_site' columns, measures also with
                 'chirp_idx'.
        :raises FileNotFoundError: If no matching files are found under any
                                   root_dir for either file type.
        """
        measures_batches: list[pd.DataFrame] = []
        species_batches:  list[pd.DataFrame] = []

        for root_dir, site in zip(root_dirs, rec_sites):

            # --- measures ---
            m_files = self._find_sonobatch_measures_files(root_dir)
            if not m_files:
                raise FileNotFoundError(
                    f"No '*_CumulativeParameters_*.txt' files found under {root_dir}"
                )
            for mf in m_files:
                df = pd.read_csv(mf, sep='\t')
                cols_to_keep = ['Path'] + SonoBatPostProcessor.RELEVANT_MEASURES_COLS
                df = df[cols_to_keep]
                df['rec_site'] = pd.Categorical(
                    [site] * len(df), dtype=self.site_dtype
                )
                measures_batches.append(df)

            # --- species ---
            s_files = self._find_sonobatch_species_id_files(root_dir)
            if not s_files:
                raise FileNotFoundError(
                    f"No '*_CumulativeSonoBatch_*.txt' files found under {root_dir}"
                )
            for sf in s_files:
                df = pd.read_csv(sf, sep='\t')
                cols_to_keep = ['Path', 'SppAccp', 'Prob', '#Maj', '#Accp']
                df = df[cols_to_keep]
                df['rec_site'] = pd.Categorical(
                    [site] * len(df), dtype=self.site_dtype
                )
                species_batches.append(df)

        df_measures = pd.concat(measures_batches, ignore_index=True)
        df_species  = pd.concat(species_batches,  ignore_index=True)

        # Normalize the 'Path' column in both DataFrames to the bare filename
        # stem (no directory, no extension).  The CumulativeSonoBatch files
        # written by SonoBat on the Windows VMs carry Windows-style paths
        # (e.g. "Y:\batch4\chopped\...\lake2_-20221226_204358_2secs.wav"),
        # while the CumulativeParameters files written on Linux carry Linux
        # paths.  Reducing both to the stem makes the PathEncoder join work
        # regardless of which machine produced the file.  The stem is already
        # the natural unique key used as 'Filename' throughout the pipeline.
        def _to_stem(path_series: pd.Series) -> pd.Series:
            # Handle both forward-slash and backslash separators.
            return (
                path_series
                .astype(str)
                .str.replace('\\', '/', regex=False)   # normalise Windows seps
                .apply(lambda p: Path(p).stem)
            )

        n_win = df_species['Path'].astype(str).str.contains('\\\\', regex=False).sum()
        if n_win:
            self.log.info(
                f"Normalizing {n_win:,} Windows-style paths in species data to stems."
            )

        df_measures['Path'] = _to_stem(df_measures['Path'])
        df_species['Path']  = _to_stem(df_species['Path'])

        # chirp_idx: 0-based rank by TimeInFile within each recording (stem)
        df_measures['chirp_idx'] = (
            df_measures
            .groupby('Path', sort=False)['TimeInFile']
            .rank(method='first')
            .sub(1)
            .astype('int32')
        )

        return df_measures, df_species

    #------------------------------------
    # _finalize_species
    #-------------------

    def _finalize_species(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply species-specific post-processing to the raw, path-encoded
        species DataFrame: compute consensus and evidence columns from raw
        SonoBat counts, add a confidence column, and normalize composite
        species names.

        Consensus is the natural fraction ``#Maj / #Accp``, which is always
        in ``(0, 1]`` because ``#Maj <= #Accp`` by SonoBat's definition.
        No scaling is needed or appropriate here.

        Evidence is ``log1p(#Accp)`` normalized by the log of a fixed ceiling
        (``ACCP_LOG_CEIL``).  This captures the real but saturating gain in
        confidence from having more accepted pulses, compresses the long right
        tail, and maps the result to ``[0, 1]`` without a fitted transformer.

        :param df: Species DataFrame with 'file_id' already assigned and
                   'Path' column already dropped.
        :return: Processed species DataFrame.
        """
        # consensus: proper fraction in (0, 1] — no scaling required
        # Guard against zero #Accp (should not occur, but be safe)
        df['Maj_scaled']  = (df['#Maj'] / df['#Accp'].replace(0, np.nan)).fillna(0.0)

        # evidence: log-compressed #Accp, normalized to [0, 1]
        df['Accp_scaled'] = (
            np.log1p(df['#Accp']) / np.log1p(SonoBatPostProcessor.ACCP_LOG_CEIL)
        ).clip(0.0, 1.0)

        df = SonoBatPostProcessor._add_confidence_column(df)
        df = SonoBatPostProcessor._normalize_composite_species(df)
        return df

    #------------------------------------
    # _merge
    #-------------------

    def _merge(
        self,
        df_measures:  pd.DataFrame,
        df_species:   pd.DataFrame,
        conf_thresh:  float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Left-join species columns onto the normalized measures DataFrame on
        'file_id', then split into a clean DataFrame and a noise DataFrame.

        **Clean DataFrame** (returned first):
            Rows whose confidence >= ``conf_thresh`` and species is a real
            SonoBat identification.  This is the main working dataset.

        **Noise DataFrame** (returned second):
            Two kinds of rows, same schema as the clean DataFrame:

            - ``species == 'unkn'``: low-confidence chirp rows
              (``0 < confidence < conf_thresh``).  Measures are real.
              Use for both RF and CNN noise/reject class.
            - ``species == 'noise'``: no-detection file rows — paths that
              appeared in the SonoBatch file but had no chirp measurements
              (NaN SppAccp).  All measure columns are NaN.  One row per
              unmatched chop.  Use for CNN noise class only.

        Logs a warning for every species file_id that finds no matching
        measures row.

        :param df_measures: Normalized measures DataFrame with 'file_id'.
        :param df_species:  Processed species DataFrame with 'file_id'.
        :param conf_thresh: Minimum confidence for the clean DataFrame.
        :return: Tuple of (df_clean, df_noise).
        """
        meas_ids    = set(df_measures['file_id'].unique())
        species_ids = set(df_species['file_id'].unique())
        unmatched   = species_ids - meas_ids

        if unmatched:
            self.log.warn(
                f"{len(unmatched)} file_id(s) in species data have no "
                f"matching measures rows — added as 'noise' rows:"
            )
            for fid in sorted(unmatched):
                path = self.path_encoder.id_to_path.get(fid, '<unknown>')
                self.log.warn(f"  file_id={fid}  path={path}")

        # Deduplicate species rows: the Cumulative files accumulate across
        # multiple SonoBat runs, so the same Path (file_id) can appear more
        # than once when multiple batches are passed as root_dirs.  Keep the
        # row with the highest confidence so the left-join below is 1:1 on
        # file_id and produces no duplicate chirp rows.
        n_species_before = len(df_species)
        df_species = (
            df_species
            .sort_values('confidence', ascending=False)
            .drop_duplicates(subset='file_id', keep='first')
        )
        n_dropped = n_species_before - len(df_species)
        if n_dropped:
            self.log.info(
                f"Dropped {n_dropped:,} duplicate species rows "
                f"(same file_id from overlapping Cumulative files); "
                f"{len(df_species):,} unique file_ids remain."
            )

        # Left-join: every chirp row gets species/confidence if available
        df_merged = df_measures.merge(
            df_species[['file_id', 'SppAccp', 'confidence']],
            on='file_id',
            how='left',
        )
        df_merged.rename(columns={'SppAccp': 'species'}, inplace=True)

        # --- Split into clean and noise ---

        # Clean: real species ID, confidence above threshold
        clean_mask = (
            df_merged['confidence'].notna() &
            (df_merged['confidence'] >= conf_thresh) &
            df_merged['species'].notna()
        )
        df_clean = df_merged[clean_mask].copy()

        # Noise part 1: low-confidence rows with real measures
        unkn_mask = (
            df_merged['confidence'].notna() &
            (df_merged['confidence'] > 0) &
            (df_merged['confidence'] < conf_thresh)
        )
        df_unkn = df_merged[unkn_mask].copy()
        df_unkn['species'] = 'unkn'

        # Noise part 2: no-detection rows (NaN confidence / NaN species)
        # One row per unmatched file_id, measures all NaN
        if unmatched:
            measure_cols = SonoBatPostProcessor.RELEVANT_MEASURES_COLS
            noise_rows = []
            for fid in sorted(unmatched):
                # rec_site: find it from the species df
                site_vals = df_species.loc[
                    df_species['file_id'] == fid, 'rec_site'
                ]
                site = site_vals.iloc[0] if len(site_vals) else None
                row = {col: np.nan for col in measure_cols}
                row['file_id']    = fid
                row['rec_site']   = site
                row['chirp_idx']  = np.nan
                row['species']    = 'noise'
                row['confidence'] = 0.0
                noise_rows.append(row)
            df_nodet = pd.DataFrame(noise_rows)
            # Restore rec_site as Categorical
            df_nodet['rec_site'] = pd.Categorical(
                df_nodet['rec_site'], dtype=self.site_dtype
            )
        else:
            df_nodet = pd.DataFrame(columns=df_clean.columns)

        df_noise = pd.concat([df_unkn, df_nodet], ignore_index=True)

        self.log.info(
            f"Split: {len(df_clean)} clean rows, "
            f"{len(df_unkn)} 'unkn' rows, "
            f"{len(df_nodet)} 'noise' rows"
        )

        return df_clean, df_noise


    
    @classmethod
    def _normalize_composite_species(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces slash-separated species strings in 'SppAccp' with their
        canonical CompositeSpecies form (alphabetically sorted members),
        so that 'Lano/Laci' and 'Laci/Lano' both become 'Laci/Lano'.

        Only the ~14% of rows containing slashes are processed. Since there
        are only O(10) distinct slash-combinations, CompositeSpecies is
        constructed exactly once per unique value, and pandas .map() handles
        the remaining 12K+ row updates in bulk.

        :param df: DataFrame with a 'SppAccp' column
        :return: DataFrame with 'SppAccp' normalized in-place
        :rtype: pd.DataFrame
        """
        slash_mask = df['SppAccp'].str.contains('/', regex=False, na=False)

        unique_slash = df.loc[slash_mask, 'SppAccp'].unique()  # ~36 values
        canon_map = {s: str(CompositeSpecies(s)) for s in unique_slash}

        df.loc[slash_mask, 'SppAccp'] = df.loc[slash_mask, 'SppAccp'].map(canon_map)

        return df

    #------------------------------------
    # _add_confidence_column
    #-------------------
    
    @classmethod
    def _add_confidence_column(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized computation of a ``confidence`` score in ``[0, 1]`` from
        SonoBat's ``Prob``, ``#Maj``, and ``#Accp`` columns.

        Formula
        -------
        ::

            confidence = Prob × (α × consensus + β × evidence)

            consensus = #Maj / #Accp          — fraction of pulses that voted
                                                for the accepted species;
                                                always in (0, 1] by definition
            evidence  = log1p(#Accp)          — log-compressed pulse count,
                        ─────────────────       normalized to [0, 1] via a
                        log1p(ACCP_LOG_CEIL)    fixed ceiling (no fitted scaler)

            α = WEIGHT_ON_CONSENSUS = 0.7
            β = WEIGHT_ON_EVIDENCE  = 0.3

        Both ``consensus`` and ``evidence`` are naturally bounded in ``[0, 1]``,
        so the bracketed term is also in ``[0, 1]``, and multiplying by
        ``Prob ∈ [0, 1]`` keeps ``confidence`` in ``[0, 1]`` without clipping —
        except for composite-species rows where slash-summed ``Prob`` can
        marginally exceed 1.0, which a final ``clip(0, 1)`` handles.

        The intermediate columns ``Maj_scaled`` and ``Accp_scaled`` are
        computed by :meth:`_finalize_species` before this method is called:

        * ``Maj_scaled``  = ``#Maj / #Accp``
        * ``Accp_scaled`` = ``log1p(#Accp) / log1p(ACCP_LOG_CEIL)``

        Handles three forms of ``Prob``:

        * ``NaN``       → ``confidence = 0.0``
        * plain float   → used directly
        * ``'x/y[/z]'`` → component probabilities are summed (composite IDs)

        :param df: DataFrame with columns ``Prob``, ``Maj_scaled``,
                   ``Accp_scaled``.
        :return: DataFrame with new ``confidence`` column added in-place.
        """
        prob_raw = df['Prob']

        # --- Resolve probability values vectorized ---

        # Mask 1: NaN rows
        nan_mask = prob_raw.isna()

        # Mask 2: slash-separated strings like '0.46/0.52'
        slash_mask = (~nan_mask) & prob_raw.astype(str).str.contains('/', regex=False)

        # Sum slash-separated probabilities per row (apply only to small subset)
        slash_probs = (
            prob_raw[slash_mask]
            .astype(str)
            .str.split('/')
            .apply(lambda parts: sum(float(p) for p in parts))
        )

        # Mask 3: plain numeric (everything else)
        plain_mask = ~(nan_mask | slash_mask)

        # Assemble resolved probability series
        prob = pd.Series(0.0, index=df.index)
        prob[plain_mask] = prob_raw[plain_mask].astype(float)
        prob[slash_mask] = slash_probs

        # --- Confidence formula ---
        # Maj_scaled  = #Maj / #Accp         (consensus, naturally in (0,1])
        # Accp_scaled = log1p(#Accp) / log1p(ACCP_LOG_CEIL)  (evidence, [0,1])
        df['confidence'] = prob * (
            cls.WEIGHT_ON_CONSENSUS * df['Maj_scaled']
            + cls.WEIGHT_ON_EVIDENCE * df['Accp_scaled']
        )

        # NaN rows get 0.0 (already 0.0 from prob init, but be explicit)
        df.loc[nan_mask, 'confidence'] = 0.0

        # Clip: only necessary for composite-species rows where summed Prob
        # marginally exceeds 1.0; all other values are bounded by construction.
        df['confidence'] = df['confidence'].clip(0.0, 1.0)

        return df

    #------------------------------------
    # _species_confidence
    #-------------------
    
    @staticmethod
    def _species_confidence(prob_info: str | float,
                           accp_n: float | int,
                           maj_n:  float | int,
                           ) -> float:
        '''
        NOTE: Use the vectorized version of this method instead:
              :meth:`_add_confidence_column`.  This row-by-row version is kept
              as a readable reference and for unit-testing individual rows.

        Combine the several pieces of confidence evidence that SonoBat
        produces into a single score in ``[0, 1]``.

        Formula::

            confidence = Prob × (α × consensus + β × evidence)

            consensus = maj_n / accp_n          — always in (0, 1] because
                                                  maj_n <= accp_n by definition
            evidence  = log1p(accp_n)           — saturating compression of
                        ─────────────────         pulse-count evidence,
                        log1p(ACCP_LOG_CEIL)      normalized to [0, 1]

            α = WEIGHT_ON_CONSENSUS = 0.7
            β = WEIGHT_ON_EVIDENCE  = 0.3

        ``prob_info`` is either a plain probability like ``0.9834``, or a
        slash-string like ``'0.46/0.52'`` for composite species IDs.  The
        latter are summed, which can produce values marginally above 1.0;
        the final ``clip`` handles that.

        :param prob_info: SonoBat ``Prob`` value — a float or slash-string.
        :param accp_n: Raw ``#Accp`` count (accepted pulses in the chop).
        :param maj_n:  Raw ``#Maj`` count (pulses voting for the accepted ID).
        :return: Confidence in ``[0, 1]``.
        '''
        if pd.isna(prob_info):
            return 0.0
        try:
            prob = float(prob_info)
        except ValueError:
            prob = sum(float(p) for p in prob_info.split('/'))

        if accp_n == 0:
            return 0.0

        consensus = maj_n / accp_n
        evidence  = (np.log1p(accp_n) /
                     np.log1p(SonoBatPostProcessor.ACCP_LOG_CEIL))

        raw = prob * (SonoBatPostProcessor.WEIGHT_ON_CONSENSUS * consensus
                      + SonoBatPostProcessor.WEIGHT_ON_EVIDENCE * evidence)
        return float(np.clip(raw, 0.0, 1.0))
    
    #------------------------------------
    # _find_sonobatch_species_id_files
    #-------------------

    def _find_sonobatch_species_id_files(self, root: str | Path) -> list[Path]:
        """
        Return all paths under *root* whose filename matches the pattern
        ``*_CumulativeSonoBatch_v<version>.txt``, replicating the behaviour of::

            find . -regextype posix-egrep -regex ".*_CumulativeSonoBatch_v[0-9.]+\\.txt"

        :param root: Directory root to search recursively.
        :return: Sorted list of matching :class:`~pathlib.Path` objects.
        """
        pattern = re.compile(r'.*_CumulativeSonoBatch_v[0-9.]+\.txt$')
        root = Path(root)
        return sorted(p for p in root.rglob('*') if pattern.match(str(p)))    

    #------------------------------------
    # _find_sonobatch_measures_files
    #-------------------

    def _find_sonobatch_measures_files(self, root: str | Path) -> list[Path]:
        """
        Return all paths under *root* whose filename matches the pattern
        ``*_CumulativeParameters_v<version>.txt``, replicating the behaviour of::

            find . -regextype posix-egrep -regex ".*_CumulativeParameters_v[0-9.]+\\.txt"

        :param root: Directory root to search recursively.
        :return: Sorted list of matching :class:`~pathlib.Path` objects.
        """
        pattern = re.compile(r'.*_CumulativeParameters_v[0-9.]+\.txt$')
        root = Path(root)
        return sorted(p for p in root.rglob('*') if pattern.match(str(p)))

# ----------------------------- Class MeasureNormalizer -----------------

NON_FEATURE_COLS = {'file_id', 'chirp_idx', 'cluster', 'TimeInFile', 'rec_site'}


class MeasureNormalizer:
    """
    Clean and normalize a SonoBat chirp measures DataFrame.

    Pipeline stages:
      1. Separate numeric feature columns from all others (non-numeric,
         and known non-feature columns such as identifiers and targets).
      2. Identify Tier 1 columns (outlier_factor > outlier_factor_thresh)
         and drop rows whose value in any Tier 1 column exceeds
         Q3 + fence_iqr_mult * IQR.
      3. Log-transform (log1p) columns with outlier_factor > log_transform_thresh.
      4. Apply RobustScaler to all numeric feature columns.
      5. Rejoin all non-feature columns (non-numeric, identifiers, targets)
         to the normalized result, aligned by index, before returning.

    The returned DataFrame from fit_transform() and transform() therefore
    contains both the normalized feature columns and all original non-feature
    columns, with outlier rows absent.

    The fitted normalizer can be saved to disk and reloaded, enabling:
      - Exact inverse-transform back to approximately original scale.
      - Applying the same scaling to new data without refitting.

    :param outlier_factor_thresh: outlier_factor above which a column is Tier 1.
    :param fence_iqr_mult: IQR multiplier for the per-row outlier fence.
    :param log_transform_thresh: outlier_factor above which log1p is applied.
    """

    def __init__(
        self,
        outlier_factor_thresh: float = 1000.0,
        fence_iqr_mult: float = 5.0,
        log_transform_thresh: float = 10.0,
    ):
        self.outlier_factor_thresh = outlier_factor_thresh
        self.fence_iqr_mult = fence_iqr_mult
        self.log_transform_thresh = log_transform_thresh

        self.log = LoggingService()

        self.numeric_cols_: list[str] = []
        self.feature_cols_: list[str] = []
        self.tier1_cols_: list[str] = []
        self.log_cols_: list[str] = []
        self.diagnostic_df_: pd.DataFrame | None = None
        self.scaler_: RobustScaler | None = None
        self.n_rows_before_: int = 0
        self.n_rows_after_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df_measures: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full normalization pipeline on the concatenated measures DataFrame.

        Returns a DataFrame containing both the normalized numeric feature columns
        and all original non-feature columns (non-numeric, identifiers, targets),
        with outlier rows removed. The caller needs no post-processing to recover
        non-feature columns.

        :param df_measures: Raw concatenated SonoBat measures DataFrame.
        :return: Normalized feature columns joined with non-feature columns,
                 index-aligned, with outlier rows absent.
        """
        self.n_rows_before_ = len(df_measures)

        numeric_df = self._extract_numeric_features(df_measures)
        self.diagnostic_df_ = self._compute_diagnostics(numeric_df)

        self.tier1_cols_ = self._identify_tier(self.outlier_factor_thresh)
        self.log_cols_   = self._identify_tier(self.log_transform_thresh)

        self.log.info(
            f"Tier 1 columns (outlier_factor > {self.outlier_factor_thresh}): "
            f"{self.tier1_cols_}"
        )
        self.log.info(
            f"Log-transform columns (outlier_factor > {self.log_transform_thresh}): "
            f"{self.log_cols_}"
        )

        filtered_df = self._filter_outlier_rows(numeric_df)
        self.n_rows_after_ = len(filtered_df)
        self.log.info(
            f"Row filtering: {self.n_rows_before_} → {self.n_rows_after_} rows "
            f"({self.n_rows_before_ - self.n_rows_after_} removed, "
            f"{100*(self.n_rows_before_ - self.n_rows_after_)/self.n_rows_before_:.1f}%)"
        )

        log_transformed_df = self._apply_log_transform(filtered_df)
        normalized_df      = self._apply_robust_scaling(log_transformed_df)

        return self._rejoin_non_feature_cols(df_measures, normalized_df)

    def transform(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the already-fitted normalizer to new data without refitting.

        Intended for new SonoBat recordings that arrive after the initial fit,
        when preserving the original normalized space is desired. No row
        filtering is applied — the caller is responsible for pre-filtering.

        Returns a DataFrame containing both normalized feature columns and
        all original non-feature columns from df_new, index-aligned.

        :param df_new: Raw measures DataFrame with the same columns as
                       the original fit data.
        :return: Normalized feature columns joined with non-feature columns.
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()

        numeric_df = df_new.select_dtypes(include=['number'])
        numeric_df = numeric_df.drop(
            columns=[c for c in NON_FEATURE_COLS if c in numeric_df.columns]
        )
        numeric_df = numeric_df.reindex(columns=self.feature_cols_, fill_value=np.nan)

        log_transformed_df = self._apply_log_transform(numeric_df)
        scaled_array = self.scaler_.transform(log_transformed_df)
        normalized_df = pd.DataFrame(
            scaled_array, columns=self.feature_cols_, index=numeric_df.index
        )

        return self._rejoin_non_feature_cols(df_new, normalized_df)

    def inverse_transform(self, df_normalized: pd.DataFrame) -> pd.DataFrame:
        """
        Recover approximately original-scale values from a normalized DataFrame.

        Operates only on the numeric feature columns present in df_normalized,
        leaving any non-feature columns (if the full rejoined DataFrame is passed)
        untouched and passed through as-is.

        Applies inverse operations in reverse pipeline order:
          1. RobustScaler inverse_transform  (undo median/IQR scaling)
          2. expm1 on log-transformed columns (undo log1p)

        Note: rows dropped during outlier filtering are not recoverable.
        Recovered values may differ slightly from originals due to
        floating-point rounding.

        :param df_normalized: DataFrame in the normalized feature space,
                              as produced by fit_transform or transform.
                              May include non-feature columns.
        :return: DataFrame with feature columns in approximately original
                 measurement units, non-feature columns passed through unchanged.
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()

        feature_cols_present = [c for c in self.feature_cols_ if c in df_normalized.columns]
        non_feature_cols_present = [
            c for c in df_normalized.columns if c not in self.feature_cols_
        ]

        unscaled_array = self.scaler_.inverse_transform(
            df_normalized[feature_cols_present]
        )
        df_unscaled = pd.DataFrame(
            unscaled_array,
            columns=feature_cols_present,
            index=df_normalized.index
        )

        cols_to_exp = [c for c in self.log_cols_ if c in df_unscaled.columns]
        if cols_to_exp:
            df_unscaled[cols_to_exp] = np.expm1(df_unscaled[cols_to_exp])

        if non_feature_cols_present:
            df_unscaled = df_unscaled.join(df_normalized[non_feature_cols_present])

        return df_unscaled

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted normalizer to disk as a standalone joblib file.

        This is an escape hatch for cases where the normalizer needs to be
        shared or inspected outside of a BatsData Parquet file.  For normal
        pipeline use, the normalizer travels inside the Parquet metadata via
        :meth:`to_dict` / :meth:`from_dict` and this method is not needed.

        :param path: Destination file path (conventionally .joblib).
        :return: None
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()
        joblib.dump(self, path)
        self.log.info(f"Normalizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'MeasureNormalizer':
        """
        Reload a previously saved MeasureNormalizer from disk.

        :param path: Path to a file saved by MeasureNormalizer.save().
        :return: Fully restored MeasureNormalizer instance.
        """
        log = LoggingService()
        normalizer = joblib.load(path)
        log.info(f"Normalizer loaded from {path}")
        return normalizer

    def report(self) -> None:
        """
        Print a human-readable summary of the normalization decisions.

        :return: None
        """
        if self.diagnostic_df_ is None:
            print("Call fit_transform() first.")
            return

        print(f"\n{'='*60}")
        print(f"MeasureNormalizer Report")
        print(f"{'='*60}")
        print(f"Rows before filtering : {self.n_rows_before_}")
        print(f"Rows after filtering  : {self.n_rows_after_}")
        print(f"Rows dropped          : {self.n_rows_before_ - self.n_rows_after_}")
        print(f"\nTier 1 (row fence applied, outlier_factor > "
              f"{self.outlier_factor_thresh}):")
        for col in self.tier1_cols_:
            of = self.diagnostic_df_.loc[col, 'outlier_factor']
            print(f"  {col:<30s}  outlier_factor={of:.1f}")
        print(f"\nLog-transformed (outlier_factor > {self.log_transform_thresh}):")
        for col in self.log_cols_:
            of = self.diagnostic_df_.loc[col, 'outlier_factor']
            print(f"  {col:<30s}  outlier_factor={of:.1f}")
        print(f"\nFeature columns after normalization ({len(self.feature_cols_)}):")
        print(f"  {self.feature_cols_}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rejoin_non_feature_cols(
        self,
        df_original: pd.DataFrame,
        df_normalized: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join non-feature columns from df_original back onto df_normalized.

        Uses df_normalized's index (the survivor set after outlier filtering)
        to slice df_original, so dropped rows are automatically absent.
        The join is left on df_normalized, guaranteeing no phantom rows appear.

        :param df_original: The raw input DataFrame passed to fit_transform
                            or transform, from which non-feature columns are drawn.
        :param df_normalized: The normalized feature-only DataFrame to join onto.
        :return: df_normalized with non-feature columns appended, index-aligned.
        """
        non_feature_cols = [
            c for c in df_original.columns if c not in self.feature_cols_
        ]
        if not non_feature_cols:
            return df_normalized

        return df_normalized.join(
            df_original.loc[df_normalized.index, non_feature_cols],
            how='left'
        )

    def to_dict(self) -> dict:
        """
        Serialize the fitted normalizer state to a JSON-safe dictionary.

        Stores the RobustScaler's center and scale arrays as plain lists so
        the result can be embedded directly in Parquet schema metadata
        without requiring pickle.

        :return: Dictionary suitable for ``json.dumps()``.
        :raises RuntimeError: If called before fit_transform.
        """
        self._assert_fitted()
        return {
            'outlier_factor_thresh': self.outlier_factor_thresh,
            'fence_iqr_mult'       : self.fence_iqr_mult,
            'log_transform_thresh' : self.log_transform_thresh,
            'feature_cols'         : self.feature_cols_,
            'numeric_cols'         : self.numeric_cols_,
            'tier1_cols'           : self.tier1_cols_,
            'log_cols'             : self.log_cols_,
            'scaler_center'        : self.scaler_.center_.tolist(),
            'scaler_scale'         : self.scaler_.scale_.tolist(),
            'n_rows_before'        : self.n_rows_before_,
            'n_rows_after'         : self.n_rows_after_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MeasureNormalizer':
        """
        Reconstruct a fitted MeasureNormalizer from a dictionary produced
        by :meth:`to_dict`.

        The restored instance is fully functional for :meth:`transform` and
        :meth:`inverse_transform`.  :meth:`report` will print a "not fitted"
        message because the diagnostic DataFrame is not serialized (it is a
        development aid, not needed for inference).

        :param d: Dictionary as produced by ``to_dict()``.
        :return: Fully restored MeasureNormalizer ready for ``transform()``
                 and ``inverse_transform()``.
        """
        obj = cls(
            outlier_factor_thresh = d['outlier_factor_thresh'],
            fence_iqr_mult        = d['fence_iqr_mult'],
            log_transform_thresh  = d['log_transform_thresh'],
        )
        obj.feature_cols_  = d['feature_cols']
        obj.numeric_cols_  = d['numeric_cols']
        obj.tier1_cols_    = d['tier1_cols']
        obj.log_cols_      = d['log_cols']
        obj.n_rows_before_ = d['n_rows_before']
        obj.n_rows_after_  = d['n_rows_after']

        scaler = RobustScaler()
        scaler.center_ = np.array(d['scaler_center'])
        scaler.scale_  = np.array(d['scaler_scale'])
        obj.scaler_    = scaler
        return obj

    def _assert_fitted(self) -> None:
        """
        Raise RuntimeError if the normalizer has not yet been fitted.

        :return: None
        :raises RuntimeError: If scaler_ is None.
        """
        if self.scaler_ is None:
            raise RuntimeError(
                "Normalizer has not been fitted. Call fit_transform() first."
            )

    def _extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only numeric columns, then drop known non-feature columns.

        :param df: Raw measures DataFrame.
        :return: Numeric feature-only DataFrame.
        """
        numeric_df = df.select_dtypes(include=['number'])
        self.numeric_cols_ = list(numeric_df.columns)
        feature_df = numeric_df.drop(
            columns=[c for c in NON_FEATURE_COLS if c in numeric_df.columns]
        )
        self.feature_cols_ = list(feature_df.columns)
        return feature_df

    def _compute_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-column diagnostics including outlier_factor.

        :param df: Numeric feature DataFrame.
        :return: Diagnostic DataFrame indexed by column name.
        """
        diag = df.agg(['mean', 'median', 'std', 'skew', 'kurt', 'min', 'max']).T
        diag['iqr'] = df.quantile(0.75) - df.quantile(0.25)
        diag['outlier_factor'] = (diag['max'] - diag['min']) / diag['iqr']
        return diag

    def _identify_tier(self, threshold: float) -> list[str]:
        """
        Return column names whose outlier_factor exceeds threshold.

        :param threshold: Minimum outlier_factor to include.
        :return: List of column names.
        """
        return list(
            self.diagnostic_df_[
                self.diagnostic_df_['outlier_factor'] > threshold
            ].index
        )

    def _filter_outlier_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where any Tier 1 column value exceeds Q3 + fence_iqr_mult * IQR.
        Logs per-column drop counts before combining masks.

        :param df: Numeric feature DataFrame.
        :return: Filtered DataFrame with original index preserved.
        """
        if not self.tier1_cols_:
            return df

        mask = pd.Series(True, index=df.index)
        for col in self.tier1_cols_:
            q3  = df[col].quantile(0.75)
            iqr = q3 - df[col].quantile(0.25)
            fence = q3 + self.fence_iqr_mult * iqr
            col_mask = df[col] <= fence
            n_flagged = (~col_mask).sum()
            self.log.info(f"  {col}: {n_flagged} rows above fence ({fence:.3f})")
            mask &= col_mask

        return df[mask].copy()

    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p to columns whose outlier_factor exceeded log_transform_thresh.

        :param df: Filtered numeric feature DataFrame.
        :return: DataFrame with log-transformed columns, index preserved.
        """
        df = df.copy()
        cols_present = [c for c in self.log_cols_ if c in df.columns]
        if cols_present:
            df[cols_present] = np.log1p(df[cols_present])
        return df

    def _apply_robust_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply RobustScaler (median/IQR) to all feature columns.

        :param df: Log-transformed numeric feature DataFrame.
        :return: Scaled DataFrame with same columns and index preserved.
        """
        self.scaler_ = RobustScaler()
        scaled_array = self.scaler_.fit_transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

# ----------------------------- Class PathEncoder -----------------

class PathEncoder:
    """
    Assign compact integer file_ids to file paths, ensuring global
    uniqueness across all DataFrames in a processing run.

    The encoder is built once from the union of all paths seen across both
    measures and species DataFrames, then applied to each DataFrame via
    :meth:`encode_df`.  This guarantees that the same path always receives
    the same file_id regardless of which DataFrame it appears in, preventing
    the collision risk that arises when each DataFrame is encoded
    independently.

    :param path_to_id: Mapping from path string to integer id.
    :param id_to_path: Reverse mapping.
    """

    def __init__(
        self,
        path_to_id: dict[str, int],
        id_to_path: dict[int, str],
    ):
        self.path_to_id = path_to_id
        self.id_to_path = id_to_path

    @classmethod
    def from_paths(
        cls,
        paths: 'np.ndarray | list[str]',
        sort_paths: bool = True,
    ) -> 'PathEncoder':
        """
        Build a PathEncoder from an array of path strings.

        :param paths: All unique paths across all DataFrames to be encoded.
        :param sort_paths: If True, assign IDs in sorted order (deterministic
                           across runs).  If False, assign by appearance order.
        :return: New PathEncoder instance.
        """
        unique_paths = list(dict.fromkeys(paths))   # deduplicate, preserve order
        if sort_paths:
            unique_paths = sorted(unique_paths)
        path_to_id = {p: i for i, p in enumerate(unique_paths)}
        id_to_path = {i: p for p, i in path_to_id.items()}
        return cls(path_to_id=path_to_id, id_to_path=id_to_path)

    def encode_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace the 'Path' column in *df* with a 'file_id' int32 column.

        :param df: DataFrame containing a 'Path' column.
        :return: Copy of *df* with 'Path' replaced by 'file_id', placed first.
        :raises KeyError: If any path in *df* is absent from this encoder's
                          mapping.
        """
        df = df.copy()
        unknown = set(df['Path'].unique()) - set(self.path_to_id)
        if unknown:
            raise KeyError(
                f"{len(unknown)} path(s) in DataFrame not found in encoder "
                f"mapping: {list(unknown)[:5]}"
            )
        df['file_id'] = df['Path'].map(self.path_to_id).astype('int32')
        df.drop(columns=['Path'], inplace=True)
        cols = ['file_id'] + [c for c in df.columns if c != 'file_id']
        return df[cols]

    def save_mapping(self, path: str | Path) -> None:
        """
        Save the id↔path mapping as a CSV for external reference.

        :param path: Output CSV file path.
        :return: None
        """
        mapping_df = pd.DataFrame(
            list(self.path_to_id.items()), columns=['Path', 'file_id']
        )
        mapping_df.to_csv(path, index=False)


#------------------------------------
# main
#-------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine SonoBat CumulativeParameters and CumulativeSonoBatch "
            "files from one or more recording sites into a single normalized "
            "Parquet dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Each --rec-sites value must correspond positionally to a
            --root-dirs value:

              --root-dirs /data2/barn /data2/lake2 --rec-sites barn lake2
        """),
    )
    parser.add_argument(
        '--dest-dir',
        required=True,
        metavar='DIR',
        help='Directory where the output .parquet file will be written.',
    )
    parser.add_argument(
        '--root-dirs',
        required=True,
        nargs='+',
        metavar='DIR',
        help='One or more root directories to search for SonoBat output files.',
    )
    parser.add_argument(
        '--rec-sites',
        required=True,
        nargs='+',
        metavar='SITE',
        help='Recording site label for each root-dir (must be same count).',
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        default=SonoBatPostProcessor.CONF_ACCEPT_THRESH_DEFAULT,
        metavar='FLOAT',
        help=(
            'Minimum confidence score [0–1] for a chirp row to be kept. '
            'Rows with NaN confidence are always dropped. '
            f'Default: {SonoBatPostProcessor.CONF_ACCEPT_THRESH_DEFAULT}.'
        ),
    )

    args = parser.parse_args()

    if len(args.root_dirs) != len(args.rec_sites):
        parser.error(
            f"--root-dirs and --rec-sites must have the same number of values "
            f"(got {len(args.root_dirs)} and {len(args.rec_sites)})."
        )

    SonoBatPostProcessor(
        root_dirs   = args.root_dirs,
        rec_sites   = args.rec_sites,
        dest_dir    = args.dest_dir,
        conf_thresh = args.conf_thresh,
    )

# ----------------------- Main -----------------------

if __name__ == "__main__":
    main()
