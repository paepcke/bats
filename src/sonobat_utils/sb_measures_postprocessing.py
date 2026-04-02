#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-31 11:29:40
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sb_measures_postprocessing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-02 11:55:41
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

Phase III — Optional incremental update:

1. New SonoBat data can be merged with a previously saved dataset by
   inverse-transforming the existing data, appending the new raw data, and
   re-fitting the normalizer from scratch.

Output: a single .parquet file via BatsData.to_parquet().  The Parquet file
carries all ancillary metadata (file_id↔path map, normalizer state) in its
schema metadata as JSON, so no sidecar files are needed.

About the conf-threshold: given that it is a mix of probability > 0.9,
but then lowered by amount of evidence, and number of chirps identified,
the confidence of ~0.50 is based on these observations:

   confidence = Prob * (0.7 * (Maj_scaled / Accp_scaled) + 0.3 * Accp_scaled)

The maximum possible value is when
    Prob=1.0
    Maj_scaled/Accp_scaled=1.0 (perfect consensus), and
    Accp_scaled=1.0 (maximum evidence percentile):
  1.0 * (0.7 * 1.0 + 0.3 * 1.0) = 1.0

A "typical good" row might look like:
    Prob=0.93,
    consensus=0.85,
    Accp_scaled=0.5 (median evidence):
  0.93 * (0.7 * 0.85 + 0.3 * 0.5) = 0.93 * (0.595 + 0.15) = 0.93 * 0.745 ≈ 0.69

A marginal row:
    Prob=0.90 (the MIN_ACCEPT_PROB),
    consensus=0.67,
    Accp_scaled=0.3:
  0.90 * (0.7 * 0.67 + 0.3 * 0.3) = 0.90 * (0.469 + 0.09) = 0.90 * 0.559 ≈ 0.50

So the distribution is roughly:

   High-confidence rows: 0.65–1.0
   Acceptable rows: 0.45–0.65
   Marginal/noisy rows: below 0.45


CLI usage:
    python sb_measures_postprocessing.py \\
        --dest-dir /path/to/output \\
        --root-dirs /data/barn /data/lake2 \\
        --rec-sites barn lake2 \\
        --conf-thresh 0.50
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd

from logging_service import LoggingService
from sklearn.preprocessing import QuantileTransformer

from sonobat_utils.bats_data import BatsData, MeasureNormalizer

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
            categories=sorted(rec_sites), ordered=False
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

        # Merge and wrap
        df_final = self._merge(df_normalized, df_species_final, self.conf_thresh)
        self.bats_data = BatsData(
            df        = df_final,
            file_map  = self.path_encoder.id_to_path,
            normalizer= self.normalizer,
            timestamp = self.timestamp,
        )
        out_path = self.dest_dir / f"bats_{self.timestamp}.parquet"
        self.bats_data.to_parquet(out_path)
        self.log.info(f"Wrote {out_path}")
            
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

        'chirp_idx' is added to the measures DataFrame: a 0-based integer
        giving the position of each chirp within its recording, derived by
        sorting on TimeInFile within each unique Path value.

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

        # chirp_idx: 0-based rank by TimeInFile within each recording (Path)
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
        species DataFrame: scale #Accp and #Maj to percentiles, add a
        confidence column, and normalize composite species names.

        :param df: Species DataFrame with 'file_id' already assigned and
                   'Path' column already dropped.
        :return: Processed species DataFrame.
        """
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
        cols_to_xform  = ['#Maj', '#Accp']
        dest_col_names = ['Maj_scaled', 'Accp_scaled']
        df[dest_col_names] = qt.fit_transform(df[cols_to_xform])

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
    ) -> pd.DataFrame:
        """
        Left-join species columns onto the normalized measures DataFrame on
        'file_id', then prune to only the species columns the final dataset
        needs, rename 'SppAccp' to 'species', and drop rows whose confidence
        is NaN or below ``conf_thresh``.

        Logs a warning for every species file_id that finds no matching
        measures row (those paths produced species IDs but no chirp measures).

        :param df_measures: Normalized measures DataFrame with 'file_id'.
        :param df_species:  Processed species DataFrame with 'file_id'.
        :param conf_thresh: Minimum confidence; rows below this are dropped.
        :return: Merged, filtered DataFrame with 'species' and 'confidence'
                 columns and no intermediate species columns.
        """
        # Warn about species rows that will find no measures match
        meas_ids    = set(df_measures['file_id'].unique())
        species_ids = set(df_species['file_id'].unique())
        unmatched   = species_ids - meas_ids
        if unmatched:
            self.log.warn(
                f"{len(unmatched)} file_id(s) in species data have no "
                f"matching measures rows and will be dropped:"
            )
            for fid in sorted(unmatched):
                path = self.path_encoder.id_to_path.get(fid, '<unknown>')
                self.log.warn(f"  file_id={fid}  path={path}")

        # Only bring over the two columns we want in the final df
        df_merged = df_measures.merge(
            df_species[['file_id', 'SppAccp', 'confidence']],
            on='file_id',
            how='left',
        )

        # Rename SppAccp -> species (already CompositeSpecies-normalized)
        df_merged.rename(columns={'SppAccp': 'species'}, inplace=True)

        # Drop rows with insufficient or missing confidence
        n_before = len(df_merged)
        df_merged = df_merged[
            df_merged['confidence'].notna() &
            (df_merged['confidence'] >= conf_thresh)
        ].copy()
        n_dropped = n_before - len(df_merged)
        self.log.info(
            f"Confidence filter (>= {conf_thresh}): "
            f"{n_before} → {len(df_merged)} rows "
            f"({n_dropped} dropped, "
            f"{100 * n_dropped / n_before:.1f}%)"
        )

        return df_merged


    
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
        This is a nearly-vectorized method for computing a 'confidence'
        float derived from SonoBat's Prob, #Accp, and #Maj values. A
        Row-by-row alternative is method _species_confidence(). See that
        method for an explanation of the semantics.

        Use this method for its (presumed) speed.

        Vectorized computation of confidence scores, adding a 'confidence'
        column to a DataFrame containing 'Prob', 'Maj_scaled', and 'Accp_scaled'.

        Handles three forms of 'Prob' values:
        - NaN         → confidence = 0.0
        - plain float → used directly
        - 'x/y[/z]'  → component probabilities are summed

        The three-mask pattern is the key to vectorizing the mixed Prob
        column. NaN rows are zeroed out, slash-strings are split and summed
        via .apply() (unavoidable for the variable-length split, but applied
        only to the small slash-string subset), and plain floats are cast
        directly — avoiding any Python-level loop over all rows.

        nan_mask computed first ensures the str.contains('/') call on mask 2
        never sees actual NaN values, which would raise or produce unexpected
        results after the .astype(str) converts them to the literal string
        "nan".

        In-place column addition (df['confidence'] = ...) matches the
        typical pandas pattern for enriching a DataFrame without copying
        it. If you prefer not to mutate the caller's DataFrame, add df =
        df.copy() at the top.

        slash_mask subset for .apply() keeps the only non-fully-vectorized
        step confined to however many rows actually have slash-strings, rather
        than running over the full DataFrame.

        :param df: DataFrame with columns 'Prob', 'Maj_scaled', 'Accp_scaled'
        :return: DataFrame with new 'confidence' column added in-place
        :rtype: pd.DataFrame
        """
        prob_raw = df['Prob']

        # --- Resolve probability values vectorized ---

        # Mask 1: NaN rows
        nan_mask = prob_raw.isna()

        # Mask 2: slash-separated strings like '0.46/0.52'
        slash_mask = (~nan_mask) & prob_raw.astype(str).str.contains('/', regex=False)

        # Sum slash-separated probabilities per row
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

        # --- Vectorized confidence formula ---
        consensus = df['Maj_scaled'] / df['Accp_scaled']
        df['confidence'] = prob * (
            cls.WEIGHT_ON_CONSENSUS * consensus
            + cls.WEIGHT_ON_EVIDENCE * df['Accp_scaled']
        )

        # NaN rows get 0.0 (already 0.0 from prob init, but be explicit)
        df.loc[nan_mask, 'confidence'] = 0.0

        return df

    #------------------------------------
    # _species_confidence
    #-------------------
    
    @staticmethod
    def _species_confidence(prob_info: str | float, # Floats are NaN values
                           scaled_accp_n: float | int,
                           scaled_maj_n: float | int
                           ) -> float:
        '''
        NOTE: Use the vectorized version of this method instead: _add_confidence_column.
              But the following comments are relevant.

        Takes the several pieces information that SonoBat produces
        to convey its species prediction confidence. Returns a single
        confidence number in [0,1]. Strategy:

        SB produces:
           Prob :  a probability that SonoBat's species ID is correct
           #Accp:  a count of pulses in SB accepted as being a chirp from any species of bat
           #Maj :  the number of those accepted chirps that agreed with the ID

        Most of the #Accp values are in the range of 4-6 accepted chirps
        in a 2-sec recording.But there are outliers of as high as 28. 
        
        So in an earlier step we scaled the raw #Accp values to be a uniform distribution of 
        percentiles, yielding values [0,1]. This way the outliers do not
        compress the many smaller number of chirps per chop into a tiny range.

        Then we think of confidence as a mix of the probability,
        the degree of species ID consensus across the chirps in a recording,
        and the amount of available evidence in the chop (i.e. the number
        of accepted chirps).

        The 'consensus' is

           consensus = scaled_maj_n / scaled_accp_n
        
        I.e. the number of chirps being the declared species divided
        by the number of chirps in the recording chop.

        The 'evidence' is the scaled number of chirps:

           confidence = Prob * (α * consensus + β * evidence)

        We set α=0.7, and β=0.3, i.e. we put some emphasis on consensus,
        and a bit less on the amount of evidence.

        prob_info is either an actual probability, like 0.9834, or
        a string with multiple probabilities, like '0.46/0.52'. These
        are the probabilities for each of multiple possible IDs, such
        as Lano/Laci. Since we create 'compound' species from such 
        split decisions, we add the split probabilities.

        :param prob_info: either a probability, or slash-separated probs
        :param scaled_accp_n: the percentile-scaled number of accepted chirps in this recording
        :param scaled_maj_n: the percentile-scaled number of chirps SB deemed to be of the ID
        :return: a single, combined confidence
        :rtype: float
        '''
        # Turn the probability from one of
        # 0.9924, NaN, '44/54' to a float:
        if pd.isna(prob_info):
            return 0.0
        try:
            prob = float(prob_info)
        except ValueError:
            # It's a string like '46/59'
            prob_parts = [float(p) for p in prob_info.split('/')]
            # Add the probabilities of identification being 
            # one of the species listed in the entry:
            prob = sum(prob_parts)
        # Weight the probability by the number of algorithmic
        # experts that accepted the SB judgement:
        consensus = scaled_maj_n / scaled_accp_n
        weighted_confidence = prob * (SonoBatPostProcessor.WEIGHT_ON_CONSENSUS * consensus \
                                      + SonoBatPostProcessor.WEIGHT_ON_EVIDENCE * scaled_accp_n)
        return weighted_confidence
    
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

              --root-dirs /data/barn /data/lake2 --rec-sites barn lake2
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
