#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-31 11:29:40
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sb_measures_postprocessing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-02 08:56:35
#
# **********************************************************

"""
Input raw data from SonoBat species identification *_CumulativeParameters_*.txt
files. Remove columns that were determined to not contribute
much to chirp measures variance.

Analyze the remaining data for outliers and skew. 

Phase I: 

1. Collect all relevant *_CumulativeParameter_*.txt files (tab-separated files); they
   contain physical measures.
2. Collect all relevant *_CumulativeSonoBatch_*.txt files; they
   contain species determinations
3. Check for species-specific anomalies, such as outliers.
4. Merge measures and species estimates.

Phase II:

Make the following adjustments:

1. Chirp measurements with amplitude-ratio or energy values exceeding Q3 + 5·IQR are
   excluded as likely recording artifacts.
2. Amplitude-ratio and acoustic energy features (HiFtoKnAmp, HiFtoUpprKnAmp, LdgToFcAmp,
   HiFtoFcAmp, KnToFcAmp, 1st5to15kHzExp, 1st10kHzExp) are
   log-transformed (log(1 + x)) to address their known right-skewed
   distributions
3. All other features are normalized using robust scaling: subtracting the median and
   dividing by the interquartile range, which is resistant to residual outliers
   and preserves the interpretability of the feature space.
4. Identifier data, such as file_id, chirp_idx, cluster, TimeInFile are excluded from
   normalization.

Phase III:

1. Optionally, merge the new normalized data with already
   existing normalized data. This will de-normalize, merge, re-process,
   and normalize the new set.

Output .csv or .feather file.
"""

from datetime import datetime
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from logging_service import LoggingService
from sklearn.preprocessing import QuantileTransformer, RobustScaler

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
    
    def __init__(self, 
                 root_dir: str | Path,
                 dest_dir: str | Path
                 ):
        self.log = LoggingService()

        self.timestamp = datetime.now().isoformat()

        self.root_dir = Path(root_dir)
        self.dest_dir = Path(dest_dir)
        
        self.rejected_entries = []
        self.composite_species: set[CompositeSpecies] = set()

        self.measures = self._collect_measures(self.root_dir, self.dest_dir)

        # Get a species identifications
        self.species_ids = self._collect_species_ids(root_dir, dest_dir)
            
    #------------------------------------
    # _collect_measures
    #-------------------
    
    def _collect_measures(self, 
                          root_dir: Path,
                          dest_dir: Path
                          ) -> pd.DataFrame:
        '''
        Starting at the root directory, find files matching *_CumulativeParameters_*.txt.
        Import all those tab-separated files into one dataframe.

        Then drop rows in which any measures are highly skewed, and
        normalize the rest:
        Three tiers of severity:
            Tier 1 — Catastrophically skewed (outlier_factor > 1000).
                     These amplitude ratio and exponential features have 
                     medians around 160 but maxima in the tens of thousands 
                     to tens of millions. The mean/median ratio for 
                     HiFtoKnAmp is ~42×. These are almost certainly 
                     measurement artifacts or physically degenerate 
                     chirps (very faint signals, clipped recordings, near-noise-floor calls).
            Tier 2 — Moderately skewed (outlier_factor 10–350): LdgToFcAmp, HiFtoFcAmp, KnToFcAmp, and a few others. Still problematic but recoverable with log transform alone.
            Tier 3 — Acceptably distributed (outlier_factor < 12): The frequency features (FreqCtr, FBak5dB, FreqKnee, etc.) and most bandwidth/slope features. These are well-behaved and just need standard scaling.        

        :param root_dir: directory where to start file search
        :return: combined data
        '''
        measures_files = self._find_sonobatch_measures_files(root_dir)
        # Each species ID file stems for a batch of recordings.
        # Each file will be loaded into one df, and its columns culled.
        # Those dfs will be added the the following list.
        # At the end we concatenate those into one df:
        measures_batches: list[pd.DataFrame] = []

        if len(measures_files) == 0:
            raise FileNotFoundError(f"Could not find '*_CumulativeParameters_v<version>.txt' files with chirp measures")
        
        for measures_file in measures_files:
            df = pd.read_csv(measures_file, sep='\t')
            # Drop a bunch of cols:
            cols_to_keep = ['Path'] + SonoBatPostProcessor.RELEVANT_MEASURES_COLS
            df = df[cols_to_keep]
            measures_batches.append(df)
        df_measures = pd.concat(measures_batches, ignore_index=True, axis='index')

        meas_normalizer = MeasureNormalizer()
        df_normalized = meas_normalizer.fit(df_measures)
        normalizer_save_path = root_dir / 'measures_normalizer.pkl'
        meas_normalizer.save(normalizer_save_path)

        # Next: turn the long paths in the Path column
        # into integers in a new column: file_id, dropping the Path
        # column:
        path_encoder = PathEncoder(df_normalized, sort_paths=True)
        df_meas_file_encoded = path_encoder.encode()
        meas_file_mapping_path = f"measures_file_id_map_{self.timestamp}.csv"
        path_encoder.save_mapping(dest_dir / meas_file_mapping_path)
        meas_df_path = f"measures_{self.timestamp}.csv"
        df_meas_file_encoded.to_csv(meas_df_path)
        return df_meas_file_encoded

    #------------------------------------
    # _collect_species_ids
    #-------------------
    
    def _collect_species_ids(self, 
                             root_dir: Path,
                             dest_dir: Path
                             ) -> pd.DataFrame:
        '''
        Starting at root_dir, find files matching the bash expression:
          find . -regextype posix-egrep -regex ".*_CumulativeSonoBatch_v[0-9.]+\.txt"
        Import all into a dataframe, and return that df.

        Relevant colums are 'Prob', 'SppAccp' '#Accp', '#Maj'. 
        Examples:
                        Prob    SppAccp     #Accp  #Maj
            3          0.9131    Laci      2.0   1.0
            4           NaN      NaN       NaN   NaN
            23        0.54/0.46  Laci/Lano  4.0   2.0
            54         NaN/NaN    NaN       NaN   NaN
          
        :param root_dir: directory where to start file search
        :return: combined data
        '''

        species_id_files = self._find_sonobatch_species_id_files(root_dir)
        # Each species ID file stems for a batch of recordings.
        # Each file will be loaded into one df, and its columns culled.
        # Those dfs will be added the the following list.
        # At the end we concatenate those into one df:
        id_batches: list[pd.DataFrame] = []

        if len(species_id_files) == 0:
            raise FileNotFoundError(f"Could not find '*_CumulativeSonoBatch_v<version>.txt' files with species IDs")
        
        for id_file in species_id_files:
            df = pd.read_csv(id_file, sep='\t')
            # Drop a bunch of cols:
            cols_to_keep = ['Path', 'SppAccp', 'Prob', '#Maj', '#Accp']
            df = df[cols_to_keep]
            id_batches.append(df)
        df_ids = pd.concat(id_batches, ignore_index=True, axis='index')

        # Scale #Accp and #Maj to percentiles 
        # (see method _species_confidence for details)
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
        cols_to_xform  = ['#Maj', '#Accp']
        dest_col_names = ['Maj_scaled', 'Accp_scaled']

        # Fit and transform the specific columns
        # We use .values to avoid index alignment issues and cast back to DF
        df[dest_col_names] = qt.fit_transform(df[cols_to_xform])

        # Convert the long Path to an integer file_id:
        path_encoder = PathEncoder(df, sort_paths=True)
        df_file_encoded = path_encoder.encode()

        # Add a 'Confidence' column in place:
        df_confidence_added = SonoBatPostProcessor._add_confidence_column(df_file_encoded)

        final_df = SonoBatPostProcessor._normalize_composite_species(df_confidence_added)

        # Save both the IDs df, and the Path <--> file_id lookup
        species_id_file_mapping_path = f"species_ids_file_id_map_{self.timestamp}.csv"
        path_encoder.save_mapping(dest_dir / species_id_file_mapping_path)
        species_id_df_path = f"species_ids_{self.timestamp}.csv"
        final_df.to_csv(species_id_df_path)

        return final_df

    #------------------------------------
    # _normalize_composite_species
    #-------------------
    
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

# ----------------------------- Class MeasureNormalizer -----------------    

NON_FEATURE_COLS = {'file_id', 'chirp_idx', 'cluster', 'TimeInFile'}
class MeasureNormalizer:
    """
    Clean and normalize a SonoBat chirp measures DataFrame.

    Pipeline stages:
      1. Separate numeric from non-numeric columns; set aside non-feature columns.
      2. Identify Tier 1 columns (outlier_factor > outlier_factor_thresh) and drop
         rows whose value in any Tier 1 column exceeds Q3 + fence_iqr_mult * IQR.
      3. Log-transform (log1p) columns with outlier_factor > log_transform_thresh.
      4. Apply RobustScaler to all remaining numeric feature columns.

    The fitted normalizer can be saved to disk and reloaded, enabling:
      - Exact inverse-transform back to (approximately) original scale.
      - Applying the same scaling to new data without refitting.

    USAGE:
    
    # --- Initial fit on your current full dataset ---
    normalizer = MeasureNormalizer()
    df_normalized = normalizer.fit_transform(df_measures)
    normalizer.report()
    normalizer.save('normalizer_v1.pkl')

    # --- Verify round-trip ---
    df_recovered = normalizer.inverse_transform(df_normalized)

    # --- Later: new SonoBat data arrives ---
    normalizer = MeasureNormalizer.load('normalizer_v1.pkl')

    # Option A (recommended while still building dataset):
    # Unscale, append, refit from scratch
    df_recovered  = normalizer.inverse_transform(df_normalized)
    df_combined   = pd.concat([df_recovered, df_new_raw], axis='index')
    normalizer2   = MeasureNormalizer()
    df_normalized = normalizer2.fit_transform(df_combined)
    normalizer2.save('normalizer_v2.pkl')

    # Option B (once dataset and model are frozen):
    # Apply existing scaler to new data only
    df_new_normalized = normalizer.transform(df_new_raw)      

    :param outlier_factor_thresh: outlier_factor above which a column is Tier 1.
    :param fence_iqr_mult: IQR multiplier for the per-row outlier fence.
    :param log_transform_thresh: outlier_factor above which log1p is applied.
    """

NON_FEATURE_COLS = {'file_id', 'chirp_idx', 'cluster', 'TimeInFile'}


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
        Persist the fitted normalizer to disk using joblib.

        Saves the complete normalizer state including the fitted RobustScaler,
        column lists, and diagnostic DataFrame. Reload with MeasureNormalizer.load().

        :param path: Destination file path (conventionally .pkl or .joblib).
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
    Replace a long Path column with compact integer file_ids.
    
    :param df: DataFrame containing a 'Path' column.
    :param sort_paths: If True, assign IDs by sorted path order (deterministic
                       across runs). If False, assign by first-appearance order.
    """

    def __init__(self, df: pd.DataFrame, sort_paths: bool = True):
        self.df = df.copy()
        self.sort_paths = sort_paths
        self.path_to_id: dict[str, int] = {}
        self.id_to_path: dict[int, str] = {}

    def encode(self) -> pd.DataFrame:
        """
        Replace the Path column with a file_id integer column.
        
        :return: Modified DataFrame with Path replaced by file_id (int32).
        """
        unique_paths = self.df['Path'].unique()
        if self.sort_paths:
            unique_paths = sorted(unique_paths)

        self.path_to_id = {p: i for i, p in enumerate(unique_paths)}
        self.id_to_path = {i: p for p, i in self.path_to_id.items()}

        self.df['file_id'] = self.df['Path'].map(self.path_to_id).astype('int32')
        self.df.drop(columns=['Path'], inplace=True)

        # Reorder: file_id first
        cols = ['file_id'] + [c for c in self.df.columns if c != 'file_id']
        return self.df[cols]

    def encode_from_existing(self, mapping_path: str) -> pd.DataFrame:
        """
        Replace Path column using a previously saved mapping.
        Raises KeyError if any Path in df is absent from the mapping.
        
        :param mapping_path: CSV file produced by save_mapping().
        :return: DataFrame with Path replaced by file_id.
        """
        mapping_df = pd.read_csv(mapping_path)
        self.path_to_id = dict(zip(mapping_df['Path'], mapping_df['file_id']))
        
        unknown = set(self.df['Path'].unique()) - set(self.path_to_id.keys())
        if unknown:
            raise KeyError(f"{len(unknown)} paths in species dataset not found "
                        f"in mapping: {list(unknown)[:5]}")
        
        self.df['file_id'] = self.df['Path'].map(self.path_to_id).astype('int32')
        self.df.drop(columns=['Path'], inplace=True)
        cols = ['file_id'] + [c for c in self.df.columns if c != 'file_id']
        return self.df[cols]

    def save_mapping(self, path: str) -> None:
        """
        Save the id↔path mapping as a CSV for later lookups.
        
        :param path: Output CSV file path.
        """
        mapping_df = pd.DataFrame(
            list(self.path_to_id.items()), columns=['Path', 'file_id']
        )
        mapping_df.to_csv(path, index=False)
                         
#------------------------------------
# main
#-------------------

def main():
    print("Hello, World!")

# ----------------------- Main -----------------------

if __name__ == "__main__":
    main()
