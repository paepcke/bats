#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-31 11:29:40
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/sb_measures_postprocessing.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-04-01 12:06:57
#
# **********************************************************

"""
Input raw data from SonoBat species identification *_Parameters_*.txt
files. Remove columns that were determined to not contribute
much to chirp measures variance.

Analyze the remaining data for outliers and skew. 

Phase I: 

1. Collect all relevant *_Parameter_*.txt files (tab-separated files); they
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
3. All features are normalized using robust scaling: subtracting the median and
   dividing by the interquartile range, which is resistant to residual outliers
   and preserves the interpretability of the feature space.
4. Identifier data, such as file_id, chirp_idx, cluster, TimeInFile are excluded from
   normalization.

Phase III:

1. Optionally, merge the new normalized data with already
   existing normalized data.

Output .csv or .feather file.
"""

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from logging_service import LoggingService
from sklearn.preprocessing import QuantileTransformer

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
    
    def _collect_measures(self, root_dir: Path) -> pd.DataFrame:
        '''
        Starting at the root directory, find files matching *_Parameters_*.txt.
        Import all those tab-separated files into one dataframe.

        :param root_dir: directory where to start file search
        :return: combined data
        '''
        pass

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
            id_batches.append(df.copy())
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

        # Add a 'Confidence' column:
        

        # Save both the IDs df, and the Path <--> file_id lookup
        species_id_file_mapping_path = f"species_ids_file_id_map_{self.timestamp}.csv"
        path_encoder.save_mapping(species_id_file_mapping_path)

    #------------------------------------
    # species_confidence
    #-------------------
    
    @staticmethod
    def species_confidence(prob_info: str | float, # Floats are NaN values
                           scaled_accp_n: float | int,
                           scaled_maj_n: float | int
                           ) -> float:
        '''
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
