# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-10 18:26:56
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-15 11:19:37

#**** file_id,chirp_idx,tightness,radius_mean,density,average_error_per_point,error_density,euclidean_distance,low_confidence,large_range,peak_detected,distance_to_prev_peak,significant_peak,cluster

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from logging_service.logging_service import LoggingService

class ChirpClusterer:

    # BARN_DATA_PATH = Path(__file__).parent / \
    #     ('src/result_analysis/data/andrewChen/analysis_results/'
    #      '2022_barn_2secs_myca_quantile_1_16/test_set_chirp_attributes.csv')

    # LAKE_DATA_PATH = Path(__file__).parent / \
    #     ('src/result_analysis/data/andrewChen/analysis_results/'
    #      '2022_lake_2secs_myca_quantile_1_28/test_set_chirp_attributes.csv')

    cluster_stat_cols = ['file_id','chirp_idx','tightness','radius_mean','density',
                         'average_error_per_point','error_density','euclidean_distance',
                         'low_confidence','large_range','peak_detected',
                         'distance_to_prev_peak','significant_peak','cluster']

    GROUPING_VAR = 'cluster'

    INCLUDED_COLS = [
        'file_id',
        'chirp_idx',
        'TimeInFile',
        'PrecedingIntrvl',
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
        'Amp1stQrtl',   # excluded from clustering
        'Amp2ndQrtl',   # excluded from clustering
        'Amp3rdQrtl',   # excluded from clustering
        'Amp4thQrtl',   # excluded from clustering
        '1st10kHzSlp',
        '1st5to15kHzSlp',
        '1st10kHzExp',
        '1st5to15kHzExp',
        'AmpK@start'
    ]

    PHYSICAL_MEASURES = [
        'TimeInFile',
        'PrecedingIntrvl',
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
        '1st10kHzSlp',
        '1st5to15kHzSlp',
        '1st10kHzExp',
        '1st5to15kHzExp'
        ]   


    #------------------------------------
    # constructor
    #-------------------

    def __init__(self, 
                 data_info: Path,
                 cluster_stats: Path,
                 k: int, 
                 outfile: Path = None):
        
        if not data_info.exists():
            raise FileNotFoundError(f"Cannot find data at {data_info}")
        if data_info.suffix == '.feather':
            data_raw = self.read_feather_file(data_info)
        elif data_info.suffix == '.csv':
            data_raw = self.read_csv_file(data_info)
        else:
            raise TypeError(f"Cannot read files of type {data_info}")

        self.log = LoggingService()

        # Remove 'unnamed' leading column, which arises
        # from .csv files that were exported from a df
        # without specifying to omit index:
        try:
            data_raw = data_raw.drop(columns=['Unnamed: 0'])
        except KeyError:
            pass

        # The cluster stats file:
        if not cluster_stats.exists():
            raise FileNotFoundError(f"Cannot find cluster stats at {cluster_stats}")
        if not cluster_stats.suffix == '.csv':
            raise TypeError(f"Cluster stats expected to be a .csv file, not {cluster_stats}")

        clusterstats = self.read_csv_file(cluster_stats)
        # Remove 'unnamed' leading column, which arises
        # from .csv files that were exported from a df
        # without specifying to omit index:
        try:
            clusterstats = clusterstats.drop(columns=['Unnamed: 0'])
        except KeyError:
            pass

        df = self.extract_data(data_raw)

        df_clustered = self.add_cluster_grouping(
            df, clusterstats, 'cluster')

        link_tbl = self.mk_link_table(df_clustered)
        node_tbl = self.mk_node_tbl(df_clustered, clusterstats)

        links_outfile = data_info.parent / f"{data_info.stem}_links.csv"
        nodes_outfile = data_info.parent / f"{data_info.stem}_nodes.csv"
        self.log.info(f"Writing link table to {links_outfile}...")
        link_tbl.to_csv(links_outfile, index=False)
        self.log.info(f"Writing nodes table to {nodes_outfile}...")
        node_tbl.to_csv(nodes_outfile, index=False)

    #------------------------------------
    # mk_link_table
    #-------------------        

    def mk_link_table(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Creates a link table for Cytoscape.

        The method starts with a df that is structured like this:

            file_id  chirp_idx   TimeInFile  ...  1st10kHzExp  1st5to15kHzExp  cluster
            0        16693          4  1514.999990  ...   129.228862      106.436745        8
            1        16694         19  1848.999988  ...   166.116571      154.967097        0
            2        16694         18  1680.000025  ...   134.551667      159.316855        0
            3        16694         17  1596.999944  ...   130.237594      135.986491        0
            4        16694         16  1506.999984  ...   205.131836      178.565429        7
            ...        ...        ...          ...  ...          ...             ...      ...
            10890    17524          6   806.000000  ...   189.795722      141.420461        0
            10891    17524          5   623.000003  ...   173.449743      197.232036        3        

       The output has format:
       
            Index(['Source', 'Target', 'TimeInFile', 'PrecedingIntrvl', 'HiFreq',
                'Bndwdth', 'FreqMaxPwr', 'PrcntMaxAmpDur', 'FreqKnee', 'PrcntKneeDur',
                'StartF', 'UpprKnFreq', 'HiFtoUpprKnAmp', 'HiFtoKnAmp', 'HiFtoFcAmp',
                'UpprKnToKnAmp', 'KnToFcAmp', 'LdgToFcAmp', 'FreqCtr', 'FFwd32dB',
                'FFwd20dB', 'FFwd15dB', 'FBak5dB', 'FFwd5dB', 'Bndw32dB', 
                '1st10kHzSlp',
                '1st5to15kHzSlp', '1st10kHzExp', '1st5to15kHzExp',
                'file_id', 'chirp_idx', 'neighbors_freq'],
                dtype='str')        

        where Source and Target are cluster identifiers. The neighbors_freq is
        a count of how often that link table links the respective
        link table row's cluster pairs.           

        :param df: _description_
        :type df: pd.DataFrame
        :return: _description_
        :rtype: pd.DataFrame
        '''
        group_var = ChirpClusterer.GROUPING_VAR
        
        # Sort once for all groups
        df = df.sort_values(['file_id', 'chirp_idx'])
        
        # Shift within groups to get Target
        df['Target'] = df.groupby('file_id')[group_var].shift(-1)
        
        # Drop the last row of each group (NaN target)
        df = df.dropna(subset=['Target'])
        
        # Rename and fix types
        df = df.rename(columns={group_var: 'Source'})
        df = df.astype({'Source': int, 'Target': int})
        
        # Reorder columns
        cols = ['Source', 'Target'] + [c for c in df.columns if c not in ('Source', 'Target')]
        df = df[cols]
        
        # Vectorized frequency count
        df['neighbors_freq'] = df.groupby(['Source', 'Target'])['Source'].transform('size').astype(int)
        
        return df

    #------------------------------------
    # mk_node_tbl
    #-------------------

    def mk_node_tbl(self, df: pd.DataFrame, clusterstats: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a node table for Cytoscape. We start with a 
        df like:

            file_id  chirp_idx   TimeInFile  ...  1st10kHzExp  1st5to15kHzExp  cluster
        0        16693          4  1514.999990  ...   129.228862      106.436745        8
        1        16694         19  1848.999988  ...   166.116571      154.967097        0
        2        16694         18  1680.000025  ...   134.551667      159.316855        0
        
        Each row in the result table will have information about 
        one of the clusters. The clusterstats are information
        from the sklearn clustering procedure, and includes:

            file_id  chirp_idx  tightness  ...  distance_to_prev_peak  significant_peak  cluster
        0        16693          4   0.847147  ...                    NaN             False        8
        1        16694         19   0.828258  ...                    NaN             False        0
        2        16694         18   0.815064  ...                    9.0             False        0        
                                 ...

        In addition to the cluster stats entering from the clusterstats,
        we add:
           - IsFirst: the number of times a chirp in each node is 
                      the first in a sequence
           - IsLast   the number of times a chirp in each node is 
                      the first in a sequence

        :param df: _description_
        :return: _description_
        '''
        cluster_profile = clusterstats.groupby('cluster').agg(
            tightness_med=('tightness', 'median'),
            population=('tightness', 'size')
        )

        # For each column, get the ratio of variance within
        # a cluster over the variance of that column overall:
        var_ratios = self.compute_variance_ratios(df)
        node_tbl_tmp = cluster_profile.join(var_ratios)

        # Compute the IsStart and IsEnd columns: 
        # Get the minimum and maximum chip_idx of each sequence

        df_plus_seq_info = self.add_seq_info(df)
        # Create a lookup table like:
        #                is_first  is_last
        #     cluster                   
        #     0             192      209
        #     1              18       23
        #     2             147       96
        #               ...
        lookup = df_plus_seq_info.groupby('cluster')[['is_first', 'is_last']].sum()
        # Add the is_first and is_last summation cols to the node table:
        node_tbl = node_tbl_tmp.join(lookup)

        # Add a 'cluster' col as needed for the node table,
        # which will be saved to .csv without the index:
        node_tbl.insert(0, 'cluster', node_tbl_tmp.index)

        return node_tbl

    #------------------------------------
    # add_seq_info
    #-------------------

    def add_seq_info(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Add three columns: 'seq_duration', 'is_first', and 'is_last'
        that indicate for each chirp whether it is the first or last
        in its sequence, and the length of the sequence in msecs.
        We use that info later in node files.

        :param df: all chirp data
        :return: augmented copy of df
        '''
        # Create the grouping object once to save processing time
        grouped = df.groupby('file_id')

        # 1. Identify the first and last chirps
        df['is_first'] = df['chirp_idx'] == grouped['chirp_idx'].transform('min')
        df['is_last']  = df['chirp_idx'] == grouped['chirp_idx'].transform('max')

        # 2. Get the end time (duration) for the entire sequence
        df['seq_duration'] = grouped['TimeInFile'].transform('max')

        return df

    #------------------------------------
    # compute_variance_ratios
    #-------------------

    def compute_variance_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each cluster (group-by), compute the ratio of within-group variance
        to overall variance for each numeric column. Report which column
        has the max and min variance ratio.
        """
        # Variances for each physical measure 
        # across all values in each column:
        numeric_cols = ChirpClusterer.PHYSICAL_MEASURES
        overall_var = df[numeric_cols].var()

        # Vectorized: compute all group variances at once → DataFrame indexed by cluster
        ratios = df.groupby('cluster')[numeric_cols].var().div(overall_var)

        res_df = pd.DataFrame({
            "MaxVarianceRatioVarName": ratios.idxmax(axis=1).values,
            "MaxVarianceRatioValue": ratios.max(axis=1).values,
            "MinVarianceRatioVarName": ratios.idxmin(axis=1).values,
            "MinVarianceRatioValue": ratios.min(axis=1).values,
            })    
        return res_df

    #------------------------------------
    # extract_data
    #-------------------

    def extract_data(self, df: pd.DataFrame) -> pd.DataFrame:

        # Drop the amplitude columns, which are not included
        # in the clustering:
        amplitude_cols = ['Amp1stQrtl', 'Amp2ndQrtl', 'Amp3rdQrtl', 'Amp4thQrtl', 'AmpK@start']
        extract = df[ChirpClusterer.INCLUDED_COLS].drop(columns=amplitude_cols)
        # Get just the Myca rows: [commented because already filtered in source df]
        # df = extract[extract['species']=='Myca']

        extract = extract.astype({'file_id': int, "chirp_idx": int})

        return extract
    
    #------------------------------------
    # add_cluster_grouping
    #-------------------

    def add_cluster_grouping(self, 
                             df: pd.DataFrame, 
                             clusterstats: pd.DataFrame, 
                             clusterstats_grp_var: str) -> pd.DataFrame:
        # Ensure that df and clusterstats have equal
        # length so we assign the proper cluster ids:
        if len(df) != len(clusterstats):
            msg = (f"Data and its clustering info must have same lengths, " 
                   f"not {len(df)} and {len(clusterstats)}, respectively")
            raise ValueError(msg)

        # Join clusterstats into df based on the composite key
        df = df.merge(
            clusterstats[['file_id', 'chirp_idx', 'cluster']], 
            on=['file_id', 'chirp_idx'], 
            how='left'
        )        
        return df

    #------------------------------------
    # mk_fake_clusters
    #-------------------    

    # NO LONGER USED
    def mk_fake_clusters(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        '''
        Assign a random cluster ID int to each row. We
        assume k clusters.

        :param df: the df to which a cluster_id will be added
        :param k: number of clusters
        :return: a df
        '''
        # 1. Determine the number of rows
        n = len(df)

        # 2. Create a sequence of 0-7 repeated to fill the length of the dataframe
        # np.tile repeats [0,1,2,3,4,5,6,7] enough times to cover n rows
        cluster_ids = np.tile(np.arange(k), (n // k) + 1)[:n]

        # 3. Shuffle the array to make it random
        np.random.shuffle(cluster_ids)

        # 4. Assign to the dataframe
        df['cluster_id'] = cluster_ids

        return df

    #------------------------------------
    # read_feather_file
    #-------------------

    def read_feather_file(self, fpath):
        df = pd.read_feather(fpath)
        return df
    
    #------------------------------------
    # read_csv_file
    #-------------------    

    def read_csv_file(self, fpath):
        df = pd.read_csv(fpath)
        return df

    #------------------------------------
    # main
    #-------------------

    @staticmethod
    def main(data_info: str | Path,
             clusterstats: str | Path,
             k: int):

        if type(data_info) == str:
            data_info = Path(data_info)
        if not isinstance(data_info, Path):
            raise TypeError(f"Data info must be file path str or Path, not {type(data_info)}")

        if type(clusterstats) == str:
            data_info = Path(clusterstats)
        if not isinstance(clusterstats, Path):
            raise TypeError(f"Cluster stats must be file path str or Path, not {type(clusterstats)}")


        inst = ChirpClusterer(data_info, clusterstats, k)

# ------------------ Main --------------------

if __name__ == "__main__":

    desc = "Given bats measures, and cluster partitioning, create link and node files for Cytoscape"
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=desc
                                     )

    parser.add_argument('-i', '--infile',
                        help='path to bat chirp attributes file',
                        default=None)

    parser.add_argument('-c', '--clusterstats',
                        help='path to file with cluster measures file',
                        default=None)

    parser.add_argument('-k', '--k',
                        type=int,
                        help='number of clusters')

    args = parser.parse_args()

    infile_p = Path(args.infile)
    if not infile_p.exists():
        print(f"Bat chirp attrs file not found: {args.infile}")

    clusterstats_p = Path(args.clusterstats)
    if not clusterstats_p.exists():
        print(f"Cluster stats file not found: {args.clusterstats}")

    clusterer = ChirpClusterer.main(
        infile_p, 
        clusterstats_p,
        args.k)
