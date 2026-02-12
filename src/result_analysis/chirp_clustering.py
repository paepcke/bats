# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-10 18:26:56
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-11 18:26:04

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

class ChirpClusterer:

    ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir", "split"] + \
                ['row_num', 'sin_hr', 'cos_hr', 'sin_day', 'cos_day', 'sin_month', 'cos_month',
                 'sin_year', 'cos_year'] + \
                ["FreqLedge","AmpK@end", "Fc", "FBak15dB  ", "FBak32dB", "EndF", "FBak20dB", "LowFreq", "Bndw20dB", 
                "CallsPerSec", "EndSlope", "SteepestSlope", "StartSlope", "Bndw15dB", "HiFtoUpprKnSlp", "HiFtoKnSlope", 
                "DominantSlope", "Bndw5dB", "PreFc500", "PreFc1000", "PreFc3000", "KneeToFcSlope", "TotalSlope", 
                "PreFc250", "CallDuration", "CummNmlzdSlp", "DurOf32dB", "SlopeAtFc", "LdgToFcSlp", "DurOf20dB", "DurOf15dB", 
                "TimeFromMaxToFc", "KnToFcDur", "HiFtoFcExpAmp", "AmpKurtosis", "LowestSlope", "KnToFcDmp", "HiFtoKnExpAmp", 
                "DurOf5dB", "KnToFcExpAmp", "RelPwr3rdTo1st", "LnExpB_StartAmp", "Filter", "HiFtoKnDmp", "LnExpB_EndAmp", 
                "HiFtoFcDmp", "AmpSkew", "LedgeDuration", "KneeToFcResidue", "PreFc3000Residue", "AmpGausR2", "PreFc1000Residue", 
                "Amp1stMean", "LdgToFcExp", "FcMinusEndF", "Amp4thMean", "HiFtoUpprKnExp", "HiFtoKnExp", "KnToFcExp", "UpprKnToKnExp", 
                "Kn-FcCurviness", "Amp2ndMean", "Quality", "HiFtoFcExp", "LnExpA_EndAmp", "RelPwr2ndTo1st", "LnExpA_StartAmp", 
                "HiFminusStartF", "Amp3rdMean", "PreFc500Residue", "Kn-FcCurvinessTrndSlp", "PreFc250Residue", "AmpVariance", "AmpMoment", 
                "meanKn-FcCurviness", "MinAccpQuality", "AmpEndLn60ExpC", "AmpStartLn60ExpC", "Preemphasis", "MaxSegLnght" ,"Max#CallsConsidered" ]


    #------------------------------------
    # constructor
    #-------------------

    def __init__(self, data_info: str | Path, k: int, outfile: Path = None):
            
        
        # Assume data file:
        if not data_info.exists():
            raise FileNotFoundError(f"Cannot find data at {data_info}")
        if data_info.suffix == '.feather':
            data_raw = self.read_feather_file(data_info)
        elif data_info.suffix == '.csv':
            data_raw = self.read_csv_file(data_info)
        else:
            raise TypeError(f"Cannot read files of type {data_info}")

        df_unclustered = self.extract_data(data_raw)
        
        df = self.mk_fake_clusters(df_unclustered, k)
        if outfile:
            print(f"Saving df to {outfile}")
            df.to_feather(outfile)

        link_tbl = self.mk_link_table(df)
        node_tbl = self.mk_node_tbl(df)

        links_outfile = data_info.parent / f"{data_info.stem}_links.csv"
        nodes_outfile = data_info.parent / f"{data_info.stem}_nodes.csv"
        print(f"Writing link table to {links_outfile}...")
        link_tbl.to_csv(links_outfile, index=False)
        print(f"Writing nodes table to {nodes_outfile}...")
        node_tbl.to_csv(nodes_outfile, index=False)

    #------------------------------------
    # mk_link_table
    #-------------------        

    def mk_link_table(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a list of DataFrames, one for each unique file_id (i.e. sequence ID)
        list_of_dfs = [group for _, group in df.groupby('file_id')]
        # Make sure each df is sorted by its chirp_idx value:
        list_of_dfs_sorted = [sub_df.sort_values('chirp_idx') for sub_df in list_of_dfs]
        # We now have:
        #         chirp_idx  cluster_id
        #    9046          0           1
        #    9047          1           4
        #    9048          2           1
        #    9049          3           5        
        #              ...

        # The link table will look like:
        #     Source    Target   LinkFreq   <measures>
        #
        link_dfs = []

        # For each df, add column: 'dst_cluster'
        for seq_df in list_of_dfs_sorted:
            seq_df['Target'] = seq_df['cluster_id'].shift(-1)
            link_cols = ['cluster_id', 'Target']
            other_cols = [col for col in seq_df.columns if col not in ['cluster_id', 'Target']]
            link_cols += other_cols
            link_df = seq_df[link_cols].rename(columns={'cluster_id': 'Source'})

            # The last chirp has no target node. So its Target 
            # is NaN. We delete that row. There should be exactly one:
            assert(link_df['Target'].isna().sum() == 1)
            link_df = link_df.dropna(subset=['Target'])
            # Ensure that node IDs are ints:
            link_df = link_df.astype({'Source': int, 'Target': int})

            link_dfs.append(link_df)

        link_tbl = pd.concat(link_dfs)

        # Now need for each link the number of times that
        # src_cluster --> dst_cluster is involved sequences:
        # Group by both Source and Target, then get the size of each group
        link_tbl['neighbors_freq'] = link_tbl.groupby(['Source', 'Target'])['Source'].transform('size').astype(int)

        return link_tbl

    #------------------------------------
    # mk_node_tbl
    #-------------------

    def mk_node_tbl(self, df: pd.DataFrame) -> pd.DataFrame:
        cluster_ids = []
        cluster_types = []
        cluster_populations = []
        k = len(df['cluster_id'].unique())
        population_counts = df['cluster_id'].value_counts()
        for cluster_id in df['cluster_id'].unique():
            cluster_ids.append(cluster_id)
            cluster_types.append(f"chirp {k}-cluster")
            cluster_populations.append(int(population_counts[cluster_id]))
        node_table = pd.DataFrame({
            'ID' : cluster_ids,
            'Population': cluster_populations,
            'Type': cluster_types
        })
        return node_table.sort_values(['ID'])

    #------------------------------------
    # extract_data
    #-------------------

    def extract_data(self, df: pd.DataFrame) -> pd.DataFrame:
        true_cols_to_drop = []
        for actual_col in df.columns:
            if actual_col in ChirpClusterer.ignore_cols:
                true_cols_to_drop.append(actual_col)
        # Drop unwanted columns:
        extract = df.drop(columns=true_cols_to_drop)
        # Get just the Myca rows:
        df = extract[extract['species']=='Myca']
        return df
    
    #------------------------------------
    # mk_fake_clusters
    #-------------------    

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
             k: int, 
             save_cluster_assignment: bool = False) -> ChirpClusterer:

        if type(data_info) == str:
            data_info = Path(data_info)
        if not isinstance(data_info, Path):
            raise TypeError(f"Data info must be file path str or Path, not {type(data_info)}")

        if save_cluster_assignment:
            data_dir = data_info.parent
            fname    = f"{data_info.stem}_clustered_{k}.feather"
            outfile  = data_dir / fname
        else:
            outfile  = None

        inst = ChirpClusterer(data_info, k, outfile=outfile)

# ------------------ Main --------------------

if __name__ == "__main__":
    k = 8
    data_fpath = Path(__file__).parent / 'data/scaled_chirps_2024-06-25T12_55_03.feather'

    clusterer = ChirpClusterer.main(data_fpath, k, save_cluster_assignment=False)
