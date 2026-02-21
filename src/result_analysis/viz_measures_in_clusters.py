# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-21 10:04:19
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-21 12:41:21

import argparse
from enum import StrEnum
import os
from pathlib import Path
import sys
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sonobat_utils.utils import Utils

class Vizzes(StrEnum):
    MEAS_IMPORTANCE_HEAT = 'meas_importance_heat'
    

class VisualizerMeasuresInClusters:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, 
                 df_info: str | Path | pd.DataFrame,
                 cluster_profile: str | Path | pd.DataFrame,
                 vizzes = [Vizzes.MEAS_IMPORTANCE_HEAT]
                 ):

        if isinstance(df_info, pd.DataFrame):
            self.df = df_info
        else:
            self.df = Utils.read_df_file(df_info)

        if isinstance(cluster_profile, pd.DataFrame):
            self.cluster_profile = cluster_profile
        else:
            self.cluster_profile = Utils.read_df_file(cluster_profile)

        num_clusters = self.df['cluster'].nunique()

        # Ensure large enough fonts:
        Utils.right_size_fontsizes()

        for viz in vizzes:
            if viz == Vizzes.MEAS_IMPORTANCE_HEAT:
                fig, ax, pivot_tbl = self.create_discrimination_heatmap(
                    self.cluster_profile, 
                    num_clusters)
            fig.show()
        

    #------------------------------------
    # create_discrimination_heatmap
    #-------------------

    def create_discrimination_heatmap(
            self,
            cluster_profile_df: pd.DataFrame, 
            num_clusters: int) -> tuple[Figure, Axes]:
        """
        Create a heatmap showing tendency (high/low) 
            for highly discriminative measures.
        :param cluster_profile_df: DataFrame with columns 
            ['cluster', 'measure', 'tendency', 'n_significant_discriminations']
        :param num_clusters: Total number of clusters in the analysis
        :return: Tuple of (fig, ax) matplotlib objects
        """
        # Filter for measures that discriminate against all other clusters
        max_discrimination = num_clusters - 1
        highly_discriminative = cluster_profile_df[cluster_profile_df['n_significant_discriminations'] == max_discrimination
        ]
        
        # Pivot to create the heatmap matrix
        pivot_data = highly_discriminative.pivot(
            index='measure', 
            columns='cluster', 
            values='tendency'
        )
        
        # Create color mapping function (modular for easy replacement)
        color_matrix = self.get_color_matrix(pivot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_data) * 0.3)))
        
        # Create heatmap
        im = ax.imshow(color_matrix, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels(pivot_data.index)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add labels
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Measure', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Tendency of Measures Discriminating Against All {max_discrimination} Other Clusters',
            fontsize=14, 
            fontweight='bold', 
            pad=20
        )
        
        # Add grid
        ax.set_xticks(np.arange(len(pivot_data.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(pivot_data.index)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.set_label('Tendency', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        
        return fig, ax, pivot_data    

    #------------------------------------
    # get_color_matrix
    #-------------------

    def get_color_matrix(self, pivot_data: pd.DataFrame) -> np.ndarray:
        """
        Convert tendency labels to numeric values for color mapping.
        Modular function - easy to replace with continuous variable later.
        
        :param pivot_data: Pivoted DataFrame with tendency values
        :return: Numpy array with numeric values (0 for 'low', 1 for 'high')
        """
        # Map 'low' to 0 (blue) and 'high' to 1 (red) using RdBu_r colormap
        color_matrix = pivot_data.replace({'low': 0, 'high': 1}).values
        return color_matrix    
    
# ------------------ Main ---------------    
if __name__ == "__main__": 

    desc = '''Visualize results of measures_in_clusters.py'''
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=desc
                                     )

    parser.add_argument('datafile',
                        help='original measures with cluster assignments')

    parser.add_argument('cluster_profile',
                        help='path to meas_towards_clusters_cluster_profiles.csv')

    args = parser.parse_args()

    datafile = Path(args.datafile)
    if not datafile.exists():
        print(f"Datafile {args.datafile} not found")
        sys.exit(1)

    cluster_profile = Path(args.cluster_profile)
    if not cluster_profile.exists():
        print(f"Datafile {args.cluster_profile} not found")
        sys.exit(1)

    
    viz = VisualizerMeasuresInClusters(datafile, cluster_profile)
    print(viz)