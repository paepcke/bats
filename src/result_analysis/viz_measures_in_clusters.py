#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-21 10:04:19
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-21 18:39:28

import argparse
from enum import StrEnum
import os
from pathlib import Path
import sys
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logging_service import LoggingService
from result_analysis.basic_charting import SimpleCharter, ChartType
from sonobat_utils.utils import Utils

class Vizzes(StrEnum):
    # Measure vs. Clusters: color is tendency (high vs. low value):
    MEAS_IMPORTANCE_TENDENCY_HEAT    = 'tendency-heat'
    # Measure vs. Clusters: color is effect size of discrimination:
    MEAS_IMPORTANCE_EFFECT_SIZE_HEAT = 'effect-size-heat'
    # Single chart with multiple histograms for 
    # one measure in one or more clusters
    MEAS_IMPORTANCE_HISTOGRAMS = 'histograms'
class VisualizerMeasuresInClusters:

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self, 
                 df_info: str | Path | pd.DataFrame,
                 cluster_profile: str | Path | pd.DataFrame,
                 vizzes = [Vizzes.MEAS_IMPORTANCE_TENDENCY_HEAT],
                 outdir: str | Path = None,
                 **kwargs
                 ):
        self.log = LoggingService()

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

        figs = []
        # Collect appropriate fname for each figure
        # in case we'll save to outdir:
        fnames = []
        for viz in vizzes:
            if viz == Vizzes.MEAS_IMPORTANCE_TENDENCY_HEAT:
                vizzer = HeatMapTendencyBinary(self.cluster_profile, num_clusters)
                fig, ax, pivot_tbl = vizzer.run()
                figs.append(fig)
                fnames.append('maes_tendency_heat.png')
            elif viz == Vizzes.MEAS_IMPORTANCE_EFFECT_SIZE_HEAT:
                vizzer = HeatMapEffectSize(self.cluster_profile, num_clusters)
                fig, ax, pivot_tbl = vizzer.run()
                figs.append(fig)
                fnames.append('maes_effect_size_heat.png')
            elif viz == Vizzes.MEAS_IMPORTANCE_HISTOGRAMS:
                # We need additional information for this 
                # chart in the kwargs. Ensure we have them:
                insufficient_info = False
                for needed in ['measure', 'clusters']:
                    if needed not in kwargs.keys():
                        msg = f"Histograms require one measure and one or more clusters; '{needed}' is missing"
                        self.log.err(msg)
                        insufficient_info = True
                        break
                if insufficient_info:
                    # Do other vizzes
                    continue
                measure = kwargs['measure']
                clusters = kwargs['clusters']
                vizzer = Histogrammer(self.df, measure, clusters)
                fig = vizzer.run()
                figs.append(fig)
                fnames.append(f"maes_measure_{measure}_histo.png")


        if outdir is not None:
            one_fig: Figure
            for i, (one_fig, one_fname) in enumerate(zip(figs, fnames)):
                outpath = Path(outdir) / one_fname
                self.log.info(f"Saving '{vizzes[i]}' to {outpath}")
                one_fig.savefig(outpath)

        plt.show()
             

# ------------------- Class HeatMapTendencyBinary -------------
class HeatMapTendencyBinary:        

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 cluster_profile_df: pd.DataFrame, 
                 num_clusters: int):
        self.cluster_profile_df = cluster_profile_df
        self.num_clusters = num_clusters

    #------------------------------------
    # create_discrimination_heatmap
    #-------------------

    def run(self) -> tuple[Figure, Axes]:
        """
        Create a heatmap showing tendency (high/low) 
            for highly discriminative measures.
        
        :param cluster_profile_df: DataFrame with columns 
            ['cluster', 'measure', 'tendency', 'n_significant_discriminations']
        :param num_clusters: Total number of clusters in the analysis
        :return: Tuple of (fig, ax, pivot_data) matplotlib objects and data
        """
        # Filter for measures that discriminate against all other clusters
        max_discrimination = self.num_clusters - 1
        highly_discriminative = self.cluster_profile_df[
            self.cluster_profile_df['n_significant_discriminations'] == max_discrimination
        ]
        
        # Pivot to create the heatmap matrix
        pivot_data = highly_discriminative.pivot(
            index='measure', 
            columns='cluster', 
            values='tendency'
        )
        
        # Create color mapping function (modular for easy replacement)
        color_matrix, cmap, norm = self.get_color_matrix(pivot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_data) * 0.3)))
        
        # Create heatmap with discrete colormap
        im = ax.imshow(color_matrix, aspect='auto', cmap=cmap, norm=norm)
        
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
        
        # Add colorbar with discrete ticks
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([-1, 1])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.set_label('Tendency', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        
        return fig, ax, pivot_data   

    #------------------------------------
    # get_color_matrix
    #-------------------

    def get_color_matrix(self, 
                         pivot_data: pd.DataFrame
                         ) -> tuple[np.ndarray, mcolors.ListedColormap, mcolors.BoundaryNorm]:
        """
        Convert tendency labels to numeric values with discrete color mapping.
        Returns color matrix and discrete colormap for binary tendency values.
        
        :param pivot_data: Pivoted DataFrame with tendency values ('low'/'high')
        :return: Tuple of (color_matrix, colormap, norm) where:
                - color_matrix: Numpy array with numeric values (-1 for 'low', 1 for 'high')
                - colormap: Discrete ListedColormap with 2 colors
                - norm: BoundaryNorm for discrete color boundaries
        """
        # Map 'low' to -1 (blue) and 'high' to 1 (red)
        color_matrix = pivot_data.replace({'low': -1, 'high': 1}).infer_objects(copy=False).values
        
        # Create discrete colormap with only 2 colors
        # Blue for low (-1), Red for high (1) - matches RdBu_r at extremes
        cmap = mcolors.ListedColormap(['#2166ac', '#b2182b'])  # blue, red
        
        # Define boundaries: anything < 0 is blue, anything > 0 is red
        bounds = [-1.5, 0, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        return color_matrix, cmap, norm

# ------------------- Class HeatMapEffectSize -------------

class HeatMapEffectSize:
    """
    Creates a heatmap showing signed effect sizes for highly discriminative measures.
    
    Effect sizes are signed by tendency:
    - Positive (red): Cluster has HIGH values for this measure
    - Negative (blue): Cluster has LOW values for this measure
    - Magnitude indicates strength of discrimination

    Example usage:
        heatmap = HeatMapEffectSize(
            cluster_profile_df=posthoc_results['cluster_profiles'],
            num_clusters=9
        )

        fig, ax, pivot_data = heatmap.run()
        plt.savefig('effect_size_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    
    :param cluster_profile_df: DataFrame with columns 
        ['cluster', 'measure', 'tendency', 'n_significant_discriminations', 'avg_effect_size']
    :param num_clusters: Total number of clusters in the analysis
    """

    #------------------------------------
    # Constructor
    #-------------------    
    
    def __init__(
        self,
        cluster_profile_df: pd.DataFrame,
        num_clusters: int
    ):
        """
        Initialize the heatmap with cluster profile data.
        
        :param cluster_profile_df: DataFrame with cluster profile statistics
        :param num_clusters: Total number of clusters
        """
        self.cluster_profile_df = cluster_profile_df
        self.num_clusters = num_clusters
        self.max_discrimination = num_clusters - 1
        
        # Will be set by run()
        self.fig = None
        self.ax = None
        self.pivot_data = None

    #------------------------------------
    # run
    #-------------------        
    
    def run(self) -> tuple[Figure, Axes, pd.DataFrame]:
        """
        Generate the signed effect size heatmap.
        
        :return: Tuple of (fig, ax, pivot_data)
        """
        # Filter for measures that discriminate against all other clusters
        highly_discriminative = self.cluster_profile_df[
            self.cluster_profile_df['n_significant_discriminations'] == self.max_discrimination
        ].copy()
        
        # Create signed effect size: positive if high tendency, negative if low
        highly_discriminative['signed_effect'] = highly_discriminative.apply(
            lambda row: row['avg_effect_size'] if row['tendency'] == 'high' 
                        else -row['avg_effect_size'],
            axis=1
        )
        
        # Pivot to create the heatmap matrix
        self.pivot_data = highly_discriminative.pivot(
            index='measure',
            columns='cluster',
            values='signed_effect'
        )
        
        # Create color matrix
        color_matrix = self.get_color_matrix(self.pivot_data)
        
        # Determine vmin/vmax for symmetric colormap
        max_abs_value = np.nanmax(np.abs(color_matrix))
        vmin, vmax = -max_abs_value, max_abs_value
        
        # Create figure
        self.fig, self.ax = plt.subplots(
            figsize=(10, max(6, len(self.pivot_data) * 0.3))
        )
        
        # Create heatmap with diverging colormap
        im = self.ax.imshow(
            color_matrix,
            aspect='auto',
            cmap='RdBu_r',  # Red for positive (high), Blue for negative (low)
            vmin=vmin,
            vmax=vmax
        )
        
        # Set ticks and labels
        self.ax.set_xticks(np.arange(len(self.pivot_data.columns)))
        self.ax.set_yticks(np.arange(len(self.pivot_data.index)))
        self.ax.set_xticklabels(self.pivot_data.columns)
        self.ax.set_yticklabels(self.pivot_data.index)
        
        # Rotate x-axis labels if needed
        plt.setp(self.ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add labels
        self.ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Measure', fontsize=12, fontweight='bold')
        self.ax.set_title(
            f'Effect Size and Direction for Measures Discriminating Against All {self.max_discrimination} Other Clusters',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Add grid
        self.ax.set_xticks(np.arange(len(self.pivot_data.columns)) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(len(self.pivot_data.index)) - 0.5, minor=True)
        self.ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            'Effect Size (signed by tendency)',
            rotation=270,
            labelpad=20,
            fontweight='bold'
        )
        
        # Add annotation showing interpretation
        self.fig.text(
            0.5, 0.02,
            'Red = High values | Blue = Low values | Intensity = Strength of discrimination',
            ha='center',
            fontsize=9,
            style='italic',
            color='gray'
        )
        
        plt.tight_layout()
        
        return self.fig, self.ax, self.pivot_data

    #------------------------------------
    # get_color_matrix
    #-------------------    
    
    def get_color_matrix(self, pivot_data: pd.DataFrame) -> np.ndarray:
        """
        Convert signed effect size values to color matrix.
        Effect sizes are already signed by tendency (positive=high, negative=low).
        
        :param pivot_data: Pivoted DataFrame with signed effect size values
        :return: Numpy array with signed effect size values
        """
        # Simply extract values - they're already signed appropriately
        color_matrix = pivot_data.values
        return color_matrix

# ------------------- Class Histogrammer -------------
class Histogrammer:        

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, df: pd.DataFrame, measure: str, clusters: list[str]):
        self.measure = measure
        # Create a dictionary with cluster as key, measures values as Series
        cluster_data = {
            f'cluster_{c}': df[df['cluster'] == c][measure].reset_index(drop=True)
            for c in clusters
        }
        self.histogram_df = pd.DataFrame(cluster_data)
        self.clusters = clusters 
    
    #------------------------------------
    # run
    #-------------------

    def run(self):
        SimpleCharter(
            self.histogram_df,
            ChartType.HISTOGRAMS,
            title=f"Measure '{self.measure}' across clusters {self.clusters}",
            xlabel=f"Measure {self.measure}"
            )


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
    
    parser.add_argument('-i', '--illustrations',
                        choices=list(Vizzes),
                        nargs='+',
                        help='Repeatable: illustrations to create')

    parser.add_argument('-o', '--outdir',
                        help='directory where to place finished figures')
    
    # Additionals only for histograms chart type:
    parser.add_argument('-c', '--clusters',
                        type=int,
                        nargs='+',
                        help='Repeatable: clusters for histograms')
    parser.add_argument('-m', '--measure',
                        help='measure for histogram')


    args = parser.parse_args()

    datafile = Path(args.datafile)
    if not datafile.exists():
        print(f"Datafile {args.datafile} not found")
        sys.exit(1)

    cluster_profile = Path(args.cluster_profile)
    if not cluster_profile.exists():
        print(f"Datafile {args.cluster_profile} not found")
        sys.exit(1)
    
    viz = VisualizerMeasuresInClusters(datafile, 
                                       cluster_profile,
                                       args.illustrations,
                                       args.outdir,
                                       **{'clusters' : args.clusters,
                                          'measure': args.measure
                                          }
                                       )
    print(viz)
