'''
Created on Apr 28, 2024

@author: paepcke
'''

from data_calcs.data_calculations import DataCalcs
from data_calcs.universal_fd import UniversalFd
from data_calcs.utils import Utils
from logging_service import LoggingService
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
sys.path.pop(0)



class DataViz:
    '''
    Visualizations such as Tsne plots.
    '''

    cmap = 'Dark2'

    #------------------------------------
    # plot_tsne
    #-------------------
    
    @staticmethod
    def plot_tsne(tsne_df, cluster_ids=None, show_plot=True):
        '''
        Given the dataframe that is output by run_tsne(),
        create a scatterplot. If cluster_ids is a list that assigns
        each TSNE point to a cluster, then the resulting plot
        will be colored by cluster membership.
        
        Typically, the cluster_labels will be the 'labels_' attribute
        of a KMeans object. But anything like [0,0,1,2,0,0,...] will work,
        as does a 2D array of R,G,B colors, or a list of hex colors. 
        See the 'c' argument of matplotlib's scatter() method.
        
        If the cluster_ids argument is None, all points will have
        the same color.
        
        This method returns the Figure object that contains the
        scatterplot. Callers may extract the plot axis through:
        
             fig.axes  ==> [<axis>]  
        
        If show_plot is False, the Figure is created but not 
        shown on screen. The caller can manipulate the plot,
        or place it in a grid with others before showing.
        
        :param tsne_df: dataframe output from run_tsne()
        :type tsne_df: pd.DataFrame
        :param cluster_ids: a list of ints or strings, same length as
            tsne_df has rows. Each id is a label for a cluster to which
            the respective Tsne point belongs.
        :type cluster_ids: union[None | list[int] | list[str]]
        :param show_plot: whether or not this method should display
            the chart on the display
        :type show_plot: bool
        :return the matplotlib Figure instance
        :rtype plt.Figure 
        '''
        
        fig, ax = plt.subplots()
        cols = tsne_df.columns
        
        if cluster_ids is None:
            ax.scatter(tsne_df[cols[0]], tsne_df[cols[1]])
        else:
            ax.scatter(tsne_df[cols[0]], 
                       tsne_df[cols[1]], 
                       c=cluster_ids,
                       cmap=DataViz.cmap
                       )
        # If you want x-labels at 45deg:
        # ax.set_xticklabels(ax.get_xticks(), rotation = 45)
        if show_plot:
            plt.show(block=False)
        return fig
        
    #------------------------------------
    # plot_perplexities_grid
    #-------------------
    
    def plot_perplexities_grid(self, perplexities, show_plot=True, block_after_show=True):
        '''
        Create multiple KMeans(Tsne) cluster charts, and
        place them in a Figure grid. Each Tsne calculation is
        performed with a different perplexity, as per the 
        argument.
        
        Returns the Figure.
        
        :param perplexities: list of perplexities to use in 
            different Tsne calculations
        :type perplexities: list[float]
        :param show_plot: whether or not to display the computed
            figure grid.
        :type show_plot: bool
        :return Figure object, ready to show
        :rtype Figure
        
        '''
        
        tsne_df   = [self.run_tsne(perplexity=perp)
                      for perp in perplexities]
        
        # Save the (by default eight) add_silhouette that each
        # KMeans computation places into self.add_silhouette;
        # create a nested array:
        add_silhouette = []
        
        kmeans_objs = []
        clusters_to_try = list(range(2,10))
        for tsne_df in tsne_df:
            kmean = self.cluster_tsne(tsne_df, cluster_range=clusters_to_try)
            kmeans_objs.append(kmean)
            add_silhouette.append(self.add_silhouette)
            
        # Find the best tsne perplexity and n_clusters by
        # building the following df:
        #
        #                     N_CLUSTERS
        #                  2      3  ...  9
        #    PERPLEXITY
        #        5.0      0.3    0.8 ... 0.4
        #       10.0           ...
        #       20.0           ...
        #       30.0           ...
        #       50.0           ...
        #        
        # Then find the max cell, which is the highest 
        # silhouette coefficient:
        #******* Get best tsne_df from best-silhouette-kmean_obj
        
        silhouette_df = pd.DataFrame(add_silhouette, 
                                     index=perplexities,
                                     columns=clusters_to_try,
                                     )
        silhouette_df.index.name = 'Perplexity'
        
        (optimal_perplexity, optimal_n_clusters) = Utils.max_df_coords(silhouette_df)


        figs       = [self.plot_tsne(tsne_df, kmeans=kmeans, show_plot=False)
                      for tsne_df, kmeans
                      in zip(tsne_df, kmeans_objs)]
        
        fig = plt.figure(tight_layout=True)
        fig.suptitle("Clustering Chirps With Varying Perplexities")
        
        # Use a grid of plots that leaves at most
        # one unused field in the lower right of the grid:
        grid_square_dims = int(np.ceil(np.sqrt(len(perplexities)))) 
        grid_spec = GridSpec(grid_square_dims, grid_square_dims)
        for fig in figs:
            grid_spec.add_axes(fig.axes[0])
        
        if show_plot:
            plt.show(block=block_after_show)
        
        
# ------------------------ Main ------------
if __name__ == '__main__':
    pass