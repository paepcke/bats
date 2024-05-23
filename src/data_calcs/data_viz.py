'''
Created on Apr 28, 2024

@author: paepcke
'''

import matplotlib.pyplot as plt
#import sys
#sys.path.pop(0)

class DataViz:
    '''
    Visualizations such as Tsne plots.
    '''

    cmap = 'Dark2'

    #------------------------------------
    # plot_tsne
    #-------------------
    
    @staticmethod
    def plot_tsne(tsne_df, cluster_ids=None, title=None, show_plot=True):
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
        :param title: optional title for the plot
        :type title: union[None | str]
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
    
    @staticmethod
    def plot_perplexities_grid(tsne_labels_title_dicts,
                               show_plot=True, 
                               block_after_show=True):
        '''
        Create multiple KMeans(Tsne) cluster charts, and
        place them in a Figure grid.
        
        Returns the Figure.
        
        :param perplexity_tsne_title_dicts: list of dicts. Each dict
            has keys 'tsne_df', 'cluster_ids', and 'title'
        :type perplexity_tsne_title_dicts: list[dict[str : union[pd.DataFrame | list[str] | str]]]  
        :return Figure object, ready to show
        :rtype Figure
        
        '''

        # Get a 2x2 grid of axes, enough for 
        # all plots:
        num_axes = len(tsne_labels_title_dicts)
        # Grid will be two cols; compute number of 
        # rows needed:
        num_axes_rows = int(num_axes / 2) + (1 if num_axes % 2 >0 else 0)
        fig, axs = plt.subplots(nrows=num_axes_rows, ncols=2, layout='tight')
        
        # Last row of plots may not be populated:
        populated_axes = axs.flatten()[:num_axes]
        
        # Draw the plots 
        for ax, content_dict in zip(populated_axes, tsne_labels_title_dicts):
            tsne_df, cluster_ids, title = content_dict.values()
            if cluster_ids is None:
                ax.scatter(tsne_df['tsne_x'], tsne_df['tsne_y'])
            else:
                ax.scatter(tsne_df['tsne_x'], 
                           tsne_df['tsne_y'], 
                           c=cluster_ids,
                           cmap=DataViz.cmap
                           )
            ax.set_title(title, fontsize=9)
        
        fig.suptitle("Tsne Perplexities and KMeans n_cluster Values", fontweight='bold')
        fig.show()
        return fig

        
# ------------------------ Main ------------
if __name__ == '__main__':
    pass