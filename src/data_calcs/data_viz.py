'''
Created on Apr 28, 2024

@author: paepcke
'''

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
#import sys
#sys.path.pop(0)

class DataViz:
    '''
    Visualizations such as Tsne plots.
    '''

    cmap = 'Dark2'

    #------------------------------------
    # simple_chart
    #-------------------
    
    @staticmethod
    def simple_chart(df,
                     xlabel='',
                     ylabel='',
                     title=None,
                     kind='line',
                     stacked=False,
                     **kwargs):
        '''
        Create one or more plots from a dataframe.
        If stacked is False, all columns will be plotted
        on the same chart. Else each column is plotted
        on a separate plot, one below the other in the
        figure. If df only contains a single column, stacked
        is ignored.
        
        The kwargs are passed to Series.plot(), and from there
        to ax.plot(). Noteworthy are scalex, and scaley to control
        scaling behavior, and the boolean choice 'legend' to ask
        for or suppress a legend. 
        
        The 'kind' argument controls the type of chart. Values are:
        
	        ‘line’    : line plot (default)
	        ‘bar’     : vertical bar plot
	        ‘barh’    : horizontal bar plot
	        ‘hist’    : histogram
	        ‘box’     : boxplot
	        ‘kde’     : Kernel Density Estimation plot
	        ‘density’ : same as ‘kde’
	        ‘area’    : area plot
	        ‘pie’     : pie plot
	        ‘scatter’ : scatter plot (DataFrame only)
	        ‘hexbin’  : hexbin plot (DataFrame only)
	
        See documentation for pandas.Series.plot for other details
        
        :param df: data to plot
        :type df: pd.DataFrame
        :param xlabel: x axis label
        :type xlabel: str
        :param ylabel: y axis label
        :type ylabel: str
        :param title: title for entire figure
        :type title: optional[str]
        :param kind: type of chart (line, vs. scatter vs. ...)
            see method comment. Default is line chart
        :type kind: optional[str]
        :param stacked: whether or not all columns are to 
            be plotted on the same chart.
        :type stacked: bool
        :return the Figure object
        :rtype matplotlib.Figure
        '''
        
        if isinstance(df, pd.DataFrame):
            cols = df.columns
        elif isinstance(df, pd.Series):
            cols = df.name
        else:
            raise TypeError(f"Data to plot must be a Pandas dataframe or series, not {df}")

        if stacked:
            fig, ax = plt.subplots(len(cols))
        else:
            fig, ax = plt.subplots(1)

        if not stacked:
            if isinstance(df, pd.DataFrame):
                # Plot all cols into one axis:
                for _ser_name, ser_vals in df.items():
                    ser_vals.plot(ax=ax, kind=kind, xlabel=xlabel, ylabel=ylabel, title=title, stacked=stacked, **kwargs)
            else:
                # Data is just a series: Don't place a legend:
                df.plot(ax=ax, kind=kind, xlabel=xlabel, ylabel=ylabel, title=title, stacked=stacked, legend=False, **kwargs)
        else:
            # Without the following you get AttributeError: 'numpy.ndarray' object has no attribute 'get_figure'
            # Unpack all the axes in the subplots
            axes = ax.ravel()
            for plot_num, (col_name, col) in enumerate(df.items()):
                col.plot(ax=axes[plot_num],
                         sharex=True,
                         sharey=True,     # Doesn't work b/c drawing in this loop
                         kind=kind, 
                         xlabel=xlabel,
                         ylabel=None,     # No individual y labels: that works
                         title=col_name,  # Title for individual subplot
                         stacked=stacked, 
                         legend=False,    # Legend replaced with subplot titles 
                         **kwargs)
            
            # Main title above the whole figure:
            fig.suptitle(title)
            # More padding between the subplots:
            fig.subplots_adjust(hspace=0.97)
            # Common Y axis label:
            fig.supylabel(ylabel)
        
        return fig    

    #------------------------------------
    # heatmap
    #-------------------
    
    @staticmethod
    def heatmap(df, xlabel_rot=None, title=None, width_height=None, save_file=None):
        '''
        Create a heatmap of a dataframe. Optionally add a title
        and/or save the figure.
        
        First removes any data that is not numeric from a
        local dataframe copy.
        
        Return the resulting Figure instance, and a list 
        of columns that had to be removed, because they were not
        numeric.  
        
         
        :param df: dataframe to visualize
        :type df: pd.DataFrame
        :param xlabel_rot: optional rotation of the xlabels in degrees
        :type xlabel_rot: union[float, int]
        :param title: optional title for the figure
        :type title: optiona[str]
        :param width_height: a 2-tuple specifying the figure size
            in inches: (width, height)
        :type width_height: optional[tuple[number, number]]
        :param save_file: optional path where to save the figure as png
        :type save_file: optional[str]
        :return the Figure object
        :rtype Figure
        '''
        
        heatmap = sb.heatmap(df, cmap='Blues')
        fig = heatmap.get_figure()
        ax  = fig.gca()
        
        if title is not None:
            if type(title) != str:
                raise TypeError(f"Title must be None, or a string, not {title}")
            fig.suptitle(title)
            
        if xlabel_rot is not None:
            if type(xlabel_rot) not in (int, float):
                raise TypeError(f"X label rotation must be numeric degrees, not {xlabel_rot}")
            ax.tick_params(axis='x', labelrotation=xlabel_rot)
        
        if width_height is not None:
            if type(width_height) != tuple or\
                type(width_height[0]) not in (int, float) or\
                type(width_height[1]) not in (int, float):
                raise TypeError(f"Figure size must be None, or a tuple of numbers for inches, not {width_height}")
            fig.set_size_inches(width_height[0], width_height[1])
            
        if save_file is not None:
            fig.savefig(save_file)
        return fig

    #------------------------------------
    # draw_xy_lines
    #-------------------
    
    @staticmethod
    def draw_xy_lines(ax, x, y, **kwargs):
        '''
        To an already existing curve, add a vertical
        and a horizontal line from the x and y axis
        to a datapoint of interest:
        
			   ^
			 y |
			   |              x
		  0.9  |------x
			   |      |
			   |  x   |
			   |      |
			   ----------------------------->
                 1... 22                    x
        
        In this example, this method was called with (22, 0.9)
        Both x and y are in data coordinates.
        
        All kwargs are passed to plot.
        
        Returns the horizontal line object, and the vertical line object.
        These may be used to change color, thickness, etc. after drawng.
        
        :param ax: matplotlib Axes on which to draw the lines
        :type ax: matplotlib.axes.Axes
        :param x: horizontal coordinate where lines should meet.
            In data coordinates
        :type x: union[float, int, str]
        :param y: vertical coordinate where lines should meet
            In data coordinates
        :type y: union[float, int, str]
        :return the horizontal and vertical lines
        :rtype list[matplotlib.lines.Line2D]
        '''
        
        # To ensure that the lines reach their respective
        # x-axis and y-axis, we need to work in axes coordinates.
        # Reason: if an axes is categorical, the line will end
        # at data coordinate 0, which may be short of the axis:
        
        ax_x, ax_y = ax.transLimits.transform((x,y))
        
        # Horizonal line at altitude y=ax_y, from 
        # x=0 to the axis coordinate equivalent of the
        # datapoint's x:
        hor_line = ax.plot([0, ax_x], [ax_y, ax_y], transform=ax.transAxes, **kwargs)
        
        # At X axis axes-coord equivalent to data point's x,
        # vertical from 0 to axes equivalent to datapoint's y
        ver_line = ax.plot([ax_x, ax_x], [0, ax_y], transform=ax.transAxes, **kwargs)
        return (hor_line, ver_line)
        

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
        if title:
            fig.suptitle(title)
        
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