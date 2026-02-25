#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-17 10:13:33
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-24 16:01:23

import argparse
from enum import StrEnum
import os
from pathlib import Path
import sys

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from sonobat_utils.utils import Utils

class ChartType(StrEnum):
    BAR = 'bar'
    HISTOGRAMS = 'histograms'

class SimpleCharter:

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 data_info: Path | pd.DataFrame, 
                 chart_type: ChartType, 
                 x_axis_col: str | None = None, 
                 y_axis_cols: str | list[str] | None = None,
                 title: str = '',
                 xlabel: str = '',
                 ylabel: str = '',
                 outfile: Path = None
                 ):
        
        if isinstance(data_info, pd.DataFrame):
            df = data_info
        else:
            df = Utils.read_df_file(data_info)

        self.title  = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        if type(y_axis_cols) == str:
            y_axis_cols = [y_axis_cols]
        # Ensure that columns are present:
        col_names = df.columns
        if x_axis_col is not None and y_axis_cols is not None:
            for col in [x_axis_col] + y_axis_cols:
                if col not in col_names:
                    raise ValueError(f"Column {col} not in the given data")

        Utils.right_size_fontsizes()

        if chart_type == ChartType.BAR:
            x_values = df[x_axis_col].unique()
            x_values.sort()
            bars = df.groupby(x_axis_col)[y_axis_cols].sum()

            fig = self.bar_chart(x_values, bars, title, xlabel, ylabel)
        
        elif chart_type == ChartType.HISTOGRAMS:
            self.fig = self.histograms(df)            
        else:
            msg = f"Only chart types {list(ChartType)} are currently implemented, not '{chart_type}'"
            raise NotImplementedError(msg)
        
        if self.fig:
            if outfile is not None:
                if outfile.exists():
                    conf = input(f"Outfile '{outfile}' exists. Replace (y/n): ")
                    if conf != 'y':
                        print('Aborting')
                        return
                plt.savefig(str(outfile), dpi=300, bbox_inches='tight', transparent=True)
            plt.show(block=True)

    #------------------------------------
    # bar_chart 
    #-------------------

    def bar_chart(self, 
                x_vals: pd.Series, 
                y_vals_df: pd.Series | pd.DataFrame,
                title: str = '',
                x_label: str = '',
                y_label: str = ''
                ) -> Figure:

        if isinstance(y_vals_df, pd.Series):
            y_vals_df = pd.DataFrame(y_vals_df)
        elif not isinstance(y_vals_df, pd.DataFrame):
            raise TypeError(f"Y values must be a pd.Series or pd.DataFrame, not {y_vals_df}")
        
        n_bars = len(y_vals_df.columns)

        x_pos = np.arange(len(x_vals))
        width = 0.8 / n_bars  # Divide available space by number of bar groups
        
        fig, ax = plt.subplots()

        # Create bars for each Series in the list
        for i, (y_colname, y_series) in enumerate(y_vals_df.items()):
            # The centering logic remains the same
            offset = width * (i - (n_bars - 1) / 2)  
            
            # We use y_series directly as the height. 
            # We use y_series.name for the label (if the Series has one).
            ax.bar(x_pos + offset, y_series, width, label=getattr(y_series, 'name', f'Group {i+1}'))

        # Optional: Set the x-ticks to match your x_vals labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_vals)

        # Labels and formatting
        if len(x_label) > 0:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label if y_label else 'Value')
        if len(title) > 0:
            ax.set_title(title)
        ax.set_xticks(x_pos)
        
        ax.set_xticklabels(x_vals)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    #------------------------------------
    # histograms
    #-------------------    

    def histograms(self, data: pd.DataFrame | pd.Series) -> Figure:
        """
        Create overlapping histograms for one or more pandas Series.
        
        :param data: pandas DataFrame (each column plotted separately) or single pandas Series
        :return: matplotlib Figure object containing the histogram plot
        """
        # Convert Series to single-column DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Limit to 4 columns
        if len(data.columns) > 4:
            data = data.iloc[:, :4]
            print("Warning: Only the first 4 columns will be plotted")
        
        # Define colors with good alpha blending
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate common bins across all columns for proper alignment
        all_values = pd.concat([data[col].dropna() for col in data.columns], ignore_index=True)
        bins = np.histogram_bin_edges(all_values, bins='auto')
        
        # Plot each column
        for i, (name, col) in enumerate(data.items()):
            ax.hist(
                col.dropna(),
                bins=bins,
                #alpha=0.5,  # Transparency for blending
                #alpha=0.8,  # Transparency for blending
                alpha=1.0,   # No blending
                color=colors[i],
                label=name,
                edgecolor='white',
                linewidth=0.5
            )        
        # Styling
        ax.set_xlabel('Value' if self.xlabel == '' else self.xlabel, fontsize=12)
        ax.set_ylabel('Frequency' if self.ylabel == '' else self.ylabel, fontsize=12)
        ax.set_title('Distribution Comparison' if self.title == '' else self.title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        return fig

# ---------------- Main -------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Simple chars like bar chart"
                                     )

    parser.add_argument('infile',
                        help='input dataframe file: .csv or .feather',
                        )
    
    parser.add_argument('chart_type',
                        choices=list(ChartType),
                        help=f"type of chart to produce; required data depends on this type",
                        )
    
    parser.add_argument('x_column',
                        help="the dataframe column for the X-axis",
                        )

    parser.add_argument('y_columns',
                        type=str,
                        nargs='+',
                        help='Repeatable: columns for the y-axis')
    
    parser.add_argument('-t', '--title',
                        help="chart title. Default: empty string",
                        default=''
                        )

    parser.add_argument('-x', '--xlabel',
                        help="label for the x axis. Default: empty string",
                        default=''
                        )

    parser.add_argument('-y', '--ylabel',
                        help="label for the y axis. Default: empty string",
                        default=''
                        )

    parser.add_argument('-o', '--outfile',
                        help="path where to save figure",
                        default=None
                        )


    args = parser.parse_args()

    SimpleCharter(Path(args.infile),
                  args.chart_type,
                  args.x_column,
                  args.y_columns,
                  args.title,
                  args.xlabel,
                  args.ylabel,
                  Path(args.outfile) if args.outfile is not None else None
                  )


