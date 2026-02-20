#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-17 10:13:33
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-17 19:02:18

import argparse
import os
from pathlib import Path
import sys

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


class SimpleCharter:

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 infile: Path, 
                 chart_type: str, 
                 x_axis_col: str, 
                 y_axis_cols: str | list[str],
                 title: str = '',
                 xlabel: str = '',
                 ylabel: str = '',
                 outfile: Path = None
                 ):
        
        if not infile.exists():
            raise FileNotFoundError(f"Path {infile} not found")
        if infile.suffix == '.feather':
            df = pd.read_feather(infile)
        elif infile.suffix == '.csv':
            df = pd.read_csv(infile)
        else:
            raise TypeError(f"Path must be .csv or .feather, not {infile}")

        if type(y_axis_cols) == str:
            y_axis_cols = [y_axis_cols]
        # Ensure that columns are present:
        col_names = df.columns
        for col in [x_axis_col] + y_axis_cols:
            if col not in col_names:
                raise ValueError(f"Column {col} not in the given data")

        # Adjust font sizes:
        # Set global sizes
        plt.rcParams.update({
            'font.size': 18,          # Base font size
            'axes.titlesize': 24,     # Title size
            'axes.labelsize': 20,     # X and Y label size
            'xtick.labelsize': 16,    # X tick label size
            'ytick.labelsize': 16,    # Y tick label size
            'legend.fontsize': 16,    # Legend size
            'lines.linewidth': 3,     # Thicker lines for visibility
            'lines.markersize': 10    # Larger markers
        })        

        if chart_type == 'bar':
            x_values = df[x_axis_col].unique()
            x_values.sort()
            bars = df.groupby(x_axis_col)[y_axis_cols].sum()

            fig = self.bar_chart(x_values, bars, title, xlabel, ylabel)
        else:
            raise NotImplementedError(f"Only chart type 'bar' is currently implemented, not '{chart_type}'")
        
        if fig:
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
                        help="any of {'bar'} (others might get added)",
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


