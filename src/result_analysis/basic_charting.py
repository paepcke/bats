# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-02-17 10:13:33
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-02-17 10:44:17

import argparse
import os
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class SimpleCharter:

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 path: Path, 
                 chart_type: str, 
                 x_axis_col: str, 
                 y_axis_cols: str | list[str],
                 title: str = '',
                 xlabel: str = '',
                 ylabel: str = ''
                 ):
        
        if not path.exists():
            raise FileNotFoundError(f"Path {path} not found")
        if path.suffix == '.feather':
            df = pd.read_feather(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise TypeError(f"Path must be .csv or .feather, not {path}")

        if type(y_axis_cols) == str:
            y_axis_cols = [y_axis_cols]
        # Ensure that columns are present:
        col_names = df.columns
        for col in [x_axis_col] + y_axis_cols:
            if col not in col_names:
                raise ValueError(f"Column {col} not in the given data")

        if chart_type == 'bar':
            self.bar_chart(df, x_axis_col, y_axis_cols, title, xlabel, ylabel)


    def bar_chart(self, 
                df: pd.DataFrame, 
                x_axis: str, 
                y_axis: list[str],
                title: str = '',
                x_label: str = '',
                y_label: str = ''
                ):
        
        x_values = df[x_axis]
        n_bars = len(y_axis)
        
        # Set up bar positions
        x_pos = np.arange(len(x_values))
        width = 0.8 / n_bars  # Divide available space by number of bar groups
        
        fig, ax = plt.subplots()
        
        # Create bars for each y column
        for i, y_col in enumerate(y_axis):
            offset = width * (i - (n_bars - 1) / 2)  # Center the bars
            ax.bar(x_pos + offset, df[y_col], width, label=y_col)
        
        # Labels and formatting
        ax.set_xlabel(x_label if x_label else x_axis)
        ax.set_ylabel(y_label if y_label else 'Value')
        ax.set_title(title if title else f'{", ".join(y_axis)} by {x_axis}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_values)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

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

    parser.add_argument('y_axis',
                        type=str,
                        nargs='+',
                        help='Repeatable: columns for the y-axis')
    
    parser.add_argument('t', '--title',
                        help="chart title. Default: empty string",
                        default=''
                        )

    parser.add_argument('x', '--xlabel',
                        help="label for the x axis. Default: empty string",
                        default=''
                        )

    parser.add_argument('y', '--ylabel',
                        help="label for the y axis. Default: empty string",
                        default=''
                        )

    args = parser.parse_args()

    SimpleCharter(args.infile,
                  args.x_column,
                  args.y_column,
                  args.title,
                  args.xlabel,
                  args.ylabel
                  )


