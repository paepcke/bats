import pandas as pd 
import argparse

def add_cli(parser):
    parser.add_argument('--input_data_path', type=str, default='data/data.csv', help='input file')


def preprocess(args):
    df = pd.read_csv(args.input_data_path)
    max_seq_len = df.groupby("Filename").size().max()
    # add a new column to the dataframe, named "TimeIndex", which is the index of the time series (i.e. the row number), for a particular Filename.
    df.fillna(0, inplace=True)
    return df, max_seq_len