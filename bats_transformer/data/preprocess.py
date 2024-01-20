import pandas as pd 
import argparse

def add_cli(parser):
    parser.add_argument('--input_data_path', type=str, default='data/data_2.csv', help='input file')


def preprocess(args):
    df = pd.read_csv(args.input_data_path)
    max_seq_len = df.groupby("Filename").size().max()
    df.fillna(0, inplace=True)
    return df, max_seq_len
