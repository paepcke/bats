import argparse
import pandas as pd
import numpy as np
import glob

minimum_length = 5
def add_cli(parser):
    parser.add_argument('-i', '--input_data_path', type=str, default='/qnap/bats/jasperridge/barn1/grouped_audio', help='input file')
    parser.add_argument('-o', '--output_data_path', type=str, default='./data.csv', help='output file')

def get_files(path):
    files = []
    for file in glob.glob(path + '/**/*Parameters_*.txt', recursive=True):
        files.append(file)
    return files

def get_df(files):
    df = pd.DataFrame()
    for file in files:
        df = pd.concat([df, (pd.read_csv(file, sep='\t'))], ignore_index = True)
    return df

def pad_group(group, max_length):
    pad_length = max_length - len(group)
    filename = group.iloc[0]["Filename"]
    padding_df = pd.DataFrame(np.nan, index=np.arange(pad_length), columns=group.columns)
    padding_df["Filename"] = filename
    group.TimeInFile = group.TimeInFile - group.TimeInFile.min()

    return pd.concat([padding_df, group], ignore_index=True)

parser = argparse.ArgumentParser()
add_cli(parser)
args = parser.parse_args()

df = get_df(get_files(args.input_data_path)).sort_values(["Filename", "TimeInFile"])
max_length = df.groupby("Filename").size().max()

#drop all entries corresponging to Filenames which have less than minimum_length entries
df = df[df.groupby("Filename").Filename.transform('size') > minimum_length]

# Apply the padding function to each group and concatenate them
padded_df = pd.concat([pad_group(group, max_length) for _, group in df.groupby('Filename')], ignore_index=True)
padded_df["TimeIndex"] = padded_df.groupby("Filename").cumcount()
padded_df.to_csv(args.output_data_path, index = False)
