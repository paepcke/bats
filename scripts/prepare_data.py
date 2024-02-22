## This script can be used to prepare the data in CSV format to feed the spacetimeformer.
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input", "-i", type = str, help = "Path to the input file")
parser.add_argument("--output", "-o", type = str, help = "Path to output file")

args = parser.parse_args();

df = pd.read_csv(args.input, sep = "\t");
df = df.sort_values(["Filename", "TimeInFile"])

max_length = df.groupby("Filename").size().max()

def pad_group(group):
    pad_length = max_length - len(group)
    filename = group.iloc[0]["Filename"]
    padding_df = pd.DataFrame(np.nan, index=np.arange(pad_length), columns=group.columns)
    padding_df["Filename"] = filename
    group.TimeInFile = group.TimeInFile - group.TimeInFile.min()

    return pd.concat([padding_df, group], ignore_index=True)

# Apply the padding function to each group and concatenate them
padded_df = pd.concat([pad_group(group) for _, group in df.groupby('Filename')], ignore_index=True)
padded_df.to_csv(args.output, index = False)

