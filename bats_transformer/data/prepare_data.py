import argparse
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

minimum_length = 5
def add_cli(parser):
    parser.add_argument('-i', '--input_data_path', type=str, default='/qnap/bats/jasperridge/barn1/grouped_audio', help='input file')
    parser.add_argument('-o', '--output_data_path', type=str, default='./data.csv', help='output file')
    parser.add_argument('-s', '--splits', type = int, default = 1)

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


def process_and_pad_groups(group, max_length, min_length):
    groups = []  # To hold the modified groups
    current_group = group.copy()
    
    # Keep removing the last row and adding to groups until reaching min_length
    while len(current_group) >= min_length:
        groups.append(current_group.copy())
        current_group = current_group[:-1]  # Remove the last row
    
    padded_groups = []
    for grp in groups:
        pad_length = max_length - len(grp)
        filename = grp.iloc[0]["Filename"]  # Assuming 'Filename' is a column in the DataFrame
        padding_df = pd.DataFrame(0, index=np.arange(pad_length), columns=grp.columns)
        padding_df["Filename"] = filename  # Fill 'Filename' column with the filename of the current group
        grp.TimeInFile = grp.TimeInFile - grp.TimeInFile.min()  # Adjust 'TimeInFile' as per the original function
        
        # Concatenate the padding dataframe and the current group
        padded_group = pd.concat([padding_df, grp], ignore_index=True)
        padded_group["TimeIndex"] = range(len(padded_group))
        padded_groups.append(padded_group)
    
    # Concatenate all padded groups into a single DataFrame
    final_df = pd.concat(padded_groups, ignore_index=True)
    
    return final_df




parser = argparse.ArgumentParser()
add_cli(parser)
args = parser.parse_args()

print("Reading files... ", end="", flush=True)
df = get_df(get_files(args.input_data_path)).sort_values(["Filename", "TimeInFile"])
print("DONE.")
max_length = df.groupby("Filename").size().max()
#get number of unique counts for Filename
num_files = len(df.groupby("Filename"))
print("number of unique files: ", num_files)

#drop all entries corresponging to Filenames which have less than minimum_length entries
print("Dropping Entries... ", end="", flush=True)
df = df[df.groupby("Filename").Filename.transform('size') > minimum_length]
print("DONE.")

# Apply the padding function to each group and concatenate them
print("Padding and repeating... ", end="", flush=True)
padded_df = pd.concat([process_and_pad_groups(group, max_length, minimum_length) 
                               for _, group in tqdm(df.groupby('Filename'))], ignore_index=True)
print("DONE.")

#write it out to n_files different files.
if(args.splits == 1):
    print("Writing to file... ", end="", flush=True)
    padded_df.to_feather(args.output_data_path)
    print("DONE.")

else:
    print("Writing to files... ", end="", flush=True)
    padded_df.drop(columns=["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir"], inplace = True)
    scaler = StandardScaler()
    padded_df = pd.DataFrame(scaler.fit_transform(padded_df), columns = padded_df.columns)

    #write padded_df to args.splits different files, each with a different subset of the rows. Also, make sure 
    #that it has rows in the multiple of max_length irrespective of the number of splits.
    num_data_points = int(len(padded_df)/max_length)
    num_data_points_per_split = num_data_points//args.splits
    for i in range(args.splits - 1):
        padded_df.iloc[i*(num_data_points_per_split)*max_length:(i+1)*(num_data_points_per_split)*max_length].to_feather(args.output_data_path + str(i) + ".feather")
    
    #write an additional file with a mapping between feather file and the number of rows.
    mapping_df = pd.DataFrame()
    mapping_df["Filename"] = [args.output_data_path + str(i) + ".feather" for i in range(args.splits - 1)]
    mapping_df["count"] = [num_data_points_per_split * max_length for i in range(args.splits - 1)]
    mapping_df.to_feather(args.output_data_path + "_mapping.feather")
    
    
    print("DONE.")    
