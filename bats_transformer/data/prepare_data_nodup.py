import argparse
import pandas as pd
import numpy as np
import glob
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from data_calcs.daytime_file_selection import DaytimeFileSelector
import os
import gc


'''
Add the command line interface arguments to specify the input data path, 
output data path, and the number of splits.
'''
def add_cli(parser):
    parser.add_argument('-i', '--input_data_path', type=str, 
                        default='/qnap/bats/jasperridge/barn1/grouped_audio', 
                        help='input file')
    
    parser.add_argument('-o', '--output_data_path', type=str, 
                        default='./data.csv', help='output file')
    
    parser.add_argument('-s', '--splits', type = int, default = 1)
    parser.add_argument('-f', '--use_feather', action='store_true', 
                        help='use feather format')
    parser.add_argument('-m', '--minimum_length', type = int, default = 5)
    parser.add_argument('-d', '--daytime', action='store_true')
    return parser

'''
Get all the files from a particular root directory
'''
def get_files(path, filter_ = (lambda x: True)):
    files = []
    for file in list(filter(filter_, glob.glob(path + '/**/*Parameters_*.txt', recursive=True))):
        files.append(file)
    return files

'''
Get the dataframe from the files. Merge all of them into a single dataframe.
'''
def get_df(files):
    df = pd.DataFrame()
    for file in files:
        df = pd.concat([df, (pd.read_csv(file, sep='\t'))], ignore_index = True)
    return df


args = add_cli(argparse.ArgumentParser()).parse_args()
minimum_length = args.minimum_length

print("Reading files... ", end="", flush=True)
filter_ = (lambda x: True)

if(args.daytime):
    filter_ = (lambda S: (lambda s: S.is_daytime_recording(s)))(DaytimeFileSelector())

df = get_df(get_files(args.input_data_path, filter = filter_)).sort_values(["Filename", "TimeInFile"])
print("Done.")


#storing the config
max_length = df.groupby("Filename").size().max()
num_files = len(df.groupby("Filename"))
print("max length: ", max_length)
print("min length: ", minimum_length)
print("number of unique files: ", num_files)

pd.DataFrame([
    {"parameter": "max_length", "value": max_length},
    {"parameter": "min_length", "value": minimum_length},
    {"parameter": "num_files", "value": num_files}
]).to_csv(args.output_data_path + "_config.csv", index=False)




#drop all entries corresponging to Filenames which have less than minimum_length entries
print("Dropping Entries... ", end="", flush=True)
df = df[df.groupby("Filename").Filename.transform('size') > minimum_length]
print("Done.")

print("Creating mapping from filename to a unique id... ", end="", flush=True)
df["file_id"] = pd.factorize(df["Filename"])[0]
filename_to_id = df.groupby("Filename")["file_id"].first().reset_index()
filename_to_id.to_csv(args.output_data_path + "_filename_to_id.csv", index=False)   
print("Done.")



df['chirp_idx'] = df.groupby('Filename').cumcount()
df.reset_index(inplace = True, drop = True)
file_id_to_chirps = df.groupby("file_id")["chirp_idx"].max().reset_index().sort_values("file_id")
file_id_to_chirps["n_samples"] = file_id_to_chirps["chirp_idx"] - minimum_length + 2 
file_id_to_chirps["cum_samples"] = file_id_to_chirps["n_samples"].cumsum()

if(args.splits == 1):
    print("Writing to file... ", end="", flush=True)
    df.to_csv(args.output_data_path)
    #write out truth values to a file
    truth_values = df
    truth_values["cntxt_sz"] = truth_values.chirp_idx 
    truth_values.to_csv(args.output_data_path + "_truth_values.csv", index=False)
    print("Done.")

else:
    print("Writing to files... ", end="\n", flush=True)

    #takes a LOOOOONG time
    df.drop(
        columns=["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 
                 'Preemphasis', 'MaxSegLnght', "ParentDir"], 
        inplace = True
    )

    columns_to_not_scale = ["file_id", "chirp_idx"]
    columns_to_scale = [col for col in df.columns if col not in columns_to_not_scale]

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    
    chunk_size = 100000
    print(len(df))
    for i in tqdm(range(0, len(df), chunk_size)):
        chunk = df.loc[i:min(len(df), i+chunk_size),:]
        scaler.partial_fit(chunk[columns_to_scale])
    
    for i in tqdm(range(0, len(df), chunk_size)):
        df.loc[i:min(len(df), i+chunk_size), columns_to_scale] = scaler.transform(df.loc[i:i+chunk_size, columns_to_scale])
    

    #storing off the scaler
    joblib.dump(scaler, args.output_data_path + "_scaler.pkl")
    print("Done. Scaler saved to ", args.output_data_path + "_scaler.pkl")

    #writing to splits
    print("Writing to splits= ", args.splits, " files...")
    print("Reseting index...")
    df = df.reset_index(drop = True)
    print("Done.")


    total_files = len(file_id_to_chirps)
    n_splits = args.splits
    files_in_a_split = ((total_files + (n_splits/2))//n_splits) #rounding up
    file_id_to_chirps["split"] = (file_id_to_chirps["file_id"]//files_in_a_split).astype(int)
    df = pd.merge(df, file_id_to_chirps[['file_id', 'split']], on = 'file_id')
    

    for split, split_df in df.groupby("split"):
        if args.use_feather:
            split_df.reset_index(drop = True).to_feather(args.output_data_path + str(split) + ".feather")
        else:
            split_df.to_csv(args.output_data_path + str(split) + ".csv", index = False)
    
    split_to_chirps = file_id_to_chirps.groupby("split")["n_samples"].sum().reset_index()
    if args.use_feather:
        split_to_chirps["Filename"] = split_to_chirps["split"].apply(lambda x: os.path.abspath(args.output_data_path + str(x) + ".feather"))
    else:
        split_to_chirps["Filename"] = split_to_chirps["split"].apply(lambda x: os.path.abspath(args.output_data_path + str(x) + ".csv"))
    split_to_chirps.to_csv(args.output_data_path + "_mapping.csv")


    print("Done.")
    truth_values = df
    truth_values["cntxt_sz"] = truth_values["chirp_idx"]

    if(args.use_feather):
        truth_values.reset_index().to_feather(args.output_data_path + "_truth_values.feather")
    else:
        truth_values.to_csv(args.output_data_path + "_truth_values.csv", index=False)
