import argparse
import pandas as pd
import numpy as np
import glob
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from data_calcs.daytime_file_selection import DaytimeFileSelector
from data_calcs.utils import Utils # for filename-friendly timestamp
import os
import gc

# Sample use:
# python src/bats_dataset/prepare_data.py -i ../audio/audio_07_22/original -o data/july_daytime_2022/splits/split -s 10 -f -m 5 -d
# python src/bats_dataset/prepare_data.py -i ../audio/audio_07_22/chopped -o data/july_daytime_chunked_quantile/splits/split -s 10 -f -m 5 --scaler quantile
# python src/bats_dataset/prepare_data.py -i ../audio/audio_07_22/chopped -o data/july_daytime_filter_epfu/splits/split -s 10 -f -m 5 --scaler quantile --species Epfu

'''
Add the command line interface arguments to specify the input data path, 
output data path, and the number of splits.
'''
def add_cli(parser):
    parser.add_argument('-i', '--input_data_path', type=str, 
                        default='/qnap/bats/jasperridge/barn1/grouped_audio', 
                        help='Folder containing subfolders with SonoBatch Parameters files')
    parser.add_argument('-o', '--output_data_path', type=str, 
                        default='./data', help='Folder for dataset and metadata')
    parser.add_argument('-s', '--splits', type = int, default = 1, help='Number of splits to shard dataset')
    parser.add_argument('-f', '--use_feather', action='store_true', 
                        help='Use .feather format (instead of .csv)')
    parser.add_argument('-m', '--minimum_length', type = int, default = 5, help='Minimum sequence length to accept')
    parser.add_argument('-d', '--daytime', action='store_true', 
                        help='Only keep files starting before sunset (instead of all files)')
    parser.add_argument('--species', type=str, help='Filter audio files by four-letter species code')
    parser.add_argument('--scaler', type=str, default='standard', help='Which scaler to use to normalize data')
    return parser

'''
Get all the files from a particular root directory
'''
def get_files(path):
    files = []
    for file in tqdm(glob.glob(path + '/**/*_Parameters_*.txt', recursive=True)):
        files.append(file)
    return files

'''
Get the species attribution dataframe from the cumulative sonobatch files
'''
def get_species_attribution_df(path):
    # get all .txt files with CumulativeSonoBatch in the name but not BatchSummary or NightlySummary
    cumulative_sonobatch_files = [
        fn for fn in sorted(os.listdir(path))
        if os.path.isfile(os.path.join(path, fn)) and fn.endswith(".txt") and "CumulativeSonoBatch" in fn and "BatchSummary" not in fn and "NightlySummary" not in fn]
    # read in the cumulative sono batch files into a single dataframe
    cumulative_sonobatch_dfs = []
    for fn in cumulative_sonobatch_files:
        df = pd.read_csv(os.path.join(path, fn), sep=None, engine="python")
        cumulative_sonobatch_dfs.append(df)
    cumulative_sonobatch_df = pd.concat(cumulative_sonobatch_dfs, ignore_index=True)
    return cumulative_sonobatch_df

'''
Get the dataframe from the files. Merge all of them into a single dataframe.
'''
def get_df(files, filter = (lambda x: True)):
    df = pd.DataFrame()
    for file in tqdm(files):
        df = pd.concat([df, (pd.read_csv(file, sep='\t'))], ignore_index = True)

    #filter column "filename" of df using filter
    df = df[df["Filename"].apply(filter)]
    return df

'''
Using the species dataframe, filter the audio dataframe to only include rows whose filename corresponds to the given species.
'''
def filter_by_species(audio_df, species_df, species):
    correct_species_files = species_df[species_df["SppAccp"] == species]["Filename"].unique()
    if species is not None:
        audio_df = audio_df[audio_df["Filename"].isin(correct_species_files)]
    return audio_df

args = add_cli(argparse.ArgumentParser()).parse_args()
minimum_length = args.minimum_length

# Ensure that the output directory exists:
out_dir = args.output_data_path
if not os.path.exists(out_dir):
    print(f"Creating output dir {out_dir}...")
    os.makedirs(out_dir)

print("Reading files... ", end="", flush=True)
filter_ = (lambda x: True)

if(args.daytime):
    filter_ = (lambda S: (lambda s: S.is_daytime_recording(s)))(DaytimeFileSelector())

df = get_df(get_files(args.input_data_path), filter = filter_).sort_values(["Filename", "TimeInFile"])

if args.species is not None:
    print(f"Filtering by species {args.species}... ", end="", flush=True)
    species_df = get_species_attribution_df(args.input_data_path)
    df = filter_by_species(df, species_df, args.species)

df.drop_duplicates(inplace = True)

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
]).to_csv(args.output_data_path + "/split_config.csv", index=False)




#drop all entries corresponging to Filenames which have less than minimum_length entries
print("Dropping Entries... ", end="", flush=True)
df = df[df.groupby("Filename").Filename.transform('size') > minimum_length]
print("Done.")

print("Creating mapping from filename to a unique id... ", end="", flush=True)
df["file_id"] = pd.factorize(df["Filename"])[0]
filename_to_id = df.groupby("Filename")["file_id"].first().reset_index()
filename_to_id.to_csv(args.output_data_path + "/split_filename_to_id.csv", index=False)   
print("Done.")



df['chirp_idx'] = df.groupby('Filename').cumcount()
df.reset_index(inplace = True, drop = True)
file_id_to_chirps = df.groupby("file_id")["chirp_idx"].max().reset_index().sort_values("file_id")
file_id_to_chirps["n_samples"] = file_id_to_chirps["chirp_idx"] - minimum_length + 2 
file_id_to_chirps["cum_samples"] = file_id_to_chirps["n_samples"].cumsum()

# Write a timestamp of the format used in
# analysis downstream to outdir/timestamp.txt:

timestamp      = Utils.file_timestamp()
out_dir        = args.output_data_path
timestamp_path = os.path.join(out_dir, 'timestamp.txt')
with open(timestamp_path, 'w') as fd:
    fd.write(timestamp)
        
if(args.splits == 1):
    print("Writing CSV file to file... ", end="", flush=True)
    df.to_csv(args.output_data_path + "/split.csv")
    #write out truth values to a file
    truth_values = df
    truth_values["cntxt_sz"] = truth_values.chirp_idx 
    truth_values.to_csv(args.output_data_path + "/split_truth_values.csv", index=False)
    print("Done.")

else:
    print("Writing to files... ", end="\n", flush=True)

    final_cols = list(df.columns)
    to_drop   = ['Filename', 'NextDirUp', 'Path', 'Version', 'Filter', 
                 'Preemphasis', 'MaxSegLngth', 'ParentDir']
    for col_to_drop in to_drop:
        if col_to_drop in final_cols:
            final_cols.remove(col_to_drop)
    df_new = df[final_cols]
    df = df_new

    columns_to_not_scale = ["file_id", "chirp_idx"]
    columns_to_scale = [col for col in df.columns if col not in columns_to_not_scale]

    if args.scaler == 'robust':
        scaler = RobustScaler()
    elif args.scaler == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')        
    else:
        scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    
    if args.scaler == 'robust' or args.scaler == 'quantile':
        scaler.fit(df[columns_to_scale])
        df.loc[:, columns_to_scale] = scaler.transform(df.loc[:, columns_to_scale])
    else:
        chunk_size = 100000
        print(len(df))
        for i in tqdm(range(0, len(df), chunk_size)):
            chunk = df.loc[i:min(len(df), i+chunk_size),:]
            scaler.partial_fit(chunk[columns_to_scale])
        
        for i in tqdm(range(0, len(df), chunk_size)):
            df.loc[i:min(len(df), i+chunk_size), columns_to_scale] = scaler.transform(df.loc[i:i+chunk_size, columns_to_scale])
    

    #storing off the scaler
    joblib.dump(scaler, args.output_data_path + "/split_scaler.pkl")
    print("Done. Scaler saved to ", args.output_data_path + "/split_scaler.pkl")

    #writing to splits
    print("Writing to splits= ", args.splits, " files...")
    print("Resetting index...")
    df = df.reset_index(drop = True)
    print("Done.")


    total_files = len(file_id_to_chirps)
    n_splits = args.splits
    files_in_a_split = ((total_files + (n_splits/2))//n_splits) #rounding up
    file_id_to_chirps["split"] = (file_id_to_chirps["file_id"]//files_in_a_split).astype(int)
    df = pd.merge(df, file_id_to_chirps[['file_id', 'split']], on = 'file_id')
    

    for split, split_df in df.groupby("split"):
        if args.use_feather:
            split_df.reset_index(drop = True).to_feather(args.output_data_path + "/split" + str(split) + ".feather")
        else:
            split_df.to_csv(args.output_data_path + "/split" + str(split) + ".csv", index = False)
    
    split_to_chirps = file_id_to_chirps.groupby("split")["n_samples"].sum().reset_index()
    if args.use_feather:
        split_to_chirps["Filename"] = split_to_chirps["split"].apply(lambda x: os.path.abspath(args.output_data_path + "/split" + str(x) + ".feather"))
    else:
        split_to_chirps["Filename"] = split_to_chirps["split"].apply(lambda x: os.path.abspath(args.output_data_path + "/split" + str(x) + ".csv"))
    split_to_chirps.to_csv(args.output_data_path + "/split_mapping.csv")


    print("Done.")
    truth_values = df
    truth_values["cntxt_sz"] = truth_values["chirp_idx"]

    if(args.use_feather):
        truth_values.reset_index().to_feather(args.output_data_path + "/split_truth_values.feather")
    else:
        truth_values.to_csv(args.output_data_path + "/split_truth_values.csv", index=False)
