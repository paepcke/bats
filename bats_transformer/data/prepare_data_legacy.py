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

minimum_length = 5

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
    parser.add_argument('--duplication', action='store_true')
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

'''
Pads groups with nan entries to make it of the same length as max_length.
'''
def pad_group(group, max_length):
    pad_length = max_length - len(group)
    filename = group.iloc[0]["Filename"]
    padding_df = pd.DataFrame(np.nan, index=np.arange(pad_length), columns=group.columns)
    padding_df["Filename"] = filename
    group.TimeInFile = group.TimeInFile - group.TimeInFile.min()

    return pd.concat([padding_df, group], ignore_index=True)

# Define a helper function for processing each group
def process_group(group):
    padded_df_tmp, truth_values_tmp = process_and_pad_groups(group, max_length, minimum_length)
    return padded_df_tmp, truth_values_tmp

'''
Takes in a particular group, and then keeps on popping the last row from it untill
it reaches the minimum length. Then, it takes all of these groups, and appends
null entries to the begining so that all of them are of the same length, 
ie max_length.
'''
def process_and_pad_groups(group, max_length, min_length):
    groups = []  # To hold the modified groups
    current_group = group.copy()
    truth_values = None

    # Keep removing the last row and adding to groups until reaching min_length
    while len(current_group) >= min_length:
        groups.append(current_group.copy())
        current_group = current_group[:-1]  # Remove the last row
    
    truth_values = group.copy()
    truth_values["cntxt_sz"] = range(len(truth_values))
    truth_values["TimeIndex"] = max_length - 1

    padded_groups = []
    for grp in groups:
        pad_length = max_length - len(grp)
        cntxt_sz = len(grp) - 1

        filename = grp.iloc[0]["Filename"]  # Assuming 'Filename' is a column in the DataFrame
        file_id = grp.iloc[0]["file_id"]
        grp["cntxt_sz"] = len(grp) - 1

        padding_df = pd.DataFrame(0, index=np.arange(pad_length), columns=grp.columns)
        padding_df["Filename"] = filename  # Fill 'Filename' column with the filename of the current group
        padding_df["file_id"] = file_id
        padding_df["cntxt_sz"] = cntxt_sz
        grp.TimeInFile = grp.TimeInFile - grp.TimeInFile.min()  # Adjust 'TimeInFile' as per the original function
        
        # Concatenate the padding dataframe and the current group
        padded_group = pd.concat([padding_df, grp], ignore_index=True)
        padded_group["TimeIndex"] = range(len(padded_group))
        padded_groups.append(padded_group)

    
    # Concatenate all padded groups into a single DataFrame
    final_df = pd.concat(padded_groups, ignore_index=True)
    
    return final_df, truth_values




args = add_cli(argparse.ArgumentParser()).parse_args()
minimum_length = args.minimum_length

print("Reading files... ", end="", flush=True)
filter_ = (lambda x: True)
selector = DaytimeFileSelector()

if(args.daytime):
    filter_ = lambda s: selector.is_daytime_recording(s)

df = get_df(get_files(args.input_data_path)).sort_values(["Filename", "TimeInFile"])
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

padded_df = None
truth_values = None
file_id_to_chirps = None

if(not args.duplication):
    df['chirp_idx'] = df.groupby('Filename').cumcount()
    padded_df = df
    padded_df.reset_index(inplace = True)
    file_id_to_chirps = df.groupby("file_id")["chirp_idx"].max().reset_index().sort_values("file_id")
    file_id_to_chirps["n_chirps"] = file_id_to_chirps["chirp_idx"] - minimum_length + 1
    file_id_to_chirps["cum_chirps"] = file_id_to_chirps["n_chirps"].cumsum()
    truth_values = padded_df
    truth_values["cntxt_sz"] = padded_df["chirp_idx"] 

else:
    # Parallelize processing using joblib and tqdm for progress tracking
    print("Padding and repeating... ", end="", flush=True)

    # Define the number of parallel jobs (e.g., the number of CPU cores)
    num_jobs = -1  # -1 means use all available cores

    # Apply parallel processing to each group
    results = Parallel(n_jobs=num_jobs)(delayed(process_group)(group) for _, group in tqdm(df.groupby("Filename")))

    # Separate the results into individual lists for concatenation
    padded_df_list = [result[0] for result in results]
    truth_values_list = [result[1] for result in results]

    # Concatenate the results
    padded_df = pd.concat(padded_df_list, ignore_index=True)
    truth_values = pd.concat(truth_values_list, ignore_index=True)

    del padded_df_list
    del truth_values_list
    del results
    del filename_to_id

    gc.collect()
print("Done.")



#write it out to n_files different files.
if(args.splits == 1):
    print("Writing to file... ", end="", flush=True)
    padded_df.to_csv(args.output_data_path)
    #write out truth values to a file
    truth_values.to_csv(args.output_data_path + "_truth_values.csv", index=False)
    print("Done.")

else:
    print("Writing to files... ", end="\n", flush=True)

    #takes a LOOOOONG time
    padded_df.drop(
        columns=["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 
                 'Preemphasis', 'MaxSegLnght', "ParentDir"], 
        inplace = True
    )

    columns_to_not_scale = ["file_id", "cntxt_sz"] if args.duplication else ["file_id", "chirp_idx"]
    columns_to_scale = [col for col in padded_df.columns if col not in columns_to_not_scale]

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    
    chunk_size = 100000
    print(len(padded_df))
    for i in tqdm(range(0, len(padded_df), chunk_size)):
        chunk = padded_df.loc[i:min(len(padded_df), i+chunk_size),:]
        scaler.partial_fit(chunk[columns_to_scale])
    
    for i in tqdm(range(0, len(padded_df), chunk_size)):
        padded_df.loc[i:min(len(padded_df), i+chunk_size), columns_to_scale] = scaler.transform(padded_df.loc[i:i+chunk_size, columns_to_scale])
    

    #storing off the scaler
    joblib.dump(scaler, args.output_data_path + "_scaler.pkl")
    print("Done. Scaler saved to ", args.output_data_path + "_scaler.pkl")

    #writing to splits
    print("Writing to splits= ", args.splits, " files...")
    print("Reseting index...")
    padded_df = padded_df.reset_index(drop = True)
    print("Done.")

    if(not args.duplication):
        total_files = len(file_id_to_chirps)
        n_splits = args.splits
        files_in_a_split = ((total_files + (n_splits/2))//n_splits) #rounding up
        file_id_to_chirps["split"] = (file_id_to_chirps["file_id"]//files_in_a_split).astype(int)
        padded_df = pd.merge(padded_df, file_id_to_chirps, on = 'file_id')
        

        for split, split_df in padded_df.groupby("split"):
            if args.use_feather:
                split_df.reset_index().to_feather(args.output_data_path + str(split) + ".feather")
            else:
                split_df.reset_index().to_csv(args.output_data_path + str(split) + ".csv", index = False)
        
        split_to_chirps = file_id_to_chirps.groupby("split")["n_chirps"].sum().reset_index()
        split_to_chirps["Filename"] = split_to_chirps["split"].apply(lambda x: os.path.abspath(args.output_data_path + str(x) + ".csv"))
        split_to_chirps.to_csv(args.output_data_path + "_mapping.csv")


    else:
        num_data_points = int(len(padded_df)/max_length)
        num_data_points_per_split = num_data_points//args.splits
        for i in tqdm(range(args.splits - 1)):
            temp_df = padded_df.iloc[i*(num_data_points_per_split)*max_length:(i+1)*(num_data_points_per_split)*max_length].reset_index(drop=True)
            
            if args.use_feather:
                temp_df.to_feather(args.output_data_path + str(i) + ".feather")
            else:
                temp_df.to_csv(args.output_data_path + str(i) + ".csv")
            
            print(temp_df.shape)
            print(max_length)
        #write an additional file with a mapping between feather file and the number of rows.
        mapping_df = pd.DataFrame()

        if args.use_feather:
            mapping_df["Filename"] = [os.path.abspath(args.output_data_path + str(i) + ".feather") for i in range(args.splits - 1)]
        else:
            mapping_df["Filename"] = [os.path.abspath(args.output_data_path + str(i) + ".csv") for i in range(args.splits - 1)]
        
        mapping_df["count"] = [num_data_points_per_split * max_length for i in range(args.splits - 1)]
        mapping_df.to_csv(args.output_data_path + "_mapping.csv")
    
    #TODO: get rid of these memory management tricks and use a better library instead...
    del padded_df
    gc.collect()
    print("Done.")
    if(truth_values is not None):
        print("Scaling truth values... ", end="", flush=True)    
        truth_values[columns_to_scale] = scaler.transform(truth_values[columns_to_scale])
        truth_values.to_csv(args.output_data_path + "_truth_values.csv", index=False)
        print("Done.")    
