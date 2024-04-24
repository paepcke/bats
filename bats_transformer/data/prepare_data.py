import argparse
import pandas as pd
import numpy as np
import glob
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

minimum_length = 5

'''
Add the command line interface arguments to specify the input data path, output data path, and the
number of splits.
'''
def add_cli(parser):
    parser.add_argument('-i', '--input_data_path', type=str, default='/qnap/bats/jasperridge/barn1/grouped_audio', help='input file')
    parser.add_argument('-o', '--output_data_path', type=str, default='./data.csv', help='output file')
    parser.add_argument('-s', '--splits', type = int, default = 1)
    parser.add_argument('-f', '--use_feather', action='store_true', help='use feather format')
    parser.add_argument('-m', '--minimum_length', type = int, default = 5)
    return parser

'''
Get all the files from a particular root directory
'''
def get_files(path):
    files = []
    for file in glob.glob(path + '/**/*Parameters_*.txt', recursive=True):
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

'''
Takes in a particular group, and then keeps on popping the last row from it untill
it reaches the minimum length. Then, it takes all of these groups, and appends
null entries to the begining so that all of them are of the same length, 
ie max_length.
'''
def process_and_pad_groups(group, max_length, min_length, filename_to_id):
    groups = []  # To hold the modified groups
    current_group = group.copy()
    
    # Keep removing the last row and adding to groups until reaching min_length
    while len(current_group) >= min_length:
        groups.append(current_group.copy())
        current_group = current_group[:-1]  # Remove the last row
    
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
    return final_df




args = add_cli(argparse.ArgumentParser()).parse_args()
minimum_length = args.minimum_length

print("Reading files... ", end="", flush=True)
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

# Apply the padding function to each group and concatenate them
print("Padding and repeating... ", end="", flush=True)
padded_df = pd.concat([process_and_pad_groups(group, max_length, minimum_length, filename_to_id) 
                               for _, group in tqdm(df.groupby('Filename'))], ignore_index=True)
print("Done.")

#write it out to n_files different files.
if(args.splits == 1):
    print("Writing to file... ", end="", flush=True)
    padded_df.to_csv(args.output_data_path)
    print("Done.")

else:
    print("Writing to files... ", end="\n", flush=True)
    print("Scaling data... ", end="", flush=True)
    padded_df.drop(columns=["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir"], inplace = True)
    columns_to_not_scale = ["file_id", "cntxt_sz"]
    columns_to_scale = [col for col in padded_df.columns if col not in columns_to_not_scale]

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    padded_df[columns_to_scale] = pd.DataFrame(scaler.fit_transform(padded_df[columns_to_scale]), columns = padded_df.columns)

    #store the parameters of the scaler somewhere (mean, variances) so that can recover dataset.
    joblib.dump(scaler, args.output_data_path + "_scaler.pkl")
    print("Done. Scaler saved to ", args.output_data_path + "_scaler.pkl")

    #write padded_df to args.splits different files, each with a different subset of the rows. Also, make sure 
    #that it has rows in the multiple of max_length irrespective of the number of splits.
    print("Writing to splits= ", args.splits, " files...")
    padded_df = padded_df.reset_index(drop = True)
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
        mapping_df["Filename"] = [args.output_data_path + str(i) + ".feather" for i in range(args.splits - 1)]
    else:
        mapping_df["Filename"] = [args.output_data_path + str(i) + ".csv" for i in range(args.splits - 1)]
    
    mapping_df["count"] = [num_data_points_per_split * max_length for i in range(args.splits - 1)]
    mapping_df.to_csv(args.output_data_path + "_mapping.csv")
    
    
    print("Done.")    
