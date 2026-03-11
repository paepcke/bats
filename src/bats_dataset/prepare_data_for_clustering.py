import argparse
import pandas as pd
import numpy as np
import glob
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os
from itertools import chain

minimum_length = 5
ignore_cols = ["FreqLedge","AmpK@end", "Fc", "FBak15dB  ", "FBak32dB", "EndF", "FBak20dB", "LowFreq", "Bndw20dB", 
               "CallsPerSec", "EndSlope", "SteepestSlope", "StartSlope", "Bndw15dB", "HiFtoUpprKnSlp", "HiFtoKnSlope", 
               "DominantSlope", "Bndw5dB", "PreFc500", "PreFc1000", "PreFc3000", "KneeToFcSlope", "TotalSlope", 
               "PreFc250", "CallDuration", "CummNmlzdSlp", "DurOf32dB", "SlopeAtFc", "LdgToFcSlp", "DurOf20dB", "DurOf15dB", 
               "TimeFromMaxToFc", "KnToFcDur", "HiFtoFcExpAmp", "AmpKurtosis", "LowestSlope", "KnToFcDmp", "HiFtoKnExpAmp", 
               "DurOf5dB", "KnToFcExpAmp", "RelPwr3rdTo1st", "LnExpB_StartAmp", "Filter", "HiFtoKnDmp", "LnExpB_EndAmp", 
               "HiFtoFcDmp", "AmpSkew", "LedgeDuration", "KneeToFcResidue", "PreFc3000Residue", "AmpGausR2", "PreFc1000Residue", 
               "Amp1stMean", "LdgToFcExp", "FcMinusEndF", "Amp4thMean", "HiFtoUpprKnExp", "HiFtoKnExp", "KnToFcExp", "UpprKnToKnExp", 
               "Kn-FcCurviness", "Amp2ndMean", "Quality", "HiFtoFcExp", "LnExpA_EndAmp", "RelPwr2ndTo1st", "LnExpA_StartAmp", 
               "HiFminusStartF", "Amp3rdMean", "PreFc500Residue", "Kn-FcCurvinessTrndSlp", "PreFc250Residue", "AmpVariance", "AmpMoment", 
               "meanKn-FcCurviness", "MinAccpQuality", "AmpEndLn60ExpC", "AmpStartLn60ExpC", "Preemphasis", "MaxSegLnght" ,"Max#CallsConsidered" ] + ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir"] + ["file_length"]

'''
Add the command line interface arguments to specify the input data path, output data path, and the
number of splits.
'''
def add_cli(parser):
    parser.add_argument('-i', '--input_data_path', type=str, default='/qnap/bats/jasperridge/barn1/grouped_audio', help='input data path')
    parser.add_argument('-o', '--output_data_path', type=str, default='./', help='output data path')
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






args = add_cli(argparse.ArgumentParser()).parse_args()
minimum_length = args.minimum_length

print("Reading files... ", end="", flush=True)
df = get_df(get_files(args.input_data_path)).sort_values(["Filename", "TimeInFile"])
print("DONE.")

df["file_length"] = df.groupby("Filename")["Filename"].transform("size")
df = df[df["file_length"] > minimum_length]

def replicate_columns(columns, length):
    return list(chain.from_iterable([f"{col}_{i}" for col in list(columns)] for i in range(length)))

for file_length, df_ in df.groupby("file_length"):
    
    final_df = None
    print("Doing for file len", file_length)
    for _, df2 in df_.groupby("Filename"):
        df3 = df2.drop(columns = ignore_cols + ['TimeInFile', 'PrecedingIntrvl'], errors = 'ignore').reset_index(drop = True)
        temp_arr = np.array(df3)
        temp_arr = temp_arr.reshape(1, -1)
        
        if(final_df is None):
            final_df = pd.DataFrame(temp_arr, columns = replicate_columns(df3.columns, file_length))
        else:
            final_df = pd.concat([final_df, pd.DataFrame(temp_arr, columns = replicate_columns(df3.columns, file_length))], axis = 0)
        
    #save arr to a csv file
    scaler = StandardScaler()
    scaler.set_output(transform='pandas')
    final_df = scaler.fit_transform(final_df)
    final_df.to_csv(os.path.join(args.output_data_path, f"{file_length}.csv"), index = False)
    

        


