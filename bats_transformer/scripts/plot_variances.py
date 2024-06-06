import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tqdm
import multiprocessing

parser = argparse.ArgumentParser(description='Plot variances of columns over chirp index')
parser.add_argument('--input', type=str, help='Path to the directory containing the model outputs', default = '/home/vdesai/data/model_outputs/daytime/')
parser.add_argument('--output', type=str, help='Path to the directory where the plots will be saved', default = '/home/vdesai/plots/daytime/')



args = parser.parse_args()

os.makedirs(args.output, exist_ok = True)




df = pd.concat([pd.read_csv(os.path.join(args.input, f)).drop(columns = ['Unnamed: 0']) for f in os.listdir(args.input) if f.endswith('.log')], ignore_index= True)


variances = df.drop(columns = ['model_id']).groupby(['file_id', 'chirp_idx']).var().reset_index()



def plot_col_variances(df, file_id, colnames):
    # Use seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Create a color palette
    colors = sns.color_palette("hsv", len(colnames))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # Increase DPI for higher resolution
    
    for i, colname in enumerate(colnames):
        df_slice = df[df.file_id == file_id]
        a = np.array(df_slice[["chirp_idx", colname]])
        
        # Plot the data
        ax.plot(a[:, 0], a[:, 1], marker='o', linestyle='-', color=colors[i], label=colname)
    
    # Add titles and labels with bold font
    ax.set_title(f'Variation of Columns over Chirp Index for file_id {file_id}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Chirp Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Values', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add legend
    #ax.legend(title="Columns", fontsize=12, title_fontsize=14)
    
    # Improve layout
    fig.tight_layout()
    
    return fig, ax

def plot_col_variances_sum(df, file_id, colnames, min = None, max = None):
    # Use seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Create a color palette
    colors = sns.color_palette("hsv", len(colnames))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # Increase DPI for higher resolution

    df_slice = df[df.file_id == file_id]

    a = np.array(df_slice[["chirp_idx"] + colnames])
    
    ax.plot(a[:, 0], np.sum(a[:, 1:], axis = 1), marker = 'o', linestyle = '-', color = 'r', label = 'sum')
    sums = np.sum(a[:, 1:], axis = 1)
    #get moving average of sums
    ax.plot(a[:, 0], pd.Series(sums).rolling(window=3).mean(), marker = 'o', linestyle = '-', color = 'g', label = 'moving average')
    
    #ax.plot(a[:, 0], np.median(a[:, 1:], axis = 1), marker = 'o', linestyle = '-', color = 'r', label = 'median')
 
    
    # Add titles and labels with bold font
    ax.set_title(f'Variation of Columns over Chirp Index for file_id {file_id}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Chirp Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Values', fontsize=14, fontweight='bold')

    if min is not None and max is not None:
        ax.set_ylim([min, max])
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add legend
    ax.legend(title="Columns", fontsize=12, title_fontsize=14)
    
    # Improve layout
    fig.tight_layout()
    
    return fig, ax

def plot_save_col_variances(df, file_id, colnames, min = None, max = None):
    fig, ax = plot_col_variances_sum(df, file_id, colnames, min, max)
    fig.savefig(os.path.join(args.output, f'{file_id}.png'), dpi=300)
    plt.close(fig)

cols = list(variances.columns)
cols.remove('file_id')
cols.remove('chirp_idx')

max_chirp_idx = variances.groupby('file_id').chirp_idx.max()
valid_file_id = max_chirp_idx[max_chirp_idx > 13].index

sums = np.sum(np.array(variances[cols]), axis = 1).reshape(-1)
min, max = np.percentile(sums, 0.05), np.percentile(sums, 99.9)

valid_file_id = list(valid_file_id)

#get number of cpu cores
pool = multiprocessing.Pool(multiprocessing.cpu_count())


pool.starmap(plot_save_col_variances, [(variances, file_id, cols, min, max) for file_id in valid_file_id])

pool.close()
pool.join()

#save the variances dataframe
variances.to_csv(os.path.join(args.output, 'variances.csv'), index = False)
