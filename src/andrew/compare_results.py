# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import os
import sys
import argparse
import matplotlib.pyplot as plt

sys.path.append("..")
from analysis_utils import *
from idiom_comparer import IdiomComparer, IdiomComparerVisualizer

# Adjust font sizes:
# Set global sizes
plt.rcParams.update({
    'font.size': 18,          # Base font size
    'axes.titlesize': 24,     # Title size
    'axes.labelsize': 20,     # X and Y label size
    'xtick.labelsize': 16,    # X tick label size
    'ytick.labelsize': 16,    # Y tick label size
    'legend.fontsize': 16,    # Legend size
    'lines.linewidth': 3,     # Thicker lines for visibility
    'lines.markersize': 10    # Larger markers
})

def idiom_comparer_pipeline(idiom_comparer, results_path):
    print("Loading in data...")
    idiom_comparer.combine_inputs()

    print("Extracting idioms from data...")
    idiom_comparer.extract_idioms()

    print("Clustering data...")
    idiom_comparer.cluster_data()
    return idiom_comparer

def main(args):
    # First define the two experiments to be compared:
    results_1 = args.results_1
    results_2 = args.results_2
    exp_1_name = args.exp_1_name
    exp_2_name = args.exp_2_name
    results_folder = args.results_folder
    results_path = f"{results_folder}/{exp_1_name}_{exp_2_name}"
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        os.mkdir(f"{results_path}/figs")

    # Run the idiom comparison pipeline on the two sets of data
    idiom_comparer = IdiomComparer(results_1, results_2, exp_1_name, exp_2_name)
    idiom_comparer = idiom_comparer_pipeline(idiom_comparer, results_path)

    print("Calculating most common subsequences...")
    # FIGURE: 12.2
    top_clusters = idiom_comparer.most_common_cluster(idiom_comparer.idiom_label_sequences)
    print(top_clusters)

    # FIGURE: 12.3
    top_transitions = idiom_comparer.most_common_transitions(idiom_comparer.idiom_label_sequences)
    print(top_transitions)

    # Stepping away from comparison for a second, this section allows us to describe the characteristics of a given cluster
    profile_ignore_columns = ["index", "OriginalIndex", "file_id", "chirp_idx", "original_df", "cluster"]
    CLUSTER_TO_PROFILE = 7
    cluster_profile = describe_cluster(idiom_comparer.idiom_chirp_attributes, 
                                       CLUSTER_TO_PROFILE, 
                                       normalize=False, 
                                       ignore_columns=profile_ignore_columns)

    print("Generating figures...") 
    visualizer = IdiomComparerVisualizer(idiom_comparer, results_path)
    visualizer.generate_figures()
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare two sets of idioms")
    parser.add_argument("--results_1", type=str, help="Path to the first set of analysis results",
                        default="./analysis_results/2022_barn_2secs_myca_quantile_1_16")
    parser.add_argument("--results_2", type=str, help="Path to the second set of analysis results",
                        default="./analysis_results/2022_lake_2secs_myca_quantile_1_28")
    parser.add_argument("--exp_1_name", type=str, default="Barn", help="Name of the first experiment (for labeling purposes)")
    parser.add_argument("--exp_2_name", type=str, default="Lake", help="Name of the second experiment (for labeling purposes)")
    parser.add_argument("--results_folder", type=str, default="./analysis_results/comparisons", help="Folder to save comparison results and figures")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)