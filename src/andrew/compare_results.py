# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib import colormaps
import matplotlib.lines
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import umap

sys.path.append("..")
from analysis_utils import *
from intermodel_confidence import ChirpClusterer

NO_AMP = 0 # set to 0 to keep all amp features, 1 to remove [Amp1stQrtl, Amp2ndQrtl, Amp3rdQrtl, Amp4thQrtl], 2 to remove all amp features

MIN_CLUSTER_K = 1
MAX_CLUSTER_K = 50

CLUSTER_METHOD = "Agglomerative"
USE_UMAP = True

K_MOST_COMMON = 10
SUBSEQ_TYPE = "all"

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

class IdiomComparer():
    def __init__(self, results_1, results_2, exp_1_name="exp_1", exp_2_name="exp_2"):
        self.results_1 = results_1
        self.results_2 = results_2
        self.exp_1_name = exp_1_name
        self.exp_2_name = exp_2_name

    def combine_inputs(self):
        assert self.results_1 and self.results_2, "IdiomComparer not properly initialized"
        chirp_attributes_1 = pd.read_csv(f"{self.results_1}/test_set_chirp_attributes.csv")
        chirp_attributes_1 = pd.read_csv(f"{self.results_1}/all_chirp_measures_scaled_quantile.csv")
        chirp_attributes_1 = pd.read_csv(f"{self.results_1}/all_chirp_measures_scaled_robust.csv")
        idiom_boundaries_1 = pd.read_csv(f"{self.results_1}/idiom_boundaries.csv", header=None)
        confidence_measures_1 = pd.read_csv(f"{self.results_1}/chirp_prediction_confidence_measures.csv")
        chirp_attributes_2 = pd.read_csv(f"{self.results_2}/test_set_chirp_attributes.csv")
        chirp_attributes_2 = pd.read_csv(f"{self.results_2}/all_chirp_measures_scaled_quantile.csv")
        chirp_attributes_2 = pd.read_csv(f"{self.results_2}/all_chirp_measures_scaled_robust.csv")
        idiom_boundaries_2 = pd.read_csv(f"{self.results_2}/idiom_boundaries.csv", header=None)
        confidence_measures_2 = pd.read_csv(f"{self.results_2}/chirp_prediction_confidence_measures.csv")

        combined_chirp_attributes = pd.concat([chirp_attributes_1, chirp_attributes_2], ignore_index=False).reset_index()
        combined_chirp_attributes.rename({"index": "OriginalIndex"}, axis=1, inplace=True)
        idiom_boundaries_attributes = pd.concat([idiom_boundaries_1, idiom_boundaries_2], ignore_index=False).reset_index()
        idiom_boundaries_attributes.rename({"index": "OriginalIndex"}, axis=1, inplace=True)
        confidence_measures_attributes = pd.concat([confidence_measures_1, confidence_measures_2], ignore_index=False).reset_index()
        confidence_measures_attributes.rename({"index": "OriginalIndex"}, axis=1, inplace=True)

        for dataframe, df_1 in [(combined_chirp_attributes, chirp_attributes_1), 
                                (idiom_boundaries_attributes, idiom_boundaries_1), 
                                (confidence_measures_attributes, confidence_measures_1)]:
            dataframe["original_df"] = dataframe.apply(lambda x: 1 if x.name < len(df_1) else 2, axis=1)

        self.combined_chirp_attributes = combined_chirp_attributes
        self.idiom_boundaries_attributes = idiom_boundaries_attributes
        return combined_chirp_attributes, idiom_boundaries_attributes, confidence_measures_attributes

    def extract_idioms(self):
        # get the idiom chirp attributes
        idiom_chirp_idxs = []
        for idx, row in self.idiom_boundaries_attributes.iterrows():
            for i in range(int(row[0]), int(row[1]) + 1):
                idiom_chirp_idxs.append((idx, i, row["original_df"]))

        idiom_chirp_attributes = pd.DataFrame()
        for idiom_idx, chirp_idx, original_df in idiom_chirp_idxs:
            attributes = self.combined_chirp_attributes.loc[(self.combined_chirp_attributes["original_df"] == original_df) & 
                                                            (self.combined_chirp_attributes["OriginalIndex"] == chirp_idx)]
            if idiom_chirp_attributes.empty:
                idiom_chirp_attributes = attributes
            else:
                idiom_chirp_attributes = pd.concat([idiom_chirp_attributes, attributes])
        idiom_chirp_attributes.reset_index(inplace=True)
        
        self.idiom_chirp_attributes = idiom_chirp_attributes
        return idiom_chirp_attributes

    def cluster_data(self):
        idiom_chirp_data = self.idiom_chirp_attributes.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy()
        clusterer = ChirpClusterer(idiom_chirp_data, NO_AMP, USE_UMAP, CLUSTER_METHOD, MIN_CLUSTER_K, MAX_CLUSTER_K, calculate_k=True)
        clusterer.prepare_data()
        linked, idiom_chirp_attributes_embedded_df, chirp_labels, filtered_chirp_data_2d, filtered_chirp_labels = clusterer.cluster_data()
        self.linked = linked
        idiom_chirp_attributes_embedded = idiom_chirp_attributes_embedded_df.to_numpy()
        self.idiom_chirp_attributes_embedded = idiom_chirp_attributes_embedded
        self.chirp_labels = chirp_labels
        self.filtered_chirp_data_2d = filtered_chirp_data_2d
        self.filtered_chirp_labels = filtered_chirp_labels

        idiom_label_sequences = self.idiom_chirp_attributes.groupby(["file_id", "original_df"]).agg(list)["cluster"].reset_index()
        self.idiom_label_sequences = idiom_label_sequences
        return idiom_chirp_attributes_embedded, chirp_labels, idiom_label_sequences

    def most_common_cluster(self, idiom_label_sequences):
        SUBSEQ_LENGTH = 1
        # find the most common cluster across both experiments
        subseq_all = most_common_subsequences(idiom_label_sequences['cluster'].values, 
                                SUBSEQ_LENGTH, 
                                subseq_type=SUBSEQ_TYPE, 
                                k=K_MOST_COMMON)
        subseq_exp1 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 1]['cluster'].values, 
                                SUBSEQ_LENGTH, 
                                subseq_type=SUBSEQ_TYPE,
                                k=K_MOST_COMMON)
        subseq_exp2 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 2]['cluster'].values, 
                                SUBSEQ_LENGTH, 
                                subseq_type=SUBSEQ_TYPE,
                                k=K_MOST_COMMON)
        subseq_all = pd.DataFrame(subseq_all, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} (all)', 'Count'])
        subseq_exp1 = pd.DataFrame(subseq_exp1, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({self.exp_1_name})', 'Count'])
        subseq_exp2 = pd.DataFrame(subseq_exp2, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({self.exp_2_name})', 'Count'])
        res = pd.concat([subseq_all, subseq_exp1, subseq_exp2], axis=1)
        res.insert(0, "Rank", range(1, len(res) + 1))
        return res

    def most_common_transitions(self, idiom_label_sequences):
        idiom_clusters_temp = []
        idiom_clusters_temp_1 = []
        idiom_clusters_temp_2 = []
        for idx, row in idiom_label_sequences.iterrows():
            idiom_clusters_temp.extend(row['cluster'])
            if row["original_df"] == 1:
                idiom_clusters_temp_1.extend(row['cluster'])
            else:
                idiom_clusters_temp_2.extend(row['cluster'])
        idiom_cluster_counts_all = Counter(idiom_clusters_temp)
        idiom_cluster_counts_1 = Counter(idiom_clusters_temp_1)
        idiom_cluster_counts_2 = Counter(idiom_clusters_temp_2)

        # find most common transitions
        SUBSEQ_LENGTH = 2
        subseq_all = most_common_subsequences(idiom_label_sequences['cluster'].values, 
                                SUBSEQ_LENGTH, 
                                subseq_type=SUBSEQ_TYPE, 
                                k=K_MOST_COMMON)
        subseq_exp1 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 1]['cluster'].values, 
                                SUBSEQ_LENGTH, 
                                subseq_type=SUBSEQ_TYPE,
                                k=K_MOST_COMMON)
        subseq_exp2 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 2]['cluster'].values, 
                                SUBSEQ_LENGTH, 
                                subseq_type=SUBSEQ_TYPE,
                                k=K_MOST_COMMON)

        # total_subseq_count_all = sum(most_common_subsequences(idiom_label_sequences['cluster'].values, SUBSEQ_LENGTH, subseq_type=SUBSEQ_TYPE).values())
        # total_subseq_count_exp1 = sum(most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 1]['cluster'].values, SUBSEQ_LENGTH, subseq_type=SUBSEQ_TYPE).values())
        # total_subseq_count_exp2 = sum(most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 2]['cluster'].values, SUBSEQ_LENGTH, subseq_type=SUBSEQ_TYPE).values())

        subseq_all = pd.DataFrame(subseq_all, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} (all)', 'Count'])
        subseq_all_proportion = subseq_all.copy()
        subseq_all_proportion['Start Cluster Count'] = subseq_all_proportion.apply(lambda x: idiom_cluster_counts_all[x['Subseq (all)'][0]], axis=1)
        subseq_all_proportion['Transition Probability'] = subseq_all_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_all[x['Subseq (all)'][0]], axis=1)
        subseq_all_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
        subseq_all_proportion = subseq_all_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        subseq_all_proportion["Transition Probability"] = subseq_all_proportion["Transition Probability"].round(3)

        subseq_exp1 = pd.DataFrame(subseq_exp1, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({self.exp_1_name})', 'Count'])
        subseq_exp1_proportion = subseq_exp1.copy()
        subseq_exp1_proportion['Start Cluster Count'] = subseq_exp1_proportion.apply(lambda x: idiom_cluster_counts_1[x[f'Subseq ({self.exp_1_name})'][0]], axis=1)
        subseq_exp1_proportion['Transition Probability'] = subseq_exp1_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_1[x[f'Subseq ({self.exp_1_name})'][0]], axis=1)
        subseq_exp1_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
        subseq_exp1_proportion = subseq_exp1_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        subseq_exp1_proportion["Transition Probability"] = subseq_exp1_proportion["Transition Probability"].round(3)

        subseq_exp2 = pd.DataFrame(subseq_exp2, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({self.exp_2_name})', 'Count'])
        subseq_exp2_proportion = subseq_exp2.copy()
        # subseq_exp2_proportion['Proportion'] = subseq_exp2_proportion['Count'] / total_subseq_count_exp2
        subseq_exp2_proportion['Start Cluster Count'] = subseq_exp2_proportion.apply(lambda x: idiom_cluster_counts_2[x[f'Subseq ({self.exp_2_name})'][0]], axis=1)
        subseq_exp2_proportion['Transition Probability'] = subseq_exp2_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_2[x[f'Subseq ({self.exp_2_name})'][0]], axis=1)
        subseq_exp2_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
        subseq_exp2_proportion = subseq_exp2_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        subseq_exp2_proportion["Transition Probability"] = subseq_exp2_proportion["Transition Probability"].round(3)

        res = pd.concat([subseq_all, subseq_exp1, subseq_exp2], axis=1)
        res.insert(0, "Rank", range(1, len(res) + 1))
        res_proportion = pd.concat([subseq_all_proportion, subseq_exp1_proportion, subseq_exp2_proportion], axis=1)
        res_proportion.insert(0, "Rank", range(1, len(res_proportion) + 1))
        res_proportion.rename(columns = {f'Subseq ({self.exp_1_name})': f'{self.exp_1_name} Subseq', 
                                         f'Subseq ({self.exp_2_name})': f'{self.exp_2_name} Subseq'}, inplace=True)
        header=[['','All','',f'{self.exp_1_name}', '', f'{self.exp_2_name}', ''], 
                ['Rank','Subseq','Proportion', 'Subseq','Proportion', 'Subseq','Proportion']]
        res_proportion.columns=header
        return res_proportion


class IdiomComparerVisualizer():
    def __init__(self, idiom_comparer, results_path):
        self.idiom_comparer = idiom_comparer
        self.results_path = results_path
        self.num_clusters = max(self.idiom_comparer.chirp_labels) - min(self.idiom_comparer.chirp_labels) + 1

        self.cmap1 = "tab10"
        cmap1_colors = colormaps["tab10"](np.linspace(0, 1, 256))
        cmap2_colors = cmap1_colors * 0.35
        cmap2_colors[:, 3] = 1
        cmap2_colors
        self.cmap2 = ListedColormap(cmap2_colors, name="Set1dark")

    def generate_figures(self):
        self.plot_all_chirp_scatter()
        self.plot_idiom_chirp_scatter()
        self.plot_idiom_cluster_scatter()
        self.plot_idiom_cluster_indiv_scatter()
        self.plot_clusters_histogram()

    def plot_all_chirp_scatter(self):
        # embed the combined chirp attributes for plotting
        combined_chirp_attributes_embedded = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        ).fit_transform(self.idiom_comparer.combined_chirp_attributes.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy())

        # Figure 1: Scatterplot of all chirps, by experiment
        # Visualize how the two experiments are distributed in a 2-D projection:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(combined_chirp_attributes_embedded[:, 0], 
                            combined_chirp_attributes_embedded[:, 1], 
                            c=self.idiom_comparer.combined_chirp_attributes["original_df"], 
                            cmap=self.cmap1,
                            vmin=1,
                            vmax=10,
                            s=5,
                            )
        plt.title(f"Distribution of chirp embeddings")# in UMAP space")
        handles, labels = scatter.legend_elements()
        plt.legend(handles, [self.idiom_comparer.exp_1_name, self.idiom_comparer.exp_2_name])
        plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
        plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_chirps.png")

    def plot_idiom_chirp_scatter(self):
        # Figure 2: Scatterplot of idiom chirps, by experiment
        # Visualize the distribution of only idiom chirps (chirps within a "whole" idiom):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.idiom_comparer.idiom_chirp_attributes_embedded[:, 0], 
                            self.idiom_comparer.idiom_chirp_attributes_embedded[:, 1], 
                            c=self.idiom_comparer.idiom_chirp_attributes["original_df"], 
                            cmap=self.cmap1,
                            vmin=1,
                            vmax=10,
                            s=5,
                            )
        plt.title(f"Distribution of idiom chirp embeddings")# in UMAP space")
        handles, labels = scatter.legend_elements()
        plt.legend(handles, [self.idiom_comparer.exp_1_name, self.idiom_comparer.exp_2_name])
        plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
        plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_idiom_chirps.png")

    def plot_idiom_cluster_scatter(self):
        # Figure 3: Scatterplot of idiom chirps, clustered, by experiment
        # plot the idiom chirps with their clusters, with the two experiments designated with different shapes
        second_exp_first_row = self.idiom_comparer.idiom_chirp_attributes.loc[
            self.idiom_comparer.idiom_chirp_attributes["original_df"] == 2].iloc[0].name

        plt.figure(figsize=(12, 8))
        scatter1 = plt.scatter(self.idiom_comparer.idiom_chirp_attributes_embedded[:second_exp_first_row, 0], 
                            self.idiom_comparer.idiom_chirp_attributes_embedded[:second_exp_first_row, 1], 
                            c=self.idiom_comparer.chirp_labels[:second_exp_first_row], 
                            cmap=self.cmap1, 
                            s=5, 
                            vmin=min(self.idiom_comparer.chirp_labels), 
                            vmax=self.num_clusters - 1
                            )
        scatter2 = plt.scatter(self.idiom_comparer.idiom_chirp_attributes_embedded[second_exp_first_row:, 0], 
                            self.idiom_comparer.idiom_chirp_attributes_embedded[second_exp_first_row:, 1], 
                            c=self.idiom_comparer.chirp_labels[second_exp_first_row:], 
                            cmap=self.cmap2, 
                            s=12, 
                            marker="s",
                            vmin=min(self.idiom_comparer.chirp_labels), 
                            vmax=self.num_clusters - 1
                            )
        plt.title(f"Distribution of idiom chirp embeddings")# in UMAP space")
        # handles, labels = scatter1.legend_elements()
        # plt.legend(handles, ["barn", "lake"])
        plt.colorbar(scatter2, label=f"Cluster Label ({self.idiom_comparer.exp_2_name})")
        plt.colorbar(scatter1, label=f"Cluster Label ({self.idiom_comparer.exp_1_name})")
        legend_symbol_barn = matplotlib.lines.Line2D([0], [0], marker='.', color='w', label='Significant Peak', markerfacecolor='black', markersize=15)
        legend_symbol_lake = matplotlib.lines.Line2D([0], [0], marker='s', color='w', label='Significant Peak', markerfacecolor='black', markersize=9)
        plt.legend([legend_symbol_barn, legend_symbol_lake], [self.idiom_comparer.exp_1_name, self.idiom_comparer.exp_2_name])
        plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
        plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_idiom_chirps_clusters.png")

    def plot_idiom_cluster_indiv_scatter(self):
        second_exp_first_row = self.idiom_comparer.idiom_chirp_attributes.loc[
            self.idiom_comparer.idiom_chirp_attributes["original_df"] == 2].iloc[0].name
        # Figure 4: Scatterplot of idiom chirps, clustered, one experiment at a time (with cluster outlines)
        
        for exp_to_plot in [1, 2]:
            plt.cla()
            plt.figure(figsize=(10, 8))
            first_exp_attributes_embedded = self.idiom_comparer.idiom_chirp_attributes_embedded[:second_exp_first_row]
            second_exp_attributes_embedded = self.idiom_comparer.idiom_chirp_attributes_embedded[second_exp_first_row:]
            for i in range(self.num_clusters):
                cluster_points = self.idiom_comparer.idiom_chirp_attributes_embedded[self.idiom_comparer.idiom_chirp_attributes['cluster'] == i]
                if len(cluster_points) > 2:
                    hull = ConvexHull(cluster_points)
                    # Plot the hull as a closed line
                    for simplex in hull.simplices:
                        plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', lw=2, alpha=1) # 'k-' for a solid black line
                
                first_exp_cluster_points = first_exp_attributes_embedded[self.idiom_comparer.idiom_chirp_attributes['cluster'][:second_exp_first_row] == i]
                second_exp_cluster_points = second_exp_attributes_embedded[self.idiom_comparer.idiom_chirp_attributes['cluster'][second_exp_first_row:] == i]
                if exp_to_plot == 1:
                    scatter = plt.scatter(first_exp_cluster_points[:, 0], first_exp_cluster_points[:, 1], s=5, label=f'Cluster {i}', alpha=1)
                else:
                    scatter = plt.scatter(second_exp_cluster_points[:, 0], second_exp_cluster_points[:, 1], s=5, label=f'Cluster {i}', alpha=1)

            plt.title(f"Distribution of idiom chirp embeddings ({self.idiom_comparer.exp_1_name if exp_to_plot == 1 else self.idiom_comparer.exp_2_name})")
            plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
            plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2")
            plt.savefig(f"{self.results_path}/figs/scatter_idiom_chirps_clusters_{self.idiom_comparer.exp_1_name if exp_to_plot == 1 else self.idiom_comparer.exp_2_name}.png")

    def plot_clusters_histogram(self):
        # Figure 5: Histogram of clusters, by experiment
        # plot a histogram of cluster indices across the two experiments
        first_exp_clusters = self.idiom_comparer.idiom_chirp_attributes.loc[
            self.idiom_comparer.idiom_chirp_attributes["original_df"] == 1]['cluster'].values
        second_exp_clusters = self.idiom_comparer.idiom_chirp_attributes.loc[
            self.idiom_comparer.idiom_chirp_attributes["original_df"] == 2]['cluster'].values
        plt.cla()
        plt.hist([first_exp_clusters, second_exp_clusters], 
                bins=(np.arange(min(self.idiom_comparer.chirp_labels), max(self.idiom_comparer.chirp_labels) + 2) - 0.5), 
                density=True,
                #  stacked=True,
                )
        plt.xticks(range(min(self.idiom_comparer.chirp_labels), max(self.idiom_comparer.chirp_labels) + 1))
        plt.title(f"Proportion of Cluster Membership in Idioms")
        plt.xlabel("Cluster Index")
        plt.ylabel("Proportion")
        plt.legend([self.idiom_comparer.exp_1_name, self.idiom_comparer.exp_2_name])
        plt.savefig(f"{self.results_path}/figs/histogram_clusters_{self.idiom_comparer.exp_1_name}_{self.idiom_comparer.exp_2_name}.png")
        plt.show()

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