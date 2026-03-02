
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from analysis_utils import *


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

 
# First define the two experiments to be compared:


results_1 = "./analysis_results/2022_barn_2secs_myca_quantile_1_16"
results_2 = "./analysis_results/2022_lake_2secs_myca_quantile_1_28"
exp_1_name = "Barn"
exp_2_name = "Lake"


chirp_attributes_1 = pd.read_csv(f"{results_1}/test_set_chirp_attributes.csv")
chirp_attributes_1 = pd.read_csv(f"{results_1}/all_chirp_measures_scaled_quantile.csv")
chirp_attributes_1 = pd.read_csv(f"{results_1}/all_chirp_measures_scaled_robust.csv")
idiom_boundaries_1 = pd.read_csv(f"{results_1}/idiom_boundaries.csv", header=None)
confidence_measures_1 = pd.read_csv(f"{results_1}/chirp_prediction_confidence_measures.csv")

chirp_attributes_2 = pd.read_csv(f"{results_2}/test_set_chirp_attributes.csv")
chirp_attributes_2 = pd.read_csv(f"{results_2}/all_chirp_measures_scaled_quantile.csv")
chirp_attributes_2 = pd.read_csv(f"{results_2}/all_chirp_measures_scaled_robust.csv")
idiom_boundaries_2 = pd.read_csv(f"{results_2}/idiom_boundaries.csv", header=None)
confidence_measures_2 = pd.read_csv(f"{results_2}/chirp_prediction_confidence_measures.csv")


combined_chirp_attributes = pd.concat([chirp_attributes_1, chirp_attributes_2], ignore_index=False).reset_index()
combined_chirp_attributes.rename({"index": "OriginalIndex"}, axis=1, inplace=True)
idiom_boundaries_attributes = pd.concat([idiom_boundaries_1, idiom_boundaries_2], ignore_index=False).reset_index()
idiom_boundaries_attributes.rename({"index": "OriginalIndex"}, axis=1, inplace=True)
confidence_measures_attributes = pd.concat([confidence_measures_1, confidence_measures_2], ignore_index=False).reset_index()
confidence_measures_attributes.rename({"index": "OriginalIndex"}, axis=1, inplace=True)
combined_chirp_attributes


for dataframe, df_1 in [(combined_chirp_attributes, chirp_attributes_1), 
                        (idiom_boundaries_attributes, idiom_boundaries_1), 
                        (confidence_measures_attributes, confidence_measures_1)]:
    dataframe["original_df"] = dataframe.apply(lambda x: 1 if x.name < len(df_1) else 2, axis=1)
combined_chirp_attributes

 
# Visualize how the two experiments are distributed in a 2-D projection:


import umap
combined_chirp_attributes_embedded = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(combined_chirp_attributes.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy())


plt.figure(figsize=(10, 8))
scatter = plt.scatter(combined_chirp_attributes_embedded[:, 0], 
                      combined_chirp_attributes_embedded[:, 1], 
                      c=combined_chirp_attributes["original_df"], 
                      cmap="tab10",
                      vmin=1,
                      vmax=10,
                      s=5,
                      )
plt.title(f"Distribution of chirp embeddings")# in UMAP space")
handles, labels = scatter.legend_elements()
plt.legend(handles, [exp_1_name, exp_2_name])
plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2");
# if not umap:
#     plt.xlim(-8800, -8000);
#     plt.ylim(-1400, -1000);


idiom_boundaries_attributes

 
# Visualize the distribution of only idiom chirps (chirps within a "whole" idiom):


# get the idiom chirp attributes
idiom_chirp_idxs = []
for idx, row in idiom_boundaries_attributes.iterrows():
    for i in range(int(row[0]), int(row[1]) + 1):
        idiom_chirp_idxs.append((idx, i, row["original_df"]))

idiom_chirp_attributes = pd.DataFrame()
for idiom_idx, chirp_idx, original_df in idiom_chirp_idxs:
    attributes = combined_chirp_attributes.loc[(combined_chirp_attributes["original_df"] == original_df) & 
                                                (combined_chirp_attributes["OriginalIndex"] == chirp_idx)]
    if idiom_chirp_attributes.empty:
        idiom_chirp_attributes = attributes
    else:
        idiom_chirp_attributes = pd.concat([idiom_chirp_attributes, attributes])
idiom_chirp_attributes.reset_index(inplace=True)
idiom_chirp_attributes


NO_AMP = 0 # set to 0 to keep all amp features, 1 to remove [Amp1stQrtl, Amp2ndQrtl, Amp3rdQrtl, Amp4thQrtl], 2 to remove all amp features

idiom_chirp_data = idiom_chirp_attributes.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy()

# remove columns that contain "Amp" in their name if NO_AMP is set
if NO_AMP == 1:
    amp_indices = [i - 3 for i, col in enumerate(idiom_chirp_attributes.columns) if col in ["Amp1stQrtl", "Amp2ndQrtl", "Amp3rdQrtl", "Amp4thQrtl"]]
    idiom_chirp_data = np.delete(idiom_chirp_data, amp_indices, axis=1)
elif NO_AMP == 2:
    amp_indices = [i - 3 for i, col in enumerate(idiom_chirp_attributes.columns) if "Amp" in col]
    idiom_chirp_data = np.delete(idiom_chirp_data, amp_indices, axis=1)


idiom_chirp_data


idiom_chirp_attributes_embedded = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(idiom_chirp_data)


# FIGURE: 10.2

plt.figure(figsize=(8, 6))
scatter = plt.scatter(idiom_chirp_attributes_embedded[:, 0], 
                      idiom_chirp_attributes_embedded[:, 1], 
                      c=idiom_chirp_attributes["original_df"], 
                      cmap="tab10",
                      vmin=1,
                      vmax=10,
                      s=5,
                      )
plt.title(f"Distribution of idiom chirp embeddings")# in UMAP space")
handles, labels = scatter.legend_elements()
plt.legend(handles, [exp_1_name, exp_2_name])
plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2");
# if not umap:
#     plt.xlim(-8800, -8000);
#     plt.ylim(-1400, -1000);

 
# Run clustering on the set of idiom chirps combined from both idioms:


MIN_K = 1
MAX_K = 50

n_clusters = find_ideal_cluster_k(idiom_chirp_attributes_embedded, MIN_K, MAX_K, plot=False)
n_clusters


clustering = "Agglomerative"
use_umap = True
NUM_CLUSTERS = n_clusters["elbow_method"]

cluster_data_input = idiom_chirp_attributes_embedded if use_umap else idiom_chirp_attributes.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy()
chirp_labels = cluster_chirps(cluster_data_input, clustering, use_umap, NUM_CLUSTERS)

chirp_labels


idiom_chirp_attributes


import matplotlib 
from matplotlib.colors import ListedColormap

cmap1_colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, 256))
cmap2_colors = cmap1_colors * 0.35
cmap2_colors[:, 3] = 1
cmap2_colors
cmap2 = ListedColormap(cmap2_colors, name="Set1dark")


# FIGURE: 11.1

# plot the idiom chirps with their clusters, with the two experiments designated with different shapes
second_exp_first_row = idiom_chirp_attributes.loc[idiom_chirp_attributes["original_df"] == 2].iloc[0].name

plt.figure(figsize=(12, 8))
scatter1 = plt.scatter(idiom_chirp_attributes_embedded[:second_exp_first_row, 0], 
                      idiom_chirp_attributes_embedded[:second_exp_first_row, 1], 
                      c=chirp_labels[:second_exp_first_row], 
                      cmap="tab10", 
                      s=5, 
                      vmin=min(chirp_labels), 
                      vmax=NUM_CLUSTERS - 1
                      )
scatter2 = plt.scatter(idiom_chirp_attributes_embedded[second_exp_first_row:, 0], 
                      idiom_chirp_attributes_embedded[second_exp_first_row:, 1], 
                      c=chirp_labels[second_exp_first_row:], 
                      cmap=cmap2, 
                      s=12, 
                      marker="s",
                      vmin=min(chirp_labels), 
                      vmax=NUM_CLUSTERS - 1
                      )
plt.title(f"Distribution of idiom chirp embeddings")# in UMAP space")
# handles, labels = scatter1.legend_elements()
# plt.legend(handles, ["barn", "lake"])
plt.colorbar(scatter2, label=f"Cluster Label ({exp_2_name})")
plt.colorbar(scatter1, label=f"Cluster Label ({exp_1_name})")
legend_symbol_barn = matplotlib.lines.Line2D([0], [0], marker='.', color='w', label='Significant Peak', markerfacecolor='black', markersize=15)
legend_symbol_lake = matplotlib.lines.Line2D([0], [0], marker='s', color='w', label='Significant Peak', markerfacecolor='black', markersize=9)
plt.legend([legend_symbol_barn, legend_symbol_lake], [exp_1_name, exp_2_name])

plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2");
# if not umap:
#     plt.xlim(-8800, -8000);
#     plt.ylim(-1400, -1000);


idiom_chirp_attributes


from scipy.spatial import ConvexHull
plt.figure(figsize=(10, 8))
idiom_chirp_attributes['cluster_idx'] = chirp_labels
exp_to_plot = 2

first_exp_attributes_embedded = idiom_chirp_attributes_embedded[:second_exp_first_row]
second_exp_attributes_embedded = idiom_chirp_attributes_embedded[second_exp_first_row:]
for i in range(NUM_CLUSTERS):
    cluster_points = idiom_chirp_attributes_embedded[idiom_chirp_attributes['cluster_idx'] == i]
    if len(cluster_points) > 2:
        hull = ConvexHull(cluster_points)
        # Plot the hull as a closed line
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', lw=2, alpha=1) # 'k-' for a solid black line
    
    first_exp_cluster_points = first_exp_attributes_embedded[idiom_chirp_attributes['cluster_idx'][:second_exp_first_row] == i]
    second_exp_cluster_points = second_exp_attributes_embedded[idiom_chirp_attributes['cluster_idx'][second_exp_first_row:] == i]
    if exp_to_plot == 1:
        scatter = plt.scatter(first_exp_cluster_points[:, 0], first_exp_cluster_points[:, 1], s=5, label=f'Cluster {i}', alpha=1)
    else:
        scatter = plt.scatter(second_exp_cluster_points[:, 0], second_exp_cluster_points[:, 1], s=5, label=f'Cluster {i}', alpha=1)

plt.title(f"Distribution of idiom chirp embeddings ({exp_1_name if exp_to_plot == 1 else exp_2_name})")
plt.xlabel(f"{'UMAP' if umap else 'PCA'} Component 1")
plt.ylabel(f"{'UMAP' if umap else 'PCA'} Component 2");

 
# Evaluate various metrics about the clustering of the two sets of idiom chirps:


idiom_label_sequences = idiom_chirp_attributes.groupby(["file_id", "original_df"]).agg(list)["cluster_idx"].reset_index()
idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 1]


# FIGURE: 12.1

# plot a histogram of cluster indices across the two experiments
first_exp_clusters = idiom_chirp_attributes.loc[idiom_chirp_attributes["original_df"] == 1]['cluster_idx'].values
second_exp_clusters = idiom_chirp_attributes.loc[idiom_chirp_attributes["original_df"] == 2]['cluster_idx'].values
plt.hist([first_exp_clusters, second_exp_clusters], 
         bins=(np.arange(min(chirp_labels), max(chirp_labels) + 2) - 0.5), 
         density=True,
        #  stacked=True,
         )
plt.xticks(range(min(chirp_labels), max(chirp_labels) + 1))
plt.title(f"Proportion of Cluster Membership in Idioms")
plt.xlabel("Cluster Index")
plt.ylabel("Proportion")
plt.legend([exp_1_name, exp_2_name]);


# FIGURE: 12.2

# find the most common cluster across both experiments
LENGTH = 1
K_MOST_COMMON = 10
SUBSEQ_TYPE = "all"
subseq_all = most_common_subsequences(idiom_label_sequences['cluster_idx'].values, 
                         LENGTH, 
                         subseq_type=SUBSEQ_TYPE, 
                         k=K_MOST_COMMON)
subseq_exp1 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 1]['cluster_idx'].values, 
                         LENGTH, 
                         subseq_type=SUBSEQ_TYPE,
                         k=K_MOST_COMMON)
subseq_exp2 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 2]['cluster_idx'].values, 
                         LENGTH, 
                         subseq_type=SUBSEQ_TYPE,
                         k=K_MOST_COMMON)
subseq_all = pd.DataFrame(subseq_all, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} (all)', 'Count'])
subseq_exp1 = pd.DataFrame(subseq_exp1, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({exp_1_name})', 'Count'])
subseq_exp2 = pd.DataFrame(subseq_exp2, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({exp_2_name})', 'Count'])
res = pd.concat([subseq_all, subseq_exp1, subseq_exp2], axis=1)
res.insert(0, "Rank", range(1, len(res) + 1))
print(res)


idiom_clusters_temp = []
idiom_clusters_temp_1 = []
idiom_clusters_temp_2 = []
for idx, row in idiom_label_sequences.iterrows():
    idiom_clusters_temp.extend(row['cluster_idx'])
    if row["original_df"] == 1:
        idiom_clusters_temp_1.extend(row['cluster_idx'])
    else:
        idiom_clusters_temp_2.extend(row['cluster_idx'])
idiom_cluster_counts_all = Counter(idiom_clusters_temp)
idiom_cluster_counts_1 = Counter(idiom_clusters_temp_1)
idiom_cluster_counts_2 = Counter(idiom_clusters_temp_2)


# FIGURE: 12.3

# find most common transitions
LENGTH = 2
subseq_all = most_common_subsequences(idiom_label_sequences['cluster_idx'].values, 
                         LENGTH, 
                         subseq_type=SUBSEQ_TYPE, 
                         k=K_MOST_COMMON)
subseq_exp1 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 1]['cluster_idx'].values, 
                         LENGTH, 
                         subseq_type=SUBSEQ_TYPE,
                         k=K_MOST_COMMON)
subseq_exp2 = most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 2]['cluster_idx'].values, 
                         LENGTH, 
                         subseq_type=SUBSEQ_TYPE,
                         k=K_MOST_COMMON)

# total_subseq_count_all = sum(most_common_subsequences(idiom_label_sequences['cluster_idx'].values, LENGTH, subseq_type=SUBSEQ_TYPE).values())
# total_subseq_count_exp1 = sum(most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 1]['cluster_idx'].values, LENGTH, subseq_type=SUBSEQ_TYPE).values())
# total_subseq_count_exp2 = sum(most_common_subsequences(idiom_label_sequences.loc[idiom_label_sequences["original_df"] == 2]['cluster_idx'].values, LENGTH, subseq_type=SUBSEQ_TYPE).values())

subseq_all = pd.DataFrame(subseq_all, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} (all)', 'Count'])
subseq_all_proportion = subseq_all.copy()
subseq_all_proportion['Start Cluster Count'] = subseq_all_proportion.apply(lambda x: idiom_cluster_counts_all[x['Subseq (all)'][0]], axis=1)
subseq_all_proportion['Transition Probability'] = subseq_all_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_all[x['Subseq (all)'][0]], axis=1)
subseq_all_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
subseq_all_proportion = subseq_all_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
subseq_all_proportion["Transition Probability"] = subseq_all_proportion["Transition Probability"].round(3)

subseq_exp1 = pd.DataFrame(subseq_exp1, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({exp_1_name})', 'Count'])
subseq_exp1_proportion = subseq_exp1.copy()
subseq_exp1_proportion['Start Cluster Count'] = subseq_exp1_proportion.apply(lambda x: idiom_cluster_counts_1[x['Subseq (Barn)'][0]], axis=1)
subseq_exp1_proportion['Transition Probability'] = subseq_exp1_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_1[x['Subseq (Barn)'][0]], axis=1)
subseq_exp1_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
subseq_exp1_proportion = subseq_exp1_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
subseq_exp1_proportion["Transition Probability"] = subseq_exp1_proportion["Transition Probability"].round(3)

subseq_exp2 = pd.DataFrame(subseq_exp2, columns=[f'{"Subseq" if SUBSEQ_TYPE == "all" else SUBSEQ_TYPE} ({exp_2_name})', 'Count'])
subseq_exp2_proportion = subseq_exp2.copy()
# subseq_exp2_proportion['Proportion'] = subseq_exp2_proportion['Count'] / total_subseq_count_exp2
subseq_exp2_proportion['Start Cluster Count'] = subseq_exp2_proportion.apply(lambda x: idiom_cluster_counts_2[x['Subseq (Lake)'][0]], axis=1)
subseq_exp2_proportion['Transition Probability'] = subseq_exp2_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_2[x['Subseq (Lake)'][0]], axis=1)
subseq_exp2_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
subseq_exp2_proportion = subseq_exp2_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
subseq_exp2_proportion["Transition Probability"] = subseq_exp2_proportion["Transition Probability"].round(3)

res = pd.concat([subseq_all, subseq_exp1, subseq_exp2], axis=1)
res.insert(0, "Rank", range(1, len(res) + 1))
res_proportion = pd.concat([subseq_all_proportion, subseq_exp1_proportion, subseq_exp2_proportion], axis=1)
res_proportion.insert(0, "Rank", range(1, len(res_proportion) + 1))
res_proportion.rename(columns = {f'Subseq ({exp_1_name})': f'{exp_1_name} Subseq', f'Subseq ({exp_2_name})': f'{exp_2_name} Subseq'}, inplace=True)
header=[['','All','',f'{exp_1_name}', '', f'{exp_2_name}', ''], ['Rank','Subseq','Proportion', 'Subseq','Proportion', 'Subseq','Proportion']]
res_proportion.columns=header
print(res_proportion)

 
# Stepping away from comparison for a second, this section allows us to describe the characteristics of a given cluster


ignore_columns = ["index", "OriginalIndex", "file_id", "chirp_idx", "original_df", "cluster_idx"]
cluster_profile = describe_cluster(idiom_chirp_attributes, 7, normalize=True, ignore_columns=ignore_columns)
cluster_profile


