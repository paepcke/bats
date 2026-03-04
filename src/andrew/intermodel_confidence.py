# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import os
import sys
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import scipy.spatial
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.signal import peak_prominences
from scipy.stats import ttest_1samp, ttest_ind
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.lines
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import umap
import networkx as nx

sys.path.append("..")
sys.path.append('../../bats_transformer/spacetimeformer')
import spacetimeformer as stf
sys.path.append('../../bats_transformer/data')
from bats_dataset import *

from analysis_utils import *  

# recall that the measures are:
# tightness, radius_mean, density, average_error_per_point, error_density, euclidean_distance
MEASURE = "radius_mean"
LOW_CONFIDENCE_PERCENTILE = 90
PREDICTION_OFFSET = 4
SIGMA = 1

NO_AMP = 0 # set to 0 to keep all amp features, 1 to remove [Amp1stQrtl, Amp2ndQrtl, Amp3rdQrtl, Amp4thQrtl], 2 to remove all amp features
MIN_CLUSTER_K = 1
MAX_CLUSTER_K = 50
CALCULATE_K = True

SAMPLE_SIZE_1 = 1000
SEQ_LENGTH_1 = 3
SAMPLE_SIZE_2 = 400
SEQ_LENGTH_2 = 5

SUBSEQ_N = 2
K_MOST_COMMON = 10

def main(args):
    # Daytime recordings (max 2 seconds) from the barn in 2022, filtered to only include Myca (Myotis californicus) chirps
    output_folder = args.output_folder
    dataset_path = args.dataset_path

    scaler_path = f"{dataset_path}/split_scaler.pkl"
    filename_to_id_path = f"{dataset_path}/split_filename_to_id.csv"

    # get all files (not directories) in output_folder
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"output_folder does not exist: {output_folder}")

    prediction_files = [
        os.path.join(output_folder, fn)
        for fn in sorted(os.listdir(output_folder))
        if os.path.isfile(os.path.join(output_folder, fn)) and fn.endswith("predictions.log")
    ]

    truth_files = [
        os.path.join(output_folder, fn)
        for fn in sorted(os.listdir(output_folder))
        if os.path.isfile(os.path.join(output_folder, fn)) and fn.endswith("ground_truths.log")
    ]

    print(f"Found {len(prediction_files)} files in {output_folder}")

    ground_truth, prediction_ensemble_measures = calculate_ensemble_measures(prediction_files, truth_files)
    prediction_ensemble_measures.sort_values(by=["file_id", "chirp_idx"], inplace=True)
    prediction_ensemble_measures.reset_index(drop=True, inplace=True)

    # get the range of file_ids
    n_points = len(prediction_ensemble_measures)
    min_file_id = prediction_ensemble_measures['file_id'].min()
    max_file_id = prediction_ensemble_measures['file_id'].max()

    # find a "low confidence threshold" and get the indices of the low confidence samples
    
    low_confidence_threshold = np.percentile(prediction_ensemble_measures['radius_mean'], LOW_CONFIDENCE_PERCENTILE)
    low_confidence_indices = prediction_ensemble_measures.index[prediction_ensemble_measures['radius_mean'] >= low_confidence_threshold].tolist()
    print(low_confidence_threshold)
    prediction_ensemble_measures["low_confidence"] = prediction_ensemble_measures['radius_mean'] >= low_confidence_threshold

    unscaled_ground_truth = unscale(ground_truth, scaler_path, cols_to_keep=["file_id", "chirp_idx"])
    
    # This section performs peak detection on the uncertainty sequences generated in the previous section.

    # Find which files have a large range by comparing each sequence's range with the total population of ranges
    uncertainty_ranges = []
    uncertainty_ranges_smoothed = []
    for file_id in tqdm(prediction_ensemble_measures['file_id'].unique()):
        sel = prediction_ensemble_measures[prediction_ensemble_measures['file_id'] == file_id]
        uncertainty_range = sel[MEASURE].max() - sel[MEASURE].min()
        uncertainty_ranges.append(uncertainty_range)
        smoothed = gaussian_filter(sel[MEASURE], sigma=SIGMA, order=0, mode='reflect')
        uncertainty_range_smoothed = smoothed.max() - smoothed.min()
        uncertainty_ranges_smoothed.append(uncertainty_range_smoothed)
    # add a column in prediction_ensemble_measures for whether or not a ttest_1samp for each uncertainty_range is significantly larger than the sample
    for file_id, uncertainty_range in zip(prediction_ensemble_measures['file_id'].unique(), uncertainty_ranges_smoothed):
        sel = prediction_ensemble_measures[prediction_ensemble_measures['file_id'] == file_id]
        ttest_res = ttest_1samp(uncertainty_ranges_smoothed, uncertainty_range, alternative='greater')
        prediction_ensemble_measures.loc[sel.index, 'large_range'] = not (ttest_res.pvalue < 0.5)

    mean_smoothed_uncertainty_range = np.mean(uncertainty_ranges_smoothed)
    print("mean_smoothed_uncertainty_range:", f"{mean_smoothed_uncertainty_range:e}")

    peak_files = []
    for file_id in tqdm(prediction_ensemble_measures['file_id'].unique()):
        peak_chirps = detect_uncertainty_peaks(file_id, prediction_ensemble_measures, MEASURE, sigma=SIGMA)
        if peak_chirps is not None and not peak_chirps.empty:
            # print(f"file_id={file_id} has peaks at chirp index: {(peak_chirps['frame_number'] + 4).tolist()}")
            for frame_number in peak_chirps['frame_number']:
                peak_files.append((file_id, frame_number + PREDICTION_OFFSET))

    # according to the (id, chirp_idx) pairs in peak_files, create a new column in prediction_ensemble_measures called "peak_detected" that is True for those pairs and False otherwise
    prediction_ensemble_measures["peak_detected"] = 0
    prev_file_id = 0
    num_peaks = 1
    for (file_id, chirp_idx) in peak_files:
        if file_id == prev_file_id:
            num_peaks += 1
        else:
            num_peaks = 1
        prediction_ensemble_measures.loc[(prediction_ensemble_measures['file_id'] == file_id) & 
                                        (prediction_ensemble_measures['chirp_idx'] == chirp_idx), 'peak_detected'] = num_peaks
        prev_file_id = file_id

    # make a column in prediction_ensemble_measures called "distance_to_next_peak" that is the number of chirps
    # until the next peak for each (file_id, chirp_idx) pair, and NaN if there is no next peak in the same file_id
    prediction_ensemble_measures["distance_to_next_peak"] = None
    for file_id in tqdm(prediction_ensemble_measures['file_id'].unique()):
        sel = prediction_ensemble_measures[prediction_ensemble_measures['file_id'] == file_id]
        peak_chirps = sel[sel['peak_detected'] > 0]['chirp_idx'].tolist()
        for idx, chirp_idx in enumerate(sel['chirp_idx']):
            next_peaks = [peak for peak in peak_chirps if peak > chirp_idx]
            if next_peaks:
                distance_to_next_peak = min(next_peaks) - chirp_idx
            else:
                distance_to_next_peak = np.nan
            prediction_ensemble_measures.loc[(prediction_ensemble_measures['file_id'] == file_id) & 
                                            (prediction_ensemble_measures['chirp_idx'] == chirp_idx), 'distance_to_next_peak'] = distance_to_next_peak

    whole_idiom_metrics = prediction_ensemble_measures[prediction_ensemble_measures["peak_detected"] >= 1]
    
    # This section does some analysis about the results of peak detection:

    # plot a histogram of chirp_idx for peak chirps (aka starting index of idioms)
    peak_chirp_indices = [chirp_idx for (file_id, chirp_idx) in peak_files] # for all peak chirps
    peak_chirp_indices = whole_idiom_metrics["chirp_idx"] # for just peak chirps at the beginning of a whole idiom
    print("mean peak chirp index:", np.mean(peak_chirp_indices), "median peak chirp index:", np.median(peak_chirp_indices))

    distance_values = prediction_ensemble_measures[prediction_ensemble_measures["peak_detected"] > 0]['distance_to_next_peak'].dropna()

    # extract the rows of prediction_ensemble_measures corresponding to files that have 2 or more peaks
    # this corresponds to the sequences that contain one whole idiom

    # this is not used for other parts of analysis
    file_id_counts = Counter([file_id for (file_id, chirp_idx) in peak_files])
    files_with_multiple_peaks = [file_id for file_id, count in file_id_counts.items() if count >= 2]
    prediction_ensemble_measures_multiple_peaks = prediction_ensemble_measures[prediction_ensemble_measures['file_id'].isin(files_with_multiple_peaks)]
    print("There are", len(files_with_multiple_peaks), "out of", len(file_id_counts), "files with 2+ peaks.")

    # from prediction_ensemble_measures, between every two peaks for the same file_id, extract the sequence of indices into a new array
    # each sequence corresponds to one whole idiom
    whole_idiom_sequences = []
    for file_id in prediction_ensemble_measures['file_id'].unique():
        sequence = prediction_ensemble_measures[prediction_ensemble_measures['file_id'] == file_id]
        peak_indices = sequence[sequence['peak_detected'] > 0].index.tolist()
        # print(peak_indices)
        for i in range(len(peak_indices) - 1):
            start_idx = peak_indices[i]
            end_idx = peak_indices[i + 1]
            # unless it is the last peak, exclude the end_idx in the sequence
            if i < len(peak_indices) - 2:
                end_idx -= 1
            seq = sequence.loc[start_idx:end_idx]
            whole_idiom_sequences.append(seq.index.tolist())

    print(len(whole_idiom_sequences))

    # transform whole_idiom_sequences to be just the start and end chirp_idx values
    whole_idiom_sequences_np = whole_idiom_sequences.copy()
    for i in range(len(whole_idiom_sequences)):
        seq = whole_idiom_sequences[i]
        start_chirp_idx = seq[0]
        end_chirp_idx = seq[-1]
        whole_idiom_sequences_np[i] = (start_chirp_idx, end_chirp_idx)
    # save the whole_idiom_sequences as a npy file
    np.save("whole_idiom_sequences.npy", whole_idiom_sequences_np)

    # from prediction_ensemble_measures, extract the sequence of indices from the start to the first peak for each file_id that has a peak
    # this is not used in analysis
    onset_idiom_sequences = []
    for file_id in prediction_ensemble_measures['file_id'].unique():
        sequence = prediction_ensemble_measures[prediction_ensemble_measures['file_id'] == file_id]
        peak_indices = sequence[sequence['peak_detected'] == True].index.tolist()
        # print(peak_indices)
        if len(peak_indices) == 0:
            continue
        start_idx = 0
        end_idx = peak_indices[0] - 1 # exclude the peak itself (it belongs to the next idiom)
        seq = sequence.loc[start_idx:end_idx]
        onset_idiom_sequences.append(seq.index.tolist())

    # print the number of rows in prediction_ensemble_measures where peak_detected is True and large_range is True
    print(prediction_ensemble_measures[(prediction_ensemble_measures['peak_detected'] == True) & (prediction_ensemble_measures['large_range'] == True)].shape[0], "chirps are peak and large range")
    # print the number of rows in prediction_ensemble_measures where peak_detected is True
    print(prediction_ensemble_measures[prediction_ensemble_measures['peak_detected'] == True].shape[0], "chirps are peak detected")
    # print the number of rows in prediction_ensemble_measures where large_range is True
    print(prediction_ensemble_measures[prediction_ensemble_measures['large_range'] == True].shape[0], "chirps are large range")
    print("There are", prediction_ensemble_measures.shape[0], "total chirps")

    # for each file_id where peak_detected is True, use split_filename_to_id_path to find each filename
    filename_to_id_df = pd.read_csv(filename_to_id_path)
    peak_filenames = []
    for (file_id, chirp_idx) in peak_files:
        matching_rows = filename_to_id_df[filename_to_id_df['file_id'] == file_id]
        if not matching_rows.empty:
            filename = matching_rows.iloc[0]['Filename']
            # use unscaled_truth_list_0 to find the TimeInFile for this file_id and chirp_idx
            time_in_file = unscaled_ground_truth[(unscaled_ground_truth['chirp_idx'] == chirp_idx) & \
                                                (unscaled_ground_truth['file_id'] == file_id)]['TimeInFile'].iloc[0]
            peak_filenames.append((filename, file_id, chirp_idx, time_in_file))

    # peak_indices = [(prediction_ensemble_measures.index[(prediction_ensemble_measures['file_id'] == file_id) & (prediction_ensemble_measures['chirp_idx'] == chirp_idx)][0]) for (file_id, chirp_idx) in peak_files]
    peak_indices = [row[0] for row in whole_idiom_metrics.iterrows()]
    print(len(peak_indices))
    
    # This next section looks at the peak points to generate files corresponding to the peak contexts and peak chirps.
    # It will also organize the data into "idioms" based on the peak locations.
    # An idiom is defined as starting from the beginning of the file or from a peak chirp to the chirp before the next peak.

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
                "meanKn-FcCurviness", "MinAccpQuality", "AmpEndLn60ExpC", "AmpStartLn60ExpC", "Preemphasis", "MaxSegLnght" ,"Max#CallsConsidered" ]
    ignore_cols += ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir", "file_id", "chirp_idx", "split"]

    data_module = stf.data.DataModule(
        datasetCls = BatsCSVDataset,
        dataset_kwargs = {
            "root_path": dataset_path,
            "prefix": "split",
            "ignore_cols": ignore_cols,
            "time_col_name": "TimeIndex",
            "val_split": 0.05,
            "test_split": 0.05,
            "context_points": None,
            "target_points": 1,
            "random_seed": 31
        },
        batch_size=64,
        workers=4,
    )

    test_data = data_module.test_dataloader()

    low_conf_contexts = []
    low_conf_truths = []

    # Grab the contexts for the first chirp for each sequence, and 
    # create a new dataframe extended from unscaled_ground_truth that includes chirp indices 0 through 3
    contexts = []
    batch_num = 0
    for batch in tqdm(test_data):
        x_t, x_c, y_t, y_c = batch
        # if the idx of the data point is in one of [low_confidence_indices, peak_indices], save the context and truth
        batch_indices = np.array(range(x_c.shape[0])) + batch_num * data_module.batch_size
        for i, idx in enumerate(batch_indices):
            # if idx in low_confidence_indices:
            if idx in peak_indices:
                low_conf_contexts.append(x_c[i].cpu().numpy())
                low_conf_truths.append(y_c[i].cpu().numpy())
        batch_num += 1

        for i in range(x_c.shape[0]):
            # remove rows that only include 0s (padding)
            x_c_nonzero = x_c[i].cpu().numpy()
            x_c_nonzero = x_c_nonzero[~np.all(x_c_nonzero == 0, axis=1)]
            if (x_c_nonzero.shape[0] == 4):
                # print(x_c_nonzero.shape)
                contexts.append(x_c_nonzero)

    contexts = np.asarray(contexts)


    if dataset_path == "../../bats_transformer/data/2022_lake_2secs_myca/splits":
        column_names = unscaled_ground_truth.columns.tolist()
        contexts_df = pd.DataFrame(contexts.reshape(contexts.shape[0] * 4, -1), columns=column_names[:-2])
        contexts_df["file_id"] = [i for i in range(min_file_id, max_file_id) for _ in range(4)]
        contexts_df["chirp_idx"] = [i for i in range(4)] * (max_file_id - min_file_id)
        unscaled_contexts_df = unscale(contexts_df, scaler_path, cols_to_keep=[])
        unscaled_contexts_df["file_id"] = [i for i in range(min_file_id, max_file_id) for _ in range(4)]
        unscaled_contexts_df["chirp_idx"] = [i for i in range(4)] * (max_file_id - min_file_id)
    else:
        column_names = unscaled_ground_truth.columns.tolist()
        contexts_df = pd.DataFrame(contexts.reshape(contexts.shape[0] * 4, -1), columns=column_names[:-2])
        contexts_df["file_id"] = [i for i in range(min_file_id, max_file_id + 1) for _ in range(4)]
        contexts_df["chirp_idx"] = [i for i in range(4)] * (max_file_id - min_file_id + 1)
        unscaled_contexts_df = unscale(contexts_df, scaler_path, cols_to_keep=[])
        unscaled_contexts_df["file_id"] = [i for i in range(min_file_id, max_file_id + 1) for _ in range(4)]
        unscaled_contexts_df["chirp_idx"] = [i for i in range(4)] * (max_file_id - min_file_id + 1)

    scaled_full_chirp_df = pd.concat([contexts_df, ground_truth], axis=0).sort_values(['file_id', 'chirp_idx'])
    full_chirp_df = pd.concat([unscaled_contexts_df, unscaled_ground_truth], axis=0).sort_values(['file_id', 'chirp_idx'])
    robust_scaler = RobustScaler()
    full_chirp_df_robust_scaled = scale(full_chirp_df, robust_scaler, cols_to_keep=["file_id", "chirp_idx"])

    low_conf_contexts = np.array(low_conf_contexts)
    low_conf_truths = np.array(low_conf_truths)
    np.save(f"{output_folder}/peak_contexts.npy", low_conf_contexts)
    np.save(f"{output_folder}/peak_truths.npy", low_conf_truths)

    # In this section, I investigate potential patterns in the identified "low-confidence" predictions through clustering.

    # get all the features that contain "Amp" in their name from unscaled_ground_truth
    amp_features = [col for col in unscaled_ground_truth.columns if "Amp" in col]

    # use sklearn hdbscan to cluster all chirps found in unscaled_ground_truth
    chirp_data = full_chirp_df_robust_scaled.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy()

    # remove columns that contain "Amp" in their name if NO_AMP is set
    if NO_AMP == 1:
        amp_indices = [i - 1 for i, col in enumerate(full_chirp_df.columns) if col in ["Amp1stQrtl", "Amp2ndQrtl", "Amp3rdQrtl", "Amp4thQrtl"]]
        chirp_data = np.delete(chirp_data, amp_indices, axis=1)
    elif NO_AMP == 2:
        amp_indices = [i - 1 for i, col in enumerate(full_chirp_df.columns) if "Amp" in col]
        chirp_data = np.delete(chirp_data, amp_indices, axis=1)

    chirp_data_embedded = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    ).fit_transform(chirp_data)

    chirp_data_embedded_df = pd.DataFrame(chirp_data_embedded, columns=['UMAP1', 'UMAP2'])

    # for each sequence in whole_idiom_sequences, use chirp_data_embedded_df to get the sequence of 2D UMAP points
    whole_idiom_umap_sequences = []
    for seq_indices in whole_idiom_sequences:
        seq_umap_points = chirp_data_embedded_df.iloc[seq_indices].to_numpy()
        whole_idiom_umap_sequences.append(seq_umap_points)
    print(len(whole_idiom_umap_sequences))

    # First, we want to see how many clusters there should be.

    # FIGURE: 5.1

    if CALCULATE_K:
        n_clusters = find_ideal_cluster_k(chirp_data_embedded, MIN_CLUSTER_K, MAX_CLUSTER_K, plot=False)
    else:
        n_clusters = {"elbow_method": 7}

    clustering = "Agglomerative"
    use_umap = True
    NUM_CLUSTERS = int(n_clusters["elbow_method"])

    cluster_data_input = chirp_data_embedded if use_umap else chirp_data
    chirp_labels = cluster_chirps(cluster_data_input, clustering, use_umap, NUM_CLUSTERS)

    # This is perhaps a separate clustering method, but visualized using a dendrogram

    # plot dendrogram of agglomerative clustering
    linked = linkage(chirp_data_embedded, 'ward')

    # Cut at specific height to get clusters
    dendrogram_clusters = fcluster(linked, t=NUM_CLUSTERS, criterion='maxclust') 
    chirp_labels = dendrogram_clusters

    pca = PCA(n_components=2)
    chirp_data_2d = pca.fit_transform(chirp_data)
    if use_umap:
        chirp_data_2d = chirp_data_embedded  # use UMAP embedding instead of PCA
    # filter out noise points (label == -1)
    filtered_chirp_data_2d = chirp_data_2d[chirp_labels != -1]
    filtered_chirp_labels = chirp_labels[chirp_labels != -1]

    # calculate the centroid of each cluster using the same colors but with a border
    unique_labels = np.unique(filtered_chirp_labels)
    centroids = []
    for label in unique_labels:
        cluster_points = filtered_chirp_data_2d[filtered_chirp_labels == label]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    whole_idiom_sequence_clusters = []
    # for each sequence in whole_idiom_umap_sequences, find the centroid of the cluster for each point in the sequence and plot a line connecting them
    for seq_points in whole_idiom_umap_sequences:
        seq_labels = []
        for point in seq_points:
            # find the nearest point in filtered_chirp_data_2d to this point
            distances = np.linalg.norm(filtered_chirp_data_2d - point, axis=1)
            nearest_idx = np.argmin(distances)
            seq_labels.append(filtered_chirp_labels[nearest_idx])
        whole_idiom_sequence_clusters.append(seq_labels)

    # get population size of each cluster in chirp_labels
    cluster_counts = Counter(chirp_labels)

    # In this section, I investigate which peak chirps (of low confidence) are *statistically* significant in terms of being a significant peak. The motivation behind this is that scipy's find_peaks doesn't consider any sort of statistical significance, just if there is a ^ shape in the plot.
    # Generally, this can be done by checking which chirps are >2sigmas over the mean uncertainty.
    # The challenge is how to get enough statistical power, specifically from sample size, as idiom sequences are only some 5-20 chirps long.
    # My approach checks the density (uncertainty) ranges of each idiom and groups similar ranges together in order to increase the sample size. But then there seems to be an issue of comparability, as it seems difficult to avoid weirdness comparing sequences with different ranges.

    peak_df = prediction_ensemble_measures[(prediction_ensemble_measures['peak_detected'] > 0) &
                                        (prediction_ensemble_measures["distance_to_next_peak"] > 0)][["file_id", "chirp_idx", MEASURE]].copy()
    peak_df["surrounding_seq_idx"] = peak_df.index.map(lambda idx: get_surrounding_peak_sequence(idx, prediction_ensemble_measures, context_size=4))
    peak_df["surrounding_seq_uncertainty"] = peak_df["surrounding_seq_idx"].map(
        lambda indices: prediction_ensemble_measures.loc[indices, MEASURE].values
    )
    peak_df["surround_seq_uncertainty_smoothed"] = peak_df["surrounding_seq_idx"].map(
        lambda indices: gaussian_filter(prediction_ensemble_measures.loc[indices, MEASURE].values, sigma=SIGMA, order=0, mode='reflect')
    )
    peak_df["seq_uncertainty_min"] = peak_df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.min())
    peak_df["seq_uncertainty_max"] = peak_df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.max())
    peak_df["seq_uncertainty_mean"] = peak_df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.mean())
    peak_df["seq_uncertainty_range"] = peak_df["seq_uncertainty_max"] - peak_df["seq_uncertainty_min"]
    # check if the seq_uncertainty_max occurs at the peak's index, i.e., the index of the row in peak_df
    peak_df["peak_at_max"] = peak_df.apply(
        lambda row: row['surround_seq_uncertainty_smoothed'][row['surrounding_seq_idx'].index(row.name)] == row['seq_uncertainty_max'],
        axis=1
    )
    peak_df["normalized_surrounding_seq"] = peak_df["surround_seq_uncertainty_smoothed"].map(
        lambda arr: (arr - arr.mean())
    )

    peak_df_2 = peak_df.loc[:,["file_id", "chirp_idx"]].copy()
    # create a row called "smoothed_uncertainty" that contains the smoothed uncertainty values for the entire sequence that the peak belongs to
    peak_df_2["smoothed_uncertainty"] = peak_df_2.apply(lambda row: get_smoothed_uncertainty_sequence(prediction_ensemble_measures, row.name, MEASURE, SIGMA), axis=1)
    peak_df_2["seq_len"] = peak_df_2["smoothed_uncertainty"].apply(lambda x: len(x) + 4)

    # add a new column to peak_df_2 called "height", the difference between the value of peak_df_2["smoothed_uncertainty"]
    # at the peak's index and the higher of the two troughs on either side of the peak (the lowest value before the values start
    # increasing on either side of the peak)    
    peak_df_2["peak_value"] = peak_df_2.apply(lambda row: row["smoothed_uncertainty"][row["chirp_idx"] - 4], axis=1)
    peak_df_2["height"] = peak_df_2.apply(lambda row: calculate_height(peak_df_2, row.name), axis=1)
    peak_df_2["prominence"] = peak_df_2.apply(lambda row: peak_prominences(row["smoothed_uncertainty"], [row["chirp_idx"] - 4])[0][0], axis=1)
    peak_df_2["range"] = peak_df_2["smoothed_uncertainty"].apply(lambda x: max(x) - min(x))
    peak_df_2["height_to_range"] = peak_df_2["height"] / peak_df_2["range"]
    peak_df_2["prominence_to_range"] = peak_df_2["prominence"] / peak_df_2["range"]

    significant_peak_ids = []
    for i in range(len(peak_df)):
        significant_peak_idxs = identify_significant_peaks(peak_df, i, 'normalized_surrounding_seq', "seq_uncertainty_range",
                                                                similarity_threshold=0.05,
                                                                #  std_threshold=2.0,
                                                                percentile_threshold=95,
                                                                verbose=0
                                                                )
        if len(significant_peak_idxs) > 0:
            significant_peak_ids.extend(significant_peak_idxs)
    # remove duplicates
    significant_peak_ids = list(set(significant_peak_ids))

    # significant_peak_ids = [idx for idx in peak_df_2[peak_df_2["prominence_to_range"] > peak_df_2["prominence_to_range"].median()].index]
    # len(significant_peak_ids)


    # for each of the significant_peak_ids, add a column to prediction_ensemble_measures indicating it is a significant peak
    prediction_ensemble_measures['significant_peak'] = prediction_ensemble_measures.index.isin(significant_peak_ids)

    significant_idiom_sequences = []
    for file_id in prediction_ensemble_measures['file_id'].unique():
        sequence = prediction_ensemble_measures[prediction_ensemble_measures['file_id'] == file_id]
        peak_indices = sequence[sequence['peak_detected'] > 0].index.tolist()
        for i in range(len(peak_indices) - 1):
            start_idx = peak_indices[i]
            if prediction_ensemble_measures.loc[start_idx, "significant_peak"] == False:
                continue
            end_idx = peak_indices[i + 1]
            # unless it is the last peak, exclude the end_idx in the sequence
            if i < len(peak_indices) - 2:
                end_idx -= 1
            seq = sequence.loc[start_idx:end_idx]
            significant_idiom_sequences.append(seq.index.tolist())
    print(len(significant_idiom_sequences))

    # for each sequence in significant_idiom_sequences, use chirp_data_embedded_df to get the sequence of 2D UMAP points
    significant_idiom_umap_sequences = []
    for seq_indices in significant_idiom_sequences:
        seq_umap_points = chirp_data_embedded_df.iloc[seq_indices].to_numpy()
        significant_idiom_umap_sequences.append(seq_umap_points)

    significant_idiom_sequence_clusters = []
    # for each sequence in significant_idiom_umap_sequences, find the centroid of the cluster for each point in the sequence and plot a line connecting them
    for seq_points in significant_idiom_umap_sequences:
        seq_labels = []
        for point in seq_points:
            # find the nearest point in filtered_chirp_data_2d to this point
            distances = np.linalg.norm(filtered_chirp_data_2d - point, axis=1)
            nearest_idx = np.argmin(distances)
            seq_labels.append(filtered_chirp_labels[nearest_idx])
        significant_idiom_sequence_clusters.append(seq_labels)

    G = nx.DiGraph()
    for seq in significant_idiom_sequence_clusters:
        clusters = seq
        for i in range(len(clusters)-1):
            if G.has_edge(clusters[i], clusters[i+1]):
                G[clusters[i]][clusters[i+1]]['weight'] += 1
            else:
                G.add_edge(clusters[i], clusters[i+1], weight=1)


    idiom_clusters_temp = []
    for seq in significant_idiom_sequence_clusters:
        idiom_clusters_temp.extend(seq)
    idiom_cluster_counts = Counter(idiom_clusters_temp)

    # Run the significance check on all chirp sequences

    # first create a dataframe like peak_df but for all chirps in prediction_ensemble_measures
    # each unique file_id is a row
    all_chirps_df = prediction_ensemble_measures.reset_index().groupby('file_id').agg({'index': lambda x: x.tolist()})
    # rename index to surrounding_seq_idx
    all_chirps_df = all_chirps_df.rename(columns={'index': 'surrounding_seq_idx'})
    all_chirps_df["surrounding_seq_uncertainty"] = all_chirps_df["surrounding_seq_idx"].map(
        lambda indices: prediction_ensemble_measures.loc[indices, MEASURE].values
    )
    all_chirps_df["surround_seq_uncertainty_smoothed"] = all_chirps_df["surrounding_seq_idx"].map(
        lambda indices: gaussian_filter(prediction_ensemble_measures.loc[indices, MEASURE].values, sigma=SIGMA, order=0, mode='reflect')
    )
    all_chirps_df["seq_uncertainty_min"] = all_chirps_df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.min())
    all_chirps_df["seq_uncertainty_max"] = all_chirps_df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.max())
    all_chirps_df["seq_uncertainty_mean"] = all_chirps_df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.mean())
    all_chirps_df["seq_uncertainty_range"] = all_chirps_df["seq_uncertainty_max"] - all_chirps_df["seq_uncertainty_min"]
    all_chirps_df["normalized_surrounding_seq"] = all_chirps_df["surround_seq_uncertainty_smoothed"].map(
        lambda arr: (arr - arr.mean())
    )
    all_chirps_df.sort_values("seq_uncertainty_range")

    all_chirps_significant_peaks = []
    for i in range(len(all_chirps_df)):
        significant_peak_idxs = identify_significant_peaks(all_chirps_df, i, 'normalized_surrounding_seq', "seq_uncertainty_range",
                                                                similarity_threshold=0.05,
                                                                #  std_threshold=2.0,
                                                                percentile_threshold=95,
                                                                verbose=0
                                                                )
        if len(significant_peak_idxs) > 0:
                all_chirps_significant_peaks.extend(significant_peak_idxs)
    # remove duplicates
    all_chirps_significant_peaks = list(set(all_chirps_significant_peaks))
    
    # In this section, I investigate whether chirps within an idiom sequence are more similar with each other than with other chirps.
    # 
    # There are perhaps a few different ways to do this.
    # 1. Find the volumn/density defined by an idiom sequence. Then pull a non-idiom sequence of the same length and find its volumn/density. Do this many times (using random samples) to produce two distributions, then run a statistical test to see if the idiom distribution mean is smaller.
    # 2. Take a random sample of idioms. Do pairwise idiom-internal chirp comparison (something like cosine similarity) to generate a distribution of idiom-internal chirp similarity. Then randomly sample other chirps and do pairwise comparison between two chirps not in the same idiom to generate another distribution. Run a statistical test to see if the idiom-internal distribution mean is smaller.

    whole_idiom_df = pd.DataFrame({"chirp_indices": whole_idiom_sequences})
    whole_idiom_df["chirp_attributes"] = whole_idiom_df["chirp_indices"].map(
        lambda indices: get_attributes_from_indices(indices, unscaled_ground_truth)
    )
    whole_idiom_df["length"] = whole_idiom_df["chirp_indices"].map(lambda indices: len(indices))

    # Approach 1: 
    # get a distribution of idiom volumes by randomly sampling sequences of chirps
    idiom_volumes = []
    for _ in range(SAMPLE_SIZE_1):
        seq = get_random_idiom_sequence(SEQ_LENGTH_1, whole_idiom_df)
        attributes = get_attributes_from_indices(seq, unscaled_ground_truth)
        volume = calculate_sequence_volume(attributes)
        idiom_volumes.append(volume)
    idiom_volumes = np.array(idiom_volumes)

    # get a distribution of non-idiom volumes by randomly sampling sequences of chirps that do not overlap with idioms
    non_idiom_volumes = []
    for _ in range(SAMPLE_SIZE_1):
        seq = get_random_sequence(SEQ_LENGTH_1, whole_idiom_sequences, unscaled_ground_truth)
        attributes = get_attributes_from_indices(seq, unscaled_ground_truth)
        volume = calculate_sequence_volume(attributes)
        non_idiom_volumes.append(volume)
    non_idiom_volumes = np.array(non_idiom_volumes)

    if SAMPLE_SIZE_1 < 20:
        print("Idiom volumes:", idiom_volumes)
        print("Non-idiom volumes:", non_idiom_volumes)

    # scale down the volumes by 1e100 to avoid overflow in t-test
    idiom_volumes /= 1e100
    non_idiom_volumes /= 1e100

    # take the log of the volumes (NOT STATISTICALLY EQUIVALENT TO SCALING)
    # idiom_volumes = np.log(idiom_volumes + 1e-12)
    # non_idiom_volumes = np.log(non_idiom_volumes + 1e-12)

    if SAMPLE_SIZE_1 < 20:
        print("Scaled idiom volumes:", idiom_volumes)
        print("Scaled non-idiom volumes:", non_idiom_volumes)

    # run a statistical t-test to compare the two distributions
    t_stat, p_value = ttest_ind(idiom_volumes, non_idiom_volumes, equal_var=False, alternative='less')
    print(f"Idiom volumes (scaled): mean = {np.mean(idiom_volumes):.4e}, std = {np.std(idiom_volumes):.4e}")
    print(f"Non-idiom volumes (scaled): mean = {np.mean(non_idiom_volumes):.4e}, std = {np.std(non_idiom_volumes):.4e}")
    print(f"T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # Approach 2:
    # get a random sample of idiom sequences; within each sequence, calculate pairwise distances (euclidean) between chirps
    idiom_pairwise_distances = []
    non_idiom_pairwise_distances = []
    for _ in tqdm(range(SAMPLE_SIZE_2)):
        seq = get_random_idiom_sequence(SEQ_LENGTH_2, whole_idiom_df)
        attributes = get_attributes_from_indices(seq, unscaled_ground_truth)
        # pick a random chirp in attributes and calculate distances to all other chirps
        ref_idx = np.random.randint(0, len(attributes))
        ref_chirp = attributes[ref_idx]
        for i, chirp in enumerate(attributes):
            if i == ref_idx:
                continue
            dist = scipy.spatial.distance.euclidean(ref_chirp, chirp)
            idiom_pairwise_distances.append(dist)

        # pick SEQ_LENGTH - 1 random chirps not in the idiom sequence and calculate distances to the same ref_chirp
        non_idiom_indices = []
        for i in range(SEQ_LENGTH_2 - 1):
            seq = get_random_sequence(1, whole_idiom_sequences, unscaled_ground_truth)
            non_idiom_indices.append(seq[0])

        non_idiom_attributes = get_attributes_from_indices(non_idiom_indices, unscaled_ground_truth)
        for chirp in non_idiom_attributes:
            dist = scipy.spatial.distance.euclidean(ref_chirp, chirp)
            non_idiom_pairwise_distances.append(dist)

    idiom_pairwise_distances = np.array(idiom_pairwise_distances)
    non_idiom_pairwise_distances = np.array(non_idiom_pairwise_distances)

    if SAMPLE_SIZE_2 <= 10:
        print("Idiom pairwise distances:", idiom_pairwise_distances)
        print("Non-idiom pairwise distances:", non_idiom_pairwise_distances)

    # run a statistical t-test to compare the two distributions
    t_stat, p_value = ttest_ind(idiom_pairwise_distances, non_idiom_pairwise_distances, equal_var=False, alternative='less')
    print(f"Idiom pairwise distances: mean = {np.mean(idiom_pairwise_distances):.4e}, std = {np.std(idiom_pairwise_distances):.4e}")
    print(f"Non-idiom pairwise distances: mean = {np.mean(non_idiom_pairwise_distances):.4e}, std = {np.std(non_idiom_pairwise_distances):.4e}")
    print(f"T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # for each seq in significant_idiom_sequences, get the triple (file_id, start_chirp, end_chirp) from prediction_ensemble_measures
    significant_idiom_triples = []
    for seq in significant_idiom_sequences:
        file_id = prediction_ensemble_measures.loc[seq[0], 'file_id']
        end_id = prediction_ensemble_measures.loc[seq[0], 'chirp_idx']
        start_id = prediction_ensemble_measures.loc[seq[-1], 'chirp_idx']
        significant_idiom_triples.append((file_id, start_id, end_id))
    print(len(significant_idiom_triples))

    scaled_full_chirp_df = scaled_full_chirp_df.loc[:,:"chirp_idx"]

    for df in full_chirp_df, scaled_full_chirp_df, full_chirp_df_robust_scaled:
        df["cluster"] = chirp_labels
        df["idiom_start"] = False
        df["idiom_end"] = False
        df["in_idiom"] = False
        for file_id, start_id, end_id in significant_idiom_triples:
            df.loc[(df["file_id"] == file_id) & (df["chirp_idx"] == start_id), "idiom_start"] = True
            df.loc[(df["file_id"] == file_id) & (df["chirp_idx"] == end_id), "idiom_end"] = True
            df.loc[(df["file_id"] == file_id) & (df["chirp_idx"] >= start_id) & (df["chirp_idx"] <= end_id), "in_idiom"] = True

    # Here we establish the final outputs to be returned from this notebook. The idea is that these outputs will represent the idioms of the given dataset/model combo, and they should be comparable between different datasets/models.

    results_folder = "./analysis_results"
    experiment_name = output_folder.split("/")[-1]
    results_path = f"{results_folder}/{experiment_name}"
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    prediction_ensemble_measures['cluster'] = None
    for idx, row in prediction_ensemble_measures.iterrows():
        file_id = row['file_id']
        chirp_idx = row['chirp_idx']
        # find the corresponding cluster for this chirp
        # print(full_chirp_df[(full_chirp_df['file_id'] == file_id) & (full_chirp_df['chirp_idx'] == chirp_idx)])
        cluster_label = full_chirp_df[(full_chirp_df['file_id'] == file_id) & (full_chirp_df['chirp_idx'] == chirp_idx)]['cluster']
        prediction_ensemble_measures.at[idx, 'cluster'] = cluster_label.iloc[0]

    unscaled_ground_truth.to_csv(f"{results_path}/test_set_chirp_attributes.csv", index=False)
    np.savetxt(f"{results_path}/idiom_boundaries.csv", whole_idiom_sequences_np, delimiter=",")
    prediction_ensemble_measures.to_csv(f"{results_path}/chirp_prediction_confidence_measures.csv", index=False)
    full_chirp_df.to_csv(f"{results_path}/all_chirp_measures.csv", index=False)
    scaled_full_chirp_df.to_csv(f"{results_path}/all_chirp_measures_scaled_quantile.csv", index=False)
    full_chirp_df_robust_scaled.to_csv(f"{results_path}/all_chirp_measures_scaled_robust.csv", index=False)


    # ALL FIGURES:

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

    # Plot histogram of our chosen uncertainty measure: radius_mean
    print(np.percentile(prediction_ensemble_measures[MEASURE], 0), 
        np.percentile(prediction_ensemble_measures[MEASURE], 25), 
        np.percentile(prediction_ensemble_measures[MEASURE], 50), 
        np.percentile(prediction_ensemble_measures[MEASURE], 75), 
        np.percentile(prediction_ensemble_measures[MEASURE], 100))
    plt.hist(prediction_ensemble_measures[MEASURE], bins=50)
    plt.title(f'Histogram of {MEASURE}')
    plt.xlabel(f'{MEASURE}')
    plt.ylabel('Count')

    plt.hist(uncertainty_ranges_smoothed)
    plt.title("Histogram of Smoothed Uncertainty Ranges")
    plt.xlabel("Smoothed Uncertainty Range")
    plt.ylabel("Count")

    # FIGURE: 3.1 and 3.2

    for file_id in tqdm(prediction_ensemble_measures['file_id'].unique()):
        if file_id in [fid for (fid, frame_number) in peak_files]:
            plot_smoothed_uncertainty(file_id, prediction_ensemble_measures, MEASURE, sigma=SIGMA, show=False)

    plt.hist(peak_chirp_indices, bins=(max(peak_chirp_indices) - min(peak_chirp_indices) + 1))
    plt.title("Histogram of Peak Chirp Indices")
    plt.xlabel("Chirp Index")
    plt.ylabel("Count")

    # FIGURE: 2.1

    # plot a histogram of histogram lengths (distance from a peak to the previous peak, only if a previous peak exists)
    # this only tracks whole idioms, as the distance from the beginning to the first peak is not measured
    plt.hist(distance_values, bins=(np.arange(min(distance_values), max(distance_values) + 2) - 0.5))
    # set the xticks to be integers from min to max distance
    plt.xticks(np.arange(min(distance_values), max(distance_values) + 1))
    plt.title("Histogram of Idiom Lengths")
    plt.xlabel("Idiom Lengths (# of chirps)")
    plt.ylabel("Count")

    # FIGURE: 4.2

    # plot histogram of chirp_labels
    # set xticks to be integers from min to max chirp label
    plt.xticks(np.arange(chirp_labels.min(), chirp_labels.max() + 1))
    plt.hist(chirp_labels, bins=np.arange(chirp_labels.min(), chirp_labels.max() + 2) - 0.5)
    plt.title(f"Histogram of {clustering} Cluster Labels")
    plt.xlabel("Cluster Label")
    plt.ylabel("Count")

    # FIGURE: 2.2
    # for a baseline reference, just plot all the chirps as a scatterplot without clustering

    colormap = "tab10" if NUM_CLUSTERS <= 10 else "tab20"
    plt.figure(figsize=(9, 8))
    scatter = plt.scatter(filtered_chirp_data_2d[:, 0], filtered_chirp_data_2d[:, 1], s=5)
    plt.title(f"All Chirps Distribution")
    plt.xlabel(f"{'UMAP' if use_umap else 'PCA'} Component 1")
    plt.ylabel(f"{'UMAP' if use_umap else 'PCA'} Component 2")

    # FIGURE: 5.3
    # plot the points in each cluster using PCA for dimensionality reduction

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(filtered_chirp_data_2d[:, 0], filtered_chirp_data_2d[:, 1], c=filtered_chirp_labels, cmap=colormap, s=5, 
                        vmin=min(chirp_labels), vmax=max(chirp_labels))
    plt.title(f"{clustering} Clustering: All Chirps")# ({'UMAP' if use_umap else 'PCA'}-reduced to 2D)")
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"{'UMAP' if use_umap else 'PCA'} Component 1")
    plt.ylabel(f"{'UMAP' if use_umap else 'PCA'} Component 2")

    # FIGURE: 9.1

    plt.figure(figsize=(10, 7))
    chirp_dendrogram = dendrogram(linked,
            truncate_mode='lastp',
            p=NUM_CLUSTERS,
            orientation='left',
            #    distance_sort='descending',
            labels=chirp_labels,
            show_leaf_counts=True
            )

    ax = plt.gca()
    y_labels = ax.get_ymajorticklabels()
    for i, y in enumerate(y_labels):
        y.set_text(f"{i+1} {y.get_text()}")
    ax.set_yticklabels(y_labels)

    plt.title('Agglomerative Dendrogram: All Chirps')
    plt.xlabel('Distance')
    ax.yaxis.set_label_position("right")
    plt.ylabel('Cluster (Chirp Count)')
    plt.show()

    plt.hist(peak_df_2["prominence_to_range"], bins=20)

    # FIGURE: 6.2

    # plot only the significant peaks on the previous scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(filtered_chirp_data_2d[:, 0], filtered_chirp_data_2d[:, 1], c=filtered_chirp_labels, cmap=colormap, s=5,
                        vmin=min(chirp_labels), vmax=max(chirp_labels))
    # overlay stars for peak_indices
    significant_peak_chirp_clusters = chirp_labels[significant_peak_ids]
    scatter_2 = plt.scatter(chirp_data_2d[significant_peak_ids, 0], chirp_data_2d[significant_peak_ids, 1], c=significant_peak_chirp_clusters, 
                            cmap=colormap, edgecolors="black", s=50, marker="*", vmin=min(chirp_labels), vmax=max(chirp_labels))
    plt.title(f"{clustering} Clustering: Chirps and Peaks")# ({'UMAP' if use_umap else 'PCA'}-reduced to 2D)")
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"{'UMAP' if use_umap else 'PCA'} Component 1")
    plt.ylabel(f"{'UMAP' if use_umap else 'PCA'} Component 2")
    legend_symbols = matplotlib.lines.Line2D([0], [0], marker='*', color='w', label='Significant Peak', markerfacecolor='black', markersize=15)
    plt.legend([legend_symbols], ["Peak Chirp"])

    # FIGURES: 8.1 and 8.2

    # find the most common 3-length label sequences in significant_idiom_sequence_clusters
    sequence_counts = most_common_subsequences(significant_idiom_sequence_clusters, SUBSEQ_N, k = K_MOST_COMMON, subseq_type="all")
    sequence_counts = most_common_subsequences(significant_idiom_sequence_clusters, SUBSEQ_N, subseq_type="all")
    prefix_counts = most_common_subsequences(significant_idiom_sequence_clusters, SUBSEQ_N, k = K_MOST_COMMON, subseq_type="prefix")
    suffix_counts = most_common_subsequences(significant_idiom_sequence_clusters, SUBSEQ_N, k = K_MOST_COMMON, subseq_type="suffix")
    subseqs = pd.DataFrame(sequence_counts, columns=['Subseq', 'Count'])
    prefixs = pd.DataFrame(prefix_counts, columns=['Prefix', 'Count'])
    suffixs = pd.DataFrame(suffix_counts, columns=['Suffix', 'Count'])

    res = pd.concat([subseqs, prefixs, suffixs], axis=1)

    res = subseqs.copy()
    res['Start Cluster Count'] = res.apply(lambda x: idiom_cluster_counts[x['Subseq'][0]], axis=1)
    res['Transition Probability'] = res.apply(lambda x: x['Count'] / idiom_cluster_counts[x['Subseq'][0]], axis=1)
    res = res.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
    res.insert(0, "Rank", range(1, len(res) + 1))
    res.drop(columns=['Count', 'Start Cluster Count'], inplace=True)
    res["Transition Probability"] = res["Transition Probability"].round(3)
    print(res)

    # FIGURE: 7.1

    # plot the graph of all chirps in filtered_chirp_data_2d with paths according to significant_idiom_sequence_clusters with
    # edge weights according to G
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(filtered_chirp_data_2d[:, 0], filtered_chirp_data_2d[:, 1], c=filtered_chirp_labels, cmap=colormap, s=5, 
                        vmin=min(chirp_labels), vmax=max(chirp_labels))
    pos = {node: centroids[i] for i, node in enumerate(unique_labels)}
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue", hide_ticks=False)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 0.1 for w in weights], alpha=0.7, hide_ticks=False)
    nx.draw_networkx_labels(G, pos, font_size=10, hide_ticks=False)

    # # draw xticks and yticks for the scatterplot
    # plt.xticks(np.arange(int(filtered_chirp_data_2d[:, 0].min()), int(filtered_chirp_data_2d[:, 0].max()) + 1, 500))
    # plt.yticks(np.arange(int(filtered_chirp_data_2d[:, 1].min()), int(filtered_chirp_data_2d[:, 1].max()) + 1, 500))

    plt.title("Cluster Transitions in Idiom Sequences")
    plt.xlabel(f"{'UMAP' if use_umap else 'PCA'} Component 1")
    plt.ylabel(f"{'UMAP' if use_umap else 'PCA'} Component 2")
    plt.show()

    # plt.hist(peak_chirp_clusters, bins=np.arange(chirp_labels.min(), chirp_labels.max() + 2) - 0.5)
    plt.hist([chirp_labels, chirp_labels[significant_peak_ids]], density=True, bins=np.arange(chirp_labels.min(), chirp_labels.max() + 2) - 0.5)
    plt.legend(["All Chirps", "Peak Chirps"])
    plt.title(f"Histogram of Clusters for All Chirps vs Peaks")
    plt.xticks(np.arange(chirp_labels.min(), chirp_labels.max() + 1))
    plt.xlabel("Cluster Label")
    plt.ylabel("Proportion")

    # FIGURE: 9.2

    # do the above plotting but with the dendrogram clustering
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(filtered_chirp_data_2d[:, 0], filtered_chirp_data_2d[:, 1], c=dendrogram_clusters, cmap=colormap, s=5,
                        vmin=min(dendrogram_clusters), vmax=max(dendrogram_clusters))
    # overlay stars for peak_indices
    significant_peak_chirp_clusters = dendrogram_clusters[significant_peak_ids]
    scatter_2 = plt.scatter(chirp_data_2d[significant_peak_ids, 0], chirp_data_2d[significant_peak_ids, 1], c=significant_peak_chirp_clusters, 
                            cmap=colormap, edgecolors="black", s=50, marker="*", vmin=min(dendrogram_clusters), vmax=max(dendrogram_clusters))
    plt.title(f"{clustering} Clustering: Chirps and Peaks")# ({'UMAP' if use_umap else 'PCA'}-reduced to 2D)")
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"{'UMAP' if use_umap else 'PCA'} Component 1")
    plt.ylabel(f"{'UMAP' if use_umap else 'PCA'} Component 2")
    legend_symbols = matplotlib.lines.Line2D([0], [0], marker='*', color='w', label='Significant Peak', markerfacecolor='black', markersize=15)
    plt.legend([legend_symbols], ["Peak Chirp"])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze idioms in chirp data/predictions")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder containing the data to analyze",
                        default="../../bats_transformer/outputs/2022_barn_2secs_myca_quantile_1_16")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset csv file containing the chirp attributes",
                        default="../../bats_transformer/data/2022_barn_2secs_myca/splits")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)