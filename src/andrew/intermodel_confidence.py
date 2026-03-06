# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import os
import sys
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.lines
from sklearn.preprocessing import RobustScaler
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
CALCULATE_K = False

CLUSTER_METHOD = "Agglomerative"
USE_UMAP = True

SAMPLE_SIZE_1 = 1000
SEQ_LENGTH_1 = 3
SAMPLE_SIZE_2 = 400
SEQ_LENGTH_2 = 5

SUBSEQ_N = 2
K_MOST_COMMON = 10

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
                "meanKn-FcCurviness", "MinAccpQuality", "AmpEndLn60ExpC", "AmpStartLn60ExpC", "Preemphasis", "MaxSegLnght" ,"Max#CallsConsidered"]
ignore_cols += ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir", "file_id", "chirp_idx", "split"]

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

class IdiomIdentifier():
    def __init__(self, prediction_files=None, truth_files=None, dataset_path=None):
        self.prediction_files = prediction_files
        self.truth_files = truth_files
        self.dataset_path = dataset_path
        self.scaler_path = f"{dataset_path}/split_scaler.pkl"
        self.filename_to_id_path = f"{dataset_path}/split_filename_to_id.csv"

        self.prediction_ensemble_measures = None
        self.ground_truth = None
        self.unscaled_ground_truth = None

    def calculate_prediction_measures(self):
        assert self.prediction_files and self.truth_files and self.scaler_path, \
            "IdiomIdentifier is not properly instantiated"
        
        self.ground_truth, self.prediction_ensemble_measures = calculate_ensemble_measures(self.prediction_files, 
                                                                                           self.truth_files)
        self.unscaled_ground_truth = unscale(self.ground_truth, self.scaler_path, cols_to_keep=["file_id", "chirp_idx"])
        self.prediction_ensemble_measures.sort_values(by=["file_id", "chirp_idx"], inplace=True)
        self.prediction_ensemble_measures.reset_index(drop=True, inplace=True)
            
        # get the range of file_ids
        # n_points = len(prediction_ensemble_measures)
        self.min_file_id = self.prediction_ensemble_measures['file_id'].min()
        self.max_file_id = self.prediction_ensemble_measures['file_id'].max()

    def _calculate_low_confidence_measure(self):
        # find a "low confidence threshold" and get the indices of the low confidence samples
        low_confidence_threshold = np.percentile(self.prediction_ensemble_measures[MEASURE], LOW_CONFIDENCE_PERCENTILE)
        # low_confidence_indices = prediction_ensemble_measures.index[prediction_ensemble_measures[MEASURE] >= low_confidence_threshold].tolist()
        self.prediction_ensemble_measures["low_confidence"] = self.prediction_ensemble_measures[MEASURE] >= low_confidence_threshold

    def detect_peaks(self):
        peak_files = []
        for file_id in tqdm(self.prediction_ensemble_measures['file_id'].unique()):
            peak_chirps = detect_uncertainty_peaks(file_id, self.prediction_ensemble_measures, MEASURE, sigma=SIGMA)
            if peak_chirps is not None and not peak_chirps.empty:
                # print(f"file_id={file_id} has peaks at chirp index: {(peak_chirps['frame_number'] + 4).tolist()}")
                for frame_number in peak_chirps['frame_number']:
                    peak_files.append((file_id, frame_number + PREDICTION_OFFSET))
        self._calculate_peak_metrics(peak_files)
        
        # peak_indices = [(self.prediction_ensemble_measures.index[(self.prediction_ensemble_measures['file_id'] == file_id) & (self.prediction_ensemble_measures['chirp_idx'] == chirp_idx)][0]) for (file_id, chirp_idx) in peak_files]
        peak_indices = [row[0] for row in self.prediction_ensemble_measures[self.prediction_ensemble_measures["peak_detected"] >= 1].iterrows()]
 
        self.peak_files = peak_files
        self.peak_indices = peak_indices

        return peak_files, peak_indices

    def _calculate_peak_metrics(self, peak_files):
        # according to the (id, chirp_idx) pairs in peak_files, create a new column in prediction_ensemble_measures called "peak_detected" that is True for those pairs and False otherwise
        self.prediction_ensemble_measures["peak_detected"] = 0
        prev_file_id = 0
        num_peaks = 1
        for (file_id, chirp_idx) in peak_files:
            if file_id == prev_file_id:
                num_peaks += 1
            else:
                num_peaks = 1
            self.prediction_ensemble_measures.loc[(self.prediction_ensemble_measures['file_id'] == file_id) & 
                                                  (self.prediction_ensemble_measures['chirp_idx'] == chirp_idx), 
                                                  'peak_detected'] = num_peaks
            prev_file_id = file_id

        # make a column in prediction_ensemble_measures called "distance_to_next_peak" that is the number of chirps
        # until the next peak for each (file_id, chirp_idx) pair, and NaN if there is no next peak in the same file_id
        self.prediction_ensemble_measures["distance_to_next_peak"] = None
        for file_id in tqdm(self.prediction_ensemble_measures['file_id'].unique()):
            sel = self.prediction_ensemble_measures[self.prediction_ensemble_measures['file_id'] == file_id]
            peak_chirps = sel[sel['peak_detected'] > 0]['chirp_idx'].tolist()
            for idx, chirp_idx in enumerate(sel['chirp_idx']):
                next_peaks = [peak for peak in peak_chirps if peak > chirp_idx]
                if next_peaks:
                    distance_to_next_peak = min(next_peaks) - chirp_idx
                else:
                    distance_to_next_peak = np.nan
                self.prediction_ensemble_measures.loc[(self.prediction_ensemble_measures['file_id'] == file_id) & 
                                                      (self.prediction_ensemble_measures['chirp_idx'] == chirp_idx), 
                                                      'distance_to_next_peak'] = distance_to_next_peak

    def identify_idiom_candidates(self):
        # from prediction_ensemble_measures, between every two peaks for the same file_id, extract the sequence of indices into a new array
        # each sequence corresponds to one whole idiom
        whole_idiom_sequences = []
        for file_id in self.prediction_ensemble_measures['file_id'].unique():
            sequence = self.prediction_ensemble_measures[self.prediction_ensemble_measures['file_id'] == file_id]
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

        # transform whole_idiom_sequences to be just the start and end chirp_idx values
        whole_idiom_sequences_np = whole_idiom_sequences.copy()
        for i in range(len(whole_idiom_sequences)):
            seq = whole_idiom_sequences[i]
            start_chirp_idx = seq[0]
            end_chirp_idx = seq[-1]
            whole_idiom_sequences_np[i] = (start_chirp_idx, end_chirp_idx)

        # from prediction_ensemble_measures, extract the sequence of indices from the start to the first peak for each file_id that has a peak
        # this is not used in analysis
        onset_idiom_sequences = []
        for file_id in self.prediction_ensemble_measures['file_id'].unique():
            sequence = self.prediction_ensemble_measures[self.prediction_ensemble_measures['file_id'] == file_id]
            peak_indices = sequence[sequence['peak_detected'] == True].index.tolist()
            # print(peak_indices)
            if len(peak_indices) == 0:
                continue
            start_idx = 0
            end_idx = peak_indices[0] - 1 # exclude the peak itself (it belongs to the next idiom)
            seq = sequence.loc[start_idx:end_idx]
            onset_idiom_sequences.append(seq.index.tolist())
        self.whole_idiom_sequences = whole_idiom_sequences
        self.whole_idiom_sequences_np = whole_idiom_sequences_np
        return whole_idiom_sequences, whole_idiom_sequences_np
    
    def identify_peak_filenames(self):
        assert self.filename_to_id_path, "No filename_to_id file given"
        assert self.peak_files, "Peaks have not yet been calculated"
        # for each file_id where peak_detected is True, use split_filename_to_id_path to find each filename
        filename_to_id_df = pd.read_csv(self.filename_to_id_path)
        peak_filenames = []
        for (file_id, chirp_idx) in self.peak_files:
            matching_rows = filename_to_id_df[filename_to_id_df['file_id'] == file_id]
            if not matching_rows.empty:
                filename = matching_rows.iloc[0]['Filename']
                # use unscaled_truth_list_0 to find the TimeInFile for this file_id and chirp_idx
                time_in_file = self.unscaled_ground_truth[(self.unscaled_ground_truth['chirp_idx'] == chirp_idx) & \
                                                          (self.unscaled_ground_truth['file_id'] == file_id)]['TimeInFile'].iloc[0]
                peak_filenames.append((filename, file_id, chirp_idx, time_in_file))

    def retrieve_sequence_contexts(self, save=False):
        assert self.peak_files, "Peaks have not yet been calculated"
        data_module = stf.data.DataModule(
            datasetCls = BatsCSVDataset,
            dataset_kwargs = {
                "root_path": self.dataset_path,
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
                if idx in self.peak_indices:
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

        num_files = self.max_file_id - self.min_file_id + 1
        # there is an issue with this dataset currently in that the final file does not start its test chirps at index 4
        if self.dataset_path == "../../bats_transformer/data/2022_lake_2secs_myca/splits":
            num_files -= 1
        else:
            column_names = self.unscaled_ground_truth.columns.tolist()
            contexts_df = pd.DataFrame(contexts.reshape(contexts.shape[0] * 4, -1), columns=column_names[:-2])
            contexts_df["file_id"] = [i for i in range(self.min_file_id, self.max_file_id + 1) for _ in range(4)]
            contexts_df["chirp_idx"] = [i for i in range(4)] * num_files
            unscaled_contexts_df = unscale(contexts_df, self.scaler_path, cols_to_keep=[])
            unscaled_contexts_df["file_id"] = [i for i in range(self.min_file_id, self.max_file_id + 1) for _ in range(4)]
            unscaled_contexts_df["chirp_idx"] = [i for i in range(4)] * num_files

        scaled_full_chirp_df = pd.concat([contexts_df, self.ground_truth], axis=0).sort_values(['file_id', 'chirp_idx'])
        full_chirp_df = pd.concat([unscaled_contexts_df, self.unscaled_ground_truth], axis=0).sort_values(['file_id', 'chirp_idx'])
        robust_scaler = RobustScaler()
        full_chirp_df_robust_scaled = scale(full_chirp_df, robust_scaler, cols_to_keep=["file_id", "chirp_idx"])
        self.full_chirp_df = full_chirp_df
        self.full_chirp_df_scaled = full_chirp_df_robust_scaled
        return full_chirp_df, full_chirp_df_robust_scaled

        # if save:
        #     low_conf_contexts = np.array(low_conf_contexts)
        #     low_conf_truths = np.array(low_conf_truths)
        #     np.save(f"{output_folder}/peak_contexts.npy", low_conf_contexts)
        #     np.save(f"{output_folder}/peak_truths.npy", low_conf_truths)

    def cluster_data(self):
        # use sklearn hdbscan to cluster all chirps found in unscaled_ground_truth
        chirp_data = self.full_chirp_df_scaled.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy()

        # remove columns that contain "Amp" in their name if NO_AMP is set
        if NO_AMP == 1:
            amp_indices = [i - 1 for i, col in enumerate(self.full_chirp_df.columns) if col in ["Amp1stQrtl", "Amp2ndQrtl", "Amp3rdQrtl", "Amp4thQrtl"]]
            chirp_data = np.delete(chirp_data, amp_indices, axis=1)
        elif NO_AMP == 2:
            amp_indices = [i - 1 for i, col in enumerate(self.full_chirp_df.columns) if "Amp" in col]
            chirp_data = np.delete(chirp_data, amp_indices, axis=1)

        chirp_data_embedded = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        ).fit_transform(chirp_data)

        chirp_data_embedded_df = pd.DataFrame(chirp_data_embedded, columns=['UMAP1', 'UMAP2'])

        # First, we want to see how many clusters there should be.
        print("Determining ideal number of clusters...")
        # FIGURE: 5.1

        if CALCULATE_K:
            n_clusters = find_ideal_cluster_k(chirp_data_embedded, MIN_CLUSTER_K, MAX_CLUSTER_K, plot=False)
        else:
            n_clusters = {"elbow_method": 7}

        self.num_clusters = int(n_clusters["elbow_method"])

        cluster_data_input = chirp_data_embedded if USE_UMAP else chirp_data
        chirp_labels = cluster_chirps(cluster_data_input, CLUSTER_METHOD, USE_UMAP, self.num_clusters)

        # This is perhaps a separate clustering method, but visualized using a dendrogram
        print("Calculating dendrogram...")
        # plot dendrogram of agglomerative clustering
        self.linked = linkage(chirp_data_embedded, 'ward')

        # Cut at specific height to get clusters
        dendrogram_clusters = fcluster(self.linked, t=self.num_clusters, criterion='maxclust') 
        chirp_labels = dendrogram_clusters

        chirp_data_2d = chirp_data_embedded
        if not USE_UMAP: # use PCA instead of UMAP
            pca = PCA(n_components=2)
            chirp_data_2d = pca.fit_transform(chirp_data)
        # filter out noise points (label == -1)
        filtered_chirp_data_2d = chirp_data_2d[chirp_labels != -1]
        filtered_chirp_labels = chirp_labels[chirp_labels != -1]
        self.chirp_data_embedded_df = chirp_data_embedded_df
        self.chirp_labels = chirp_labels
        self.filtered_chirp_data_2d = filtered_chirp_data_2d
        self.filtered_chirp_labels = filtered_chirp_labels
        return chirp_data_embedded_df, chirp_labels, filtered_chirp_data_2d, filtered_chirp_labels
        
        # # for each sequence in whole_idiom_sequences, use chirp_data_embedded_df to get the sequence of 2D UMAP points
        # whole_idiom_umap_sequences = []
        # for seq_indices in self.whole_idiom_sequences:
        #     seq_umap_points = chirp_data_embedded_df.iloc[seq_indices].to_numpy()
        #     whole_idiom_umap_sequences.append(seq_umap_points)
        # whole_idiom_sequence_clusters = []
        # # for each sequence in whole_idiom_umap_sequences, find the centroid of the cluster for each point in the sequence and plot a line connecting them
        # for seq_points in whole_idiom_umap_sequences:
        #     seq_labels = []
        #     for point in seq_points:
        #         # find the nearest point in filtered_chirp_data_2d to this point
        #         distances = np.linalg.norm(filtered_chirp_data_2d - point, axis=1)
        #         nearest_idx = np.argmin(distances)
        #         seq_labels.append(filtered_chirp_labels[nearest_idx])
        #     whole_idiom_sequence_clusters.append(seq_labels)

    def identify_significant_peaks(self):
        # peak_df = prediction_ensemble_measures[(prediction_ensemble_measures['peak_detected'] > 0) &
        #                                       (prediction_ensemble_measures["distance_to_next_peak"] > 0)][["file_id", "chirp_idx", MEASURE]].copy()
        # peak_df = identify_significant_peaks_by_range(peak_df, prediction_ensemble_measures, MEASURE, SIGMA)
        # significant_peak_ids = []
        # for i in range(len(peak_df)):
        #     significant_peak_idxs = identify_significant_peaks(peak_df, i, 'normalized_surrounding_seq', "seq_uncertainty_range",
        #                                                             similarity_threshold=0.05,
        #                                                             #  std_threshold=2.0,
        #                                                             percentile_threshold=95,
        #                                                             verbose=0
        #                                                             )
        #     if len(significant_peak_idxs) > 0:
        #         significant_peak_ids.extend(significant_peak_idxs)
        # # remove duplicates
        # significant_peak_ids = list(set(significant_peak_ids))
        
        peak_df = self.prediction_ensemble_measures[(self.prediction_ensemble_measures['peak_detected'] > 0) &
                                            (self.prediction_ensemble_measures["distance_to_next_peak"] > 0)][["file_id", "chirp_idx", MEASURE]].copy()
        peak_df = identify_significant_peaks_by_prominence(peak_df, self.prediction_ensemble_measures, MEASURE, SIGMA)

        significant_peak_ids = [idx for idx in peak_df[peak_df["prominence_to_range"] > peak_df["prominence_to_range"].median()].index]

        print("Generating significant peak sequences...")
        # for each of the significant_peak_ids, add a column to prediction_ensemble_measures indicating it is a significant peak
        self.prediction_ensemble_measures['significant_peak'] = self.prediction_ensemble_measures.index.isin(significant_peak_ids)

        significant_idiom_sequences = []
        for file_id in self.prediction_ensemble_measures['file_id'].unique():
            sequence = self.prediction_ensemble_measures[self.prediction_ensemble_measures['file_id'] == file_id]
            peak_indices = sequence[sequence['peak_detected'] > 0].index.tolist()
            for i in range(len(peak_indices) - 1):
                start_idx = peak_indices[i]
                if self.prediction_ensemble_measures.loc[start_idx, "significant_peak"] == False:
                    continue
                end_idx = peak_indices[i + 1]
                # unless it is the last peak, exclude the end_idx in the sequence
                if i < len(peak_indices) - 2:
                    end_idx -= 1
                seq = sequence.loc[start_idx:end_idx]
                significant_idiom_sequences.append(seq.index.tolist())

        # for each sequence in significant_idiom_sequences, use chirp_data_embedded_df to get the sequence of 2D UMAP points
        significant_idiom_umap_sequences = []
        for seq_indices in significant_idiom_sequences:
            seq_umap_points = self.chirp_data_embedded_df.iloc[seq_indices].to_numpy()
            significant_idiom_umap_sequences.append(seq_umap_points)

        significant_idiom_sequence_clusters = []
        # for each sequence in significant_idiom_umap_sequences, find the centroid of the cluster for each point in the sequence and plot a line connecting them
        for seq_points in significant_idiom_umap_sequences:
            seq_labels = []
            for point in seq_points:
                # find the nearest point in filtered_chirp_data_2d to this point
                distances = np.linalg.norm(self.filtered_chirp_data_2d - point, axis=1)
                nearest_idx = np.argmin(distances)
                seq_labels.append(self.filtered_chirp_labels[nearest_idx])
            significant_idiom_sequence_clusters.append(seq_labels)

        self.significant_peak_ids = significant_peak_ids
        self.significant_idiom_sequence_clusters = significant_idiom_sequence_clusters
        self.significant_idiom_sequences = significant_idiom_sequences
        self.peak_df = peak_df
        return significant_peak_ids, significant_idiom_sequences, significant_idiom_sequence_clusters

    def identify_most_common_subsequences(self):
        # FIGURES: 8.1 and 8.2
        idiom_clusters_temp = []
        for seq in self.significant_idiom_sequence_clusters:
            idiom_clusters_temp.extend(seq)
        idiom_cluster_counts = Counter(idiom_clusters_temp)

        # find the most common 3-length label sequences in significant_idiom_sequence_clusters
        sequence_counts = most_common_subsequences(self.significant_idiom_sequence_clusters, 
                                                   SUBSEQ_N, k = K_MOST_COMMON, subseq_type="all")
        sequence_counts = most_common_subsequences(self.significant_idiom_sequence_clusters, 
                                                   SUBSEQ_N, subseq_type="all")
        prefix_counts = most_common_subsequences(self.significant_idiom_sequence_clusters, 
                                                 SUBSEQ_N, k = K_MOST_COMMON, subseq_type="prefix")
        suffix_counts = most_common_subsequences(self.significant_idiom_sequence_clusters, 
                                                 SUBSEQ_N, k = K_MOST_COMMON, subseq_type="suffix")
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
        return res

    def evaluate_idiom_similarity(self):
        # Approach 1: 
        # get a distribution of idiom volumes by randomly sampling sequences of chirps
        compare_idiom_similarity_by_volume(self.whole_idiom_sequences, self.unscaled_ground_truth, SAMPLE_SIZE_1, SEQ_LENGTH_1)

        # Approach 2:
        # get a random sample of idiom sequences; within each sequence, calculate pairwise distances (euclidean) between chirps
        compare_idiom_similarity_by_pairwise_distance(self.whole_idiom_sequences, self.unscaled_ground_truth, SAMPLE_SIZE_2, SEQ_LENGTH_2)

    def output_results(self, results_path):
        # for each seq in significant_idiom_sequences, get the triple (file_id, start_chirp, end_chirp) from prediction_ensemble_measures
        significant_idiom_triples = []
        for seq in self.significant_idiom_sequences:
            file_id = self.prediction_ensemble_measures.loc[seq[0], 'file_id']
            end_id = self.prediction_ensemble_measures.loc[seq[0], 'chirp_idx']
            start_id = self.prediction_ensemble_measures.loc[seq[-1], 'chirp_idx']
            significant_idiom_triples.append((file_id, start_id, end_id))

        for df in self.full_chirp_df, self.full_chirp_df_scaled:
            df["cluster"] = self.chirp_labels
            df["idiom_start"] = False
            df["idiom_end"] = False
            df["in_idiom"] = False
            for file_id, start_id, end_id in significant_idiom_triples:
                df.loc[(df["file_id"] == file_id) & (df["chirp_idx"] == start_id), "idiom_start"] = True
                df.loc[(df["file_id"] == file_id) & (df["chirp_idx"] == end_id), "idiom_end"] = True
                df.loc[(df["file_id"] == file_id) & (df["chirp_idx"] >= start_id) & (df["chirp_idx"] <= end_id), "in_idiom"] = True

        # Here we establish the final outputs to be returned from this notebook. 
        # The idea is that these outputs will represent the idioms of the given dataset/model combo, 
        # and they should be comparable between different datasets/models.

        self.prediction_ensemble_measures['cluster'] = None
        for idx, row in self.prediction_ensemble_measures.iterrows():
            file_id = row['file_id']
            chirp_idx = row['chirp_idx']
            # find the corresponding cluster for this chirp
            # print(self.full_chirp_df[(self.full_chirp_df['file_id'] == file_id) & (self.full_chirp_df['chirp_idx'] == chirp_idx)])
            cluster_label = self.full_chirp_df[(self.full_chirp_df['file_id'] == file_id) & (self.full_chirp_df['chirp_idx'] == chirp_idx)]['cluster']
            self.prediction_ensemble_measures.at[idx, 'cluster'] = cluster_label.iloc[0]

        self.unscaled_ground_truth.to_csv(f"{results_path}/test_set_chirp_attributes.csv", index=False)
        np.savetxt(f"{results_path}/idiom_boundaries.csv", self.whole_idiom_sequences_np, delimiter=",")
        self.prediction_ensemble_measures.to_csv(f"{results_path}/chirp_prediction_confidence_measures.csv", index=False)
        self.full_chirp_df.to_csv(f"{results_path}/all_chirp_measures.csv", index=False)
        self.full_chirp_df_scaled.to_csv(f"{results_path}/all_chirp_measures_scaled_robust.csv", index=False)

class IdiomIdentifierVisualizer():
    def __init__(self, idiom_identifier, results_path):
        self.idiom_identifier = idiom_identifier
        self.results_path = results_path

    def plot_uncertainty_histogram(self):
        # Figure 1: histogram of our chosen uncertainty measure: radius_mean
        # print(np.percentile(self.prediction_ensemble_measures[MEASURE], 0), 
        #    np.percentile(self.prediction_ensemble_measures[MEASURE], 25), 
        #    np.percentile(self.prediction_ensemble_measures[MEASURE], 50), 
        #    np.percentile(self.prediction_ensemble_measures[MEASURE], 75), 
        #    np.percentile(self.prediction_ensemble_measures[MEASURE], 100))
        plt.hist(self.idiom_identifier.prediction_ensemble_measures[MEASURE], bins=50)
        plt.title(f'Histogram of {MEASURE}')
        plt.xlabel(f'{MEASURE}')
        plt.ylabel('Count')
        plt.savefig(f"{self.results_path}/figs/histogram_{MEASURE}.png")
        plt.show()

    def plot_smoothed_uncertainty_histogram(self):
        # Figure 2: histogram of smoothed uncertainty ranges
        # Find which files have a large range by comparing each sequence's range with the total population of ranges
        uncertainty_ranges = []
        uncertainty_ranges_smoothed = []
        for file_id in tqdm(self.idiom_identifier.prediction_ensemble_measures['file_id'].unique()):
            sel = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures['file_id'] == file_id]
            uncertainty_range = sel[MEASURE].max() - sel[MEASURE].min()
            uncertainty_ranges.append(uncertainty_range)
            smoothed = gaussian_filter(sel[MEASURE], sigma=SIGMA, order=0, mode='reflect')
            uncertainty_range_smoothed = smoothed.max() - smoothed.min()
            uncertainty_ranges_smoothed.append(uncertainty_range_smoothed)
        # add a column in prediction_ensemble_measures for whether or not a ttest_1samp for each uncertainty_range is significantly larger than the sample
        for file_id, uncertainty_range in zip(self.idiom_identifier.prediction_ensemble_measures['file_id'].unique(), uncertainty_ranges_smoothed):
            sel = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures['file_id'] == file_id]
            ttest_res = ttest_1samp(uncertainty_ranges_smoothed, uncertainty_range, alternative='greater')
            self.idiom_identifier.prediction_ensemble_measures.loc[sel.index, 'large_range'] = not (ttest_res.pvalue < 0.5)

        plt.hist(uncertainty_ranges_smoothed)
        plt.title("Histogram of Smoothed Uncertainty Ranges")
        plt.xlabel("Smoothed Uncertainty Range")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_{MEASURE}_smoothed.png")
        plt.show()

    def plot_uncertainty_plots(self):
        # Figure 3: uncertainty plots
        for file_id in tqdm(self.idiom_identifier.prediction_ensemble_measures['file_id'].unique()):
            if file_id in [fid for (fid, frame_number) in self.idiom_identifier.peak_files]:
                plot_smoothed_uncertainty(file_id, self.idiom_identifier.prediction_ensemble_measures, MEASURE, sigma=SIGMA, show=False)
        plt.savefig(f"{self.results_path}/figs/uncertainty_plots.png")
        plt.show()

    def plot_peak_indices_histogram(self):
        # Figure 4: histogram of peak indices
        # plot a histogram of chirp_idx for peak chirps (aka starting index of idioms)
        peak_chirp_indices = [chirp_idx for (file_id, chirp_idx) in self.idiom_identifier.peak_files] # for all peak chirps
        peak_chirp_indices = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures["peak_detected"] >= 1]["chirp_idx"] # for just peak chirps at the beginning of a whole idiom
        plt.hist(peak_chirp_indices, bins=(max(peak_chirp_indices) - min(peak_chirp_indices) + 1))
        plt.title("Histogram of Peak Chirp Indices")
        plt.xlabel("Chirp Index")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_peak_indices.png")
        plt.show()

    def plot_idiom_length_histogram(self):
        # Figure 5: histogram of idiom lengths
        # plot a histogram of histogram lengths (distance from a peak to the previous peak, only if a previous peak exists)
        # this only tracks whole idioms, as the distance from the beginning to the first peak is not measured
        distance_values = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures["peak_detected"] > 0]['distance_to_next_peak'].dropna()
        plt.hist(distance_values, bins=(np.arange(min(distance_values), max(distance_values) + 2) - 0.5))
        # set the xticks to be integers from min to max distance
        plt.xticks(np.arange(min(distance_values), max(distance_values) + 1))
        plt.title("Histogram of Idiom Lengths")
        plt.xlabel("Idiom Lengths (# of chirps)")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_idiom_lengths.png")
        plt.show()

    def plot_cluster_histogram(self):
        # Figure 6: histogram of clusters
        # plot histogram of chirp_labels
        # set xticks to be integers from min to max chirp label
        plt.xticks(np.arange(self.idiom_identifier.chirp_labels.min(), self.idiom_identifier.chirp_labels.max() + 1))
        plt.hist(self.idiom_identifier.chirp_labels, bins=np.arange(self.idiom_identifier.chirp_labels.min(), self.idiom_identifier.chirp_labels.max() + 2) - 0.5)
        plt.title(f"Histogram of {CLUSTER_METHOD} Cluster Labels")
        plt.xlabel("Cluster Label")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_clusters.png")
        plt.show()

    def plot_chirp_scatter(self):
        # Figure 7: scatterplot of chirps, unclustered
        # for a baseline reference, just plot all the chirps as a scatterplot without clustering
        plt.figure(figsize=(9, 8))
        scatter = plt.scatter(self.idiom_identifier.filtered_chirp_data_2d[:, 0], 
                              self.idiom_identifier.filtered_chirp_data_2d[:, 1], 
                              s=5)
        plt.title(f"All Chirps Distribution")
        plt.xlabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 1")
        plt.ylabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_all_chirps.png")

    def plot_clustered_chirp_scatter(self):
        # Figure 8: scatterplot of chirps, clustered
        # plot the points in each cluster using PCA for dimensionality reduction
        colormap = "tab10" if self.idiom_identifier.num_clusters <= 10 else "tab20"
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.idiom_identifier.filtered_chirp_data_2d[:, 0], 
                              self.idiom_identifier.filtered_chirp_data_2d[:, 1], 
                              c=self.idiom_identifier.filtered_chirp_labels, cmap=colormap, s=5, 
                              vmin=min(self.idiom_identifier.chirp_labels), vmax=max(self.idiom_identifier.chirp_labels))
        plt.title(f"{CLUSTER_METHOD} Clustering: All Chirps")# ({'UMAP' if USE_UMAP else 'PCA'}-reduced to 2D)")
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 1")
        plt.ylabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_all_chirps_clustered.png")

    def plot_chirp_dendrogram(self):
        # Figure 9: dendrogram of chirps
        plt.figure(figsize=(10, 7))
        chirp_dendrogram = dendrogram(self.idiom_identifier.linked,
                truncate_mode='lastp',
                p=self.idiom_identifier.num_clusters,
                orientation='left',
                #    distance_sort='descending',
                labels=self.idiom_identifier.chirp_labels,
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
        plt.savefig(f"{self.results_path}/figs/dendrogram_clustering.png")

    def plot_prominence_histogram(self):
        # Figure 10: histogram of prominence to range ratio
        plt.hist(self.idiom_identifier.peak_df["prominence_to_range"], bins=20)
        plt.title("Histogram of Peak Prominence to Range Ratio")
        plt.xlabel("Prominence to Range Ratio")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_peak_prominence_to_range.png")
        plt.show()

    def plot_chirp_peak_scatter(self):
        # Figure 11: scatterplot of chirps, clustered, with significant peaks highlighted
        # plot only the significant peaks on the previous scatter plot
        colormap = "tab10" if self.idiom_identifier.num_clusters <= 10 else "tab20"
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.idiom_identifier.filtered_chirp_data_2d[:, 0], 
                              self.idiom_identifier.filtered_chirp_data_2d[:, 1], 
                              c=self.idiom_identifier.filtered_chirp_labels, cmap=colormap, s=5,
                              vmin=min(self.idiom_identifier.chirp_labels), vmax=max(self.idiom_identifier.chirp_labels))
        # overlay stars for peak_indices
        significant_peak_chirp_clusters = self.idiom_identifier.chirp_labels[self.idiom_identifier.significant_peak_ids]
        scatter_2 = plt.scatter(self.idiom_identifier.filtered_chirp_data_2d[self.idiom_identifier.significant_peak_ids, 0], 
                                self.idiom_identifier.filtered_chirp_data_2d[self.idiom_identifier.significant_peak_ids, 1], 
                                c=significant_peak_chirp_clusters, 
                                cmap=colormap, edgecolors="black", s=50, marker="*", 
                                vmin=min(self.idiom_identifier.chirp_labels), vmax=max(self.idiom_identifier.chirp_labels))
        plt.title(f"{CLUSTER_METHOD} Clustering: Chirps and Peaks")# ({'UMAP' if USE_UMAP else 'PCA'}-reduced to 2D)")
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 1")
        plt.ylabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 2")
        legend_symbols = matplotlib.lines.Line2D([0], [0], marker='*', color='w', label='Significant Peak', markerfacecolor='black', markersize=15)
        plt.legend([legend_symbols], ["Peak Chirp"])
        plt.savefig(f"{self.results_path}/figs/scatter_chirps_and_peaks.png")

    def plot_cluster_transition_scatter(self):
        # Figure 12: scatterplot of chirps, clustered, with transition graph
        # calculate the centroid of each cluster using the same colors but with a border
        unique_labels = np.unique(self.idiom_identifier.filtered_chirp_labels)
        centroids = []
        for label in unique_labels:
            cluster_points = self.idiom_identifier.filtered_chirp_data_2d[self.idiom_identifier.filtered_chirp_labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

        G = nx.DiGraph()
        for seq in self.idiom_identifier.significant_idiom_sequence_clusters:
            clusters = seq
            for i in range(len(clusters)-1):
                if G.has_edge(clusters[i], clusters[i+1]):
                    G[clusters[i]][clusters[i+1]]['weight'] += 1
                else:
                    G.add_edge(clusters[i], clusters[i+1], weight=1)

        # plot the graph of all chirps in filtered_chirp_data_2d with paths according to significant_idiom_sequence_clusters with
        # edge weights according to G
        plt.figure(figsize=(7, 6))
        colormap = "tab10" if self.idiom_identifier.num_clusters <= 10 else "tab20"
        scatter = plt.scatter(self.idiom_identifier.filtered_chirp_data_2d[:, 0],
                              self.idiom_identifier.filtered_chirp_data_2d[:, 1], 
                              c=self.idiom_identifier.filtered_chirp_labels, cmap=colormap, s=5, 
                              vmin=min(self.idiom_identifier.chirp_labels), vmax=max(self.idiom_identifier.chirp_labels))
        pos = {node: centroids[i] for i, node in enumerate(unique_labels)}
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue", hide_ticks=False)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 0.1 for w in weights], alpha=0.7, hide_ticks=False)
        nx.draw_networkx_labels(G, pos, font_size=10, hide_ticks=False)
        plt.title("Cluster Transitions in Idiom Sequences")
        plt.xlabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 1")
        plt.ylabel(f"{'UMAP' if USE_UMAP else 'PCA'} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_chirps_and_transitions.png")

    def plot_chirp_v_peak_cluster_histogram(self):
        # Figure 13: histogram of clusters for all chirps vs peaks
        # plt.hist(peak_chirp_clusters, bins=np.arange(chirp_labels.min(), chirp_labels.max() + 2) - 0.5)
        plt.hist([self.idiom_identifier.chirp_labels, self.idiom_identifier.chirp_labels[self.idiom_identifier.significant_peak_ids]], 
                 density=True, bins=np.arange(self.idiom_identifier.chirp_labels.min(), self.idiom_identifier.chirp_labels.max() + 2) - 0.5)
        plt.legend(["All Chirps", "Peak Chirps"])
        plt.title(f"Histogram of Clusters for All Chirps vs Peaks")
        plt.xticks(np.arange(self.idiom_identifier.chirp_labels.min(), self.idiom_identifier.chirp_labels.max() + 1))
        plt.xlabel("Cluster Label")
        plt.ylabel("Proportion")
        plt.savefig(f"{self.results_path}/figs/histogram_clusters_all_vs_peaks.png")
        plt.show()

    def generate_figures(self):
        self.plot_uncertainty_histogram()
        self.plot_smoothed_uncertainty_histogram()
        self.plot_uncertainty_plots()
        self.plot_peak_indices_histogram()
        self.plot_idiom_length_histogram()
        self.plot_cluster_histogram()
        self.plot_chirp_scatter()
        self.plot_clustered_chirp_scatter()
        self.plot_chirp_dendrogram()
        self.plot_prominence_histogram()
        self.plot_chirp_peak_scatter()
        self.plot_cluster_transition_scatter()
        self.plot_chirp_v_peak_cluster_histogram()

def idiom_identifier_pipeline(idiom_identifier, results_path):
    print("Calculating ensemble measures...")
    idiom_identifier.calculate_prediction_measures()

    # This section performs peak detection on the uncertainty sequences generated in the previous section.
    print("Performing peak detection...")
    peak_files, _ = idiom_identifier.detect_peaks()

    # This section does some analysis about the results of peak detection:
    print("Identifying idiom candidates...")
    idiom_identifier.identify_idiom_candidates()

    # This next section looks at the peak points to generate files corresponding to the peak contexts and peak chirps.
    # It will also organize the data into "idioms" based on the peak locations.
    # An idiom is defined as starting from the beginning of the file or from a peak chirp to the chirp before the next peak.
    print("Retrieving contexts for chirp sequences...")
    idiom_identifier.retrieve_sequence_contexts()

    # In this section, I investigate potential patterns in the identified "low-confidence" predictions through clustering.
    print("Clustering chirps based on physical measures...")
    idiom_identifier.cluster_data()

    # In this section, I investigate which peak chirps (of low confidence) are *statistically* significant in terms of being a significant peak. 
    # The motivation behind this is that scipy's find_peaks doesn't consider any sort of statistical significance, just if there is a ^ shape in the plot.
    # Generally, this can be done by checking which chirps are >2sigmas over the mean uncertainty.
    # The challenge is how to get enough statistical power, specifically from sample size, as idiom sequences are only some 5-20 chirps long.
    # My approach checks the density (uncertainty) ranges of each idiom and groups similar ranges together in order to increase the sample size. 
    # But then there seems to be an issue of comparability, as it seems difficult to avoid weirdness comparing sequences with different ranges.
    print("Identifying significant peaks...")
    idiom_identifier.identify_significant_peaks()    
    
    print("Outputting results...")
    idiom_identifier.output_results(results_path)

def main(args):
    output_folder = args.output_folder
    dataset_path = args.dataset_path
    results_folder = args.results_folder

    experiment_name = output_folder.split("/")[-1]
    results_path = f"{results_folder}/{experiment_name}"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if not os.path.exists(f"{results_path}/figs"):
        os.mkdir(f"{results_path}/figs")
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
    print(f"Loaded {len(prediction_files)} files in {output_folder}")

    # Run the idiom identification pipeline on the data
    idiom_identifier = IdiomIdentifier(prediction_files=prediction_files, 
                                       truth_files=truth_files, 
                                       dataset_path=dataset_path)
    idiom_identifier_pipeline(idiom_identifier, results_path)
    
    # Useful statistics on the data:
    prediction_ensemble_measures = idiom_identifier.prediction_ensemble_measures
    file_id_counts = Counter([file_id for (file_id, chirp_idx) in idiom_identifier.peak_files])
    files_with_multiple_peaks = [file_id for file_id, count in file_id_counts.items() if count >= 2]
    # prediction_ensemble_measures_multiple_peaks = prediction_ensemble_measures[prediction_ensemble_measures['file_id'].isin(files_with_multiple_peaks)]
    print("There are", prediction_ensemble_measures.shape[0], "total chirps")
    print("There are", len(files_with_multiple_peaks), "out of", len(file_id_counts), "files with 2+ peaks.")
    # print the number of rows in prediction_ensemble_measures where peak_detected is True
    print(prediction_ensemble_measures[prediction_ensemble_measures['peak_detected'] == True].shape[0], "chirps are peak detected")

    # Figure: 
    # Identify the most common transitions
    most_common_transitions = idiom_identifier.identify_most_common_subsequences()
    print(most_common_transitions)

    # ALL FIGURES:
    print("Generating figures...")    
    # idiom_identifier.generate_figures(results_path)
    visualizer = IdiomIdentifierVisualizer(idiom_identifier, results_path)
    visualizer.generate_figures()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze idioms in chirp data/predictions")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder containing the data to analyze",
                        default="../../bats_transformer/outputs/2022_barn_2secs_myca_quantile_1_16")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset csv file containing the chirp attributes",
                        default="../../bats_transformer/data/2022_barn_2secs_myca/splits")
    parser.add_argument("--results_folder", type=str, help="Path to the folder where results will be saved",
                        default="./analysis_results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)