# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import sys
from collections import Counter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.lines
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import networkx as nx

sys.path.append("..")
sys.path.append('../../bats_transformer/spacetimeformer')
import spacetimeformer as stf
sys.path.append('../../bats_transformer/data')
from bats_dataset import *

from analysis_utils import *  
from chirp_clusterer import ChirpClusterer

# recall that the measures are:
# tightness, radius_mean, density, average_error_per_point, error_density, euclidean_distance

IGNORE_COLS = ["FreqLedge", "AmpK@end", "Fc", "FBak15dB  ", "FBak32dB", 
               "EndF", "FBak20dB", "LowFreq", "Bndw20dB", "CallsPerSec", 
               "EndSlope", "SteepestSlope", "StartSlope", "Bndw15dB", "HiFtoUpprKnSlp", 
               "HiFtoKnSlope", "DominantSlope", "Bndw5dB", "PreFc500", "PreFc1000", 
               "PreFc3000", "KneeToFcSlope", "TotalSlope", "PreFc250", "CallDuration", 
               "CummNmlzdSlp", "DurOf32dB", "SlopeAtFc", "LdgToFcSlp", "DurOf20dB", 
               "DurOf15dB", "TimeFromMaxToFc", "KnToFcDur", "HiFtoFcExpAmp", "AmpKurtosis", 
               "LowestSlope", "KnToFcDmp", "HiFtoKnExpAmp", "DurOf5dB", "KnToFcExpAmp", 
               "RelPwr3rdTo1st", "LnExpB_StartAmp", "Filter", "HiFtoKnDmp", "LnExpB_EndAmp", 
                "HiFtoFcDmp", "AmpSkew", "LedgeDuration", "KneeToFcResidue", "PreFc3000Residue", 
                "AmpGausR2", "PreFc1000Residue", "Amp1stMean", "LdgToFcExp", "FcMinusEndF", 
                "Amp4thMean", "HiFtoUpprKnExp", "HiFtoKnExp", "KnToFcExp", "UpprKnToKnExp", 
                "Kn-FcCurviness", "Amp2ndMean", "Quality", "HiFtoFcExp", "LnExpA_EndAmp", 
                "RelPwr2ndTo1st", "LnExpA_StartAmp", "HiFminusStartF", "Amp3rdMean", "PreFc500Residue", 
                "Kn-FcCurvinessTrndSlp", "PreFc250Residue", "AmpVariance", "AmpMoment", "meanKn-FcCurviness", 
                "MinAccpQuality", "AmpEndLn60ExpC", "AmpStartLn60ExpC", "Preemphasis", "MaxSegLnght", 
                "Max#CallsConsidered"] + \
                ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 
                 'Preemphasis', 'MaxSegLnght', "ParentDir", "file_id", "chirp_idx", 
                 "split"]

class IdiomIdentifier():
    """
    Identify candidate bat call idioms from prediction uncertainty patterns.

    This class analyzes ensemble prediction uncertainty across chirps to
    detect peaks in model disagreement. These peaks are used as boundaries
    to segment sequences of chirps that may correspond to behavioral
    "idioms" or structured call sequences.

    The workflow typically consists of:
        1. Computing ensemble prediction measures.
        2. Detecting uncertainty peaks.
        3. Segmenting chirp sequences between peaks.

    Attributes
    ----------
    prediction_files : list[str]
        Paths to model prediction files used to compute ensemble measures.

    truth_files : list[str]
        Paths to ground-truth files corresponding to the predictions.

    prediction_ensemble_measures : pandas.DataFrame
        DataFrame containing ensemble uncertainty measures per chirp.

    ground_truth : pandas.DataFrame
        Ground-truth values aligned with predictions.

    unscaled_ground_truth : pandas.DataFrame
        Ground-truth data after reversing feature scaling.

    peak_files : list[tuple]
        List of detected peaks represented as `(file_id, chirp_idx)` pairs.

    peak_indices : list[int]
        Indices of detected peaks within `prediction_ensemble_measures`.

    whole_idiom_sequences : list[list[int]]
        Lists of DataFrame indices representing chirp sequences between peaks.

    whole_idiom_sequences_np : list[tuple]
        Start and end indices for each detected idiom sequence.
    """
    def __init__(self, prediction_files=None, truth_files=None, dataset_path=None,
                 measure="radius_mean", low_conf_percentile=90, prediction_offset=4, sigma=1,
                 **kwargs):
        """
        Initialize the IdiomIdentifier.

        Parameters
        ----------
        prediction_files : list[str], optional
            Paths to prediction files produced by model inference.

        truth_files : list[str], optional
            Paths to ground-truth data corresponding to the predictions.

        dataset_path : str, optional
            Path to the dataset directory containing scaling information
            and filename-to-ID mappings.
        """
        self.prediction_files = prediction_files
        self.truth_files = truth_files
        self.dataset_path = dataset_path
        self.scaler_path = f"{dataset_path}/split_scaler.pkl"
        self.filename_to_id_path = f"{dataset_path}/split_filename_to_id.csv"

        self.measure = measure
        self.low_conf_percentile = low_conf_percentile
        self.prediction_offset = prediction_offset
        self.sigma = sigma

        self.prediction_ensemble_measures = None
        self.ground_truth = None
        self.unscaled_ground_truth = None

        self.clusterer = ChirpClusterer(**kwargs)

    def calculate_prediction_measures(self):
        """
        Compute ensemble prediction uncertainty measures.

        This method aggregates predictions from multiple models to compute
        uncertainty metrics (e.g., prediction variance or radius measures)
        for each chirp. It also restores unscaled ground-truth values and
        prepares the data for downstream peak detection.

        Stores
        ------
        ground_truth : pandas.DataFrame
            Ground-truth values aligned with predictions.

        prediction_ensemble_measures : pandas.DataFrame
            Ensemble uncertainty measures for each chirp.

        unscaled_ground_truth : pandas.DataFrame
            Ground-truth data with scaling reversed.
        """
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
        """
        Identify low-confidence predictions.

        A threshold is computed from the selected uncertainty measure
        using a percentile cutoff. Chirps with values exceeding this
        threshold are marked as low-confidence predictions.
        """
        # find a "low confidence threshold" and get the indices of the low confidence samples
        low_confidence_threshold = np.percentile(self.prediction_ensemble_measures[self.measure], self.low_conf_percentile)
        # low_confidence_indices = prediction_ensemble_measures.index[prediction_ensemble_measures[self.measure] >= low_confidence_threshold].tolist()
        self.prediction_ensemble_measures["low_confidence"] = self.prediction_ensemble_measures[self.measure] >= low_confidence_threshold

    def detect_peaks(self):
        """
        Detect peaks in prediction uncertainty.

        This method scans each file's uncertainty signal and identifies
        local peaks that represent spikes in model disagreement. These
        peaks are treated as candidate boundaries between behavioral
        sequences.

        Returns
        -------
        peak_files : list[tuple]
            Detected peaks as `(file_id, chirp_idx)` pairs.

        peak_indices : list[int]
            Corresponding indices in the prediction DataFrame.
        """
        peak_files = []
        for file_id in tqdm(self.prediction_ensemble_measures['file_id'].unique()):
            peak_chirps = detect_uncertainty_peaks(file_id, self.prediction_ensemble_measures, self.measure, sigma=self.sigma)
            if peak_chirps is not None and not peak_chirps.empty:
                # print(f"file_id={file_id} has peaks at chirp index: {(peak_chirps['frame_number'] + 4).tolist()}")
                for frame_number in peak_chirps['frame_number']:
                    peak_files.append((file_id, frame_number + self.prediction_offset))
        self._calculate_peak_metrics(peak_files)
        
        # peak_indices = [(self.prediction_ensemble_measures.index[(self.prediction_ensemble_measures['file_id'] == file_id) & (self.prediction_ensemble_measures['chirp_idx'] == chirp_idx)][0]) for (file_id, chirp_idx) in peak_files]
        peak_indices = [row[0] for row in self.prediction_ensemble_measures[self.prediction_ensemble_measures["peak_detected"] >= 1].iterrows()]
 
        self.peak_files = peak_files
        self.peak_indices = peak_indices

        return peak_files, peak_indices

    def _calculate_peak_metrics(self, peak_files):
        """
        Compute peak-related metrics for detected uncertainty spikes.

        This method annotates the prediction DataFrame with peak indicators
        and computes additional metrics such as the distance from each
        chirp to the next detected peak within the same file.

        Parameters
        ----------
        peak_files : list[tuple]
            Detected peaks represented as `(file_id, chirp_idx)` pairs.
        """
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
        """
        Segment chirp sequences into candidate idioms.

        Using the detected peaks as boundaries, this method extracts
        contiguous chirp sequences between peaks. Each sequence is
        considered a potential behavioral idiom.

        Returns
        -------
        whole_idiom_sequences : list[list[int]]
            Lists of DataFrame indices representing each candidate idiom.

        whole_idiom_sequences_np : list[tuple]
            Start and end indices for each idiom sequence.
        """
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
        """
        Retrieve filenames and timestamps for detected peaks.

        Using the dataset filename-to-ID mapping, this method associates
        each detected peak with its original audio filename and timestamp
        within the file.
        """
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
        """
        Retrieve contextual sequences surrounding detected peaks.

        This method loads the original dataset and extracts temporal
        windows around detected peaks or idiom candidates for further
        analysis or visualization.

        Parameters
        ----------
        save : bool, optional
            If True, save the retrieved sequence contexts to disk.
        """
        assert self.peak_files, "Peaks have not yet been calculated"
        data_module = stf.data.DataModule(
            datasetCls = BatsCSVDataset,
            dataset_kwargs = {
                "root_path": self.dataset_path,
                "prefix": "split",
                "ignore_cols": IGNORE_COLS,
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
        """
        Cluster chirps using acoustic feature vectors.

        This method extracts scaled chirp features from the full chirp dataset
        and applies the `ChirpClusterer` pipeline to perform dimensionality
        reduction and clustering. The resulting cluster labels and embeddings
        are stored for downstream analysis and visualization.

        Returns
        -------
        chirp_data_embedded_df : pandas.DataFrame
            DataFrame containing the 2-D embedding of chirps (UMAP or PCA).

        chirp_labels : numpy.ndarray
            Cluster label assigned to each chirp.

        filtered_chirp_data_2d : numpy.ndarray
            Two-dimensional coordinates of chirps excluding noise points.

        filtered_chirp_labels : numpy.ndarray
            Cluster labels corresponding to the filtered chirps.

        Stores
        ------
        linked : numpy.ndarray
            Hierarchical clustering linkage matrix.

        chirp_data_embedded_df : pandas.DataFrame
            Embedded chirp coordinates.

        chirp_labels : numpy.ndarray
            Cluster assignments for all chirps.

        filtered_chirp_data_2d : numpy.ndarray
            Embedded coordinates excluding noise points.

        filtered_chirp_labels : numpy.ndarray
            Cluster labels corresponding to filtered coordinates.
        """
        chirp_data = self.full_chirp_df_scaled.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy()
        self.clusterer.chirp_attributes = self.full_chirp_df_scaled
        self.clusterer.chirp_data = chirp_data
        self.clusterer.prepare_data()
        linked, chirp_data_embedded_df, chirp_labels, filtered_chirp_data_2d, filtered_chirp_labels = self.clusterer.cluster_data()
        self.linked = linked
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
        """
        Identify statistically significant uncertainty peaks.

        This method evaluates peaks detected in the uncertainty signal and
        determines which peaks are statistically significant based on their
        prominence relative to surrounding uncertainty values.

        Significant peaks represent chirps where model disagreement is
        unusually high and may indicate meaningful behavioral transitions.

        Returns
        -------
        significant_peak_ids : list[int]
            Indices of chirps identified as significant peaks.

        significant_idiom_sequences : list[list[int]]
            Idiom sequences containing significant peaks.

        significant_idiom_sequence_clusters : list[list[int]]
            Cluster label sequences corresponding to the significant idioms.

        Stores
        ------
        significant_peak_ids : list[int]
            Indices of statistically significant peak chirps.

        significant_idiom_sequences : list[list[int]]
            Chirp index sequences corresponding to significant idioms.

        significant_idiom_sequence_clusters : list[list[int]]
            Cluster label sequences for each significant idiom.

        peak_df : pandas.DataFrame
            DataFrame containing information about detected peaks.
        """
        # peak_df = prediction_ensemble_measures[(prediction_ensemble_measures['peak_detected'] > 0) &
        #                                       (prediction_ensemble_measures["distance_to_next_peak"] > 0)][["file_id", "chirp_idx", self.measure]].copy()
        # peak_df = identify_significant_peaks_by_range(peak_df, prediction_ensemble_measures, self.measure, self.sigma)
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
                                            (self.prediction_ensemble_measures["distance_to_next_peak"] > 0)][["file_id", "chirp_idx", self.measure]].copy()
        peak_df = identify_significant_peaks_by_prominence(peak_df, self.prediction_ensemble_measures, self.measure, self.sigma)

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

    def output_results(self, results_path):
        """
        Save idiom analysis results to disk.

        This method exports the final datasets generated by the idiom
        identification pipeline, including chirp attributes, cluster
        assignments, idiom boundaries, and prediction uncertainty measures.

        Parameters
        ----------
        results_path : str
            Directory where output files will be written.

        Outputs
        -------
        test_set_chirp_attributes.csv
            Unscaled chirp feature dataset.

        idiom_boundaries.csv
            Start and end indices for detected idiom sequences.

        chirp_prediction_confidence_measures.csv
            Ensemble prediction uncertainty measures per chirp.

        all_chirp_measures.csv
            Full dataset of chirp features.

        all_chirp_measures_scaled_robust.csv
            Robust-scaled chirp feature dataset.
        """
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

    @classmethod
    def add_cli(cls, parser):
        ChirpClusterer.add_cli(parser)
        parser.add_argument("--measure", type=str, default="radius_mean")
        parser.add_argument("--low_conf_percentile", type=int, default=90)
        parser.add_argument("--prediction_offset", type=int, default=4)
        parser.add_argument("--sigma", type=int, default=1)

class IdiomIdentifierVisualizer():
    """
    Visualization utilities for analyzing and interpreting results from an
    IdiomIdentifier pipeline.

    This class generates a variety of plots that summarize the uncertainty,
    clustering behavior, peak detection, and idiom sequence patterns found in
    bat chirp data. The visualizations help diagnose model uncertainty,
    identify statistically significant peaks, and understand transitions
    between clustered chirp types.

    Figures produced include:

    - Uncertainty distribution histograms
    - Smoothed uncertainty range analysis
    - Uncertainty time-series plots
    - Peak chirp index distributions
    - Idiom length distributions
    - Cluster label histograms
    - Dimensionality-reduced chirp scatter plots
    - Clustered chirp visualizations
    - Hierarchical clustering dendrograms
    - Peak prominence statistics
    - Chirp–peak overlay scatterplots
    - Cluster transition graphs between idiom sequences
    - Comparison of cluster distributions for all chirps vs peak chirps

    Parameters
    ----------
    idiom_identifier : IdiomIdentifier
        The IdiomIdentifier instance containing processed chirp data,
        prediction ensemble measures, clustering outputs, and detected peaks.

    results_path : str
        Directory path where generated figures will be saved. Figures are
        typically stored in the `figs/` subdirectory of this path.

    Attributes
    ----------
    idiom_identifier : IdiomIdentifier
        Reference to the analysis object containing all processed data.

    results_path : str
        Path where visualization outputs are written.

    num_clusters : int
        Total number of clusters identified in the chirp clustering step.
    """
    def __init__(self, idiom_identifier, results_path):
        """
        Initialize the IdiomIdentifierVisualizer.

        Parameters
        ----------
        idiom_identifier : IdiomIdentifier
            An initialized and processed IdiomIdentifier object containing
            prediction measures, clustering results, peak detection results,
            and sequence data.

        results_path : str
            Directory where generated figures will be saved.
        """
        self.idiom_identifier = idiom_identifier
        self.results_path = results_path
        self.num_clusters = max(self.idiom_identifier.chirp_labels) - min(self.idiom_identifier.chirp_labels) + 1

    def plot_uncertainty_histogram(self):
        """
        Plot a histogram of the selected uncertainty measure.

        This visualization displays the distribution of the chosen uncertainty
        metric (e.g., `radius_mean`) across all chirp predictions generated by
        the ensemble model.

        The histogram helps characterize the overall uncertainty landscape
        and can be used to determine thresholds for identifying low-confidence
        predictions.

        The resulting figure is saved to:
            figs/histogram_<self.idiom_identifier.measure>.png
        """
        # Figure 1: histogram of our chosen uncertainty measure: radius_mean
        # print(np.percentile(self.prediction_ensemble_measures[self.idiom_identifier.measure], 0), 
        #    np.percentile(self.prediction_ensemble_measures[self.idiom_identifier.measure], 25), 
        #    np.percentile(self.prediction_ensemble_measures[self.idiom_identifier.measure], 50), 
        #    np.percentile(self.prediction_ensemble_measures[self.idiom_identifier.measure], 75), 
        #    np.percentile(self.prediction_ensemble_measures[self.idiom_identifier.measure], 100))
        plt.figure(figsize=(8, 6))
        plt.hist(self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.measure], bins=50)
        plt.title(f'Histogram of {self.idiom_identifier.measure}')
        plt.xlabel(f'{self.idiom_identifier.measure}')
        plt.ylabel('Count')
        plt.savefig(f"{self.results_path}/figs/histogram_{self.idiom_identifier.measure}.png")
        plt.show()

    def plot_smoothed_uncertainty_histogram(self):
        """
        Plot a histogram of smoothed uncertainty ranges across files.

        For each file in the dataset, the uncertainty sequence is smoothed using
        a Gaussian filter. The range (max - min) of the smoothed uncertainty is
        then calculated and compared across files.

        A one-sample t-test is applied to determine whether a file's uncertainty
        range is statistically larger than the population distribution of ranges.
        Results are stored in the `large_range` column of the prediction
        ensemble measures DataFrame.

        This visualization helps identify files that exhibit unusually large
        uncertainty fluctuations.

        The resulting figure is saved to:
            figs/histogram_<self.idiom_identifier.measure>_smoothed.png
        """
        # Figure 2: histogram of smoothed uncertainty ranges
        # Find which files have a large range by comparing each sequence's range with the total population of ranges
        uncertainty_ranges = []
        uncertainty_ranges_smoothed = []
        for file_id in tqdm(self.idiom_identifier.prediction_ensemble_measures['file_id'].unique()):
            sel = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures['file_id'] == file_id]
            uncertainty_range = sel[self.idiom_identifier.measure].max() - sel[self.idiom_identifier.measure].min()
            uncertainty_ranges.append(uncertainty_range)
            smoothed = gaussian_filter(sel[self.idiom_identifier.measure], sigma=self.idiom_identifier.sigma, order=0, mode='reflect')
            uncertainty_range_smoothed = smoothed.max() - smoothed.min()
            uncertainty_ranges_smoothed.append(uncertainty_range_smoothed)
        # add a column in prediction_ensemble_measures for whether or not a ttest_1samp for each uncertainty_range is significantly larger than the sample
        for file_id, uncertainty_range in zip(self.idiom_identifier.prediction_ensemble_measures['file_id'].unique(), uncertainty_ranges_smoothed):
            sel = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures['file_id'] == file_id]
            ttest_res = ttest_1samp(uncertainty_ranges_smoothed, uncertainty_range, alternative='greater')
            self.idiom_identifier.prediction_ensemble_measures.loc[sel.index, 'large_range'] = not (ttest_res.pvalue < 0.5)

        plt.figure(figsize=(8, 6))
        plt.hist(uncertainty_ranges_smoothed)
        plt.title("Histogram of Smoothed Uncertainty Ranges")
        plt.xlabel("Smoothed Uncertainty Range")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_{self.idiom_identifier.measure}_smoothed.png")
        plt.show()

    def plot_uncertainty_plots(self):
        """
        Generate uncertainty time-series plots for files containing peak detections.

        For each file containing a detected peak chirp, this method visualizes
        the smoothed uncertainty trajectory across the chirp sequence.

        These plots help visually inspect whether detected peaks correspond to
        meaningful spikes in model uncertainty.

        The resulting combined figure is saved to:
            figs/uncertainty_plots.png
        """
        # Figure 3: uncertainty plots
        plt.figure(figsize=(8, 6))
        for file_id in tqdm(self.idiom_identifier.prediction_ensemble_measures['file_id'].unique()):
            if file_id in [fid for (fid, frame_number) in self.idiom_identifier.peak_files]:
                plot_smoothed_uncertainty(file_id, 
                                          self.idiom_identifier.prediction_ensemble_measures, 
                                          self.idiom_identifier.measure, 
                                          sigma=self.idiom_identifier.sigma, 
                                          show=False)
        plt.title("Uncertainty plots")
        plt.savefig(f"{self.results_path}/figs/uncertainty_plots.png")
        plt.show()

    def plot_peak_indices_histogram(self):
        """
        Plot a histogram of peak chirp indices.

        This visualization shows the distribution of chirp indices where peaks
        were detected within sequences. Peaks represent potential starting points
        of idioms (distinct chirp sequences).

        The histogram helps determine whether peaks occur preferentially at
        certain positions within sequences.

        The resulting figure is saved to:
            figs/histogram_peak_indices.png
        """
        # Figure 4: histogram of peak indices
        # plot a histogram of chirp_idx for peak chirps (aka starting index of idioms)
        peak_chirp_indices = [chirp_idx for (file_id, chirp_idx) in self.idiom_identifier.peak_files] # for all peak chirps
        peak_chirp_indices = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures["peak_detected"] >= 1]["chirp_idx"] # for just peak chirps at the beginning of a whole idiom
        plt.figure(figsize=(8, 6))
        plt.hist(peak_chirp_indices, bins=(max(peak_chirp_indices) - min(peak_chirp_indices) + 1))
        plt.title("Histogram of Peak Chirp Indices")
        plt.xlabel("Chirp Index")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_peak_indices.png")
        plt.show()

    def plot_idiom_length_histogram(self):
        """
        Plot a histogram of detected idiom lengths.

        Idiom length is defined as the number of chirps between consecutive
        detected peaks. This metric characterizes the temporal structure of
        identified idiom sequences.

        The histogram displays the frequency of different idiom lengths,
        allowing analysis of typical sequence durations.

        The resulting figure is saved to:
            figs/histogram_idiom_lengths.png
        """
        # Figure 5: histogram of idiom lengths
        # plot a histogram of histogram lengths (distance from a peak to the previous peak, only if a previous peak exists)
        # this only tracks whole idioms, as the distance from the beginning to the first peak is not measured
        distance_values = self.idiom_identifier.prediction_ensemble_measures[self.idiom_identifier.prediction_ensemble_measures["peak_detected"] > 0]['distance_to_next_peak'].dropna()
        plt.figure(figsize=(8, 6))
        plt.hist(distance_values, bins=(np.arange(min(distance_values), max(distance_values) + 2) - 0.5))
        # set the xticks to be integers from min to max distance
        plt.xticks(np.arange(min(distance_values), max(distance_values) + 1))
        plt.title("Histogram of Idiom Lengths")
        plt.xlabel("Idiom Lengths (# of chirps)")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_idiom_lengths.png")
        plt.show()

    def plot_cluster_histogram(self):
        """
        Plot a histogram of chirp cluster assignments.

        This visualization shows the frequency distribution of cluster labels
        produced during the clustering stage of the IdiomIdentifier pipeline.

        It provides insight into the relative size and balance of discovered
        chirp clusters.

        The resulting figure is saved to:
            figs/histogram_clusters.png
        """
        # Figure 6: histogram of clusters
        # plot histogram of chirp_labels
        # set xticks to be integers from min to max chirp label
        plt.figure(figsize=(8, 6))
        plt.xticks(np.arange(self.idiom_identifier.chirp_labels.min(), self.idiom_identifier.chirp_labels.max() + 1))
        plt.hist(self.idiom_identifier.chirp_labels, bins=np.arange(self.idiom_identifier.chirp_labels.min(), self.idiom_identifier.chirp_labels.max() + 2) - 0.5)
        plt.title(f"Histogram of {self.idiom_identifier.clusterer.cluster_method} Cluster Labels")
        plt.xlabel("Cluster Label")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_clusters.png")
        plt.show()

    def plot_chirp_scatter(self):
        """
        Plot a 2D scatterplot of all chirps without cluster labels.

        The chirp feature vectors are projected into two dimensions using
        either UMAP or PCA. This plot provides a baseline visualization
        of the overall chirp feature distribution before clustering.

        The resulting figure is saved to:
            figs/scatter_all_chirps.png
        """
        # Figure 7: scatterplot of chirps, unclustered
        # for a baseline reference, just plot all the chirps as a scatterplot without clustering
        plt.figure(figsize=(9, 8))
        scatter = plt.scatter(self.idiom_identifier.filtered_chirp_data_2d[:, 0], 
                              self.idiom_identifier.filtered_chirp_data_2d[:, 1], 
                              s=5)
        plt.title(f"All Chirps Distribution")
        plt.xlabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 1")
        plt.ylabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_all_chirps.png")

    def plot_clustered_chirp_scatter(self):
        """
        Plot a 2D scatterplot of chirps colored by cluster label.

        Chirp feature vectors are projected into two dimensions using UMAP
        or PCA. Each point is colored according to its assigned cluster
        label, allowing visual inspection of cluster separation and
        structure.

        The resulting figure is saved to:
            figs/scatter_all_chirps_clustered.png
        """
        # Figure 8: scatterplot of chirps, clustered
        # plot the points in each cluster using PCA for dimensionality reduction
        colormap = "tab10" if self.num_clusters <= 10 else "tab20"
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.idiom_identifier.filtered_chirp_data_2d[:, 0], 
                              self.idiom_identifier.filtered_chirp_data_2d[:, 1], 
                              c=self.idiom_identifier.filtered_chirp_labels, cmap=colormap, s=5, 
                              vmin=min(self.idiom_identifier.chirp_labels), vmax=max(self.idiom_identifier.chirp_labels))
        plt.title(f"{self.idiom_identifier.clusterer.cluster_method} Clustering: All Chirps")# ({'UMAP' if USE_UMAP else 'PCA'}-reduced to 2D)")
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 1")
        plt.ylabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_all_chirps_clustered.png")

    def plot_chirp_dendrogram(self):
        """
        Plot a hierarchical clustering dendrogram for chirps.

        The dendrogram visualizes the hierarchical relationships between
        chirp feature vectors generated during agglomerative clustering.

        Only the top clusters are shown using truncation for readability.
        Each leaf corresponds to a chirp cluster and includes the number
        of chirps contained within that cluster.

        The resulting figure is saved to:
            figs/dendrogram_clustering.png
        """
        # Figure 9: dendrogram of chirps
        plt.figure(figsize=(11, 8))
        chirp_dendrogram = dendrogram(self.idiom_identifier.linked,
                truncate_mode='lastp',
                p=int(self.num_clusters),
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
        """
        Plot a histogram of peak prominence-to-range ratios.

        The prominence-to-range ratio measures how prominent a detected
        peak is relative to the uncertainty range of the sequence.

        This visualization helps assess the statistical strength of
        detected peaks and distinguish meaningful peaks from noise.

        The resulting figure is saved to:
            figs/histogram_peak_prominence_to_range.png
        """
        # Figure 10: histogram of prominence to range ratio
        plt.cla()
        plt.figure(figsize=(8, 6))
        plt.hist(self.idiom_identifier.peak_df["prominence_to_range"], bins=20)
        plt.title("Histogram of Peak Prominence to Range Ratio")
        plt.xlabel("Prominence to Range Ratio")
        plt.ylabel("Count")
        plt.savefig(f"{self.results_path}/figs/histogram_peak_prominence_to_range.png")
        plt.show()

    def plot_chirp_peak_scatter(self):
        """
        Plot clustered chirps with significant peaks highlighted.

        Chirps are displayed in a 2D projection and colored according
        to their cluster labels. Chirps identified as statistically
        significant peaks are highlighted with star markers.

        This visualization helps identify whether peak chirps tend to
        concentrate in particular clusters.

        The resulting figure is saved to:
            figs/scatter_chirps_and_peaks.png
        """
        # Figure 11: scatterplot of chirps, clustered, with significant peaks highlighted
        # plot only the significant peaks on the previous scatter plot
        colormap = "tab10" if self.num_clusters <= 10 else "tab20"
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
        plt.title(f"{self.idiom_identifier.clusterer.cluster_method} Clustering: Chirps and Peaks")# ({'UMAP' if USE_UMAP else 'PCA'}-reduced to 2D)")
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 1")
        plt.ylabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 2")
        legend_symbols = matplotlib.lines.Line2D([0], [0], marker='*', color='w', label='Significant Peak', markerfacecolor='black', markersize=15)
        plt.legend([legend_symbols], ["Peak Chirp"])
        plt.savefig(f"{self.results_path}/figs/scatter_chirps_and_peaks.png")

    def plot_cluster_transition_scatter(self):
        """
        Visualize cluster transitions between chirps in idiom sequences.

        This method constructs a directed graph where nodes represent
        chirp clusters and edges represent observed transitions between
        clusters within significant idiom sequences.

        Edge weights correspond to the frequency of transitions.
        Cluster centroids are used as node positions in the 2D space.

        This visualization reveals structural patterns in how chirp
        clusters transition within idiom sequences.

        The resulting figure is saved to:
            figs/scatter_chirps_and_transitions.png
        """
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
        plt.figure(figsize=(8, 6))
        colormap = "tab10" if self.num_clusters <= 10 else "tab20"
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
        plt.xlabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 1")
        plt.ylabel(f"{self.idiom_identifier.clusterer.reduc_method.upper()} Component 2")
        plt.savefig(f"{self.results_path}/figs/scatter_chirps_and_transitions.png")

    def plot_chirp_v_peak_cluster_histogram(self):
        """
        Compare cluster distributions for all chirps versus peak chirps.

        This histogram overlays two distributions:

        - Cluster labels for all chirps
        - Cluster labels for chirps identified as significant peaks

        The comparison helps determine whether peak chirps are associated
        with specific clusters or occur uniformly across clusters.

        The resulting figure is saved to:
            figs/histogram_clusters_all_vs_peaks.png
        """
        # Figure 13: histogram of clusters for all chirps vs peaks
        # plt.hist(peak_chirp_clusters, bins=np.arange(chirp_labels.min(), chirp_labels.max() + 2) - 0.5)
        plt.cla()
        plt.figure(figsize=(8, 6))
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
        """
        Generate the full set of analysis visualizations.

        This method sequentially runs all plotting functions in the
        IdiomIdentifierVisualizer to produce a complete collection of
        figures summarizing uncertainty behavior, peak detection,
        clustering structure, and idiom sequence transitions.

        All generated figures are saved to the `figs/` directory within
        the specified results path.
        """
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
