import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import os
from tqdm import tqdm
import joblib
from scipy import special
from scipy.spatial.distance import pdist
from gap_statistic import OptimalK
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from scipy.cluster.hierarchy import fcluster
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn import cluster
from collections import Counter
from scipy.ndimage import gaussian_filter
import sys
sys.path.append("..")
from peak_detection.data_series_analyzer import DataSeriesAnalyzer

def unscale(df, scaler_path, cols_to_keep=[]):
    scaler = joblib.load(open(f'{scaler_path}', 'rb'))
    non_scaler_columns = {}
    temp_df = df.copy()
    for idx, column in enumerate(temp_df.columns):
        if column not in scaler.feature_names_in_:
            non_scaler_columns[idx] = temp_df[column]
            temp_df.drop(columns=[column], inplace=True)

    all_features = list(scaler.feature_names_in_)
    column_idxs = [] # indices of columns in predictions that are in scaler.feature_names_in_
    for column in scaler.feature_names_in_:
        if column not in temp_df.columns:
            temp_df[column] = 0
        else:
            column_idxs.append(all_features.index(column))

    temp_df = temp_df[scaler.feature_names_in_]

    df_np = temp_df.to_numpy()
    df_np = df_np[:, :-1]

    inverted_predictions = scaler.inverse_transform(temp_df)

    filtered_inverted_predictions = inverted_predictions[:, column_idxs]

    original_df = pd.DataFrame(filtered_inverted_predictions, columns=[temp_df.columns[i] for i in column_idxs])

    for col in cols_to_keep:
        original_df[col] = df[col]
    return original_df

def group_ensemble_data(prediction_df_list, index):
    preds_i = np.array([pred.iloc[index] for pred in prediction_df_list])
    return preds_i

def compute_cluster_measures(X, radius_for_density=None):
    """
    Compute cluster measures for an array X of shape (n_points, n_features).
    Returns a dict with:
      - n, dim
      - centroid
      - distances (to centroid)
      - radius (max distance to centroid)
      - avg_radius (mean distance to centroid)
      - std_radius
      - diameter (max pairwise distance)
      - avg_pairwise_distance
      - tightness = 1 / (1 + avg_pairwise_distance)  (higher -> tighter)
      - density = n_points / vol_ball(radius)  (uses 'radius' or radius_for_density if provided)
    """
    X = X[:, 2:len(X) - 3]
    X = np.asarray(X, dtype=float)
    
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_points, n_features)")
    n, d = X.shape

    centroid = X.mean(axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    radius = float(distances.max()) if n > 0 else 0.0
    avg_radius = float(distances.mean()) if n > 0 else 0.0
    std_radius = float(distances.std(ddof=0)) if n > 0 else 0.0

    if n > 1:
        pairwise = pdist(X, metric="euclidean")
        diameter = float(pairwise.max())
        avg_pairwise = float(pairwise.mean())
    else:
        pairwise = np.array([])
        diameter = 0.0
        avg_pairwise = 0.0

    # tightness: inverse of typical pairwise distance (bounded)
    tightness = 1.0 / (1.0 + avg_pairwise) if avg_pairwise >= 0 else np.nan

    # density: n / volume_of_d_ball(radius)
    r = radius_for_density if (radius_for_density is not None) else radius
    if r <= 0:
        density = np.inf if n > 0 else 0.0
    else:
        # Volume of d-dimensional ball: V_d(r) = pi^(d/2) / Gamma(d/2 + 1) * r^d
        vol_unit_ball = (np.pi ** (d / 2.0)) / special.gamma(d / 2.0 + 1.0)
        vol = vol_unit_ball * (r ** d)
        density = n / vol if vol > 0 else np.inf

    return {
        "n_points": n,
        "dim": d,
        "centroid": centroid,
        "distances_to_centroid": distances,
        "radius_max": radius,
        "radius_mean": avg_radius,
        "radius_std": std_radius,
        "diameter_pairwise": diameter,
        "avg_pairwise_distance": avg_pairwise,
        "tightness": tightness,
        "volume": vol,
        "density": density,
    }

def calculate_ensemble_measures(prediction_files, truth_files):
    prediction_list = []
    truth_list = []
    absolute_errors = []
    euclidean_distances = []
    for pred_file, truth_file in zip(prediction_files, truth_files):
        pred_df = pd.read_csv(pred_file)
        truth_df = pd.read_csv(truth_file)
        prediction_list.append(pred_df)
        truth_list.append(truth_df)

        # select numeric columns and keep only common columns
        pred_num = pred_df.select_dtypes(include=[np.number])
        truth_num = truth_df.select_dtypes(include=[np.number])
        common_cols = pred_num.columns.intersection(truth_num.columns)

        if len(common_cols) == 0:
            # nothing numeric in common, append empty dataframe
            absolute_errors.append(pd.DataFrame())
        else:
            # align row counts (use the shorter one) and reset index for safe subtraction
            n = min(len(pred_num), len(truth_num))
            pred_slice = pred_num.loc[: n - 1, common_cols].reset_index(drop=True)
            truth_slice = truth_num.loc[: n - 1, common_cols].reset_index(drop=True)

            # error = truth - prediction, keep absolute per-cell errors
            errors = truth_slice - pred_slice
            abs_errors = errors.abs()

            euclidean_distances.append((errors * errors).sum(axis=1).pow(0.5))

            absolute_errors.append(abs_errors)

    euclidean_distances = np.array(euclidean_distances)
    for error_df in absolute_errors:
        error_df.drop(["Unnamed: 0", "TimeIndex", "file_id", "chirp_idx"], axis=1, inplace=True, errors="ignore")

    tightness = []
    radii = []
    densities = []
    error_densities = []

    for i in range(len(prediction_list[0])):
        pred_cluster_i = group_ensemble_data(prediction_list, i)
        measures = compute_cluster_measures(pred_cluster_i)
        tightness.append(measures["tightness"])
        radii.append(measures["radius_mean"])
        densities.append(measures["density"])

        error_cluster_i = group_ensemble_data(absolute_errors, i)
        error_measures = compute_cluster_measures(error_cluster_i)
        error_densities.append(error_measures["density"])
        # print(measures["tightness"])

    average_error_per_point = []
    for error_df in absolute_errors:
        if not error_df.empty:
            avg_error = error_df.mean(axis=1)
            average_error_per_point.append(avg_error.values)
        else:
            average_error_per_point.append(np.nan)  # or some other placeholder for empty dataframes
    average_error_per_point = np.array(average_error_per_point)

    # copy file_id and chirp_idx from prediction_list as ints
    prediction_ensemble_measures = prediction_list[0][['file_id', 'chirp_idx']].copy().astype(int)
    # prediction_ensemble_measures.insert(0, 'id', prediction_list[0].index)
    prediction_ensemble_measures['tightness'] = tightness
    prediction_ensemble_measures['radius_mean'] = radii
    prediction_ensemble_measures['density'] = densities
    prediction_ensemble_measures['average_error_per_point'] = average_error_per_point.mean(axis=0)
    prediction_ensemble_measures['error_density'] = error_densities
    prediction_ensemble_measures['euclidean_distance'] = euclidean_distances.mean(axis=0)
    return truth_list[0], prediction_ensemble_measures

def plot_uncertainty_by_file(file_id, df, ax=None, figsize=(12, 8), metrics=[], show=True, save_filename=None):
    """
    Plot tightness vs chirp_idx for a given file_id using the existing `prediction_ensemble_measures`.
    Returns the matplotlib Axes instance.
    """
    sel = df[df["file_id"] == file_id].copy()
    if sel.empty:
        raise ValueError(f"No rows found for file_id={file_id}")

    sel = sel.sort_values("chirp_idx")
    x = sel["chirp_idx"]
    # if chirp_idx are whole numbers, cast to int for nicer x ticks
    if np.allclose(x % 1, 0):
        x = x.astype(int)
    # y = sel["tightness"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if type(metrics) == str:
        metrics = [metrics]

    if "tightness" in metrics:
        ax.plot(x, sel["tightness"], marker="o", linestyle="-", color="C0")
    # Color points based on the value of sel["low_confidence"]
    # colors = sel["low_confidence"].map({True: "red", False: "blue"})
    if "radius_mean" in metrics:
        ax.plot(x, sel["radius_mean"] / -2, marker="", linestyle="-", color="C1")
    # ax.scatter(x, sel["radius_mean"] / -2, c=colors, marker="o", label="radius_mean (colored by low_confidence)")
    if "density" in metrics:
        ax.plot(x, np.log(sel["density"]) / 100, marker="o", linestyle="-", color="C2")
    if "average_error_per_point" in metrics:
        ax.plot(x, sel["average_error_per_point"], marker="o", linestyle="-", color="C3")
    if "error_density" in metrics:
        ax.plot(x, np.log(sel["error_density"]) / 100, marker="o", linestyle="-", color="C4")
    if "euclidean_distance" in metrics:
        ax.plot(x, sel["euclidean_distance"] / 5, marker="o", linestyle="-", color="C5")
    ax.set_xlabel("chirp_idx")
    ax.set_ylabel("tightness")

    # ensure x-axis ticks are integer indices (avoid fractional chirp_idx labels)
    x_min = int(np.floor(x.min()))
    x_max = int(np.ceil(x.max()))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    # limit number of ticks to avoid overcrowding
    if x_max - x_min <= 20:
        ticks = np.arange(x_min, x_max + 1)
    else:
        step = max(1, (x_max - x_min) // 20)
        ticks = np.arange(x_min, x_max + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])

    ax.set_title(f"tightness vs chirp_idx (file_id={file_id})")
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.set_ylim(-1,0.1)

    # label the three plotted series (they were plotted above in order) and show legend
    labels = ["tightness", "-radius_mean", "log(pred_density) / 100", "log(error_density) / 100", "mean euclidean distance / 5"]
    for line, lbl in zip(ax.lines[-len(labels):], labels):
        line.set_label(lbl)
    ax.legend(fontsize="small", framealpha=0.5)

    if show:
        plt.show()

    if save_filename:
        fig_filename = f"{save_filename}.png"
        ax.figure.savefig(fig_filename)
        # print(f"Saved figure to {fig_filename}")
        plt.close()
    return ax

def plot_smoothed_uncertainty(file_id, df, measure, sigma=1, ylim=None, show=True):
    sel = df[df['file_id'] == file_id]
    smoothed = gaussian_filter(sel[measure], sigma=sigma, order=0, mode='reflect')
    plt.plot(sel['chirp_idx'], -np.log(smoothed))
    plt.title(f"{measure} vs chirp_idx: file={file_id}")
    plt.xlabel("chirp_idx")
    plt.ylabel(f"-ln({measure})")
    if ylim is not None:
        plt.ylim(ylim)
    if show:
        plt.show()

def find_ideal_cluster_k(data, min_k, max_k, plot=False):
    n_clusters = dict()
    
    # calculate gap statistic using optimalK
    optimalK = OptimalK(n_jobs=4, parallel_backend='joblib')
    optimal_k_n_clusters = optimalK(data, cluster_array=np.arange(min_k, max_k))

    n_clusters["gap_statistic"] = optimal_k_n_clusters
    # get the index of the first row in optimalK.gap_df["diff"] that is positive
    first_positive_diff_idx = optimalK.gap_df[optimalK.gap_df["diff"] > 0].index[0]
    n_clusters["diff_statistic"] = first_positive_diff_idx + 1 # add 1 because index starts at 0

    if plot:
        # plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value)
        optimalK.plot_results()

    # Use the elbow method to find the number of clusters where the within-cluster sum of squares (WCSS) starts to level off
    # Use agglomerative clustering to compute WCSS for k=1 to 50
    wcss = []
    for i in tqdm(range(min_k, max_k)):
        agglom = cluster.AgglomerativeClustering(n_clusters=i)
        agglom.fit(data)
        labels = agglom.labels_
        centroids = np.array([data[labels == j].mean(axis=0) for j in range(i)])
        wcss.append(sum(np.min(cdist(data, centroids, 'euclidean'), axis=1)) / data.shape[0])
    if plot:
        plt.plot(range(min_k, max_k), wcss)
        plt.xticks(np.arange(min_k, max_k + 1, step=(max_k - min_k) // 10))
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.show()

    # identify the elbow point in the WCSS plot using the "knee" method
    knee_locator = KneeLocator(range(min_k, max_k), wcss, curve='convex', direction='decreasing')
    elbow_k = knee_locator.knee
    n_clusters["elbow_method"] = elbow_k
    return n_clusters

def cluster_chirps(data, clustering_method, umap, n_clusters):
    # getting the linkage tree allows us to specify the number of clusters we want afterwards
    if clustering_method == "HDBSCAN":
        if umap:
            hdb = HDBSCAN(min_cluster_size=5, 
                                min_samples=5, 
                                metric="euclidean",
                                max_cluster_size=1000,
                                cluster_selection_epsilon=7e-7,
                                )
            hdb.fit(data)
            Z = hdb.single_linkage_tree_.to_numpy()
            chirp_labels = fcluster(Z, n_clusters, criterion='maxclust')
            # chirp_labels = hdb.fit_predict(chirp_data_embedded)
        else: 
            hdb = HDBSCAN(min_cluster_size=5, 
                                min_samples=5, 
                                metric="euclidean",
                                max_cluster_size=1000,
                                cluster_selection_epsilon=2e-4,
                                )
            Z = hdb.fit(data)
            Z = hdb.single_linkage_tree_.to_numpy()
            chirp_labels = fcluster(Z, n_clusters, criterion='maxclust')
            # chirp_labels = hdb.fit_predict(chirp_data)
    elif clustering_method == "Agglomerative":
        if umap:
            agglomerative = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                            # distance_threshold=20,
                                                            linkage="ward",
                                                            )
            chirp_labels = agglomerative.fit_predict(data)
        else:
            agglomerative = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                            # distance_threshold=100000,
                                                            linkage="ward",
                                                            )
            chirp_labels = agglomerative.fit_predict(data)
    else:
        raise ValueError("clustering_method must be one of: 'HDBSCAN', 'Agglomerative'")
    
    return chirp_labels

def detect_uncertainty_peaks(file_id, df, measure, sigma=3):
    """
    Detect peaks in tightness for a given file_id using DataSeriesAnalyzer.
    Returns a list of chirp_idx where peaks were detected.
    """
    sel = df[df["file_id"] == file_id].copy()
    # sort sel by chirp_idx
    sel = sel.sort_values("chirp_idx")
    data = sel[measure]
    # reverse polarity of values if measure is density (because we want to match peaks with low density moments)
    if measure == "density":
        data = -sel[measure]
    data.index = np.arange(len(data))
    
    analyzer = DataSeriesAnalyzer(data)
    # Get a df with columns:
    # 'frame_number', 'prominence', 'plateau', 'smoothed_content_val', 'content_vals'
    peaks_info = analyzer.analyze(sigma=sigma)
    # print(peaks_info)
    return peaks_info

# Function: get_surrounding_peak_sequence
# ----------------------------
# Given the index of a peak chirp, this function retrieves a sequence of chirp indices
# centered around the peak, extending a specified number of chirps before and after.
# Parameters:
# - peak_idx: Index of the peak chirp in dataframe
# - df: DataFrame containing chirp data with 'file_id' and 'chirp_idx' columns
# - context_size: Number of chirps to include before and after the peak
# Returns:
# - List of chirp indices surrounding the peak
def get_surrounding_peak_sequence(peak_idx, df, context_size=5):
    peak_row = df.loc[peak_idx]
    file_id = peak_row['file_id']
    chirp_idx = peak_row['chirp_idx']
    
    sel = df[df['file_id'] == file_id].sort_values('chirp_idx').reset_index()
    peak_pos = sel[sel['chirp_idx'] == chirp_idx].index[0]

    start_pos = max(0, peak_pos - context_size)
    end_pos = min(len(sel) - 1, peak_pos + context_size)
    
    surrounding_indices = sel.loc[start_pos:end_pos, 'index'].tolist()
    return surrounding_indices

# Function: identify_significant_peaks
# ------------------------------------------
# Given a DataFrame of idiom sequences with their uncertainty measures, identify significant peaks in a specified measure
# for a particular sequence (row) based on similarity to other sequences and statistical significance
# Parameters:
# - idiom_df: DataFrame containing idiom sequences and their uncertainty measures
# - row_idx: Index of the row in idiom_df to analyze
# - measure: Column name in idiom_df representing the uncertainty measure to analyze (e.g., 'normalized_uncertainty_seq')
# - similarity_threshold: Fractional threshold to determine similar sequences based on uncertainty range (default 0.1)
# - significance_threshold: Z-score threshold to identify significant peaks (default 2.0)
# Returns:
# - significant_peaks: Indices of the significant peaks in the specified measure for the given row
from scipy.stats import normaltest

def identify_significant_peaks(idiom_df, row_idx, measure, range,
                               similarity_threshold=0.1,
                               std_threshold=None,
                               percentile_threshold=None,
                               verbose=0):
    row = idiom_df.iloc[row_idx]
    # get sequences with similar uncertainty ranges
    similar_seqs = idiom_df[(idiom_df[range] >= (1 - similarity_threshold) * row[range]) & \
                            (idiom_df[range] <= (1 + similarity_threshold) * row[range])]
    # if similar_seqs.shape[0] < 2:
    #     return []
    if verbose > 0:
        print("Number of similar sequences found:", similar_seqs.shape[0], "total sample size:", sum([len(seq) for seq in similar_seqs[measure]]))

    # pool together the normalized uncertainty sequences to form a distribution
    pooled_values = []
    for seq in similar_seqs[measure]:
        pooled_values.extend([float(x) for x in seq])
    pooled_values = np.array(pooled_values)

    # run a normality check on pooled_values
    normaltest_result = normaltest(pooled_values)
    if verbose > 0:
        print("Normality test p-value for pooled values:", normaltest_result.pvalue)

    # calculate the z-score of the original sequence's normalized density values against the pooled distribution
    original_seq_values = np.array([float(x) for x in row[measure]])
    
    if percentile_threshold:
        # calculate the specified percentile of the pooled distribution
        percentile = np.percentile(pooled_values, percentile_threshold)
        if verbose > 0:
            print("chirps that pass percentile threshold:", np.where(original_seq_values > percentile)[0].shape[0])
        significant_peaks = np.where(original_seq_values > percentile)[0]

    elif std_threshold:
        z_scores = (original_seq_values - np.mean(pooled_values)) / (np.std(pooled_values) + 1e-12)
        if verbose > 0:
            print("Z-scores of original sequence:", z_scores)

        # identify peaks in the z-scores that are above the significance threshold
        significant_peaks = np.where(z_scores > std_threshold)[0]
    else:
        return []
    significant_peaks = [row["surrounding_seq_idx"][peak_idx] for peak_idx in significant_peaks]

    return significant_peaks

# Function: get_random_sequence
# ---------------------------
# This function retrieves a random sequence of 'n' chirp indices from the unscaled_truth_df,
# ensuring that none of the indices in the selected sequence overlap with any indices
# present in the existing_sequences list.
# Parameters:
# - n: The desired length of the chirp index sequence to retrieve.
# - existing_sequences: A list of lists, where each inner list contains chirp indices
#   that should not be included in the new sequence.
# - unscaled_truth_df: A DataFrame containing chirp data
# Return: A list of 'n' chirp indices that do not overlap with existing_sequences.

def get_random_sequence(n, existing_sequences, unscaled_truth_df):
    existing_set = set()
    for seq in existing_sequences:
        existing_set.update(seq)

    file_ids = unscaled_truth_df['file_id'].unique()
    while True:
        file_id = np.random.choice(file_ids)
        file_chirps = unscaled_truth_df[unscaled_truth_df['file_id'] == file_id]
        if file_chirps.shape[0] < n:
            continue
        start_idx = np.random.randint(0, file_chirps.shape[0] - n + 1)
        seq_indices = file_chirps.index[start_idx:start_idx + n].tolist()
        if not any(idx in existing_set for idx in seq_indices):
            return seq_indices
        
# Function: get_random_idiom_sequence
# ---------------------------
# This function retrieves a random idiom sequence of length n from the idiom_df
# Parameters:
# - n: The desired length of the idiom sequence to retrieve.
# - idiom_df: A DataFrame containing idiom sequences and their chirp indices.
# Return: A list of 'n' chirp indices that make up the idiom sequence

def get_random_idiom_sequence(n, idiom_df):
    idiom_sequences = idiom_df[idiom_df['length'] >= n]['chirp_indices'].index
    if not idiom_sequences.any():
        raise ValueError("No idiom sequences of sufficient length found.")
    selected_idiom = idiom_df.iloc[np.random.choice(idiom_sequences)]['chirp_indices']
    start_idx = np.random.randint(0, len(selected_idiom) - n + 1)
    return selected_idiom[start_idx:start_idx + n]

# Function: get_attributes_from_indices
# --------------------------------------
# This function retrieves chirp attributes from the unscaled_truth_df
# based on a provided list of chirp indices.
# Parameters:
# - indices: A list of chirp indices for which to retrieve attributes.
# - unscaled_truth_df: A DataFrame containing chirp data.
# Return: A list of chirp attributes corresponding to the provided indices.

def get_attributes_from_indices(indices, unscaled_truth_df):
    return [unscaled_truth_df.iloc[i]["PrecedingIntrvl":"AmpK@start"].values for i in indices]

# Function: calculate_sequence_volume
# --------------------------------------
# This function calculates the spherical volume in hyperspace occupied by a sequence of chirp attributes.
# Parameters:
# - attributes: A list of chirp attributes, where each attribute is a numpy array.
# Return: The calculated volume as a float.

def calculate_sequence_volume(attributes):
    centroid = np.mean(attributes, axis=0)
    distances = np.linalg.norm(attributes - centroid, axis=1)
    radius = np.max(distances)
    dim = attributes[0].shape[0]
    volume = (np.pi ** (dim / 2)) / special.gamma((dim / 2) + 1) * (radius ** dim)
    return volume

# Function: most_common_subsequences
# --------------------------------------
# 
def most_common_subsequences(sequences, length, subseq_type="all", k=None):
    assert subseq_type in ["all", "prefix", "suffix"], "subseq_type must be in [all, prefix, suffix]"
    subseq_count = Counter()
    for seq in sequences:
        if subseq_type == "all": 
            for i in range(len(seq) - (length - 1)):
                subseq_count[tuple(seq[i:i + length])] += 1
        if subseq_type == "prefix":
            if len(seq) >= length:
                prefix = tuple(seq[:length])
                subseq_count[prefix] += 1
        if subseq_type == "suffix":
            if len(seq) >= length:
                suffix = tuple(seq[-length:])
                subseq_count[suffix] += 1
    if k:
        return subseq_count.most_common(k)
    else:
        return subseq_count

# Function: quantile_normalize
# ----------------------
# quantile_normalize uses a QuantileTransformer to normalize a list of columns in a DataFrame to a normal distribution,
# where values are transformed into their corresponding z-score.
# @param df: the pandas DataFrame whose columns are being normalized
# @param columns: the columns to normalize
# @return: a pandas DataFrame whose columns are scaled, all other columns of df are kept constant
def quantile_normalize(df, columns):
    scaled_df = df.copy()
    scaler = QuantileTransformer(output_distribution='normal')     
    scaler.set_output(transform="pandas")
    scaler.fit(df[columns])
    scaled_df.loc[:, columns] = scaler.transform(df.loc[:, columns])
    return scaled_df

# Function: describe_cluster
# ----------------------------
# describe_cluster gets the average of all chirp attributes within a given cluster, producing the "average chirp" that
# represents the given cluster.
# @param df: a pandas DataFrame() containing chirp attributes and cluster ids
# @param cluster_idx: the cluster id to be averaged over
# @param normalize: whether to normalize the chirp attributes using a QuantileTransformer to produce z-score-like measures
# return: a pandas Series() consisting of the average value for each chirp attribute
def describe_cluster(df, cluster_idx, normalize=False, ignore_columns=None):
    cols_to_use = [col for col in df.columns if col not in ignore_columns]
    if normalize:
        df_to_describe = quantile_normalize(df, cols_to_use)
    else:
        df_to_describe = df
    cluster_data = df_to_describe[df_to_describe["cluster_idx"] == cluster_idx]
    cluster_avg = cluster_data.loc[:, cols_to_use].mean(axis=0)
    return cluster_avg

# =================================================
# Old methods of finding peaks in uncertainty:
# Example usage:
# drops = find_tightness_drops(sample_file_id, method='relative', rel_threshold=0.02, window=2, return_type='df')
def _find_tightness_drops(
    file_id,
    df,
    window=2,
    method="relative",
    rel_threshold=0.5,
    z_thresh=2.0,
    min_abs_drop=None,
    min_neighbors=1,
    return_type="chirp_idx",
):
    """
    Find chirp_idx within a file where tightness shows a significant decrease
    compared to surrounding chirp_idx values.

    Parameters
    - file_id: value in df['file_id'] to select rows for.
    - df: dataframe with columns ['file_id','chirp_idx','tightness'] (defaults to prediction_ensemble_measures).
    - window: number of neighbors before and after to consider (int).
    - method: one of
        * 'relative'   -> (local_mean - t) / (local_mean) >= rel_threshold
        * 'zscore'     -> (local_mean - t) / local_std >= z_thresh
        * 'absolute'   -> (local_mean - t) >= min_abs_drop
    - rel_threshold: fraction threshold for 'relative' method (default 0.5).
    - z_thresh: threshold in sigma for 'zscore' method (default 2.0).
    - min_abs_drop: minimum absolute drop used by 'absolute' method (if None, must be provided)
    - min_neighbors: minimum number of neighbor points required to evaluate (default 1).
    - return_type: 'chirp_idx' (list), 'df' (rows with diagnostics), or 'indices' (original df indices).

    Returns
    - depending on return_type
    """
    sel = df[df["file_id"] == file_id].copy()
    if sel.empty:
        raise ValueError(f"No rows found for file_id={file_id}")

    # keep original dataframe index so we can return indices if requested
    sel = sel.sort_values("chirp_idx").reset_index().rename(columns={"index": "orig_index"})
    t = sel["tightness"].astype(float).values
    n = len(t)
    eps = 1e-12

    results = []
    for i in range(n):
        # neighbor window: exclude current i
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbor_idx = list(range(lo, i)) + list(range(i + 1, hi))
        if len(neighbor_idx) < min_neighbors:
            results.append((i, False, np.nan, np.nan, np.nan))
            continue

        neigh_vals = t[neighbor_idx]
        local_mean = float(np.nanmean(neigh_vals))
        local_med = float(np.nanmedian(neigh_vals))
        local_std = float(np.nanstd(neigh_vals, ddof=0))

        drop_amount = local_mean - t[i]
        drop_fraction = drop_amount / (local_mean + eps)
        drop_z = (local_mean - t[i]) / (local_std + eps)

        is_drop = False
        if method == "relative":
            is_drop = (drop_fraction >= rel_threshold)
        elif method == "zscore":
            is_drop = (drop_z >= z_thresh)
        elif method == "absolute":
            if min_abs_drop is None:
                raise ValueError("min_abs_drop must be provided for method='absolute'")
            is_drop = (drop_amount >= min_abs_drop)
        else:
            raise ValueError("method must be one of: 'relative', 'zscore', 'absolute'")

        results.append((i, bool(is_drop), drop_amount, drop_fraction, drop_z))

    res_df = pd.DataFrame(
        results, columns=["pos", "is_drop", "drop_amount", "drop_fraction", "drop_z"]
    )
    # attach chirp_idx and original index and tightness
    res_df["chirp_idx"] = sel["chirp_idx"].values
    res_df["tightness"] = sel["tightness"].values
    res_df["orig_index"] = sel["orig_index"].values

    if return_type == "df":
        return res_df.loc[res_df["is_drop"]].reset_index(drop=True)
    elif return_type == "chirp_idx":
        return res_df.loc[res_df["is_drop"], "chirp_idx"].tolist()
    elif return_type == "indices":
        return res_df.loc[res_df["is_drop"], "orig_index"].tolist()
    else:
        raise ValueError("return_type must be one of: 'df', 'chirp_idx', 'indices'")
    
# Example:
# outliers = find_tightness_outliers(sample_file_id, method="iqr", return_type="df")
def _find_tightness_outliers(
    file_id,
    df,
    method="iqr",
    iqr_multiplier=1.5,
    z_thresh=3.0,
    return_type="df",
):
    """
    Identify chirp_idx rows for a given file_id whose 'tightness' is an outlier.

    Parameters
    - file_id: value in df['file_id'] to select rows for.
    - df: dataframe with columns ['file_id','chirp_idx','tightness'] (defaults to prediction_ensemble_measures).
    - method: 'iqr' (default), 'zscore', or 'modified_zscore' (based on MAD).
    - iqr_multiplier: multiplier for IQR rule (only for method='iqr').
    - z_thresh: threshold for zscore / modified zscore (absolute value).
    - return_type: 'df' (returns DataFrame with an 'is_outlier' column),
                   'chirp_idx' (returns list of chirp_idx values flagged),
                   'indices' (returns DataFrame indices).

    Returns
    - depending on return_type
    """
    sel = df[df["file_id"] == file_id].copy()
    if sel.empty:
        raise ValueError(f"No rows found for file_id={file_id}")

    x = sel["tightness"].astype(float)

    if method == "iqr":
        q1 = np.nanpercentile(x, 25)
        q3 = np.nanpercentile(x, 75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        is_outlier = (x < lower) | (x > upper)
    elif method == "zscore":
        mu = np.nanmean(x)
        sigma = np.nanstd(x, ddof=0)
        if sigma == 0:
            is_outlier = np.zeros_like(x, dtype=bool)
        else:
            z = (x - mu) / sigma
            is_outlier = np.abs(z) > z_thresh
    elif method == "modified_zscore":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if mad == 0:
            is_outlier = np.zeros_like(x, dtype=bool)
        else:
            mod_z = 0.6745 * (x - med) / mad
            is_outlier = np.abs(mod_z) > z_thresh
    else:
        raise ValueError("method must be one of: 'iqr', 'zscore', 'modified_zscore'")

    sel["is_outlier"] = is_outlier.values

    if return_type == "df":
        return sel
    elif return_type == "chirp_idx":
        return sel.loc[sel["is_outlier"], "chirp_idx"].tolist()
    elif return_type == "indices":
        return sel.index[sel["is_outlier"]].tolist()
    else:
        raise ValueError("return_type must be one of: 'df', 'chirp_idx', 'indices'")