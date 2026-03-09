# -*- coding: utf-8 -*-
# @Author: Andrew Chen

from collections import Counter
import numpy as np
import pandas as pd
from scipy import special
from scipy.ndimage import gaussian_filter
from scipy.signal import peak_prominences
from scipy.spatial.distance import pdist, cdist, euclidean
from scipy.stats import ttest_ind, normaltest
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from tqdm import tqdm
import joblib
from gap_statistic import OptimalK
from kneed import KneeLocator
from hdbscan import HDBSCAN

from peak_detection.data_series_analyzer import DataSeriesAnalyzer

def scale(df, scaler, cols_to_keep=[]):
    """
    Scale numeric columns of a DataFrame using a provided sklearn scaler.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing features to scale.
    scaler : sklearn-like transformer
        Fitted or unfitted scaler object implementing `fit` and `transform`.
    cols_to_keep : list of str, optional
        Column names that should NOT be scaled.

    Returns
    -------
    pandas.DataFrame
        DataFrame with selected columns scaled in-place.
    """
    columns_to_scale = [col for col in df.columns if col not in cols_to_keep]
    scaler.fit(df[columns_to_scale])
    df.loc[:, columns_to_scale] = scaler.transform(df.loc[:, columns_to_scale])
    return df

def unscale(df, scaler_path, cols_to_keep=[]):
    """
    Invert scaling transformation using a saved scaler.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled DataFrame to be inverse transformed.
    scaler_path : str
        Path to a joblib-saved scaler.
    cols_to_keep : list of str, optional
        Columns to copy directly from the input df without inverse scaling.

    Returns
    -------
    pandas.DataFrame
        DataFrame restored to original feature scale.
    """
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
    """
    Extract ensemble predictions at a given row index.

    Parameters
    ----------
    prediction_df_list : list of pandas.DataFrame
        List of prediction DataFrames.
    index : int
        Row index to extract from each DataFrame.

    Returns
    -------
    numpy.ndarray
        Array of stacked predictions across ensemble members.
    """
    preds_i = np.array([pred.iloc[index] for pred in prediction_df_list])
    return preds_i

def compute_cluster_measures(X, radius_for_density=None):
    """
    Compute geometric and density-based statistics for a cluster of points.

    Parameters
    ----------
    X : array-like of shape (n_points, n_features)
        Data points in feature space.
    radius_for_density : float, optional
        Radius to use for density calculation. If None, uses max radius.

    Returns
    -------
    dict
        Dictionary containing cluster statistics including centroid,
        radii, pairwise distances, tightness, volume, and density.
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
    """
    Compute ensemble-based uncertainty and error measures across multiple
    prediction files relative to corresponding ground-truth files.

    This function aggregates predictions from multiple ensemble members,
    computes per-point geometric cluster statistics (e.g., tightness,
    radius, density), and evaluates prediction error metrics relative to
    ground truth. Metrics are computed per chirp index across ensemble
    members.

    Parameters
    ----------
    prediction_files : list of str
        List of file paths to CSV files containing model predictions.
        Each file must contain numeric prediction columns and identifier
        columns including 'file_id' and 'chirp_idx'.
    truth_files : list of str
        List of file paths to CSV files containing corresponding ground
        truth values. Must align one-to-one with `prediction_files`.

    Returns
    -------
    tuple
        (truth_df, prediction_ensemble_measures)

        truth_df : pandas.DataFrame
            The first loaded ground-truth DataFrame (used as reference).

        prediction_ensemble_measures : pandas.DataFrame
            DataFrame indexed like the first prediction file, containing:
            - file_id (int)
            - chirp_idx (int)
            - tightness (float): inverse pairwise-distance-based compactness
            - radius_mean (float): mean distance to cluster centroid
            - density (float): estimated hypersphere density of predictions
            - average_error_per_point (float): mean absolute error across ensemble
            - error_density (float): density of absolute error cluster
            - euclidean_distance (float): mean Euclidean error across ensemble

    Notes
    -----
    - Only numeric columns common to both prediction and truth files are
      used for error computation.
    - Cluster measures are computed using `compute_cluster_measures` on
      ensemble-stacked predictions at each chirp index.
    - Density is estimated using the volume of a hypersphere defined by
      the maximum centroid distance.
    """
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
    Plot selected uncertainty and error metrics across chirp indices for a given file.

    Parameters
    ----------
    file_id : int or str
        Identifier of the file whose chirp-level uncertainty should be plotted.
    df : pandas.DataFrame
        DataFrame containing at least the columns:
        ['file_id', 'chirp_idx'] and one or more uncertainty metrics such as
        'tightness', 'radius_mean', 'density',
        'average_error_per_point', 'error_density', 'euclidean_distance'.
    ax : matplotlib.axes.Axes, optional
        Existing axes object to draw on. If None, a new figure and axes
        are created.
    figsize : tuple of int, optional
        Size of the figure when creating a new one.
    metrics : str or list of str, optional
        Metric name(s) to plot. If a string is provided, it is converted
        to a single-element list.
    show : bool, optional
        If True, calls `plt.show()` after plotting.
    save_filename : str, optional
        If provided, saves the figure to '<save_filename>.png'.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the generated plot.
    """
    sel = df[df["file_id"] == file_id].copy()
    if sel.empty:
        raise ValueError(f"No rows found for file_id={file_id}")

    sel = sel.sort_values("chirp_idx")
    x = sel["chirp_idx"]
    # if chirp_idx are whole numbers, cast to int for nicer x ticks
    if np.allclose(x % 1, 0):
        x = x.astype(int)

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
    ax.set_ylabel("uncertainty")

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

    ax.set_title(f"uncertainty vs chirp_idx (file_id={file_id})")
    ax.grid(True, linestyle="--", alpha=0.6)

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
    """
    Plot a Gaussian-smoothed uncertainty measure across chirp indices
    for a specific file.

    Parameters
    ----------
    file_id : int or str
        Identifier of the file to visualize.
    df : pandas.DataFrame
        DataFrame containing 'file_id', 'chirp_idx', and the specified measure.
    measure : str
        Column name representing the uncertainty metric to smooth and plot.
    sigma : float, optional
        Standard deviation for Gaussian smoothing.
    ylim : tuple, optional
        y-axis limits as (min, max).
    show : bool, optional
        If True, displays the plot.

    Returns
    -------
    None
    """
    sel = df[df['file_id'] == file_id]
    smoothed = gaussian_filter(sel[measure], sigma=sigma, order=0, mode='reflect')
    if measure == "density":
        plt.plot(sel['chirp_idx'], -np.log(smoothed))
    else:
        plt.plot(sel["chirp_idx"], smoothed)
    plt.title(f"Uncertainty by chirp prediction: file={file_id}")
    plt.xlabel("Chirp Index")
    if measure == "density":
        plt.ylabel(f"-ln({measure if measure != 'radius_mean' else 'distance'})")
    else:
        plt.ylabel(f"{measure if measure != 'radius_mean' else 'distance'}")
    if ylim is not None:
        plt.ylim(ylim)
    if show:
        plt.show()

def find_ideal_cluster_k(data, min_k, max_k, plot=False):
    """
    Estimate the optimal number of clusters using multiple heuristics.

    This function applies:
    - Gap statistic
    - First positive gap-difference rule
    - Elbow method (via WCSS and knee detection)

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Feature matrix to cluster.
    min_k : int
        Minimum number of clusters to evaluate.
    max_k : int
        Maximum number of clusters to evaluate (exclusive upper bound).
    plot : bool, optional
        If True, visualizes gap statistic and elbow curves.

    Returns
    -------
    dict
        Dictionary containing:
        - 'gap_statistic'
        - 'diff_statistic'
        - 'elbow_method'
    """
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

    # Use the elbow method to find the number of clusters where the within-cluster sum of squares (WCSS) 
    # starts to level off
    # Use agglomerative clustering to compute WCSS from min_k to max_k
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
    """
    Cluster chirp feature representations using HDBSCAN or Agglomerative clustering.

    Parameters
    ----------
    data : array-like
        Feature matrix to cluster.
    clustering_method : {'HDBSCAN', 'Agglomerative'}
        Clustering algorithm to use.
    umap : bool
        Indicates whether the data has been UMAP-embedded (affects
        clustering hyperparameters).
    n_clusters : int
        Desired number of clusters (used for hierarchical extraction
        or agglomerative clustering).

    Returns
    -------
    numpy.ndarray
        Array of cluster labels for each sample.
    """
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
    Detect significant peaks in an uncertainty measure using
    DataSeriesAnalyzer.

    Parameters
    ----------
    file_id : int or str
        Identifier of the file to analyze.
    df : pandas.DataFrame
        DataFrame containing 'file_id', 'chirp_idx', and the specified measure.
    measure : str
        Column name representing the uncertainty metric.
    sigma : float, optional
        Smoothing parameter passed to the peak analyzer.

    Returns
    -------
    pandas.DataFrame
        DataFrame describing detected peaks, including frame index,
        prominence, and related statistics.
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
    return peaks_info

def get_surrounding_peak_sequence(peak_idx, df, context_size=5):
    """
    Retrieve indices surrounding a detected peak within the same file.

    Parameters
    ----------
    peak_idx : int
        Index of the peak row in the DataFrame.
    df : pandas.DataFrame
        DataFrame containing 'file_id' and 'chirp_idx'.
    context_size : int, optional
        Number of chirps to include before and after the peak.

    Returns
    -------
    list of int
        Original DataFrame indices representing the surrounding sequence.
    """

    peak_row = df.loc[peak_idx]
    file_id = peak_row['file_id']
    chirp_idx = peak_row['chirp_idx']
    
    sel = df[df['file_id'] == file_id].sort_values('chirp_idx').reset_index()
    peak_pos = sel[sel['chirp_idx'] == chirp_idx].index[0]

    start_pos = max(0, peak_pos - context_size)
    end_pos = min(len(sel) - 1, peak_pos + context_size)
    
    surrounding_indices = sel.loc[start_pos:end_pos, 'index'].tolist()
    return surrounding_indices

def identify_significant_peaks(idiom_df, row_idx, measure, range,
                               similarity_threshold=0.1,
                               std_threshold=None,
                               percentile_threshold=None,
                               verbose=0):
    """
    Identify statistically significant peaks in an uncertainty sequence
    relative to similar sequences.

    Parameters
    ----------
    idiom_df : pandas.DataFrame
        DataFrame containing uncertainty sequences and metadata.
    row_idx : int
        Index of the row to analyze.
    measure : str
        Column containing sequences of uncertainty values.
    range : str
        Column name representing overall sequence range for similarity filtering.
    similarity_threshold : float, optional
        Fractional tolerance for selecting similar sequences.
    std_threshold : float, optional
        Z-score threshold for significance detection.
    percentile_threshold : float, optional
        Percentile threshold for peak detection.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    list of int
        Indices corresponding to significant peak positions.
    """
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

def get_random_sequence(n, existing_sequences, unscaled_truth_df):
    """
    Retrieve a random contiguous chirp index sequence that does not
    overlap with previously selected sequences.

    Parameters
    ----------
    n : int
        Desired sequence length.
    existing_sequences : list of list of int
        Previously selected index sequences to avoid overlapping.
    unscaled_truth_df : pandas.DataFrame
        DataFrame containing chirp data with 'file_id'.

    Returns
    -------
    list of int
        Randomly selected sequence of indices.
    """
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
        
def get_random_idiom_sequence(n, idiom_df):
    """
    Retrieve a random contiguous subsequence from idiom-defined chirp sequences.

    Parameters
    ----------
    n : int
        Desired subsequence length.
    idiom_df : pandas.DataFrame
        DataFrame containing idiom sequences in column 'chirp_indices'
        and sequence lengths in column 'length'.

    Returns
    -------
    list
        Subsequence of chirp indices.
    """

    idiom_sequences = idiom_df[idiom_df['length'] >= n]['chirp_indices'].index
    if not idiom_sequences.any():
        raise ValueError("No idiom sequences of sufficient length found.")
    selected_idiom = idiom_df.iloc[np.random.choice(idiom_sequences)]['chirp_indices']
    start_idx = np.random.randint(0, len(selected_idiom) - n + 1)
    return selected_idiom[start_idx:start_idx + n]

def get_attributes_from_indices(indices, unscaled_truth_df):
    """
    Extract chirp attribute vectors for given indices.

    Parameters
    ----------
    indices : list of int
        DataFrame row indices.
    unscaled_truth_df : pandas.DataFrame
        DataFrame containing chirp attribute columns.

    Returns
    -------
    list of numpy.ndarray
        Attribute arrays corresponding to each index.
    """
    return [unscaled_truth_df.iloc[i]["PrecedingIntrvl":"AmpK@start"].values for i in indices]

def calculate_sequence_volume(attributes):
    """
    Compute the hyperspherical volume occupied by a set of attribute vectors.

    Parameters
    ----------
    attributes : array-like
        Collection of attribute vectors.

    Returns
    -------
    float
        Estimated volume of the minimal bounding hypersphere.
    """

    centroid = np.mean(attributes, axis=0)
    distances = np.linalg.norm(attributes - centroid, axis=1)
    radius = np.max(distances)
    dim = attributes[0].shape[0]
    volume = (np.pi ** (dim / 2)) / special.gamma((dim / 2) + 1) * (radius ** dim)
    return volume

def most_common_subsequences(sequences, length, subseq_type="all", k=None):
    """
    Identify the most frequent subsequences of specified length.

    Parameters
    ----------
    sequences : list of sequences
        Collection of sequences to analyze.
    length : int
        Length of subsequences to count.
    subseq_type : {'all', 'prefix', 'suffix'}, optional
        Whether to count all subsequences, only prefixes, or only suffixes.
    k : int, optional
        Number of top results to return.

    Returns
    -------
    list of tuple
        List of (subsequence, count) pairs sorted by frequency.
    """
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
                
    return subseq_count.most_common(k)

def get_smoothed_uncertainty_sequence(df, peak_idx, measure, sigma=1):
    """
    Return a smoothed uncertainty sequence for the file containing a peak.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing uncertainty measures.
    peak_idx : int
        Index of a peak row.
    measure : str
        Column name of uncertainty measure.
    sigma : float, optional
        Gaussian smoothing parameter.

    Returns
    -------
    list of float
        Smoothed uncertainty values.
    """
    # Find the start and end indices of the sequence that contains the peak
    file_id = df.iloc[peak_idx]["file_id"]
    chirp_idx = df.iloc[peak_idx]["chirp_idx"]
    
    # Find all rows in the same file and chirp
    sequence_rows = df[(df["file_id"] == file_id)]
    # Apply Gaussian smoothing to the MEASURE column of these rows
    sequence_rows[measure] = gaussian_filter(sequence_rows[measure].values, sigma=sigma, order=0, mode='reflect')
    
    # Return the smoothed uncertainty values for this sequence
    return sequence_rows[measure].tolist()

def calculate_height(df, peak_idx):
    """
    Calculate the prominence-like height of a detected peak relative to
    surrounding troughs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'chirp_idx' and 'smoothed_uncertainty'.
    peak_idx : int
        Index of the peak row.

    Returns
    -------
    float
        Height difference between peak and nearest trough.
    """

    adjusted_idx = df.loc[peak_idx]["chirp_idx"] - 4
    sequence = df.loc[peak_idx]["smoothed_uncertainty"]
    peak_value = sequence[adjusted_idx]
    
    # Find the troughs on either side of the peak
    left_trough_idx = None
    right_trough_idx = None
    
    # Look for troughs to the left of the peak
    for i in range(adjusted_idx - 1, -1, -1):
        if sequence[i] > sequence[i + 1]:
            left_trough_idx = i + 1
            break
    
    # Look for troughs to the right of the peak
    for i in range(adjusted_idx + 1, len(sequence)):
        if sequence[i] > sequence[i - 1]:
            right_trough_idx = i - 1
            break
    
    if left_trough_idx is not None and right_trough_idx is not None:
        left_trough_uncertainty = sequence[left_trough_idx]
        right_trough_uncertainty = sequence[right_trough_idx]
        max_trough_uncertainty = max(left_trough_uncertainty, right_trough_uncertainty)
        return peak_value - max_trough_uncertainty
    elif left_trough_idx is not None:
        return peak_value - sequence[left_trough_idx]
    elif right_trough_idx is not None:
        return peak_value - sequence[right_trough_idx]
    else:
        return 0

def quantile_normalize(df, columns):
    """
    Apply quantile normalization to selected columns, mapping them to a
    standard normal distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to normalize.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized columns.
    """

    scaled_df = df.copy()
    scaler = QuantileTransformer(output_distribution='normal')     
    scaler.set_output(transform="pandas")
    scaler.fit(df[columns])
    scaled_df.loc[:, columns] = scaler.transform(df.loc[:, columns])
    return scaled_df

def describe_cluster(df, cluster_idx, normalize=False, ignore_columns=None):
    """
    Compute the average feature vector representing a specified cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing feature columns and 'cluster_idx'.
    cluster_idx : int
        Cluster identifier to summarize.
    normalize : bool, optional
        If True, applies quantile normalization before averaging.
    ignore_columns : list of str, optional
        Columns to exclude from averaging.

    Returns
    -------
    pandas.Series
        Mean feature values for the specified cluster.
    """
    cols_to_use = [col for col in df.columns if col not in ignore_columns]
    if normalize:
        df_to_describe = quantile_normalize(df, cols_to_use)
    else:
        df_to_describe = df
    cluster_data = df_to_describe[df_to_describe["cluster"] == cluster_idx]
    cluster_avg = cluster_data.loc[:, cols_to_use].mean(axis=0)
    return cluster_avg

def identify_significant_peaks_by_range(df, reference_df, measure, sigma=1):
    df["surrounding_seq_idx"] = df.index.map(lambda idx: get_surrounding_peak_sequence(idx, reference_df, context_size=4))
    df["surrounding_seq_uncertainty"] = df["surrounding_seq_idx"].map(
        lambda indices: reference_df.loc[indices, measure].values
    )
    df["surround_seq_uncertainty_smoothed"] = df["surrounding_seq_idx"].map(
        lambda indices: gaussian_filter(reference_df.loc[indices, measure].values, sigma=sigma, order=0, mode='reflect')
    )
    df["seq_uncertainty_min"] = df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.min())
    df["seq_uncertainty_max"] = df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.max())
    df["seq_uncertainty_mean"] = df["surround_seq_uncertainty_smoothed"].map(lambda arr: arr.mean())
    df["seq_uncertainty_range"] = df["seq_uncertainty_max"] - df["seq_uncertainty_min"]
    # check if the seq_uncertainty_max occurs at the peak's index, i.e., the index of the row in df
    df["peak_at_max"] = df.apply(
        lambda row: row['surround_seq_uncertainty_smoothed'][row['surrounding_seq_idx'].index(row.name)] == row['seq_uncertainty_max'],
        axis=1
    )
    df["normalized_surrounding_seq"] = df["surround_seq_uncertainty_smoothed"].map(
        lambda arr: (arr - arr.mean())
    )
    return df

def identify_significant_peaks_by_prominence(df, reference_df, measure, sigma=1):
    # create a row called "smoothed_uncertainty" that contains the smoothed uncertainty values for the entire sequence that the peak belongs to
    df["smoothed_uncertainty"] = df.apply(lambda row: get_smoothed_uncertainty_sequence(reference_df, row.name, measure, sigma), axis=1)
    df["seq_len"] = df["smoothed_uncertainty"].apply(lambda x: len(x) + 4)

    # add a new column to df called "height", the difference between the value of df["smoothed_uncertainty"]
    # at the peak's index and the higher of the two troughs on either side of the peak (the lowest value before the values start
    # increasing on either side of the peak)    
    df["peak_value"] = df.apply(lambda row: row["smoothed_uncertainty"][row["chirp_idx"] - 4], axis=1)
    df["height"] = df.apply(lambda row: calculate_height(df, row.name), axis=1)
    df["prominence"] = df.apply(lambda row: peak_prominences(row["smoothed_uncertainty"], [row["chirp_idx"] - 4])[0][0], axis=1)
    df["range"] = df["smoothed_uncertainty"].apply(lambda x: max(x) - min(x))
    df["height_to_range"] = df["height"] / df["range"]
    df["prominence_to_range"] = df["prominence"] / df["range"]
    return df

def compare_idiom_similarity_by_volume(idiom_sequences, ground_truth, sample_size, seq_length):
    idiom_df = pd.DataFrame({"chirp_indices": idiom_sequences})
    idiom_df["chirp_attributes"] = idiom_df["chirp_indices"].map(
        lambda indices: get_attributes_from_indices(indices, ground_truth)
    )
    idiom_df["length"] = idiom_df["chirp_indices"].map(lambda indices: len(indices))

    idiom_volumes = []
    for _ in range(sample_size):
        seq = get_random_idiom_sequence(seq_length, idiom_df)
        attributes = get_attributes_from_indices(seq, ground_truth)
        volume = calculate_sequence_volume(attributes)
        idiom_volumes.append(volume)
    idiom_volumes = np.array(idiom_volumes)

    # get a distribution of non-idiom volumes by randomly sampling sequences of chirps that do not overlap with idioms
    non_idiom_volumes = []
    for _ in range(sample_size):
        seq = get_random_sequence(seq_length, idiom_sequences, ground_truth)
        attributes = get_attributes_from_indices(seq, ground_truth)
        volume = calculate_sequence_volume(attributes)
        non_idiom_volumes.append(volume)
    non_idiom_volumes = np.array(non_idiom_volumes)

    if sample_size < 20:
        print("Idiom volumes:", idiom_volumes)
        print("Non-idiom volumes:", non_idiom_volumes)

    # scale down the volumes by 1e100 to avoid overflow in t-test
    idiom_volumes /= 1e100
    non_idiom_volumes /= 1e100

    # take the log of the volumes (NOT STATISTICALLY EQUIVALENT TO SCALING)
    # idiom_volumes = np.log(idiom_volumes + 1e-12)
    # non_idiom_volumes = np.log(non_idiom_volumes + 1e-12)

    if sample_size < 20:
        print("Scaled idiom volumes:", idiom_volumes)
        print("Scaled non-idiom volumes:", non_idiom_volumes)

    # run a statistical t-test to compare the two distributions
    t_stat, p_value = ttest_ind(idiom_volumes, non_idiom_volumes, equal_var=False, alternative='less')
    print(f"Idiom volumes (scaled): mean = {np.mean(idiom_volumes):.4e}, std = {np.std(idiom_volumes):.4e}")
    print(f"Non-idiom volumes (scaled): mean = {np.mean(non_idiom_volumes):.4e}, std = {np.std(non_idiom_volumes):.4e}")
    print(f"T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

def compare_idiom_similarity_by_pairwise_distance(idiom_sequences, ground_truth, sample_size, seq_length):
    idiom_df = pd.DataFrame({"chirp_indices": idiom_sequences})
    idiom_df["chirp_attributes"] = idiom_df["chirp_indices"].map(
        lambda indices: get_attributes_from_indices(indices, ground_truth)
    )
    idiom_df["length"] = idiom_df["chirp_indices"].map(lambda indices: len(indices))
    
    idiom_pairwise_distances = []
    non_idiom_pairwise_distances = []
    for _ in tqdm(range(sample_size)):
        seq = get_random_idiom_sequence(seq_length, idiom_df)
        attributes = get_attributes_from_indices(seq, ground_truth)
    # pick a random chirp in attributes and calculate distances to all other chirps
    ref_idx = np.random.randint(0, len(attributes))
    ref_chirp = attributes[ref_idx]
    for i, chirp in enumerate(attributes):
        if i == ref_idx:
            continue
        dist = euclidean(ref_chirp, chirp)
        idiom_pairwise_distances.append(dist)

    # pick SEQ_LENGTH - 1 random chirps not in the idiom sequence and calculate distances to the same ref_chirp
    non_idiom_indices = []
    for i in range(seq_length - 1):
        seq = get_random_sequence(1, idiom_sequences, ground_truth)
        non_idiom_indices.append(seq[0])

    non_idiom_attributes = get_attributes_from_indices(non_idiom_indices, ground_truth)
    for chirp in non_idiom_attributes:
        dist = euclidean(ref_chirp, chirp)
        non_idiom_pairwise_distances.append(dist)

    idiom_pairwise_distances = np.array(idiom_pairwise_distances)
    non_idiom_pairwise_distances = np.array(non_idiom_pairwise_distances)

    if sample_size <= 10:
        print("Idiom pairwise distances:", idiom_pairwise_distances)
        print("Non-idiom pairwise distances:", non_idiom_pairwise_distances)

    # run a statistical t-test to compare the two distributions
    t_stat, p_value = ttest_ind(idiom_pairwise_distances, non_idiom_pairwise_distances, equal_var=False, alternative='less')
    print(f"Idiom pairwise distances: mean = {np.mean(idiom_pairwise_distances):.4e}, std = {np.std(idiom_pairwise_distances):.4e}")
    print(f"Non-idiom pairwise distances: mean = {np.mean(non_idiom_pairwise_distances):.4e}, std = {np.std(non_idiom_pairwise_distances):.4e}")
    print(f"T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

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