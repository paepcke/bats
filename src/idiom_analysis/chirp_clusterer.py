# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import umap
from analysis_utils import *  

class ChirpClusterer():
    """
    Cluster bat chirps based on extracted acoustic features.

    The ChirpClusterer performs preprocessing, dimensionality reduction,
    and clustering on a set of chirp feature vectors. It is designed to
    identify groups of similar chirps ("chirp types") by embedding the
    feature space into a lower-dimensional representation and applying
    hierarchical clustering.

    The class supports removing amplitude-related features, automatic
    cluster-number estimation using an elbow method, and clustering in
    either the original feature space or a reduced UMAP embedding.

    Attributes
    ----------
    chirp_data : pandas.DataFrame or numpy.ndarray
        Original matrix of chirp feature vectors where rows correspond
        to chirps and columns correspond to acoustic features.

    chirp_data_embedded : numpy.ndarray
        Two-dimensional embedding of the chirp features produced during
        preprocessing (typically via UMAP).

    chirp_data_embedded_df : pandas.DataFrame
        DataFrame representation of the embedded chirp data used for
        plotting and downstream analysis.

    chirp_labels : numpy.ndarray
        Cluster label assigned to each chirp after clustering.

    filtered_chirp_data_2d : numpy.ndarray
        Two-dimensional embedded coordinates excluding noise points.

    filtered_chirp_labels : numpy.ndarray
        Cluster labels corresponding to the filtered embedded chirps.

    linked : numpy.ndarray
        Linkage matrix representing the hierarchical clustering tree.

    """

    def __init__(self, no_amp=0, reduc_method="umap", cluster_method="Agglomerative", 
                 min_k = 1, max_k=50, calculate_k=True, k=8):
        """
        Initialize the ChirpClusterer.

        This class performs dimensionality reduction and clustering on bat chirp
        feature vectors in order to group similar chirps into clusters. It supports
        optional removal of amplitude-related features, UMAP embedding for
        visualization and clustering, and automatic selection of the number of
        clusters.

        Parameters
        ----------
        chirp_data : pandas.DataFrame or numpy.ndarray
            Matrix of chirp feature vectors where each row corresponds to a chirp
            and each column represents an extracted acoustic feature.

        no_amp : int, optional
            Controls removal of amplitude-related features:
            - 0: keep all amplitude features
            - 1: remove quartile amplitude features (Amp1stQrtl–Amp4thQrtl)
            - 2: remove all features containing "Amp" in their name.

        reduc_method: str, optional


        cluster_method : str, optional
            Name of the clustering algorithm used by the helper function
            `cluster_chirps` (e.g., "Agglomerative").

        min_k : int, optional
            Minimum number of clusters considered when searching for the optimal
            number of clusters.

        max_k : int, optional
            Maximum number of clusters considered when searching for the optimal
            number of clusters.

        calculate_k : bool, optional
            If True, automatically determine the ideal number of clusters using an
            elbow-based method. If False, use the provided `k`.

        k : int, optional
            Fixed number of clusters to use when `calculate_k` is False.
        """
        self.chirp_data = None
        self.chirp_attributes = None
        self.no_amp = no_amp
        self.reduc_method = reduc_method
        self.cluster_method = cluster_method
        self.min_k = min_k
        self.max_k = max_k
        self.calculate_k = calculate_k
        self.k = k
        print(f"""
Made a ChirpClusterer with:
no_amp {self.no_amp}
reduc_method {self.reduc_method}
min_k {self.min_k}
calculate_k {self.calculate_k}
""")

    def prepare_data(self):
        """
        Prepare chirp feature data for clustering.

        This method preprocesses the chirp feature matrix by optionally removing
        amplitude-related features depending on the `no_amp` setting. After feature
        selection, it performs dimensionality reduction using UMAP to embed the
        chirp data into a 2-dimensional space.

        The resulting embedding is stored both as a NumPy array and as a Pandas
        DataFrame for convenience in visualization and downstream analysis.

        Stores
        ------
        self.chirp_data_embedded : numpy.ndarray
            Two-dimensional UMAP embedding of the chirp feature matrix.

        self.chirp_data_embedded_df : pandas.DataFrame
            DataFrame containing the UMAP embedding with columns ['UMAP1', 'UMAP2'].
        """
        if self.no_amp == 1:
            amp_indices = [i - 1 for i, col in enumerate(self.chirp_attributes.columns) if col in ["Amp1stQrtl", "Amp2ndQrtl", "Amp3rdQrtl", "Amp4thQrtl"]]
            self.chirp_data = np.delete(self.chirp_data, amp_indices, axis=1)
        elif self.no_amp == 2:
            amp_indices = [i - 1 for i, col in enumerate(self.chirp_attributes.columns) if "Amp" in col]
            self.chirp_data = np.delete(self.chirp_data, amp_indices, axis=1)

        chirp_data_embedded = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        ).fit_transform(self.chirp_data)

        chirp_data_embedded_df = pd.DataFrame(chirp_data_embedded, columns=['UMAP1', 'UMAP2'])
        self.chirp_data_embedded = chirp_data_embedded
        self.chirp_data_embedded_df = chirp_data_embedded_df

    def cluster_data(self):
        """
        Cluster the prepared chirp embeddings.

        This method determines the number of clusters (either automatically via an
        elbow method or using a fixed value), then performs clustering on the chirp
        data. Agglomerative hierarchical clustering is used to construct a
        dendrogram, and clusters are extracted by cutting the tree at the specified
        number of clusters.

        If UMAP is enabled, clustering operates on the UMAP embedding; otherwise
        the original feature space is used, with PCA applied for 2-D visualization.

        Noise points (label == -1) are filtered out when generating the
        visualization-ready data.

        Returns
        -------
        linked : numpy.ndarray
            Linkage matrix representing the hierarchical clustering tree.

        chirp_data_embedded_df : pandas.DataFrame
            DataFrame containing the 2-D embedding used for visualization.

        chirp_labels : numpy.ndarray
            Cluster labels assigned to each chirp.

        filtered_chirp_data_2d : numpy.ndarray
            Two-dimensional chirp coordinates excluding noise points.

        filtered_chirp_labels : numpy.ndarray
            Cluster labels corresponding to the filtered chirp coordinates.

        Stores
        ------
        self.linked
            Hierarchical clustering linkage matrix.

        self.chirp_labels
            Cluster assignments for all chirps.

        self.filtered_chirp_data_2d
            2-D coordinates of chirps excluding noise points.

        self.filtered_chirp_labels
            Labels corresponding to the filtered chirp coordinates.
        """

        # First, we want to see how many clusters there should be.
        print("Determining ideal number of clusters...")
        # FIGURE: 5.1

        if self.calculate_k:
            n_clusters = find_ideal_cluster_k(self.chirp_data_embedded, self.min_k, self.max_k, plot=False)
        else:
            n_clusters = {"elbow_method": self.k}

        num_clusters = int(n_clusters["elbow_method"])

        cluster_data_input = self.chirp_data_embedded if self.reduc_method == "umap" else self.chirp_data
        chirp_labels = cluster_chirps(cluster_data_input, self.cluster_method, self.reduc_method == "umap", num_clusters)

        # This is perhaps a separate clustering method, but visualized using a dendrogram
        print("Calculating dendrogram...")
        # plot dendrogram of agglomerative clustering
        self.linked = linkage(self.chirp_data_embedded, 'ward')

        # Cut at specific height to get clusters
        dendrogram_clusters = fcluster(self.linked, t=num_clusters, criterion='maxclust') 
        chirp_labels = dendrogram_clusters

        chirp_data_2d = self.chirp_data_embedded
        if self.reduc_method == "pca": # use PCA instead of UMAP
            pca = PCA(n_components=2)
            chirp_data_2d = pca.fit_transform(self.chirp_data)
        # filter out noise points (label == -1)
        filtered_chirp_data_2d = chirp_data_2d[chirp_labels != -1]
        filtered_chirp_labels = chirp_labels[chirp_labels != -1]
        self.chirp_labels = chirp_labels
        self.filtered_chirp_data_2d = filtered_chirp_data_2d
        self.filtered_chirp_labels = filtered_chirp_labels
        return self.linked, self.chirp_data_embedded_df, chirp_labels, filtered_chirp_data_2d, filtered_chirp_labels

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--no_amp", type=int, default=0)
        parser.add_argument("--min_cluster_k", type=int, default=1)
        parser.add_argument("--max_cluster_k", type=int, default=50)
        parser.add_argument("--cluster_method", type=str, default="Agglomerative")
        parser.add_argument("--reduc_method", type=str, default="umap")
        parser.add_argument("--calculate_k", action="store_true")
        parser.add_argument("--cluster_k", type=int, default=8)