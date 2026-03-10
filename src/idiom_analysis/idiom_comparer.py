# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import sys
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
from chirp_clusterer import ChirpClusterer

class IdiomComparer():
    """
    Compare idiom structures between two experimental datasets.

    The IdiomComparer class loads chirp feature data and idiom boundary
    annotations from two experiment result directories, extracts chirps
    belonging to idioms, performs clustering on the chirp features, and
    computes statistics describing idiom structure.

    It is primarily used to analyze differences in idiom composition and
    transition patterns between two experimental conditions.

    Attributes
    ----------
    results_1 : str
        Path to the first experiment's analysis results directory.
    results_2 : str
        Path to the second experiment's analysis results directory.
    exp_1_name : str, optional
        Name used to label the first experiment in plots and outputs.
        Default is "exp_1".
    exp_2_name : str, optional
        Name used to label the second experiment in plots and outputs.
        Default is "exp_2".
    clusterer : ChirpClusterer
        Object to cluster chirps.
    combined_chirp_attributes : pandas.DataFrame
        Combined chirp feature dataset from both experiments.
    idiom_boundaries_attributes : pandas.DataFrame
        Idiom boundary annotations from both experiments.
    idiom_chirp_attributes : pandas.DataFrame
        Subset of chirp attributes corresponding only to chirps within idioms.
    idiom_chirp_attributes_embedded : ndarray
        Low-dimensional embedding of idiom chirp features used for visualization.
    chirp_labels : ndarray
        Cluster labels assigned to each idiom chirp.
    idiom_label_sequences : pandas.DataFrame
        Sequences of cluster labels representing idioms for each file.
    """
    def __init__(self, results_1, results_2, exp_1_name="exp_1", exp_2_name="exp_2", 
                 **kwargs):
        """
        Initialize the IdiomComparer

        Parameters
        ----------
        results_1 : str, optional
            Path to first set of prediction files produced by model inference.

        results_2 : str, optional
            Path to second set of prediction files produced by model inference.

        exp_1_name : str, optional
            Name of dataset corresponding to first set of predictions.

        exp_2_name : str, optional
            Name of dataset corresponding to second set of predictions.
        """
        self.results_1 = results_1
        self.results_2 = results_2
        self.exp_1_name = exp_1_name
        self.exp_2_name = exp_2_name
        self.clusterer = ChirpClusterer(**kwargs)

    def combine_inputs(self):
        """
        Load and merge chirp attributes, idiom boundaries, and confidence measures
        from the two experiment result directories.

        The datasets are concatenated and a column (`original_df`) is added to
        indicate which experiment each row originated from.

        Returns
        -------
        tuple
            A tuple containing:

            combined_chirp_attributes : pandas.DataFrame
                Concatenated chirp feature data from both experiments.

            idiom_boundaries_attributes : pandas.DataFrame
                Concatenated idiom boundary annotations.

            confidence_measures_attributes : pandas.DataFrame
                Combined chirp prediction confidence measures.
        """
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
        """
        Extract chirp feature rows corresponding to idiom segments.

        Idiom boundaries specify start and end chirp indices for each idiom.
        This method retrieves the chirp feature rows within those ranges
        and constructs a new dataframe containing only chirps that occur
        within idioms.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing chirp attributes for chirps that belong to
            detected idioms.
        """
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
        """
        Perform clustering on idiom chirp features.

        Chirp features are extracted from the idiom chirp dataset and passed
        to a clustering pipeline (`ChirpClusterer`). The clustering process
        produces cluster labels and a low-dimensional embedding of the chirp
        features for visualization.

        The method also constructs sequences of cluster labels for each idiom,
        enabling downstream analysis of idiom structure and transitions.

        Returns
        -------
        tuple
            idiom_chirp_attributes_embedded : ndarray
                Embedded representation of idiom chirp features.

            chirp_labels : ndarray
                Cluster labels assigned to each idiom chirp.

            idiom_label_sequences : pandas.DataFrame
                Dataframe containing sequences of cluster labels for each idiom.
        """
        idiom_chirp_data = self.idiom_chirp_attributes.loc[:, 'PrecedingIntrvl':'AmpK@start'].to_numpy()
        self.clusterer.chirp_attributes = self.idiom_chirp_attributes
        self.clusterer.chirp_data = idiom_chirp_data
        self.clusterer.prepare_data()
        linked, idiom_chirp_attributes_embedded_df, chirp_labels, filtered_chirp_data_2d, filtered_chirp_labels = self.clusterer.cluster_data()
        
        self.linked = linked
        idiom_chirp_attributes_embedded = idiom_chirp_attributes_embedded_df.to_numpy()
        self.idiom_chirp_attributes_embedded = idiom_chirp_attributes_embedded
        self.chirp_labels = chirp_labels
        self.filtered_chirp_data_2d = filtered_chirp_data_2d
        self.filtered_chirp_labels = filtered_chirp_labels

        idiom_label_sequences = self.idiom_chirp_attributes.groupby(["file_id", "original_df"]).agg(list)["cluster"].reset_index()
        self.idiom_label_sequences = idiom_label_sequences
        return idiom_chirp_attributes_embedded, chirp_labels, idiom_label_sequences

    @classmethod
    def add_cli(cls, parser):
        """
        Add command-line arguments for subsequence analysis.

        This method registers CLI arguments used to configure subsequence
        analysis when running scripts from the command line.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser to which the subsequence analysis options
            will be added.
        """
        ChirpClusterer.add_cli(parser)



class IdiomComparerVisualizer():
    """
    Generate visualizations for idiom clustering comparisons between experiments.

    This class produces a set of figures that visualize the distribution of
    chirps and idioms across two experiments. It relies on a fully processed
    `IdiomComparer` object containing combined chirp attributes, idiom
    embeddings, and cluster assignments.

    The visualizations include:

    - Scatterplots of all chirps from both experiments
    - Scatterplots of idiom chirps only
    - Cluster-colored idiom embeddings
    - Per-experiment cluster visualizations with convex hull boundaries
    - Histograms showing cluster membership proportions

    Parameters
    ----------
    idiom_comparer : IdiomComparer
        A processed IdiomComparer instance containing extracted idioms,
        embeddings, and cluster labels.
    results_path : str
        Path to the output directory where generated figures will be saved.

    Attributes
    ----------
    idiom_comparer : IdiomComparer
        Reference to the IdiomComparer object containing the analysis data.
    results_path : str
        Directory where visualization outputs are written.
    num_clusters : int
        Number of clusters detected in the idiom chirp dataset.
    cmap1 : str
        Primary colormap used for experiment 1 visualizations.
    cmap2 : matplotlib.colors.ListedColormap
        Darkened variant of the primary colormap used for experiment 2.
    """
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
        """
        Generate all comparison figures.

        This method runs the full visualization pipeline, producing several
        scatterplots and summary plots that illustrate how chirps and idioms
        are distributed across clusters and experiments.

        The generated figures include:

        1. Scatterplot of all chirps from both experiments
        2. Scatterplot of idiom-only chirps
        3. Cluster-colored scatterplot of idiom embeddings
        4. Separate cluster scatterplots for each experiment
        5. Histogram of cluster membership proportions

        All figures are saved to the ``figs`` subdirectory of ``results_path``.
        """
        self.plot_all_chirp_scatter()
        self.plot_idiom_chirp_scatter()
        self.plot_idiom_cluster_scatter()
        self.plot_idiom_cluster_indiv_scatter()
        self.plot_clusters_histogram()

    def plot_all_chirp_scatter(self):
        """
        Plot a 2-D embedding of all chirps from both experiments.

        Chirp feature vectors are embedded into two dimensions using UMAP
        and plotted as a scatterplot. Points are colored according to the
        experiment they originated from.

        This visualization provides an overview of how chirp distributions
        from the two experiments overlap or separate in feature space.

        The resulting figure is saved as:

        ``scatter_chirps.png``
        """
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
        """
        Plot a 2-D embedding of idiom chirps only.

        This scatterplot shows the embedded representation of chirps that
        belong to detected idioms. Points are colored by experiment to
        visualize whether idiom-related chirps differ between datasets.

        The embedding is produced during the clustering stage of the
        IdiomComparer pipeline.

        The resulting figure is saved as:

        ``scatter_idiom_chirps.png``
        """
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
        """
        Plot clustered idiom chirps for both experiments in a shared embedding.

        Idiom chirps are displayed in a 2-D embedding with colors indicating
        cluster membership. Different marker styles distinguish chirps from
        the two experiments.

        This plot highlights:

        - The spatial separation of clusters
        - How cluster membership differs between experiments
        - The relative distribution of idiom chirps in feature space

        The resulting figure is saved as:

        ``scatter_idiom_chirps_clusters.png``
        """
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
        """
        Plot cluster distributions separately for each experiment.

        For each experiment, idiom chirps are plotted in the embedded space
        and colored by cluster label. Convex hulls are drawn around clusters
        when sufficient points are present, highlighting the spatial extent
        of each cluster.

        This visualization allows direct comparison of cluster structure
        within each experiment independently.

        Two figures are produced:

        - ``scatter_idiom_chirps_clusters_<exp_1_name>.png``
        - ``scatter_idiom_chirps_clusters_<exp_2_name>.png``
        """
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
        """
        Plot the distribution of cluster membership across experiments.

        This histogram compares the relative frequency of cluster labels
        among idiom chirps from the two experiments. The distributions are
        normalized to show proportions rather than raw counts.

        The plot helps reveal whether certain chirp clusters occur more
        frequently in one experiment than the other.

        The resulting figure is saved as:

        ``histogram_clusters_<exp_1_name>_<exp_2_name>.png``
        """
        # Figure 5: Histogram of clusters, by experiment
        # plot a histogram of cluster indices across the two experiments
        first_exp_clusters = self.idiom_comparer.idiom_chirp_attributes.loc[
            self.idiom_comparer.idiom_chirp_attributes["original_df"] == 1]['cluster'].values
        second_exp_clusters = self.idiom_comparer.idiom_chirp_attributes.loc[
            self.idiom_comparer.idiom_chirp_attributes["original_df"] == 2]['cluster'].values
        plt.cla()
        plt.figure(figsize=(8, 6))
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