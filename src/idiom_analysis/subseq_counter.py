# -*- coding: utf-8 -*-
# @Author: Andrew Chen

from collections import Counter
import pandas as pd    

from analysis_utils import IdiomUtils

class SubseqCounter():
    """
    Analyze and compare common subsequences within idiom cluster sequences.

    This class provides utilities for identifying frequently occurring
    subsequences (e.g., cluster transitions) within sequences of cluster
    labels representing idioms. It can compute the most common subsequences
    within a single set of sequences or compare subsequence statistics
    between two experimental datasets.

    Transition probabilities are estimated by normalizing subsequence counts
    by the total number of occurrences of the starting cluster in the
    corresponding sequences.

    Parameters
    ----------
    sequences : list[list[int]] or pandas.DataFrame
        Collection of cluster label sequences representing idioms.
        For comparison tasks this is typically a DataFrame containing
        columns such as ``cluster`` (the sequence) and ``original_df``
        indicating the experiment source.

    exp_1_name : str, optional
        Label used to identify the first experiment in comparison outputs.

    exp_2_name : str, optional
        Label used to identify the second experiment in comparison outputs.

    Attributes
    ----------
    sequences : list[list[int]] or pandas.DataFrame
        Input idiom cluster sequences.

    exp_1_name : str or None
        Name of the first experiment.

    exp_2_name : str or None
        Name of the second experiment.
    """
    def __init__(self, sequences, exp_1_name=None, exp_2_name=None):
        """
        Initialize the SubseqCounter.

        Parameters
        ----------
        sequences : list[list[int]] or pandas.DataFrame
            Collection of cluster label sequences representing idioms.

        exp_1_name : str, optional
            Name of the first experiment for comparison tasks.

        exp_2_name : str, optional
            Name of the second experiment for comparison tasks.
        """
        self.sequences = sequences
        self.exp_1_name = exp_1_name
        self.exp_2_name = exp_2_name

    def _format_subseq_df(self, sequence_counts, col_name, base_rate_fn, round_n):
        """
        Format subsequence counts and compute transition probabilities.

        This helper method converts a list of subsequences and their counts
        into a structured DataFrame. It also computes transition probabilities
        by dividing the subsequence count by the total number of occurrences
        of the subsequence's starting cluster.

        Parameters
        ----------
        sequence_counts : list[tuple]
            List of subsequences and their counts, typically returned by
            ``most_common_subsequences``.

        col_name : str
            Name of the column containing the subsequence values
            (e.g., "Subseq", "Prefix", or "Suffix").

        base_rate_fn : callable
            Function used to determine the total count of the starting cluster
            for computing transition probabilities.

        round_n : int
            Number of decimal places used when rounding the transition
            probability values.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing subsequences, counts, and transition
            probabilities sorted by probability.
        """
        subseqs = pd.DataFrame(sequence_counts, columns=[col_name, 'Count'])
        subseqs['Start Cluster Count'] = subseqs.apply(base_rate_fn, axis=1)
        subseqs['Transition Probability'] = subseqs.apply(lambda x: x['Count'] / x["Start Cluster Count"], axis=1)
        subseqs = subseqs.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        subseqs.drop(columns=['Start Cluster Count'], inplace=True)
        subseqs["Transition Probability"] = subseqs["Transition Probability"].round(round_n)
        return subseqs

    def identify_most_common_subsequences(self, length=2, k_most_common=None, calc_prob=True, round_n=3):
        """
        Identify the most common subsequences within idiom cluster sequences.

        This method analyzes the cluster label sequences of idioms and counts
        frequently occurring subsequences of a specified length. Subsequence
        statistics are calculated for three categories:

        - All subsequences within idiom sequences
        - Prefix subsequences (beginning of idioms)
        - Suffix subsequences (end of idioms)

        Transition probabilities are computed by normalizing subsequence
        counts by the total occurrences of the starting cluster.

        Parameters
        ----------
        length : int, default=2
            Length of the subsequences to analyze.

        k_most_common : int or None, optional
            Number of most frequent subsequences to return. If None,
            all subsequences are included.

        calc_prob : bool, default=True
            Whether to compute transition probabilities.

        round_n : int, default=3
            Number of decimal places used when rounding probabilities.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing ranked subsequences along with their counts
            and transition probabilities for all, prefix, and suffix patterns.
        """
        idiom_clusters_temp = []
        prefix_clusters_temp = []
        suffix_clusters_temp = []
        for seq in self.sequences:
            idiom_clusters_temp.extend(seq)
            prefix_clusters_temp.append(seq[0])
            suffix_clusters_temp.append(seq[-1])
        idiom_cluster_counts = Counter(idiom_clusters_temp)
        prefix_cluster_counts = Counter(prefix_clusters_temp)
        suffix_cluster_counts = Counter(suffix_clusters_temp)

        # find the most common 3-length label sequences in significant_idiom_sequence_clusters
        sequence_counts = IdiomUtils.most_common_subsequences(self.sequences, 
                                                   length, k = k_most_common, subseq_type="all")
        prefix_counts = IdiomUtils.most_common_subsequences(self.sequences, 
                                                 length, k = k_most_common, subseq_type="prefix")
        suffix_counts = IdiomUtils.most_common_subsequences(self.sequences, 
                                                 length, k = k_most_common, subseq_type="suffix")
        
        subseqs = self._format_subseq_df(sequence_counts, "Subseq", lambda x: idiom_cluster_counts[x['Subseq'][0]], round_n)
        prefixs = self._format_subseq_df(prefix_counts, "Prefix", lambda x: prefix_cluster_counts[x['Prefix'][0]], round_n)
        suffixs = self._format_subseq_df(suffix_counts, "Suffix", lambda x: suffix_cluster_counts[x['Suffix'][-1]], round_n)

        res = pd.concat([subseqs, prefixs, suffixs], axis=1)
        res.insert(0, "Rank", range(1, len(res) + 1))

        return res

    def compare_most_common_subsequences(self, length=2, k_most_common=None, subseq_type="all", calc_prob=True, round_n=3):
        """
        Compare the most common subsequences between two experiments.

        This method identifies frequently occurring cluster subsequences
        within idioms and compares their statistics across two datasets.
        Subsequence counts and transition probabilities are computed for:

        - all idioms combined
        - idioms from experiment 1
        - idioms from experiment 2

        The resulting table allows direct comparison of cluster transition
        patterns between experiments.

        Parameters
        ----------
        length : int, default=2
            Length of the subsequences to analyze.

        k_most_common : int or None, optional
            Number of most common subsequences to include. If None,
            all subsequences are returned.

        subseq_type : {"all", "prefix", "suffix"}, default="all"
            Type of subsequences to extract.

        calc_prob : bool, default=True
            Whether to compute transition probabilities.

        round_n : int, default=3
            Number of decimal places used when rounding probabilities.

        Returns
        -------
        pandas.DataFrame
            A formatted table containing ranked subsequences and their
            counts and transition probabilities for the combined dataset
            and each individual experiment.
        """
        idiom_clusters_temp = []
        idiom_clusters_temp_1 = []
        idiom_clusters_temp_2 = []
        for idx, row in self.sequences.iterrows():
            idiom_clusters_temp.extend(row['cluster'])
            if row["original_df"] == 1:
                idiom_clusters_temp_1.extend(row['cluster'])
            else:
                idiom_clusters_temp_2.extend(row['cluster'])
        idiom_cluster_counts_all = Counter(idiom_clusters_temp)
        idiom_cluster_counts_1 = Counter(idiom_clusters_temp_1)
        idiom_cluster_counts_2 = Counter(idiom_clusters_temp_2)

        # find most common transitions
        subseq_all = IdiomUtils.most_common_subsequences(self.sequences['cluster'].values, 
                                length, 
                                subseq_type=subseq_type, 
                                k=k_most_common)
        subseq_exp1 = IdiomUtils.most_common_subsequences(self.sequences.loc[self.sequences["original_df"] == 1]['cluster'].values, 
                                length, 
                                subseq_type=subseq_type, 
                                k=k_most_common)
        subseq_exp2 = IdiomUtils.most_common_subsequences(self.sequences.loc[self.sequences["original_df"] == 2]['cluster'].values, 
                                length, 
                                subseq_type=subseq_type, 
                                k=k_most_common)

        subseq_all = self._format_subseq_df(subseq_all, "Subseq", lambda x: idiom_cluster_counts_all[x['Subseq'][0]], round_n)
        subseq_exp1 = self._format_subseq_df(subseq_exp1, "Subseq", lambda x: idiom_cluster_counts_1[x['Subseq'][0]], round_n)
        subseq_exp2 = self._format_subseq_df(subseq_exp2, "Subseq", lambda x: idiom_cluster_counts_2[x['Subseq'][0]], round_n)
        
        res = pd.concat([subseq_all, subseq_exp1, subseq_exp2], axis=1)
        res.insert(0, "Rank", range(1, len(res) + 1))
        header=[['','All','','',f'{self.exp_1_name}','','',f'{self.exp_2_name}','',''], 
                ['Rank','Subseq','Count','Proportion','Subseq','Count','Proportion','Subseq','Count','Proportion']]
        res.columns=header
        return res

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
        parser.add_argument("--subseq_n", type=int, default=2, help="Length of subsequences to count")
        parser.add_argument("--subseq_k", type=int, default=None, help="Number of most common subsequences to output")
        parser.add_argument("--subseq_type", type=str, default="all", 
                            help="What type of subsequence: [\"all\", \"prefix\", \"suffix\"]")
        parser.add_argument("--subseq_calc_prob", type=int, default=None, 
                            help="Whether to calculate probabilities (in addition to absolute counts)")