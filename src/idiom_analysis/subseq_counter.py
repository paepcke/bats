from collections import Counter
import pandas as pd    

from analysis_utils import *

class SubseqCounter():
    def __init__(self, sequences, exp_1_name=None, exp_2_name=None):
        self.sequences = sequences
        self.exp_1_name = exp_1_name
        self.exp_2_name = exp_2_name

    def _format_subseq_df(self, sequence_counts, col_name, base_rate_fn, round_n):
        subseqs = pd.DataFrame(sequence_counts, columns=[col_name, 'Count'])
        subseqs['Start Cluster Count'] = subseqs.apply(base_rate_fn, axis=1)
        subseqs['Transition Probability'] = subseqs.apply(lambda x: x['Count'] / x["Start Cluster Count"], axis=1)
        subseqs = subseqs.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        subseqs.drop(columns=['Start Cluster Count'], inplace=True)
        subseqs["Transition Probability"] = subseqs["Transition Probability"].round(round_n)
        return subseqs

    def identify_most_common_subsequences(self, length=2, k_most_common=None, calc_prob=True, round_n=3):
        """
        Identify common cluster transition patterns within idiom sequences.

        This method analyzes the cluster label sequences of significant idioms
        and counts the most frequent subsequences of a fixed length. These
        subsequences represent recurring patterns of chirp types that may
        correspond to common behavioral motifs.

        Returns
        -------
        list[tuple]
            The most common cluster subsequences and their frequencies.
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
        sequence_counts = most_common_subsequences(self.sequences, 
                                                   length, k = k_most_common, subseq_type="all")
        prefix_counts = most_common_subsequences(self.sequences, 
                                                 length, k = k_most_common, subseq_type="prefix")
        suffix_counts = most_common_subsequences(self.sequences, 
                                                 length, k = k_most_common, subseq_type="suffix")
        
        subseqs = self._format_subseq_df(sequence_counts, "Subseq", lambda x: idiom_cluster_counts[x['Subseq'][0]], round_n)
        prefixs = self._format_subseq_df(prefix_counts, "Prefix", lambda x: prefix_cluster_counts[x['Prefix'][0]], round_n)
        suffixs = self._format_subseq_df(suffix_counts, "Suffix", lambda x: suffix_cluster_counts[x['Suffix'][-1]], round_n)

        res = pd.concat([subseqs, prefixs, suffixs], axis=1)
        res.insert(0, "Rank", range(1, len(res) + 1))

        return res

    def compare_most_common_subsequences(self, length=2, k_most_common=None, subseq_type="all", calc_prob=True, round_n=3):
        """
        Identify the most common cluster-to-cluster transitions within idioms.

        Transition subsequences of length two are extracted from idiom cluster
        sequences and counted to determine how frequently one cluster is
        followed by another. Transition probabilities are also calculated by
        normalizing counts by the total occurrences of the starting cluster.

        Statistics are reported for:

        - all idioms combined
        - experiment 1
        - experiment 2

        Parameters
        ----------
        idiom_label_sequences : pandas.DataFrame
            Dataframe containing idiom cluster label sequences grouped by file.
        length: int
            number of chirp in each subsequence
            
        Returns
        -------
        pandas.DataFrame
            A formatted dataframe showing the most common cluster subsequences
            and their estimated transition probabilities.
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
        subseq_all = most_common_subsequences(self.sequences['cluster'].values, 
                                length, 
                                subseq_type=subseq_type, 
                                k=k_most_common)
        subseq_exp1 = most_common_subsequences(self.sequences.loc[self.sequences["original_df"] == 1]['cluster'].values, 
                                length, 
                                subseq_type=subseq_type, 
                                k=k_most_common)
        subseq_exp2 = most_common_subsequences(self.sequences.loc[self.sequences["original_df"] == 2]['cluster'].values, 
                                length, 
                                subseq_type=subseq_type, 
                                k=k_most_common)

        subseq_all = self._format_subseq_df(subseq_all, "Subseq", lambda x: idiom_cluster_counts_all[x['Subseq'][0]], round_n)
        # subseq_all = pd.DataFrame(subseq_all, columns=[f'{"Subseq" if subseq_type == "all" else subseq_type} (all)', 'Count'])
        # subseq_all_proportion = subseq_all.copy()
        # subseq_all_proportion['Start Cluster Count'] = subseq_all_proportion.apply(lambda x: idiom_cluster_counts_all[x['Subseq (all)'][0]], axis=1)
        # subseq_all_proportion['Transition Probability'] = subseq_all_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_all[x['Subseq (all)'][0]], axis=1)
        # subseq_all_proportion.drop(['Start Cluster Count'], axis=1, inplace=True)
        # subseq_all_proportion = subseq_all_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        # subseq_all_proportion["Transition Probability"] = subseq_all_proportion["Transition Probability"].round(round_n)

        subseq_exp1 = self._format_subseq_df(subseq_exp1, "Subseq", lambda x: idiom_cluster_counts_1[x['Subseq'][0]], round_n)
        # subseq_exp1 = pd.DataFrame(subseq_exp1, columns=[f'{"Subseq" if subseq_type == "all" else subseq_type} ({self.exp_1_name})', 'Count'])
        # subseq_exp1_proportion = subseq_exp1.copy()
        # subseq_exp1_proportion['Start Cluster Count'] = subseq_exp1_proportion.apply(lambda x: idiom_cluster_counts_1[x[f'Subseq ({self.exp_1_name})'][0]], axis=1)
        # subseq_exp1_proportion['Transition Probability'] = subseq_exp1_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_1[x[f'Subseq ({self.exp_1_name})'][0]], axis=1)
        # subseq_exp1_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
        # subseq_exp1_proportion = subseq_exp1_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        # subseq_exp1_proportion["Transition Probability"] = subseq_exp1_proportion["Transition Probability"].round(round_n)

        subseq_exp2 = self._format_subseq_df(subseq_exp2, "Subseq", lambda x: idiom_cluster_counts_2[x['Subseq'][0]], round_n)
        # subseq_exp2 = pd.DataFrame(subseq_exp2, columns=[f'{"Subseq" if subseq_type == "all" else subseq_type} ({self.exp_2_name})', 'Count'])
        # subseq_exp2_proportion = subseq_exp2.copy()
        # subseq_exp2_proportion['Start Cluster Count'] = subseq_exp2_proportion.apply(lambda x: idiom_cluster_counts_2[x[f'Subseq ({self.exp_2_name})'][0]], axis=1)
        # subseq_exp2_proportion['Transition Probability'] = subseq_exp2_proportion.apply(lambda x: x['Count'] / idiom_cluster_counts_2[x[f'Subseq ({self.exp_2_name})'][0]], axis=1)
        # subseq_exp2_proportion.drop(["Count", 'Start Cluster Count'], axis=1, inplace=True)
        # subseq_exp2_proportion = subseq_exp2_proportion.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        # subseq_exp2_proportion["Transition Probability"] = subseq_exp2_proportion["Transition Probability"].round(round_n)

        res = pd.concat([subseq_all, subseq_exp1, subseq_exp2], axis=1)
        # res.insert(0, "Rank", range(1, len(res) + 1))
        # res_proportion = pd.concat([subseq_all_proportion, subseq_exp1_proportion, subseq_exp2_proportion], axis=1)
        res.insert(0, "Rank", range(1, len(res) + 1))
        # res_proportion.rename(columns = {f'Subseq ({self.exp_1_name})': f'{self.exp_1_name} Subseq', 
        #                                  f'Subseq ({self.exp_2_name})': f'{self.exp_2_name} Subseq'}, inplace=True)
        header=[['','All','','',f'{self.exp_1_name}','','',f'{self.exp_2_name}','',''], 
                ['Rank','Subseq','Count','Proportion','Subseq','Count','Proportion','Subseq','Count','Proportion']]
        res.columns=header
        return res

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--subseq_n", type=int, default=2)
        parser.add_argument("--subseq_k", type=int, default=None)
        parser.add_argument("--subseq_calc_prob", type=int, default=None)