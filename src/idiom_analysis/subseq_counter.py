from collections import Counter
import pandas as pd    

from analysis_utils import *

class SubseqCounter():
    def __init__(self, sequences):
        self.sequences = sequences

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
        
        subseqs = pd.DataFrame(sequence_counts, columns=['Subseq', 'Count'])
        subseqs['Start Cluster Count'] = subseqs.apply(lambda x: idiom_cluster_counts[x['Subseq'][0]], axis=1)
        subseqs['Transition Probability'] = subseqs.apply(lambda x: x['Count'] / x["Start Cluster Count"], axis=1)
        subseqs = subseqs.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        subseqs.drop(columns=['Start Cluster Count'], inplace=True)
        subseqs["Transition Probability"] = subseqs["Transition Probability"].round(round_n)

        prefixs = pd.DataFrame(prefix_counts, columns=['Prefix', 'Count'])
        prefixs['Start Cluster Count'] = prefixs.apply(lambda x: prefix_cluster_counts[x['Prefix'][0]], axis=1)
        prefixs['Transition Probability'] = prefixs.apply(lambda x: x['Count'] / x["Start Cluster Count"], axis=1)
        prefixs = prefixs.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        prefixs.drop(columns=['Start Cluster Count'], inplace=True)
        prefixs["Transition Probability"] = prefixs["Transition Probability"].round(round_n)

        suffixs = pd.DataFrame(suffix_counts, columns=['Suffix', 'Count'])
        suffixs['Start Cluster Count'] = suffixs.apply(lambda x: suffix_cluster_counts[x['Suffix'][-1]], axis=1)
        suffixs['Transition Probability'] = suffixs.apply(lambda x: x['Count'] / x["Start Cluster Count"], axis=1)
        suffixs = suffixs.sort_values('Transition Probability', ascending=False).reset_index(drop=True)
        suffixs.drop(columns=['Start Cluster Count'], inplace=True)
        suffixs["Transition Probability"] = suffixs["Transition Probability"].round(round_n)

        res = pd.concat([subseqs, prefixs, suffixs], axis=1)
        res.insert(0, "Rank", range(1, len(res) + 1))

        return res

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--subseq_n", type=int, default=2)
        parser.add_argument("--subseq_k", type=int, default=None)
        parser.add_argument("--subseq_calc_prob", type=int, default=None)