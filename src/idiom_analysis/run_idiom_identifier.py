# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import os
import argparse
from collections import Counter
import matplotlib.pyplot as plt

from idiom_identifier import IdiomIdentifier, IdiomIdentifierVisualizer
from subseq_counter import SubseqCounter

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

def idiom_identifier_pipeline(idiom_identifier, results_path):
    print("Calculating ensemble measures...")
    idiom_identifier.calculate_prediction_measures()

    # This section performs peak detection on the uncertainty sequences generated in the previous section.
    print("Performing peak detection...")
    idiom_identifier.detect_peaks()

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
    return idiom_identifier

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
                                       dataset_path=dataset_path,
                                       measure=args.measure, low_conf_percentile=args.low_conf_percentile, 
                                       prediction_offset=args.prediction_offset, sigma=args.sigma,
                                       no_amp=args.no_amp, reduc_method=args.reduc_method, cluster_method=args.cluster_method,
                                       min_k=args.min_cluster_k, max_k = args.max_cluster_k, 
                                       calculate_k = args.calculate_k, k=args.cluster_k)
    idiom_identifier = idiom_identifier_pipeline(idiom_identifier, results_path)

    # Useful statistics on the data:
    prediction_ensemble_measures = idiom_identifier.prediction_ensemble_measures
    file_id_counts = Counter([file_id for (file_id, chirp_idx) in idiom_identifier.peak_files])
    files_with_multiple_peaks = [file_id for file_id, count in file_id_counts.items() if count >= 2]
    # prediction_ensemble_measures_multiple_peaks = prediction_ensemble_measures[prediction_ensemble_measures['file_id'].isin(files_with_multiple_peaks)]
    print("There are", prediction_ensemble_measures.shape[0], "total chirps")
    print("There are", len(files_with_multiple_peaks), "out of", len(file_id_counts), "files with 2+ peaks.")
    # print the number of rows in prediction_ensemble_measures where peak_detected is True
    print(prediction_ensemble_measures[prediction_ensemble_measures['peak_detected'] == True].shape[0], "chirps are peak detected")

    # FIGURES: 8.1 and 8.2
    # Identify the most common transitions
    subseq_counter = SubseqCounter(idiom_identifier.significant_idiom_sequence_clusters)
    most_common_transitions = subseq_counter.identify_most_common_subsequences(length=args.subseq_n, 
                                                                               k_most_common=args.subseq_k,
                                                                               calc_prob=args.subseq_calc_prob)
    print(most_common_transitions.iloc[:10, [0, 1, 3]])

    # ALL FIGURES:
    print("Generating figures...")
    visualizer = IdiomIdentifierVisualizer(idiom_identifier, results_path)
    visualizer.generate_figures()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze idioms in chirp data/predictions")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder containing the data to analyze",
                        default="/home/ayc227/bats/bats_transformer/outputs/2022_barn_2secs_myca_quantile_1_16")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset csv file containing the chirp attributes",
                        default="/home/ayc227/bats/bats_transformer/data/2022_barn_2secs_myca/splits")
    parser.add_argument("--results_folder", type=str, help="Path to the folder where results will be saved",
                        default="/home/ayc227/bats/src/idiom_analysis/analysis_results")
    IdiomIdentifier.add_cli(parser)
    SubseqCounter.add_cli(parser)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)