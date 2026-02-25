#!/usr/bin/env bash

# Idiom-internal chirps:
   src/sonobat_utils/data_selection.py \
   $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_augmented.csv \
   --population_type idiom-internal \
   --rand_selector all \
   --outfile $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_idiom_internal.csv

   # -> 645 rows

# Idiom-starts chirps:
   src/sonobat_utils/data_selection.py \
   $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures.csv \
   --population_type idiom-starts \
   --rand_selector all \
   --outfile $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_idiom_starts.csv

   # -> 144 rows

# Idiom-ends chirps:
   src/sonobat_utils/data_selection.py \
   $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_augmented.csv \
   --population_type idiom-ends \
   --rand_selector all \
   --outfile $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_idiom_ends.csv

   # -> 144 rows

# Idiom-internal chirps:
   src/sonobat_utils/data_selection.py \
   $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_augmented.csv \
   --population_type idiom-internal \
   --rand_selector all \
   --outfile $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_idiom_internal.csv \

   # -> 645 rows

# Non-Idiom chirps: as many as there are IDIOM_STARTS:
   src/sonobat_utils/data_selection.py \
   $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_augmented.csv \
   --population_type non-idiom-random \
   --rand_selector match-idiom-start \
   --outfile $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_match_idiom_start_pop.csv

   # -> 144 rows

# Non-Idiom chirps: as many as there are IN_IDIOM:
   src/sonobat_utils/data_selection.py \
   $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_augmented.csv \
   --population_type non-idiom-random \
   --rand_selector match-in-idiom \
   --outfile $HOME/VSCodeWorkspaces/bats/src/result_analysis/data/andrewChen/analysis_results/2022_barn_2secs_myca_quantile_1_16/all_chirp_measures_match_in_idiom_pop.csv

   # -> 931 rows

