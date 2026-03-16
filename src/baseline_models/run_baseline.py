# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import argparse
import spacetimeformer as stf
from bats_dataset.bats_dataset import BatsCSVDataset, BatsCSVDatasetWithMetadata

from global_mean_predictor import GlobalMeanPredictor
from local_mean_predictor import LocalMeanPredictor
from repeat_last_predictor import RepeatLastPredictor
from informer import InformerPredictor

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
BATCH_SIZE = 16

def main(args):
    if args.model == "informer":
        data_module = stf.data.DataModule(
            datasetCls = BatsCSVDatasetWithMetadata,
            dataset_kwargs = {
                "root_path": args.dataset_path,
                "prefix": "split",
                "ignore_cols": IGNORE_COLS,
                "time_col_name": "TimeIndex",
                "val_split": 0.05,
                "test_split": 0.05,
                "context_points": None,
                "target_points": 1,
                "random_seed": args.random_seed,
                "metadata_cols": ["file_id", "chirp_idx"]
            },
            batch_size = BATCH_SIZE,
            workers = 1
        )
    else:
        data_module = stf.data.DataModule(
            datasetCls = BatsCSVDataset,
            dataset_kwargs = {
                "root_path": args.dataset_path,
                "prefix": "split",
                "ignore_cols": IGNORE_COLS,
                "time_col_name": "TimeIndex",
                "val_split": 0.05,
                "test_split": 0.05,
                "context_points": None,
                "target_points": 1,
                "random_seed": args.random_seed
            },
            batch_size=BATCH_SIZE,
            workers=4,
        )

    train_data = data_module.train_dataloader()
    val_data = data_module.val_dataloader()
    test_data = data_module.test_dataloader()

    if args.model == "repeat-last":
        baseline = RepeatLastPredictor()
    elif args.model == "local-mean":
        baseline = LocalMeanPredictor()
    elif args.model == "global-mean":
        baseline = GlobalMeanPredictor(args.sample_size, BATCH_SIZE)
    elif args.model == "informer":
        baseline = InformerPredictor(num_features=len(train_data.dataset.target_cols), num_epochs=args.n_epochs, 
                                     seq_len=args.seq_len, batch_size=BATCH_SIZE, model_path=args.model_path)
    else:
        ValueError("value for --model must be one of [\"global-mean\", \"local-mean\", \"repeat-last\", \"informer\"]")

    baseline.train(train_data)
    mse_df = baseline.test(test_data)

    if args.model != "informer":
        average_loss_per_row = mse_df.mean(axis=1)
        average_loss_per_row_no_outlier = mse_df.drop("UpprKnToKnAmp", axis=0).mean(axis=1)
        print(average_loss_per_row.mean(), average_loss_per_row_no_outlier.mean())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run baseline strategy: repeat last")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset",
                        default="/home/ayc227/bats/bats_transformer/data/2022_barn_2secs_myca/splits")
    parser.add_argument("--model", type=str, help="Which type of baseline [\"global-mean\", \"local-mean\", \"repeat-last\", \"informer\"]")
    parser.add_argument("--random_seed", type=int, help="Random seed for dataset",
                        default=31)
    parser.add_argument("--sample_size", type=int, 
                        help="Size of sample (instead of using whole population) for global mean")
    InformerPredictor.add_cli(parser)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)