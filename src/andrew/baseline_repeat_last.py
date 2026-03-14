# -*- coding: utf-8 -*-
# @Author: Andrew Chen

import argparse
import spacetimeformer as stf
from bats_dataset.bats_dataset import BatsCSVDataset
from tqdm import tqdm
import numpy as np
import pandas as pd

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

def main(args):
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
        batch_size=64,
        workers=4,
    )

    train_data = data_module.train_dataloader()
    val_data = data_module.val_dataloader()
    test_data = data_module.test_dataloader()

    truths = []
    preds = []
    errors = []

    for batch in tqdm(test_data):
        x_t, x_c, y_t, y_c = batch
        mask = x_t > 0
        lengths = mask.sum(dim=1)
        feature_sums = x_c.sum(dim=1)
        for i, row in enumerate(feature_sums):
            pred = x_c[i, -1, :]
            truths.append(y_c[i].numpy()[0])
            preds.append(pred.numpy())
            errors.append((y_c[i] - pred).numpy()[0])

    truths, preds, errors = np.array(truths), np.array(preds), np.array(errors)

    target_columns = train_data.dataset.target_cols

    mae = np.abs(errors).mean(axis=0)
    mse = (errors * errors).mean(axis=0)

    mse_df = pd.DataFrame(np.array([target_columns, mse]).T)
    mse_df = mse_df.set_index(0)
    mse_df[1] = mse_df[1].astype(float)
    # mse_df[1] = mse_df[1].round(6)

    average_loss_per_row = mse_df.mean(axis=1)
    average_loss_per_row_no_outlier = mse_df.drop("UpprKnToKnAmp", axis=0).mean(axis=1)
    print(average_loss_per_row.mean(), average_loss_per_row_no_outlier.mean())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run baseline strategy: repeat last")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset",
                        default="/home/ayc227/bats/bats_transformer/data/2022_barn_2secs_myca/splits")
    parser.add_argument("--random_seed", type=int, help="Random seed for dataset",
                        default=31)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)