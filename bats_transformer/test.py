from argparse import ArgumentParser
import torch
import numpy as np

import pytorch_lightning as pl
import spacetimeformer as stf
import pandas as pd

from pytorch_lightning.loggers import WandbLogger
from data import preprocess
import time
import tqdm
from itertools import chain
from data.bats_dataset import *
from pytorch_lightning.callbacks import LearningRateMonitor

from utils import *

parser = ArgumentParser()
stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
stf.callbacks.TimeMaskedLossCallback.add_cli(parser)
stf.data.DataModule.add_cli(parser)
preprocess.add_cli(parser)

parser.add_argument("--model_path", type=str, default = None)
parser.add_argument("--ignore_cols", nargs='+', type=str, default = [])
parser.add_argument("--log_file", type=str, required=True)


config = parser.parse_args()
print(f"Batch size: {config.batch_size}")
args = config
ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir"] + config.ignore_cols

#This is like... complete hacks


data_module = stf.data.DataModule(
    datasetCls = BatsCSVDatasetWithMetadata,
    dataset_kwargs = {
        "root_path": args.input_data_path,
        "prefix": "split",
        "ignore_cols": ignore_cols,
        "time_col_name": "TimeIndex",
        "val_split": 0.05,
        "test_split": 0.05,
        "context_points": None,
        "target_points": 1,
    },
    batch_size = config.batch_size,
    workers = config.workers,
    overfit = args.overfit
)


x_dim = 1
yc_dim = len(data_module.train_dataloader().dataset.target_cols)
yt_dim = yc_dim
max_seq_len = data_module.train_dataloader().dataset.seq_length

print(f"{x_dim = }, {yc_dim = }, {yt_dim = }")

config.null_value = None
config.pad_value = None

model = stf.spacetimeformer_model.Spacetimeformer_Forecaster(max_seq_len = 54).load_from_checkpoint(checkpoint_path=args.model_path)

model.set_null_value(config.null_value);

#how to get Filename, Cntxt_sz here?
for batch in data_module.train_dataloader(): 
    data, meta_data = batch
    print(meta_data)