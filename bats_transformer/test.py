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
from data.bats_dataset_better import *
from pytorch_lightning.callbacks import LearningRateMonitor

from itertools import chain
from utils import *

parser = ArgumentParser()
stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
stf.callbacks.TimeMaskedLossCallback.add_cli(parser)
stf.data.DataModule.add_cli(parser)
preprocess.add_cli(parser)

parser.add_argument("--model_path", type=str, default = None)
parser.add_argument("--ignore_cols", nargs='+', type=str, default = [])
parser.add_argument("--log_file", type=str, required=True)
parser.add_argument("--telegram_updates", action="store_true")

config = parser.parse_args()
print(f"Batch size: {config.batch_size}")
args = config
ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir", "file_id", "chirp_idx", "split"] + config.ignore_cols

#This is like... complete hacks


data_module = stf.data.DataModule(
    datasetCls = BatsCSVDatasetWithMetadata,
    dataset_kwargs = {
        "root_path": args.input_data_path,
        "prefix": "split",
        "ignore_cols": ignore_cols,
        "metadata_cols": ["file_id", "chirp_idx"],
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

#how to move model to gpu?
if(len(args.gpus)):
    model = model.to(torch.device(f"cuda:{args.gpus[0]}"))

dummy_dataset = data_module.train_dataloader().dataset
time_cols = [dummy_dataset.time_col_name]
target_cols = dummy_dataset.target_cols
metadata_cols = dummy_dataset.metadata_cols


predictions_list = []
for batch in tqdm.tqdm(chain(
                data_module.train_dataloader(),
                data_module.val_dataloader(), 
                data_module.test_dataloader()
            )):
    
    x_c_batch, y_c_batch, x_t_batch, y_t_batch, metadata = batch
    y_hat_t = spacetimeformer_predict(model, x_c_batch, y_c_batch, x_t_batch)

    predictions_list += [np.squeeze(np.concatenate((x_t_batch.numpy(), y_hat_t.numpy(), metadata.numpy()), axis=2))]


predictions = pd.concat(
    [pd.DataFrame(d, columns = time_cols + target_cols + metadata_cols) for d in predictions_list], 
    ignore_index = True
)

predictions["model_id"] = args.model_path    
predictions.to_csv(args.log_file)

#ping on telegram after inference is done
if(args.telegram_updates):
    send_telegram_message("inference for {} is done".format(args.model_path))