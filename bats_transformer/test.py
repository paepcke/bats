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

from itertools import chain
from utils import *
# from telegram_utils import *

parser = ArgumentParser()
stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
stf.callbacks.TimeMaskedLossCallback.add_cli(parser)
stf.data.DataModule.add_cli(parser)
preprocess.add_cli(parser)

parser.add_argument("--model_path", type=str, default = None)
parser.add_argument("--ignore_cols", nargs='+', type=str, default = [])
parser.add_argument("--log_file", type=str, required=True)
parser.add_argument("--telegram_updates", action="store_true")
parser.add_argument("--shuffle", action="store_true", help='Whether to shuffle the data during testing (should match training)')
parser.add_argument("--random_seed", type=int, default=42)

config = parser.parse_args()
print(f"Batch size: {config.batch_size}")
args = config
ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir", "file_id", "chirp_idx", "split"] + config.ignore_cols


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
        "shuffle": args.shuffle,
        "random_seed": args.random_seed
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
model.set_null_value(config.null_value)

#how to move model to gpu?
if(len(args.gpus)):
    model = model.to(torch.device(f"cuda:{args.gpus[0]}"))

dummy_dataset = data_module.train_dataloader().dataset
time_cols = [dummy_dataset.time_col_name]
target_cols = dummy_dataset.target_cols
metadata_cols = dummy_dataset.metadata_cols

print("time_cols", time_cols)
print("target_cols", target_cols)
print("metadata_cols", metadata_cols)

ground_truths_list = []
predictions_list = []
losses_list = []
# for batch in tqdm.tqdm(chain(
#                 data_module.train_dataloader(),
#                 data_module.val_dataloader(), 
#                 data_module.test_dataloader()
#             )):
for batch in tqdm.tqdm(data_module.test_dataloader()):
    
    x_c_batch, y_c_batch, x_t_batch, y_t_batch, metadata = batch
    y_hat_t = spacetimeformer_predict(model, x_c_batch, y_c_batch, x_t_batch)
    loss = spacetimeformer_predict_calculate_loss(model, x_c_batch, y_c_batch, x_t_batch, y_t_batch)

    ground_truths_list += [np.squeeze(np.concatenate((x_t_batch.numpy(), y_t_batch.numpy(), metadata.numpy()), axis=2))]
    predictions_list += [np.squeeze(np.concatenate((x_t_batch.numpy(), y_hat_t.numpy(), metadata.numpy()), axis=2))]
    losses_list += [np.squeeze(np.concatenate((x_t_batch.numpy(), loss.numpy(), metadata.numpy()), axis=2))]

ground_truths = pd.concat(
    [pd.DataFrame(d, columns = time_cols + target_cols + metadata_cols) for d in ground_truths_list], 
    ignore_index = True
)

predictions = pd.concat(
    [pd.DataFrame(d, columns = time_cols + target_cols + metadata_cols) for d in predictions_list], 
    ignore_index = True
)

losses = pd.concat(
    [pd.DataFrame(d, columns = time_cols + target_cols + metadata_cols) for d in losses_list], 
    ignore_index = True
)

ground_truths["model_id"] = args.model_path
predictions["model_id"] = args.model_path
losses["model_id"] = args.model_path

ground_truth_path = args.log_file.replace(".log", "_ground_truths.log")
predictions_path = args.log_file.replace(".log", "_predictions.log")
losses_path = args.log_file.replace(".log", "_losses.log")

ground_truths.to_csv(ground_truth_path)
predictions.to_csv(predictions_path)
losses.to_csv(losses_path)

#ping on telegram after inference is done
if(args.telegram_updates):
    send_telegram_message("inference for {} is done".format(args.model_path))