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
from data.bats_dataset_better import BatsCSVDataset
from pytorch_lightning.callbacks import LearningRateMonitor

from utils import *

parser = ArgumentParser()
stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
stf.callbacks.TimeMaskedLossCallback.add_cli(parser)
stf.data.DataModule.add_cli(parser)
preprocess.add_cli(parser)

parser.add_argument("--wandb", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--plot_samples", type=int, default=8)
parser.add_argument("--attn_plot", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--accumulate", type=int, default=1)
parser.add_argument("--val_check_interval", type=float, default=1.0)
parser.add_argument("--limit_val_batches", type=float, default=1.0)
parser.add_argument("--no_earlystopping", action="store_true")
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--trials", type=int, default=1, help="How many consecutive trials to run")
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--log_file", type=str, default="/home/vdesai/bats_data/logs/log.txt")
parser.add_argument("--predictions_path", type=str, default="/home/vdesai/bats_data/predictions.csv")
parser.add_argument("--originals_path", type=str, default="/home/vdesai/bats_data/originals.csv")
parser.add_argument("--mse_log_path", type=str, default="/home/vdesai/bats_data/mse_log.csv")
parser.add_argument("--pca_components", type=int, default=0)
parser.add_argument("--construct_full_output", action ="store_true");
parser.add_argument("--checkpoint_val_loss", action = 'store_true');
parser.add_argument("--telegram_updates", action="store_true")
parser.add_argument("--model_path", required=False)

#take a list of string as input from cli
parser.add_argument("--ignore_cols", nargs='+', type=str, default = [])



config = parser.parse_args()
print(f"Batch size: {config.batch_size}")
args = config
ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir", "file_id", "chirp_idx", "split"] + config.ignore_cols


wandb_logger = WandbLogger(name=f"{args.run_name}", save_dir="/home/vdesai/bats_data/logs/") if args.wandb else None


data_module = stf.data.DataModule(
    datasetCls = BatsCSVDataset,
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
    #get output of subprocess in a variable
    batch_size = config.batch_size,
    workers = config.workers,
    overfit = args.overfit
)


x_dim = 1
yc_dim = len(data_module.train_dataloader().dataset.target_cols)
yt_dim = yc_dim

max_seq_len = data_module.train_dataloader().dataset.seq_length

print(f"{x_dim = }, {yc_dim = }, {yt_dim = }")
# Example DataLoader check
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

assert train_loader is not None, "Training DataLoader is None"
assert val_loader is not None, "Validation DataLoader is None"
assert test_loader is not None, "Test DataLoader is None"

assert len(train_loader.dataset) > 0, "Training dataset is empty"
assert len(val_loader.dataset) > 0, "Validation dataset is empty"
assert len(test_loader.dataset) > 0, "Test dataset is empty"

print("Length of splits", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
config.null_value = None
config.pad_value = None
seed = args.random_seed
max_epochs = args.max_epochs

pl.seed_everything(seed)
# initialize the spacetimeformer model
print("Initializing model... ", flush = True)
t = time.time()

model = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            d_queries_keys=config.d_qk,
            d_values=config.d_v,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_attn_out=config.dropout_attn_out,
            dropout_attn_matrix=config.dropout_attn_matrix,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            pos_emb_type=config.pos_emb_type,
            use_final_norm=not config.no_final_norm,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            attn_time_windows=config.attn_time_windows,
            use_shifted_time_windows=config.use_shifted_time_windows,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            class_loss_imp=config.class_loss_imp,
            recon_loss_imp=config.recon_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
            pad_value=config.pad_value,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_val=not config.no_val,
            use_time=not config.no_time,
            use_space=not config.no_space,
            use_given=not config.no_given,
            recon_mask_skip_all=config.recon_mask_skip_all,
            recon_mask_max_seq_len=config.recon_mask_max_seq_len,
            recon_mask_drop_seq=config.recon_mask_drop_seq,
            recon_mask_drop_standard=config.recon_mask_drop_standard,
            recon_mask_drop_full=config.recon_mask_drop_full,
        )
print("Done.[Time taken: {}]".format(time.time() - t))

model.set_null_value(config.null_value);
print("Initializing trainer.. ", flush = True)
t = time.time()

callbacks = None

if(args.checkpoint_val_loss):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='/home/vdesai/bats_data/models/',
        filename=args.run_name + '-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}',
        save_top_k=-1,
    )

    callbacks = [checkpoint_callback]

print("Callbacks: ", callbacks)
trainer = pl.Trainer(
        gpus=args.gpus,
        logger=wandb_logger if args.wandb else None,
        accelerator="dp",
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
        overfit_batches=20 if args.debug else 0,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        limit_val_batches=args.limit_val_batches,
        max_epochs=max_epochs,
        callbacks = callbacks
)

print("Done. [Time taken: {}]".format(time.time() - t))

start = time.time()
trainer.fit(model, datamodule=data_module)
end = time.time()

print(f"Time taken to train: {end - start} seconds")
print("With config: ")
print(config)

#with open(args.log_file, "a") as f:
#    f.write(f"Ignore cols: {ignore_cols}\n")
#    f.write(f"Time taken to train: {end - start} seconds\n")
#    f.write("With config: \n")
#    f.write(f"{config}\n")


# Saving model checkpoint
if not args.model_path:
    model_path = f"/home/vdesai/bats_data/transformer/models/{args.run_name}.ckpt"
else:
    model_path = args.model_path
trainer.save_checkpoint(model_path)
print(model.device)

#ping on telegram after training is done
if(args.telegram_updates):
    send_telegram_message("training for {} is done".format(args.run_name))
