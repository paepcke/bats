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
parser.add_argument("--pca_components", type=int, default=10)

#take a list of string as input from cli
parser.add_argument("--ignore_cols", nargs='+', type=str, default = [])

config = parser.parse_args()
args = config
ignore_cols = ["Filename", "ParentDir", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght'] + config.ignore_cols
config.ignore_cols = ignore_cols
#take this as input from cli

#reading dataframe
df_original, df, max_seq_len, pca = preprocess.preprocess(config)

bats_time_series = stf.data.CSVTimeSeries(
                        raw_df = df,
                        time_col_name = "TimeIndex",
                        time_features = ["hour", "minute", "seconds"],
                        ignore_cols = None,
                        val_split = 0.1,
                        test_split = 0.1
                    )
# create a dataloader with the bats_time_series object
bats_dataset = stf.data.CSVTorchDset(
                    csv_time_series = bats_time_series,
                    split = "train",
                    context_points = max_seq_len - 2,
                    target_points = 2, 
                    time_resolution = 1
                )

bats_time_series_original = stf.data.CSVTimeSeries(
                        raw_df = df_original.drop(columns=config.ignore_cols),
                        time_col_name = "TimeIndex",
                        time_features = ["hour", "minute", "seconds"],
                        ignore_cols = None,
                        val_split = 0.1,
                        test_split = 0.1
                    )

bats_dataset_original = stf.data.CSVTorchDset(
                    csv_time_series = bats_time_series_original,
                    split = "train",
                    context_points = max_seq_len - 2,
                    target_points = 2, 
                    time_resolution = 1
                )


wandb_logger = WandbLogger(name=f"{args.run_name}", save_dir="/home/vdesai/bats_data/logs/") if args.wandb else None
x_dim = bats_time_series.time_cols.size
yc_dim = len(bats_time_series.target_cols)
yt_dim = len(bats_time_series.target_cols)

print(f"{x_dim = }, {yc_dim = }, {yt_dim = }")

data_module = stf.data.DataModule(
    datasetCls = stf.data.CSVTorchDset,
    dataset_kwargs = {
        "csv_time_series": bats_time_series,
        "context_points": max_seq_len - 2,
        "target_points": 2,
        "time_resolution": 1,
    },
    batch_size = config.batch_size,
    workers = config.workers,
    overfit = args.overfit
)

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


scaler = bats_time_series.apply_scaling
inverse_scaler = bats_time_series.reverse_scaling
config.null_value = None
config.pad_value = None
seed = args.random_seed
max_epochs = args.max_epochs

pl.seed_everything(seed)
# initialize the spacetimeformer model
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

model.set_inv_scaler(inverse_scaler);
model.set_scaler(scaler);
model.set_null_value(config.null_value);

trainer = pl.Trainer(
        gpus=args.gpus,
        #callbacks=callbacks,
        logger=wandb_logger if args.wandb else None,
        accelerator="dp",
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
        overfit_batches=20 if args.debug else 0,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        limit_val_batches=args.limit_val_batches,
        max_epochs=max_epochs
)

start = time.time()
trainer.fit(model, datamodule=data_module)
end = time.time()

print(f"Time taken to train: {end - start} seconds")
print("With config: ")
print(config)

with open(args.log_file, "a") as f:
    f.write(f"Ignore cols: {ignore_cols}\n")
    f.write(f"Time taken to train: {end - start} seconds\n")
    f.write("With config: \n")
    f.write(f"{config}\n")
# Saving model checkpoint
model_path = f"/home/vdesai/bats_data/models/{args.run_name}.ckpt"
trainer.save_checkpoint(model_path)

batch_size = 64
df_columns = list(bats_time_series_original.time_cols) + list(bats_time_series_original.target_cols)
predictions = pd.DataFrame(columns = ["FileIndex"] + df_columns)
originals = pd.DataFrame(columns = ["FileIndex"] + df_columns) 
i = 0

for batch_index in tqdm.tqdm(range(0, (len(bats_dataset)), batch_size)):
    # Process each batch
    batch = [bats_dataset[j] for j in range(batch_index, min(batch_index + batch_size, len(bats_dataset)))]
    batch_original = [bats_dataset_original[j] for j in range(batch_index, min(batch_index + batch_size, len(bats_dataset_original)))]

    # Stack tensors for batch processing
    x_c_batch = torch.stack([item[0] for item in batch])
    y_c_batch = torch.stack([torch.from_numpy(model._inv_scaler(item[1].numpy())).float() for item in batch])
    x_t_batch = torch.stack([item[2] for item in batch])
    y_t_batch = torch.stack([torch.from_numpy(model._inv_scaler(item[3].numpy())).float() for item in batch])

    x_c_batch_original = torch.stack([item[0] for item in batch_original])
    y_c_batch_original = torch.stack([torch.from_numpy(bats_time_series_original.reverse_scaling(item[1].numpy())).float() for item in batch_original])
    x_t_batch_original = torch.stack([item[2] for item in batch_original])
    y_t_batch_original = torch.stack([torch.from_numpy(bats_time_series_original.reverse_scaling(item[3].numpy())).float() for item in batch_original])

    # Model prediction for each batch
    yhat_t_batch = model.predict(x_c_batch, y_c_batch, x_t_batch)

    #take the inverse pca transform, back into original space
    y_t_batch = torch.tensor( pca.inverse_transform(y_t_batch.numpy()))
    y_c_batch = torch.tensor( pca.inverse_transform(y_c_batch.numpy()))
    yhat_t_batch = torch.tensor(pca.inverse_transform(yhat_t_batch.numpy()))

    #the originals are already in the original space
    y_c_batch_original = y_c_batch_original
    y_t_batch_original = y_t_batch_original

    #TODO: vectorize this piece of code?
    for j in range(len(batch)):
        # Concatenating tensors for DataFrame creation
        predictions_data = torch.cat((x_c_batch[j], y_c_batch[j]), dim=1)
        predictions_data = torch.cat((predictions_data, torch.cat((x_t_batch[j], yhat_t_batch[j]), dim=1)), dim=0)

        originals_data = torch.cat((x_c_batch[j], y_c_batch_original[j]), dim=1)
        originals_data = torch.cat((originals_data, torch.cat((x_t_batch[j], y_t_batch_original[j]), dim=1)), dim=0)

        # Create DataFrame and append
        predictions_df = pd.DataFrame(predictions_data.numpy(), columns=df_columns)
        predictions_df["FileIndex"] = i
        predictions = pd.concat([predictions, predictions_df], ignore_index=True)

        originals_df = pd.DataFrame(originals_data.numpy(), columns=df_columns)
        originals_df["FileIndex"] = i
        originals = pd.concat([originals, originals_df], ignore_index=True)
        
        i += 1

#Now that we have gone through all of the files, it is time to save these where they belong
#now we have originals and predictions, time to evaluate the performace of the model.

Y = originals.to_numpy(dtype=np.float64)
Yhat = predictions.to_numpy(dtype=np.float64)

mean_Y = np.mean(Y, axis = 0)
mean_Yhat = np.mean(Yhat, axis = 0)
sig_1 = np.std(Y, axis = 0)
sig_2 = np.std(Yhat, axis = 0)

Y = (Y - mean_Y)/sig_1
Yhat = (Yhat - mean_Yhat)/sig_2
error = ((Y - Yhat)**2)


#drop the rows in error where all the values are zero
error = error[~np.all(error == 0, axis=1)]

#also drop columns where there are nan values
error = error[:, ~np.any(np.isnan(error), axis=0)]
MSE_df = pd.DataFrame(error, columns = df_columns)
MSE_df.describe().T.to_csv(config.mse_log_path)

error = np.mean(error, axis = 1)
print(f"25th percentile: {np.percentile(error, 25)}")
print(f"50th percentile: {np.percentile(error, 50)}")
print(f"75th percentile: {np.percentile(error, 75)}")
print(f"Min: {np.min(error)}")
print(f"Max: {np.max(error)}")
print(f"Mean: {np.mean(error)}")

print(f"25th percentile: {np.percentile(error, 25)}", file=open(args.log_file, "a"))
print(f"50th percentile: {np.percentile(error, 50)}", file=open(args.log_file, "a"))
print(f"75th percentile: {np.percentile(error, 75)}", file=open(args.log_file, "a"))
print(f"Min: {np.min(error)}", file=open(args.log_file, "a"))
print(f"Max: {np.max(error)}", file=open(args.log_file, "a"))
print(f"Mean: {np.mean(error)}", file=open(args.log_file, "a"))

predictions.to_csv(config.predictions_path)
originals.to_csv(config.originals_path)
