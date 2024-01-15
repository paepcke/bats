import torch
import numpy as np
import pytorch_lightning as pl
import spacetimeformer as stf
import pandas as pd

from data import preprocess

parser = ArgumentParser()
stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
stf.callback.TimeInFileLossCallback.add_cli(parser)
stf.data.DataModule.add_cli(parser)
preprocess.add_cli(parser)

parser.add_argument("--model_path", type=str, action="store_true")
config = parser.parse_args()
args = config
model_path = args.model_path

#reading dataframe
df, max_seq_len = preprocess.preprocess(config)
df, max_seq_len = preprocess.preprocess(config)
bats_time_series = stf.data.CSVTimeSeries(
                        raw_df = df,
                        time_col_name = "TimeInFile",
                        time_features = ["minute"],
                        ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght'],
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

model = model.load_from_checkpoint(checkpoint_path=model_path)

#now that model is loaded, run it on the test data set, save the results, and then evaluate the variance in the output

#run the model on the test data set, and dump the output of the model to a csv file

