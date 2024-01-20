import torch
import numpy as np
import pytorch_lightning as pl
import spacetimeformer as stf
import pandas as pd

from data import preprocess
import argparse
parser = argparse.ArgumentParser()

stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
stf.callbacks.TimeMaskedLossCallback.add_cli(parser)
stf.data.DataModule.add_cli(parser)
preprocess.add_cli(parser)

parser.add_argument("--model_path", type=str, default="/home/vdesai/bats/bats_transformer/models/random_seed_42.ckpt")
parser.add_argument("--predictions_path", type=str, default="/home/vdesai/bats/bats_transformer/predictions.csv")
parser.add_argument("--originals_path", type=str, default="/home/vdesai/bats/bats_transformer/originals.csv")


config = parser.parse_args()
args = config
model_path = args.model_path

#reading dataframe
df, max_seq_len = preprocess.preprocess(config)
df, max_seq_len = preprocess.preprocess(config)

bats_time_series = stf.data.CSVTimeSeries(
                        raw_df = df,
                        time_col_name = "TimeIndex",
                        time_features = ["hour", "minute", "second"],
                        ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght'],
                        val_split = 0.1,
                        test_split = 0.1
                    )
# create a dataloader with the bats_time_series object
print("B")
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

scaler = bats_time_series.apply_scaling
inverse_scaler = bats_time_series.reverse_scaling
config.null_value = None
config.pad_value = None

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



model = model.load_from_checkpoint(checkpoint_path=model_path)
model.set_inv_scaler(inverse_scaler);
model.set_scaler(scaler);
model.set_null_value(config.null_value);

df_columns = list(bats_time_series.time_cols) + list(bats_time_series.target_cols)
predictions = pd.DataFrame(columns = ["FileIndex"] + df_columns)
originals = pd.DataFrame(columns = ["FileIndex"] + df_columns) 
i = 0

for (x_c, y_c, x_t, y_t) in bats_dataset:
    y_c = torch.from_numpy(model._inv_scaler(y_c.numpy())).float()
    y_t = torch.from_numpy(model._inv_scaler(y_t.numpy())).float()
    yhat_t = model.predict(x_c.unsqueeze(0), y_c.unsqueeze(0), x_t.unsqueeze(0))
    
    predictions_df = pd.DataFrame(torch.cat((x_c, y_c), dim=1).numpy(), columns = df_columns)
    predictions_df = pd.concat([predictions_df, pd.DataFrame(torch.cat((x_t, yhat_t.squeeze(0)), dim=1).numpy(), columns = df_columns)], ignore_index = True)
    predictions_df["FileIndex"] = i


    originals_df = pd.DataFrame(torch.cat((x_c, y_c), dim=1).numpy(), columns = df_columns)
    originals_df = pd.concat([originals_df, pd.DataFrame(torch.cat((x_t, y_t), dim=1).numpy(), columns = df_columns)], ignore_index = True)
    originals_df["FileIndex"] = i
    originals   = pd.concat([originals, originals_df], ignore_index=True)
    predictions = pd.concat([predictions, predictions_df], ignore_index=True)
    #print(originals)
    #print(predictions)
    i += 1
#Now that we have gone through all of the files, it is time to save these where they belong

predictions.to_csv(config.predictions_path)
originals.to_csv(config.originals_path)

