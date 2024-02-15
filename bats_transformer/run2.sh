#!/bin/bash
RUN_NAME=$1
DATA_PATH=/home/vdesai/bats_data/training_files/data.csv
MAX_EPOCHS=5
GPUS=1
RANDOM_SEED=$2
D_MODEL=20
D_QK=20
D_V=20
D_FF=100
ENC_LAYERS=2
DEC_LAYERS=2
N_HEADS=1

python3 train.py \
    --input_data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHS \
    --gpus $GPUS \
    --random_seed $RANDOM_SEED \
    --run_name $RUN_NAME \
    --d_model $D_MODEL \
    --d_qk $D_QK \
    --d_v $D_V \
    --d_ff $D_FF \
    --enc_layers $ENC_LAYERS \
    --dec_layers $DEC_LAYERS \
    --n_heads $N_HEADS \
    --predictions_path /home/vdesai/bats_logs_new/predictions/$RUN_NAME.predictions \
    --originals_path /home/vdesai/bats_logs_new/originals/$RUN_NAME.originals \
    --mse_log_path /home/vdesai/bats_logs_new/logs/$RUN_NAME.mse.csv \
    --log_file /home/vdesai/bats_logs_new/logs/$RUN_NAME.log

