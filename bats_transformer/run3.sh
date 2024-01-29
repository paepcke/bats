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
    --predictions_path /home/vdesai/bats_data/predictions/$RUN_NAME.predictions \
    --originals_path /home/vdesai/bats_data/originals/$RUN_NAME.originals \
    --log_file /home/vdesai/bats_data/logs/$RUN_NAME.log \
    --ignore_cols DominantSlope Amp3rdMean DurOf20dB HiFtoFcDmp PreFc500 PreFc3000 Bndw5dB AmpEndLn60ExpC PreFc1000 HiFtoUpprKnSlp AmpK@end Bndwdth RelPwr3rdTo1st FreqKnee PreFc250Residue Bndw15dB Amp2ndMean AmpKurtosis KnToFcAmp KnToFcExpAmp AmpSkew EndF HiFminusStartF DurOf32dB AmpGausR2 FFwd5dB HiFtoKnDmp HiFtoKnSlope HiFtoFcExp PrcntMaxAmpDur ParentDir PreFc250 TimeFromMaxToFc HiFtoUpprKnAmp Fc LdgToFcSlp KneeToFcSlope SteepestSlope LdgToFcAmp MinAccpQuality SlopeAtFc Amp4thQrtl AmpK@start HiFtoFcAmp LedgeDuration EndSlope LnExpB_EndAmp FFwd32dB FBak32dB AmpStartLn60ExpC DurOf5dB PreFc500Residue

