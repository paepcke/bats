#!/bin/bash
RUN_NAME=$1
DATA_PATH=/home/vdesai/bats_data/training_files/data.feather
MAX_EPOCHS=20
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
    --predictions_path /home/vdesai/bats_logs_new/predictions/$RUN_NAME.predictions.feather \
    --originals_path /home/vdesai/bats_logs_new/originals/$RUN_NAME.originals.feather \
    --mse_log_path /home/vdesai/bats_logs_new/logs/$RUN_NAME.mse.feather \
    --log_file /home/vdesai/bats_logs_new/logs/$RUN_NAME.log \
    --ignore_cols FreqLedge AmpK@end Fc "FBak15dB  " FBak32dB EndF FBak20dB LowFreq Bndw20dB CallsPerSec EndSlope SteepestSlope StartSlope Bndw15dB HiFtoUpprKnSlp HiFtoKnSlope DominantSlope Bndw5dB PreFc500 PreFc1000 PreFc3000 KneeToFcSlope TotalSlope PreFc250 CallDuration CummNmlzdSlp DurOf32dB SlopeAtFc LdgToFcSlp DurOf20dB DurOf15dB TimeFromMaxToFc KnToFcDur HiFtoFcExpAmp AmpKurtosis LowestSlope KnToFcDmp HiFtoKnExpAmp DurOf5dB KnToFcExpAmp RelPwr3rdTo1st LnExpB_StartAmp Filter HiFtoKnDmp LnExpB_EndAmp HiFtoFcDmp AmpSkew LedgeDuration KneeToFcResidue PreFc3000Residue AmpGausR2 PreFc1000Residue Amp1stMean LdgToFcExp FcMinusEndF Amp4thMean HiFtoUpprKnExp HiFtoKnExp KnToFcExp UpprKnToKnExp Kn-FcCurviness Amp2ndMean Quality HiFtoFcExp LnExpA_EndAmp RelPwr2ndTo1st LnExpA_StartAmp HiFminusStartF Amp3rdMean PreFc500Residue Kn-FcCurvinessTrndSlp PreFc250Residue AmpVariance AmpMoment meanKn-FcCurviness MinAccpQuality AmpEndLn60ExpC AmpStartLn60ExpC Preemphasis MaxSegLnght Max#CallsConsidered 

