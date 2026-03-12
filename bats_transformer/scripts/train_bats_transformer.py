from utils import run_train_py
import argparse

# sample usage: python scripts/train_bats_transformer.py --random_seeds {31..31} --data_path ./data/filter_test/splits --model_path models/model --shuffle_data
# python scripts/train_bats_transformer.py --random_seeds {31..31} --data_path ./data/july_daytime_chunked_quantile/splits --model_path models/july_daytime_test/model
# python scripts/train_bats_transformer.py --random_seeds {31..31} --data_path ./data/2022_barn_daytime_2secs/splits --model_path models/2022_barn_daytime_2secs_test/model

ignore_cols = ["FreqLedge","AmpK@end", "Fc", "FBak15dB  ", "FBak32dB", "EndF", "FBak20dB", "LowFreq", "Bndw20dB", 
               "CallsPerSec", "EndSlope", "SteepestSlope", "StartSlope", "Bndw15dB", "HiFtoUpprKnSlp", "HiFtoKnSlope", 
               "DominantSlope", "Bndw5dB", "PreFc500", "PreFc1000", "PreFc3000", "KneeToFcSlope", "TotalSlope", 
               "PreFc250", "CallDuration", "CummNmlzdSlp", "DurOf32dB", "SlopeAtFc", "LdgToFcSlp", "DurOf20dB", "DurOf15dB", 
               "TimeFromMaxToFc", "KnToFcDur", "HiFtoFcExpAmp", "AmpKurtosis", "LowestSlope", "KnToFcDmp", "HiFtoKnExpAmp", 
               "DurOf5dB", "KnToFcExpAmp", "RelPwr3rdTo1st", "LnExpB_StartAmp", "Filter", "HiFtoKnDmp", "LnExpB_EndAmp", 
               "HiFtoFcDmp", "AmpSkew", "LedgeDuration", "KneeToFcResidue", "PreFc3000Residue", "AmpGausR2", "PreFc1000Residue", 
               "Amp1stMean", "LdgToFcExp", "FcMinusEndF", "Amp4thMean", "HiFtoUpprKnExp", "HiFtoKnExp", "KnToFcExp", "UpprKnToKnExp", 
               "Kn-FcCurviness", "Amp2ndMean", "Quality", "HiFtoFcExp", "LnExpA_EndAmp", "RelPwr2ndTo1st", "LnExpA_StartAmp", 
               "HiFminusStartF", "Amp3rdMean", "PreFc500Residue", "Kn-FcCurvinessTrndSlp", "PreFc250Residue", "AmpVariance", "AmpMoment", 
               "meanKn-FcCurviness", "MinAccpQuality", "AmpEndLn60ExpC", "AmpStartLn60ExpC", "Preemphasis", "MaxSegLnght" ,"Max#CallsConsidered" ]

parser = argparse.ArgumentParser(description='Train bats transformer model')
parser.add_argument('--random_seeds', type=int, nargs='+', help='Random seeds to use for training')
parser.add_argument('--gpu', type=int, help='GPU to use for training')
parser.add_argument('--data_path', type=str, 
                    default='/home/vdesai/data/training_data/daytime/splits', 
                    help='Path to the data')
parser.add_argument('--shuffle_data', action='store_true', help='Whether to shuffle the data during training')
parser.add_argument('--quantize', action="store_true", help='Whether to use quantized model (experimental)')
parser.add_argument('--model_path', required = False, help='Folder to output trained models')
parser.add_argument('--ignore_cols', nargs='+', type=str, default = ignore_cols, help='Which chirp attributes to remove from data')

args = parser.parse_args()

model_path = args.model_path if args.model_path is not None else f"bats_tranformer_seed_nodup_data"
    
d_model = 100
# d_qk = 128
# layers = 2
# n_heads = 4


for random_seed in args.random_seeds:
    print("Running for random seed ", random_seed)
    run_train_py(run_name = f"model_{random_seed}", 
                quantize = args.quantize,
                random_seed = random_seed, Dmodel = d_model, 
                # Dqk = d_qk, layers = layers, heads = n_heads,
                shuffle_data = args.shuffle_data,
                ignore_cols = ignore_cols, 
                additional_flags = ["--telegram_updates"], 
                gpus = args.gpu, data_path = args.data_path, 
                model_path = args.model_path + "/model_" + str(random_seed) if args.model_path else None)