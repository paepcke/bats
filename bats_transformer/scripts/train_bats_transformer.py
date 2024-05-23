from utils import run_train_py
import argparse

parser = argparse.ArgumentParser(description='Train bats transformer model')
parser.add_argument('--random_seeds', type=int, nargs='+', help='Random seeds to use for training')
parser.add_argument('--gpu', type=int, help='GPU to use for training')
parser.add_argument('--data_path', type=str, 
                    default='/home/vdesai/data/training_data/daytime/splits', 
                    help='Path to the data')
parser.add_argument('--model_path', required = False)


args = parser.parse_args()

model_path = args.model_path if args.model_path is not None else f"bats_tranformer_seed_nodup_data"
    
d_model = 100;

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

for random_seed in args.random_seeds:
    print("Running for random seed ", random_seed)
    run_train_py(run_name = f"{model_path.split('/')[-1]}_{random_seed}", 
                random_seed = random_seed, Dmodel = d_model, 
                ignore_cols = ignore_cols, 
                additional_flags = ["--telegram_updates"], 
                gpus = args.gpu, data_path = args.data_path, 
                model_path = args.model_path + "_" + str(random_seed) if args.model_path else None)