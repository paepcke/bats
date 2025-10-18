from utils import run_test_py
import os
import argparse

# sample usage: python scripts/test_bats_transformer.py --model_paths models/daytime_files_new_10_11/models_{31..31} --out_dir outputs/test/ --input_data_path data/daytime_files_new/splits --gpus 0 --shuffle_data

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

parser = argparse.ArgumentParser()
parser.add_argument("--model_paths", nargs='+', type=str, default = [])
parser.add_argument("--ignore_cols", nargs='+', type=str, default = ignore_cols)
parser.add_argument("--out_dir", type=str, default = "/home/vdesai/data/model_outputs/daytime")
parser.add_argument("--input_data_path", type=str, default='/home/vdesai/data/training_data/daytime/splits')
parser.add_argument("--shuffle_data", action='store_true', help='Whether to shuffle the data during testing (should match training)')
parser.add_argument("--gpus", type=str, default = '0')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

for model_path in args.model_paths:
    run_test_py(model_path = model_path,
                data_path = args.input_data_path,
                log_file = os.path.join(os.path.abspath(args.out_dir), f"{model_path.split('/')[-1]}.log"),
                ignore_cols = args.ignore_cols,
                gpus = args.gpus, additional_flags=['--telegram_updates'])