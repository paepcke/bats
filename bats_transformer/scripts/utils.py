import sys
import subprocess

def run_train_py(run_name, 
                 random_seed, Dmodel = 20, Dqk = 20, 
                 layers = 2, heads = 1,
                 ignore_cols = [],
                 data_path = None,
                 gpus = None,
                 predictions_path = None,
                 originals_path = None,
                 mse_log_path = None,
                 log_file = None,
                 additional_flags = None):
    if data_path is None:
        data_path = "/home/vdesai/bats_data/training_files/splits_feather"
    
    max_epochs = 10
    
    d_model = Dmodel
    d_qk = Dqk
    d_v = Dqk
    d_ff = 4*d_model
    enc_layers = layers
    dec_layers = layers
    n_heads = heads

    if not gpus:
        gpus = 0
    if not predictions_path:
        predictions_path = f"/home/vdesai/bats_data/transformer/predictions/{run_name}.predictions"
    
    if not originals_path:
        originals_path = f"/home/vdesai/bats_data/transformer/originals/{run_name}.originals"
    
    if not mse_log_path:
        mse_log_path = f"/home/vdesai/bats_data/transformer/logs/{run_name}.mse.csv"
    
    if not log_file:
        log_file = f"/home/vdesai/bats_data/transformer/logs/{run_name}.log"

    command = [
        "python3", "train.py",
        "--input_data_path", data_path,
        "--max_epochs", str(max_epochs),
        "--gpus", str(gpus),
        "--random_seed", str(random_seed),
        "--run_name", run_name,
        "--d_model", str(d_model),
        "--d_qk", str(d_qk),
        "--d_v", str(d_v),
        "--d_ff", str(d_ff),
        "--enc_layers", str(enc_layers),
        "--dec_layers", str(dec_layers),
        "--n_heads", str(n_heads),
        "--predictions_path", predictions_path,
        "--originals_path", originals_path,
        "--mse_log_path", mse_log_path,
        "--log_file", log_file
    ]

    if(len(ignore_cols) > 0):
        command.append("--ignore_cols")
        command += ignore_cols
    
    if(additional_flags):
        command += additional_flags
    #get output of subprocess on console
    subprocess.run(command)

def run_test_py(model_path, 
                data_path = None,
                ignore_cols = [],
                log_file = "./log.txt",
                gpus = None, 
                additional_flags = None):
    if data_path is None:
        data_path = "/home/vdesai/bats_data/inference_files/measures/"
    
    if not gpus:
        gpus = 0
    
    if not log_file:
        log_file = f"/home/vdesai/bats_data/transformer/logs/{model_path}.log"

    command = [
        "python3", "test.py",
        "--input_data_path", data_path,
        "--gpus", str(gpus),
        "--log_file", log_file,
        "--model_path", model_path
    ]

    if(len(ignore_cols) > 0):
        command.append("--ignore_cols")
        command += ignore_cols
    
    if(additional_flags):
        command += additional_flags
    #get output of subprocess on console
    subprocess.run(command)