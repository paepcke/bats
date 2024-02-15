import sys
import subprocess

def run_training(run_name, random_seed, Dmodel = 20, Dqk = 20, layers = 2, heads = 1):
    data_path = "/home/vdesai/bats_data/training_files/data.csv"
    max_epochs = 5
    gpus = 0
    d_model = Dmodel
    d_qk = Dqk
    d_v = Dqk
    d_ff = 4*d_model
    enc_layers = layers
    dec_layers = layers
    n_heads = heads

    predictions_path = f"/home/vdesai/bats_logs_new/predictions/{run_name}.predictions"
    originals_path = f"/home/vdesai/bats_logs_new/originals/{run_name}.originals"
    mse_log_path = f"/home/vdesai/bats_logs_new/logs/{run_name}.mse.csv"
    log_file = f"/home/vdesai/bats_logs_new/logs/{run_name}.log"

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

    subprocess.run(command)

d_model_list = [20,40,60,80,100]
Dqk_list = [20,40,60,80,100]
layers_list = [2,3,4,5,6]
heads_list = [1,2,3,4,5]

print("Running experiments, changing d_model...")
for d_model in d_model_list:
    run_training(f"d_model_{d_model}", 1, Dmodel = d_model)

print("Running experiments, changing Dqk...")
for Dqk in Dqk_list:
    run_training(f"Dqk_{Dqk}", 1, Dqk = Dqk)

print("Running experiments, changing layers...")
for layers in layers_list:
    run_training(f"layers_{layers}", 1, layers = layers)

print("Running experiments, changing heads...")
for heads in heads_list:
    run_training(f"heads_{heads}", 1, heads = heads)

print("Okay, a bit of all of them...")
run_training("all", 1, Dmodel = 80, Dqk = 60, layers = 4, heads = 4)
