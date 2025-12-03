from huggingface_hub import hf_hub_download
import torch
from transformers import InformerConfig, InformerModel, InformerForPrediction
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# os.chdir("bats_transformer")
# print(os.getcwd())
sys.path.append('../bats_transformer')

import spacetimeformer as stf
from data.bats_dataset import BatsCSVDataset, BatsCSVDatasetWithMetadata

ignore_cols = ["Filename", "NextDirUp", 'Path', 'Version', 'Filter', 'Preemphasis', 'MaxSegLnght', "ParentDir", "file_id", "chirp_idx", "split"] + \
              ["FreqLedge","AmpK@end", "Fc", "FBak15dB  ", "FBak32dB", "EndF", "FBak20dB", "LowFreq", "Bndw20dB", 
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
ignore_cols += ["Amp1stQrtl", "Amp2ndQrtl", "Amp3rdQrtl", "Amp4thQrtl", "HiFtoUpprKnAmp", "HiFtoKnAmp", "HiFtoFcAmp", "UpprKnToKnAmp", "KnToFcAmp", "LdgToFcAmp"]

num_epochs = 10
batch_size = 16
seq_len = 21
num_features = 32  # number of features in your dataset
prediction_length = 1
target_cols = ["HiFreq", "Bndwdth"]

model_path_prefix = "models/baseline_model/ft_10_no_amp"
if not os.path.exists(model_path_prefix):
    os.mkdir(model_path_prefix)
if not os.path.exists(f"{model_path_prefix}/plots"):
    os.mkdir(f"{model_path_prefix}/plots")

data_module = stf.data.DataModule(
    datasetCls = BatsCSVDatasetWithMetadata,
    dataset_kwargs = {
        "root_path": "./data/july_daytime_chunked_quantile/splits",
        "prefix": "split",
        "ignore_cols": ignore_cols,
        "metadata_cols": ["file_id", "chirp_idx"],
        "time_col_name": "TimeIndex",
        "val_split": 0.05,
        "test_split": 0.05,
        "context_points": None,
        "target_points": 1,
        # "target_cols": target_cols
    },
    batch_size = batch_size,
    workers = 1
)
train_dataloader = data_module.train_dataloader()
test_dataloader = data_module.test_dataloader()

num_features = len(train_dataloader.dataset.target_cols)
print("num_features", num_features) 

config = InformerConfig(
    input_size=num_features,
    context_length=seq_len - 1,     # encoder sequence length
    lags_sequence=[1],
    prediction_length=prediction_length,     # forecast horizon
    num_time_features=1,
    d_model=512,    # hidden size (transformer dimension)
    n_heads=8,      # number of attention heads
    e_layers=2,     # number of encoder layers
    d_layers=1,     # number of decoder layers
    d_ff=2048,      # feed-forward dimension
    dropout=0.05,   # dropout rate
    factor=5,       # ProbSparse attention factor (Informer-specific)
    activation="gelu",  # activation function
    output_attention=False,  # set True if you want attention weights returned
)

# Randomly initializing a model (with random weights) from the configuration
model = InformerForPrediction(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = model.to(device)

# 4. Sample DataLoader loop (you must build your own DataLoader)
# losses = []
# for epoch in tqdm(range(num_epochs)):
#     model.train()
#     step = 0
#     for batch in tqdm(train_dataloader):
#         x_t, x_c, y_t, y_c, _ = batch
#         actual_batch_size = x_c.shape[0]
#         # print(x_t)
#         # print(x_c)
#         # print(y_t)
#         # print(y_c)
#         # print([x.shape for x in batch])
#         # batch[past_values] shape: (batch_size, seq_len, features)
#         # batch[future_values] shape: (batch_size, pred_len, features)
#         past_values = x_c.to(device)
#         future_values = y_c.to(device)
#         # If using time features or observed masks:
#         past_time_features = x_t.to(device)
#         future_time_features = y_t.to(device)
#         past_observed_mask = torch.ones((actual_batch_size, seq_len, num_features), device=device)
#         # if past_observed_mask is None:
#         #     past_observed_mask = torch.ones((batch_size, seq_len, num_features), device=device)
#         # print(past_values.shape)
#         # print(future_values.shape)
#         # print(past_time_features.shape)
#         # print(future_time_features.shape)
#         # print(past_observed_mask.shape)

#         optimizer.zero_grad()
#         outputs = model(
#             past_values=past_values,
#             future_values=future_values,
#             past_time_features=past_time_features,
#             future_time_features=future_time_features,
#             past_observed_mask=past_observed_mask,
#             future_observed_mask=torch.ones_like(future_values)
#             # include other optional args if you have them
#         )

#         # The model may return a dict or tuple; adjust accordingly:
#         # For example: outputs.last_hidden_state of shape (batch_size, pred_len, c_out)
#         preds = outputs
#         loss = preds.loss
#         # print(preds)
#         # quit(0)

#         # loss = criterion(preds, future_values)
#         loss.backward()
#         optimizer.step()

#         if step % 20 == 0:
#             print(sum(losses[-20:]) / 20)
#         step += 1
#         losses.append(loss.item())
#         # quit(0)
#         # print(f"Epoch {epoch+1}/{num_epochs} — Training Loss: {avg_loss:.6f}")
# plt.plot(losses)
# plt.savefig(f"{model_path_prefix}/plots/loss.png")
# model.save_pretrained(model_path_prefix)
# config.save_pretrained(model_path_prefix)

print("Running inference on test data...")
model.eval()

predictions = []
truths = []
confidences = []
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        x_t, x_c, y_t, y_c, metadata = batch
        
        # Get actual batch size for this batch
        actual_batch_size = x_c.shape[0]
        
        # Move to device
        past_values = x_c.to(device)
        future_values = y_c.to(device)
        past_time_features = x_t.to(device)
        future_time_features = y_t.to(device)
        
        # Debug: Print shapes
        if batch_idx == 0:
            print(f"\nBatch shapes:")
            print(f"  past_values: {past_values.shape}")
            print(f"  past_time_features: {past_time_features.shape}")
            print(f"  future_time_features: {future_time_features.shape}")
        
        # Ensure time features have correct shape [batch, seq_len, num_time_features]
        # if past_time_features.dim() == 2:
        #     past_time_features = past_time_features.unsqueeze(-1)
        # if future_time_features.dim() == 2:
        #     future_time_features = future_time_features.unsqueeze(-1)
        
        # Create observed mask with actual batch size
        past_observed_mask = torch.ones(
            (actual_batch_size, seq_len, num_features), 
            device=device
        )
        
        # Generate predictions
        outputs = model.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
        )
        
        preds = outputs.sequences.mean(dim=1)
        confs = outputs.sequences.std(dim=1).mean(dim=2)

        # Print first batch results
        if batch_idx == 0:
            print(f"\nOutput shape: {outputs.sequences.shape}")
            print(f"Predictions (first 3 samples):")
            print(outputs.sequences.mean(dim=1)[:3])
            print(f"\nGround truth (first 3 samples):")
            print(future_values[:3])
            
            # Calculate and print error for first batch
            mse = torch.mean((preds - future_values) ** 2)
            print(f"\nMSE on first batch: {mse.item():.6f}")

        print(confs.shape, metadata.shape)

        preds = torch.cat((preds, metadata), dim=2)
        confs = torch.cat((torch.unsqueeze(confs, 1), metadata), dim=2)
        future_values = torch.cat((future_values, metadata), dim=2)

        predictions.extend(list(np.squeeze(preds.numpy())))
        truths.extend(list(np.squeeze(future_values.numpy().squeeze())))
        confidences.extend(list(confs.numpy().squeeze()))

        
        
        # Process all batches (you can accumulate predictions here)
        # if batch_idx >= 5:  # Just process first few batches for demo
        #     break

# print(confidences)
# header = "TimeInFile,PrecedingIntrvl,HiFreq,Bndwdth,FreqMaxPwr,PrcntMaxAmpDur,FreqKnee,PrcntKneeDur,StartF,FreqCtr,FFwd32dB,FFwd20dB,FFwd15dB,FBak5dB,FFwd5dB,Bndw32dB,Amp1stQrtl,Amp2ndQrtl,Amp3rdQrtl,Amp4thQrtl,AmpK@start,UpprKnFreq,HiFtoUpprKnAmp,HiFtoKnAmp,HiFtoFcAmp,UpprKnToKnAmp,KnToFcAmp,LdgToFcAmp"

# ground_truths = pd.concat(
#     [pd.DataFrame(d, columns = time_cols + target_cols + metadata_cols) for d in truths], 
#     ignore_index = True
# )
# ground_truths.to_csv(f"{model_path_prefix}/_ground_truths.log")
# quit(0)

preds_np = np.array(predictions)
np.savetxt(f"{model_path_prefix}/_predictions.log", preds_np, delimiter=",", header=",".join(train_dataloader.dataset.target_cols + train_dataloader.dataset.metadata_cols))#",".join(target_cols))
truths_np = np.array(truths)
np.savetxt(f"{model_path_prefix}/_ground_truths.log", truths_np, delimiter=",", header=",".join(train_dataloader.dataset.target_cols + train_dataloader.dataset.metadata_cols))#",".join(target_cols))
confidences_np = np.array(confidences)
np.savetxt(f"{model_path_prefix}/_confidences.log", confidences_np, delimiter=",", header="mean std across features," + ",".join(train_dataloader.dataset.metadata_cols))#",".join(target_cols))

print("\nInference completed successfully!")
# quit(0)