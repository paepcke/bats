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
from bats_dataset.bats_dataset import BatsCSVDataset, BatsCSVDatasetWithMetadata

class InformerPredictor():
    def __init__(self, num_features, num_epochs=10, batch_size=16,
                 seq_len=46, prediction_length=1, model_path="./"):
        self.num_features = num_features
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.prediction_length = prediction_length
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(f"{model_path}/plots"):
            os.mkdir(f"{model_path}/plots")

        self.config = InformerConfig(
            input_size=self.num_features,
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
        self.model = InformerForPrediction(self.config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def train(self, data):
        # 4. Sample DataLoader loop (you must build your own DataLoader)
        losses = []
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            step = 0
            for batch in tqdm(data):
                x_t, x_c, y_t, y_c, _ = batch
                actual_batch_size = x_c.shape[0]
                # print(x_t)
                # print(x_c)
                # print(y_t)
                # print(y_c)
                # print([x.shape for x in batch])
                # batch[past_values] shape: (batch_size, seq_len, features)
                # batch[future_values] shape: (batch_size, pred_len, features)
                past_values = x_c.to(self.device)
                future_values = y_c.to(self.device)
                # If using time features or observed masks:
                past_time_features = x_t.to(self.device)
                future_time_features = y_t.to(self.device)
                past_observed_mask = torch.ones((actual_batch_size, self.seq_len, self.num_features), device=self.device)
                # if past_observed_mask is None:
                #     past_observed_mask = torch.ones((batch_size, seq_len, self.num_features), device=self.device)
                # print(past_values.shape)
                # print(future_values.shape)
                # print(past_time_features.shape)
                # print(future_time_features.shape)
                # print(past_observed_mask.shape)

                self.optimizer.zero_grad()
                outputs = self.model(
                    past_values=past_values,
                    future_values=future_values,
                    past_time_features=past_time_features,
                    future_time_features=future_time_features,
                    past_observed_mask=past_observed_mask,
                    future_observed_mask=torch.ones_like(future_values)
                    # include other optional args if you have them
                )

                # The model may return a dict or tuple; adjust accordingly:
                # For example: outputs.last_hidden_state of shape (batch_size, pred_len, c_out)
                preds = outputs
                loss = preds.loss
                # print(preds)
                # quit(0)

                # loss = criterion(preds, future_values)
                loss.backward()
                self.optimizer.step()

                if step % 20 == 0:
                    print(sum(losses[-20:]) / 20)
                step += 1
                losses.append(loss.item())
                # quit(0)
                # print(f"Epoch {epoch+1}/{self.num_epochs} — Training Loss: {avg_loss:.6f}")
        plt.plot(losses)
        plt.savefig(f"{self.model_path}/plots/loss.png")
        self.model.save_pretrained(self.model_path)
        self.config.save_pretrained(self.model_path)

    def test(self, data):
        print("Running inference on test data...")
        self.model.eval()

        predictions = []
        truths = []
        confidences = []
        mses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data)):
                x_t, x_c, y_t, y_c, metadata = batch
                
                # Get actual batch size for this batch
                actual_batch_size = x_c.shape[0]
                
                # Move to device
                past_values = x_c.to(self.device)
                future_values = y_c.to(self.device)
                past_time_features = x_t.to(self.device)
                future_time_features = y_t.to(self.device)
                
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
                    (actual_batch_size, self.seq_len, self.num_features), 
                    device=self.device
                )
                
                # Generate predictions
                outputs = self.model.generate(
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
                
        preds_np = np.array(predictions)
        np.savetxt(f"{self.model_path}/_predictions.log", preds_np, delimiter=",", header=",".join(data.dataset.target_cols + data.dataset.metadata_cols))#",".join(target_cols))
        truths_np = np.array(truths)
        np.savetxt(f"{self.model_path}/_ground_truths.log", truths_np, delimiter=",", header=",".join(data.dataset.target_cols + data.dataset.metadata_cols))#",".join(target_cols))
        confidences_np = np.array(confidences)
        np.savetxt(f"{self.model_path}/_confidences.log", confidences_np, delimiter=",", header="mean std across features," + ",".join(data.dataset.metadata_cols))#",".join(target_cols))

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--n_epochs", type=int, default=10)
        parser.add_argument("--seq_len", type=int, default=46)
        parser.add_argument("--model_path", type=str, default="/home/ayc227/bats/bats_transformer/models/informer_test")