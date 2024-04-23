from argparse import ArgumentParser
import torch
import numpy as np

import pytorch_lightning as pl
import spacetimeformer as stf
import pandas as pd

from pytorch_lightning.loggers import WandbLogger
from data import preprocess
import time
import tqdm
import requests
import os

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"];
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"];
      
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.json()

def send_telegram_image(image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    data = {
        "chat_id": CHAT_ID,
    }
    files = {
        "photo": open(image_path, "rb")
    }
    response = requests.post(url, data=data, files=files)
    return response.json()

def spacetimeformer_predict(
        model,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        og_device = y_c.device
        # move to model device
        x_c = x_c.to(model.device).float()
        x_t = x_t.to(model.device).float()
        # move y_c to cpu if it isn't already there, scale, and then move back to the model device
        y_c = y_c.to(model.device).float()
        # create dummy y_t of zeros
        yhat_t = (
            torch.zeros((x_t.shape[0], x_t.shape[1], model.d_yt)).to(model.device).float()
        )

        with torch.no_grad():
            # gradient-free prediction
            normalized_preds, *_ = model.forward(
                x_c, y_c, x_t, yhat_t, **model.eval_step_forward_kwargs
            )

        
        return normalized_preds.to(og_device).float();

def spacetimeformer_predict_calculate_loss(
        model,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> torch.Tensor:
        og_device = y_c.device
        # move to model device
        x_c = x_c.to(model.device).float()
        x_t = x_t.to(model.device).float()
        # move y_c to cpu if it isn't already there, scale, and then move back to the model device
        y_c = y_c.to(model.device).float()
        # create dummy y_t of zeros
        yhat_t = (
            torch.zeros((x_t.shape[0], x_t.shape[1], model.d_yt)).to(model.device).float()
        )

        with torch.no_grad():
            # gradient-free prediction
            normalized_preds, *_ = model.forward(
                x_c, y_c, x_t, yhat_t, **model.eval_step_forward_kwargs
            )

        error = normalized_preds - y_t;
        error = error**2;
        
        return error.to(og_device).float();

def spacetimeformer_predict_calculate_loss_pca(
        model,
        pca,
        scaler,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> torch.Tensor:
        og_device = y_c.device
        # move to model device
        x_c = x_c.to(model.device).float()
        x_t = x_t.to(model.device).float()
        # move y_c to cpu if it isn't already there, scale, and then move back to the model device
        y_c = y_c.to(model.device).float()
        # create dummy y_t of zeros
        yhat_t = (
            torch.zeros((x_t.shape[0], x_t.shape[1], model.d_yt)).to(model.device).float()
        )

        with torch.no_grad():
            # gradient-free prediction
            normalized_preds, *_ = model.forward(
                x_c, y_c, x_t, yhat_t, **model.eval_step_forward_kwargs
            )

        normalized_preds = scaler(pca.inverse_transform(
            model._inv_scaler(normalized_preds).cpu().numpy()
        ));

        error = normalized_preds - y_t.cpu().numpy();
        error = error**2;
        
        return error