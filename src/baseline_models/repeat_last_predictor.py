from tqdm import tqdm
import numpy as np
import pandas as pd

class RepeatLastPredictor():
    def __init__(self):
        pass

    def train(self, data=None):
        pass

    def test(self, data):
        truths = []
        preds = []
        errors = []

        for batch in tqdm(data):
            x_t, x_c, y_t, y_c = batch
            mask = x_t > 0
            lengths = mask.sum(dim=1)
            feature_sums = x_c.sum(dim=1)
            for i, row in enumerate(feature_sums):
                pred = x_c[i, -1, :]
                truths.append(y_c[i].numpy()[0])
                preds.append(pred.numpy())
                errors.append((y_c[i] - pred).numpy()[0])

        truths, preds, errors = np.array(truths), np.array(preds), np.array(errors)

        target_columns = data.dataset.target_cols

        mae = np.abs(errors).mean(axis=0)
        mse = (errors * errors).mean(axis=0)

        mse_df = pd.DataFrame(np.array([target_columns, mse]).T)
        mse_df = mse_df.set_index(0)
        mse_df[1] = mse_df[1].astype(float)
        return mse_df