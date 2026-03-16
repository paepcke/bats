from tqdm import tqdm
import numpy as np
import pandas as pd

class GlobalMeanPredictor():
    def __init__(self, sample_size=None, batch_size=16):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.global_mean = None

    def train(self, data):
        means = np.zeros(32)
        num_batches = 0
        for batch in tqdm(data):
            x_t, x_c, y_t, y_c = batch
            # print(y_c.shape)
            # print(y_c.squeeze().mean(dim=0).numpy())
            means += y_c.squeeze().mean(dim=0).numpy()
            num_batches += (y_c.shape[0] / 64)
            # print(means)
            if self.sample_size and self.sample_size < self.batch_size * num_batches:
                break
        means /= num_batches
        self.global_mean = means

    def test(self, data):
        truths = []
        errors = []
        preds = []

        for batch in tqdm(data):
            x_t, x_c, y_t, y_c = batch
            for row in y_c:
                truths.append(row.numpy()[0])
                preds.append(self.global_mean)
                errors.append((row - self.global_mean).numpy()[0])

        truths, preds, errors = np.array(truths), np.array(preds), np.array(errors)

        target_columns = data.dataset.target_cols

        mae = np.abs(errors).mean(axis=0)
        mse = (errors * errors).mean(axis=0)

        mse_df = pd.DataFrame(np.array([target_columns, mse]).T)
        mse_df = mse_df.set_index(0)
        mse_df[1] = mse_df[1].astype(float)
        # mse_df[1] = mse_df[1].round(6)
        return mse_df
