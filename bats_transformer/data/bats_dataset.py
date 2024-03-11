import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

import spacetimeformer as stf

import matplotlib.pyplot as plt


'''
Custom class for dealing with the special case of our sonobats dataset.
'''

class BatsCSVDataset(Dataset):
    def __init__(self, 
                 root_path = '/home/vdesai/bats_data/training_files/splits',
                 prefix = 'split',
                 ignore_cols = [],
                 time_col_name = "TimeIndex",
                 val_split = 0.15, 
                 test_split = 0.15, 
                 normalize = True, 
                 drop_all_nan = False,
                 time_features = ["year", "month", "day", "hour", "minute"], 
                 context_points = 31,
                 target_points = 1
    ):
        assert root_path is not None
        assert prefix is not None

        self.root_path = root_path
        self.prefix = prefix
        
        self.mapping_df = pd.read_csv(f"{root_path}/{prefix}_mapping.feather")
        self.num_chirps_ = self.mapping_df.iloc[0]["count"]
        self.num_files_ = self.mapping_df.shape[0]

        self.mapping_df["cumulative_count"] = self.mapping_df["count"].cumsum()
        self.time_col_name = time_col_name

        self.ignore_cols = ignore_cols
        self.context_points = context_points
        self.target_points = target_points

        if not target_cols:
            target_cols = pd.read_feather(
                            os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"])
                        ).columns.tolist()
            target_cols.remove(time_col_name)
            
            for col in ignore_cols:
                target_cols.remove(col)
        
        self.target_cols = target_cols

    
    def run_sanity_check(self):
        df = pd.read_feather(os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"]))
        assert self.time_col_name in df.columns
    
    def __len__(self):
        return self.num_chirps_ * self.num_files_
    
    def _torch(self, *dfs):
        return tuple(torch.from_numpy(x.values).float() for x in dfs)

    def __getitem__(self, idx):
        file_idx = self.mapping_df[self.mapping_df["cumulative_count"] <= idx].shape[0] - 1
        chirp_idx = idx - self.mapping_df.iloc[file_idx]["cumulative_count"]
        filename = self.mapping_df.iloc[file_idx]["Filename"]
        df = pd.read_feather(os.path.join(self.root_path, filename))
        if self.ignore_cols:
            df.drop(columns=self.ignore_cols, inplace=True)
        
        series_slice = df.iloc[chirp_idx]
        
        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :],
        )

        ctxt_x = ctxt_slice[self.time_col_name]
        trgt_x = trgt_slice[self.time_col_name]

        ctxt_y = ctxt_slice[self.target_cols]
        trgt_y = trgt_slice[self.target_cols]

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)

    

    


