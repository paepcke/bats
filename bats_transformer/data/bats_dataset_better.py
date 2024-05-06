import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import joblib #this is used to "unpickle" the scaler

import spacetimeformer as stf

import matplotlib.pyplot as plt


class BatsCSVDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_path = '/home/vdesai/bats_data/new_dataset/splits',
                 prefix = 'split',
                 ignore_cols = [],
                 target_cols = [],
                 time_col_name = "TimeIndex",
                 val_split = 0.1, 
                 test_split = 0.1, 
                 context_points = 57,
                 target_points = 1,
                 split = "train"
    ):
        assert root_path is not None
        assert prefix is not None

        self.root_path = root_path
        self.prefix = prefix
        
        self.mapping_df = pd.read_csv(f"{root_path}/{prefix}_mapping.csv")
        self.config_df = pd.read_csv(f"{root_path}/{prefix}_config.csv")

        self.max_length = self.config_df[self.config_df.parameter == "max_length"]["value"].values[0]
        self.min_length = self.config_df[self.config_df.parameter == "min_length"]["value"].values[0]

        assert context_points is None or target_points is None

        if context_points is None and target_points is None:
            context_points = self.max_length - 1
            target_points = 1

        elif context_points is None:
            context_points = self.max_length - target_points

        else:
            target_points = self.max_length - context_points
                    
        self.seq_length = context_points + target_points
        self.time_col_name = time_col_name
        self.ignore_cols = ignore_cols
        self.context_points = context_points
        self.target_points = target_points
        self.split = split
        self.metadata_cols = ["Filename", "Cntxt_sz"]

        self.val_split = val_split
        self.test_split = test_split
        self.train_split = 1 - val_split - test_split

        assert self.train_split > 0

        self.run_sanity_check()        
        
        self.mapping_df["cumulative_count"] = self.mapping_df["n_samples"].cumsum() - self.mapping_df["n_samples"]

        self.total_chirps = self.mapping_df["n_samples"].sum()        
        self.train_chirps = int(self.total_chirps * self.train_split)
        self.val_chirps = int(self.total_chirps * self.val_split)
        self.test_chirps = int(self.total_chirps * self.test_split)                

        ## Loading the scaler
        self.scaler = joblib.load(f"{root_path}/{prefix}_scaler.pkl")
        self.scaler.set_output(transform = "pandas")
        self.scaler_cols = list(self.scaler.get_feature_names_out())
        
        if not target_cols:
            target_cols = pd.read_feather(
                            os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"].split("/")[-1])
                        ).columns.tolist()
            if time_col_name in target_cols:
                target_cols.remove(time_col_name)
            
            for col in ignore_cols:
                if col in target_cols:
                    target_cols.remove(col)
        
        self.target_cols = target_cols
        self.split = split

    
    def run_sanity_check(self):
        #reading a single df to make sure the time column is in there.
        df = pd.read_feather(os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"].split("/")[-1]))
        #assert self.time_col_name in df.columns

        #check that every file in the mapping df actually exists
        for filename in self.mapping_df["Filename"]:
            assert os.path.exists(os.path.join(self.root_path, filename.split("/")[-1]))
        
        #check that the count in the mapping df actually is equal to the number of rows in the file
        for idx, row in self.mapping_df.iterrows():
            df = pd.read_feather(os.path.join(self.root_path, row["Filename"].split("/")[-1]))
            n_samples = row["n_samples"]
            file_id_to_samples = df.groupby("file_id")["chirp_idx"].max().reset_index()
            file_id_to_samples["n_samples"] = file_id_to_samples["chirp_idx"] - self.min_length + 2
            total_samples = file_id_to_samples["n_samples"].sum()
            assert n_samples == total_samples
        
    
    def __len__(self):
        return {
            "train": self.train_chirps,
            "val": self.val_chirps,
            "test": self.test_chirps
        }[self.split]
        
    
    #add one more dimension to the tensor if only 1 dimensional
    def _torch(self, *dfs):
        return tuple(
                torch.from_numpy(x.values).float().unsqueeze(1) if len(x.shape) == 1 
                else torch.from_numpy(x.values).float() for x in dfs
            )
    
    def make_len(self, df, seq_len):
        #pad with rows containing zeros to make df of length seq_len
        if len(df) < seq_len:
            df = pd.concat([pd.DataFrame(np.zeros((seq_len - len(df), len(df.columns))), columns = df.columns), df], axis = 0)

        df[self.time_col_name] = StandardScaler().fit_transform(np.arange(seq_len).reshape(-1,1))
        return df
    
    def __getitem__(self, idx):
        if self.split == "val":
            idx += self.train_chirps
        elif self.split == "test":
            idx += self.train_chirps + self.val_chirps

        split_to_use = self.mapping_df[self.mapping_df["cumulative_count"] <= idx].iloc[-1]
        filename = split_to_use["Filename"]
        sample_idx = idx - split_to_use["cumulative_count"]

        df = pd.read_feather(filename)
        
        #get the file index from mapping_df
        file_idx = self.mapping_df[self.mapping_df["cumulative_count"] <= idx].index[-1]
        chirp_idx = idx - self.mapping_df.iloc[file_idx]["cumulative_count"]
        chirp_idx *= self.seq_length

        filename = self.mapping_df.iloc[file_idx]["Filename"]
        df = pd.read_feather(os.path.join(self.root_path, filename.split("/")[-1]))

        file_id_to_samples = df.groupby("file_id")["chirp_idx"].max().reset_index()
        
        file_id_to_samples["n_samples"] = file_id_to_samples["chirp_idx"] - self.min_length + 2
        file_id_to_samples["cum_samples"] = file_id_to_samples["n_samples"].cumsum() - file_id_to_samples["n_samples"]
        print(file_id_to_samples)
        
        file_id_to_use_ = file_id_to_samples[file_id_to_samples["cum_samples"] <= sample_idx].iloc[-1]
        file_id_to_use = file_id_to_use_["file_id"]
        chirps_to_use = sample_idx - file_id_to_use_["cum_samples"]
        df_slice = df[df.file_id == file_id_to_use].copy()

        if self.ignore_cols:
            df_slice.drop(columns=self.ignore_cols, inplace=True, errors = 'ignore')

        series_slice = self.make_len(df_slice.iloc[:-chirps_to_use] if chirps_to_use > 0 else df_slice, self.seq_length)

        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :]
        )

        ctxt_x = ctxt_slice[self.time_col_name]
        trgt_x = trgt_slice[self.time_col_name]

        ctxt_y = ctxt_slice[self.target_cols]
        trgt_y = trgt_slice[self.target_cols]

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)

    #Function which scales the data back into the original space
    def scale_data(self, data):
        return pd.DataFrame(
                self.scaler.inverse_transform(pd.DataFrame(data, columns = self.scaler_cols), copy = True),
                columns = self.scaler_cols,
        )[data.columns]