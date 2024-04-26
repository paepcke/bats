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

        self.mapping_df["cumulative_count"] = ((self.mapping_df["count"] // self.seq_length).cumsum() 
                                             - (self.mapping_df["count"] // self.seq_length))

        self.total_chirps = self.mapping_df["count"].sum() // self.seq_length        
        self.train_chirps = int(self.total_chirps * self.train_split)
        self.val_chirps = int(self.total_chirps * self.val_split)
        self.test_chirps = int(self.total_chirps * self.test_split)                



        if not target_cols:
            target_cols = pd.read_feather(
                            os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"].split("/")[-1])
                        ).columns.tolist()
            target_cols.remove(time_col_name)
            
            for col in ignore_cols:
                if col in target_cols:
                    target_cols.remove(col)
        
        self.target_cols = target_cols
        self.split = split

    
    def run_sanity_check(self):
        #reading a single df to make sure the time column is in there.
        df = pd.read_feather(os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"].split("/")[-1]))
        assert self.time_col_name in df.columns

        #check that every file in the mapping df actually exists
        for filename in self.mapping_df["Filename"]:
            assert os.path.exists(os.path.join(self.root_path, filename.split("/")[-1]))
        
        #check that the count in the mapping df actually is equal to the number of rows in the file
        for idx, row in self.mapping_df.iterrows():
            df = pd.read_feather(os.path.join(self.root_path, row["Filename"].split("/")[-1]))
            assert row["count"] == df.shape[0]
            assert df.shape[0] % (self.context_points + self.target_points) == 0
        
    
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

    def __getitem__(self, idx):
        if self.split == "val":
            idx += self.train_chirps
        elif self.split == "test":
            idx += self.train_chirps + self.val_chirps
        
        #get the file index from mapping_df
        file_idx = self.mapping_df[self.mapping_df["cumulative_count"] <= idx].index[-1]
        chirp_idx = idx - self.mapping_df.iloc[file_idx]["cumulative_count"]
        chirp_idx *= self.seq_length

        filename = self.mapping_df.iloc[file_idx]["Filename"]
        df = pd.read_feather(os.path.join(self.root_path, filename.split("/")[-1]))

        if self.ignore_cols:
            df.drop(columns=self.ignore_cols, inplace=True, errors = 'ignore')
        
        series_slice = df.reset_index(drop=True).iloc[chirp_idx:chirp_idx + self.context_points + self.target_points]
        assert(len(series_slice) == self.context_points + self.target_points), f"{idx}, {file_idx}, {chirp_idx}, {filename}, {len(series_slice)}, {self.context_points + self.target_points}"
        #print(self.time_col_name, self.context_points, series_slice)
        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :]
        )

        ctxt_x = ctxt_slice[self.time_col_name]
        trgt_x = trgt_slice[self.time_col_name]

        ctxt_y = ctxt_slice[self.target_cols]
        trgt_y = trgt_slice[self.target_cols]

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)

    
    


class BatsCSVDatasetWithMetadata(Dataset):
    def __init__(self, 
                 root_path = '/home/vdesai/bats_data/training_files/splits',
                 prefix = 'split',
                 ignore_cols = [],
                 target_cols = [],
                 metadata_cols = [],
                 time_col_name = "TimeIndex",
                 val_split = 0.1, 
                 test_split = 0.1, 
                 context_points = 57,
                 target_points = 1,
                 split = "train"
    ):
        assert root_path is not None
        assert prefix is not None
        assert len(metadata_cols) > 0, ("If you do not have any metadata to keep track of,"+
                                        "just use BatsCSVDataset instead")
        
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
        self.metadata_cols = metadata_cols

        self.val_split = val_split
        self.test_split = test_split
        self.train_split = 1 - val_split - test_split

        assert self.train_split > 0

        self.run_sanity_check()        

        self.mapping_df["cumulative_count"] = ((self.mapping_df["count"] // self.seq_length).cumsum() 
                                             - (self.mapping_df["count"] // self.seq_length))

        self.total_chirps = self.mapping_df["count"].sum() // self.seq_length        
        self.train_chirps = int(self.total_chirps * self.train_split)
        self.val_chirps = int(self.total_chirps * self.val_split)
        self.test_chirps = int(self.total_chirps * self.test_split)                



        if not target_cols:
            target_cols = pd.read_feather(
                            os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"].split("/")[-1])
                        ).columns.tolist()
            target_cols.remove(time_col_name)
            
            for col in ignore_cols:
                if col in target_cols:
                    target_cols.remove(col)
            
            for col in metadata_cols:
                if col in target_cols:
                    target_cols.remove(col)

        #assert that target_cols and metadata_cols have no entries in common
        assert len(set(target_cols).intersection(set(metadata_cols))) == 0
        assert len(set(target_cols).intersection(set(ignore_cols))) == 0
        assert len(set(metadata_cols).intersection(set(ignore_cols))) == 0

        self.target_cols = target_cols
        self.split = split

    
    def run_sanity_check(self):
        #reading a single df to make sure the time column is in there.
        df = pd.read_feather(os.path.join(self.root_path, self.mapping_df.iloc[0]["Filename"].split("/")[-1]))
        assert self.time_col_name in df.columns

        #check that every file in the mapping df actually exists
        for filename in self.mapping_df["Filename"]:
            assert os.path.exists(os.path.join(self.root_path, filename.split("/")[-1]))
        
        #check that the count in the mapping df actually is equal to the number of rows in the file
        for idx, row in self.mapping_df.iterrows():
            df = pd.read_feather(os.path.join(self.root_path, row["Filename"].split("/")[-1]))
            assert row["count"] == df.shape[0]
            assert df.shape[0] % (self.context_points + self.target_points) == 0
        
    
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
    
    def __getitem__(self, idx):
        if self.split == "val":
            idx += self.train_chirps
        elif self.split == "test":
            idx += self.train_chirps + self.val_chirps
        
        #get the file index from mapping_df
        file_idx = self.mapping_df[self.mapping_df["cumulative_count"] <= idx].index[-1]
        chirp_idx = idx - self.mapping_df.iloc[file_idx]["cumulative_count"]
        chirp_idx *= self.seq_length

        filename = self.mapping_df.iloc[file_idx]["Filename"]
        df = pd.read_feather(os.path.join(self.root_path, filename.split("/")[-1]))

        if self.ignore_cols:
            df.drop(columns=self.ignore_cols, inplace=True, errors = 'ignore')
        
        series_slice = df.reset_index(drop=True).iloc[chirp_idx:chirp_idx + self.context_points + self.target_points]
        assert(len(series_slice) == self.context_points + self.target_points), f"{idx}, {file_idx}, {chirp_idx}, {filename}, {len(series_slice)}, {self.context_points + self.target_points}"
        #print(self.time_col_name, self.context_points, series_slice)
        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :]
        )

        ctxt_x = ctxt_slice[self.time_col_name]
        trgt_x = trgt_slice[self.time_col_name]

        ctxt_y = ctxt_slice[self.target_cols]
        trgt_y = trgt_slice[self.target_cols]
        
        
        metadata = trgt_slice[self.metadata_cols]
        #print(ctxt_x.shape, ctxt_y.shape, trgt_x.shape, trgt_y.shape, file_idx, filename, chirp_idx)
        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y, metadata)

    


