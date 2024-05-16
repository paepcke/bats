import pandas as pd
import numpy as np

class SplitDataFrame():
    def __init__(self, 
                 path, 
                 prefix = 'split',
    ):
        assert path is not None
        assert prefix is not None

        self.mapping_df = pd.read_csv(f"{root_path}/{prefix}_mapping.csv")