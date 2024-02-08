import pandas as pd 
import argparse

def add_cli(parser):
    parser.add_argument('--input_data_path', type=str, default='data/data_2.csv', help='input file')


def preprocess(args):
    
    df = pd.read_csv(args.input_data_path)
    max_seq_len = df.groupby("Filename").size().max()
    df.fillna(0, inplace=True)
    if not args.pca_components:
        return df, max_seq_len
    else:
        from sklearn.decomposition import PCA
        print(args.ignore_cols)
        pca = PCA(n_components=args.pca_components)
        pca.fit(df.drop(columns=args.ignore_cols + ["TimeIndex"]))
        df_reduced = pd.DataFrame(pca.transform(df.drop(columns=args.ignore_cols + ["TimeIndex"])), columns=["pca_"+str(i) for i in range(args.pca_components)])

        #only append the time column
        df_reduced["TimeIndex"] = df["TimeIndex"]

        #return the reduced dataframe, and information such that you can construct the original dataframe
        return df, df_reduced, max_seq_len, pca
