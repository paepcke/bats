from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import joblib
from argparse import ArgumentParser

# sample usage: python src/andrew/scripts/unscale_predictions.py --scaler_path bats_transformer/data/daytime_files_new/splits/split_scaler.pkl --pred_path bats_transformer/outputs/sample_loss_output.log --output_path bats_transformer/outputs/unscaled_predictions.csv

parser = ArgumentParser()
parser.add_argument('--scaler_path', type=str, default='../../bats_transformer/data/daytime_files_new/splits/split_scaler.pkl', help='Path to the saved scaler')
parser.add_argument('--pred_path', type=str, default='../../bats_transformer/outputs/sample_loss_output.log', help='Path to the predictions CSV file')
parser.add_argument('--output_path', type=str, default='../../bats_transformer/outputs/unscaled_predictions.csv', help='Path to save the unscaled predictions CSV file')
args = parser.parse_args()

scaler = joblib.load(open(args.scaler_path, 'rb'))

scaler.feature_names_in_

predictions = pd.read_csv(args.pred_path)

non_scaler_columns = {}

for idx, column in enumerate(predictions.columns):
    if column not in scaler.feature_names_in_:
        non_scaler_columns[idx] = predictions[column]
        predictions.drop(columns=[column], inplace=True)

all_features = list(scaler.feature_names_in_)
column_idxs = [] # indices of columns in predictions that are in scaler.feature_names_in_
for column in scaler.feature_names_in_:
    if column not in predictions.columns:
        predictions[column] = 0
    else:
        column_idxs.append(all_features.index(column))

predictions = predictions[scaler.feature_names_in_]

predictions_np = predictions.to_numpy()
predictions_np = predictions_np[:, :-1]

inverted_predictions = scaler.inverse_transform(predictions)

filtered_inverted_predictions = inverted_predictions[:, column_idxs]

filtered_inverted_df = pd.DataFrame(filtered_inverted_predictions, columns=[predictions.columns[i] for i in column_idxs])

for idx, col_data in non_scaler_columns.items():
    filtered_inverted_df.insert(idx, col_data.name, col_data.values)
print(filtered_inverted_df)

filtered_inverted_df.to_csv(args.output_path, index=False)