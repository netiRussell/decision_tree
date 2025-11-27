"""
This file applies the learned trees onto the data
"""

import pandas as pd
import torch
from dt import selectDevice
from rf import RandomForest

config = {
  "label_name": "satfin", # either satjob or satfin
  "batch_size": 2500, # num of samples to predict per batch
}

if __name__ == "__main__":
  # Select device
  device = selectDevice()

  # Load in the dataset
  dataset = pd.read_parquet("/Users/ruslanabdulin/Desktop/CSUN/Fall25/COMP541/data/df_preprocessed_with_predictions.parquet")
  #dataset.info(verbose=True)

  # Get the input and the target rows out for the dataset:
  X = dataset.loc[(dataset[config['label_name']].isna()), ~dataset.columns.isin(['satfin', 'satjob'])]
  mask = dataset[config['label_name']].isna()
  rows_to_fill = dataset.index[mask]

  # Ensure that target will not be empty
  print(mask.shape)
  if(mask.shape[0] == 0):
    raise RuntimeError("Running prediction on the label without missing values")

  # Load in the model (Static method, no init needed) 
  rf = RandomForest._load(f"./{config['label_name']}/RandomForest.pkl", device=device)


  # Build full X, y tensors
  X = torch.tensor( X.to_numpy(dtype='float32'), dtype=torch.float32, device=device )

  # Predict in batches 
  print("Prediction has been started")
  num_batches = X.shape[0] // config['batch_size']
  for i in range(num_batches+1):
    print(f"Current batch: {i} out of {num_batches}")
    start = i * config['batch_size']
    end = start + config['batch_size'] 
    prediction = rf.predict(X[start:end]).cpu().numpy()
    dataset.loc[rows_to_fill[start:end], config['label_name']] = prediction

  # Saving the dataset
  print("Saving the updated dataset...")
  dataset.to_parquet( "/Users/ruslanabdulin/Desktop/CSUN/Fall25/COMP541/data/df_preprocessed_with_predictions2.parquet", index=False )
  print("Done.")