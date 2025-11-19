import pandas as pd
import torch
from dt import selectDevice
from rf import RandomForest

if __name__ == "__main__":
  # Select device
  device = selectDevice()

  # Load in the dataset
  dataset = pd.read_parquet("/Users/ruslanabdulin/Desktop/CSUN/Fall25/COMP541/data/df_preprocessed.parquet")
  #dataset.info(verbose=True)

  # Define the model
  rf = RandomForest(device, n_trees=10, min_samples_split=2, max_depth=20, num_features=20)

  # Get the training samples out for the satjob:
    # All of the ones that have a non-NaN value for the 'satjob' feature
  satjob_subset = dataset.loc[dataset['satjob'].notna()]

  # Build full X, y tensors
  X_all_satjob = torch.tensor(
      satjob_subset.drop(['satjob', 'satfin'], axis=1).to_numpy(dtype='float32'),
      dtype=torch.float32,
      device=device
  )
  y_all_satjob = torch.tensor(
      satjob_subset['satjob'].to_numpy(dtype='int64'),
      dtype=torch.int64,
      device=device
  )

  #  Train / test split 
  n_samples = X_all_satjob.shape[0]
  perm = torch.randperm(n_samples, device=device)  # shuffle indices
  
  # 80% train, 20% test
  train_size = int(0.8 * n_samples)
  train_idx = perm[:train_size]
  test_idx  = perm[train_size:]

  satjob_train_X, satjob_train_target = X_all_satjob[train_idx], y_all_satjob[train_idx]
  satjob_test_X,  satjob_test_target  = X_all_satjob[test_idx],  y_all_satjob[test_idx]

  # Train
  # print(f"Number of samples in X:{len(satjob_train_X)}, in Y:{len(satjob_train_target)}")
  print("Training has been started")
  rf.fit(satjob_train_X, satjob_train_target)

  # Test the accuracy for satjob
  print("Testing has been started")
  prediction = rf.predict(satjob_test_X)
  print(f"Accuracy: {torch.sum(prediction == satjob_test_target)/len(satjob_test_target)}")

  # Demonstrate the resulting tree
  #print("\n\n", rf)

  # Save
  #rf.save("decision_tree.pkl")
  #print("Decision Tree has been saved")