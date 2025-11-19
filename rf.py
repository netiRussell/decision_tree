# Random forest
import numpy as np
import pandas as pd
import torch # Useful to potentially bring computations to CUDA
import pickle

from dt import DecisionTree

class RandomForest():

  def __init__(self, device, n_trees=10, min_samples_split=2, max_depth=100, num_features=None):
    # Stopping criterias
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth

    # Other parameters
    self.n_trees = n_trees
    self.num_features = num_features
    self.trees = [None]*self.n_trees
    self.device = device

  def fit(self, X, target):
    # Main logic
    for i in range(self.n_trees):
      print(f"Current tree{i}")

      # Create a tree
      tree = DecisionTree(self.device, self.min_samples_split, self.max_depth, self.num_features )
      # Get a subset of the data
      X_datasubset, target_datasubset = self._get_subsets(X, target) 
      # Fit the tree
      tree.fit(X_datasubset, target_datasubset)
      self.trees[i] = tree

  def predict(self, X):
    # Get predictions from each tree
    predictions = torch.tensor([tree.predict(X) for tree in self.trees], dtype=torch.int64, device=self.device)

    # Swap axes to have array per prediction of all trees, not per tree
    predictions = torch.swapaxes(predictions, 0, 1)

    # The majority vote per prediction
    return torch.tensor([self._most_common_label(pred) for pred in predictions], dtype=torch.int64, device=self.device)

  # ----------------------
  # -- Static Functions --
  # ----------------------
  def _get_subsets(self, X, target):
    n_samples = X.shape[0]

    # random indices WITH replacement
    idxs = torch.randint(
        low=0,
        high=n_samples,
        size=(n_samples,),
        device=X.device,
        dtype=torch.long
    )
    return X[idxs], target[idxs]
  
  def _most_common_label(self, target):
    values, counts = torch.unique(target, return_counts=True)  
    return values[torch.argmax(counts)].item()
  
  def _save(self, path: str):
        """Save trained tree to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
  
  @staticmethod
  def _load(path: str, device=None):
      """Load trained Random Forest from disk."""
      with open(path, "rb") as f:
          rf: "RandomForest" = pickle.load(f)
      if device is not None:
          rf.device = device
      return rf
