# Random forest
import torch # Useful to potentially bring computations to CUDA
import pickle

from dt import DecisionTree

class RandomForest():

  def __init__(self, device=None, num_trees=10, min_samples_split=2, max_depth=100, num_features=None, feature_names=None):
    # Stopping criterias
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth

    # Other parameters
    self.num_trees = num_trees
    self.num_features = num_features
    self.trees = [None]*self.num_trees
    self.device = device
    self.feature_names = feature_names

  def fit(self, X, target):
    # Main logic
    for i in range(self.num_trees):
      #print(f"Current tree{i}")

      # Create a tree
      tree = DecisionTree(self.device, self.min_samples_split, self.max_depth, self.num_features, self.feature_names )
      # Get a subset of the data
      X_datasubset, target_datasubset = self._get_subsets(X, target) 
      # Fit the tree
      tree.fit(X_datasubset, target_datasubset)
      self.trees[i] = tree

  def predict(self, X):
    # Get predictions from each tree
    tree_preds = [tree.predict(X) for tree in self.trees]

    # Stack into shape (num_trees, n_samples)
    predictions = torch.stack(tree_preds, dim=0)

    # Swap axes to have array per prediction of all trees, not per tree
    predictions = torch.swapaxes(predictions, 0, 1)

    # The majority vote per prediction
    return torch.tensor([self._most_common_label(pred) for pred in predictions], dtype=torch.int64, device=self.device)

  def to_dict(self):
    # Turn the RF to Python dictionary for the visualization
    return {
        "num_trees": self.num_trees,
        "min_samples_split": self.min_samples_split,
        "max_depth": self.max_depth,
        "num_features": self.num_features,
        "trees": [tree.to_dict() for tree in self.trees]
    }

  def __repr__(self):
    # [RECURSIVE]
    lines = [f"RandomForest(num_trees={self.num_trees}, "
              f"min_samples_split={self.min_samples_split}, "
              f"max_depth={self.max_depth}, "
              f"num_features={self.num_features})"]
    for i, tree in enumerate(self.trees):
        lines.append(f"\n=== Tree {i} ===")
        lines.append(repr(tree))
    return "\n".join(lines)

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
