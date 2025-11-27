'''
Brief explanation of the Decision Tree's logic:
- The model learns a tree by splitting data according to mathematical rules
- The split choice is motivated by the most optimal information gain
- While the uncertainty(how mixed with different targets the result is) is measured with Gini Impurity
- The model must compare every possible split

- Information gain = E(parent) - (sum_of_every_E(child_i) * weights)
where E = entropy
- Gini Impurity

- Possible questions that define splits are any possible combinations feature >= value

- DT is a greedy algorithm => no guarantee for the most optimal result
'''


import numpy as np
import pandas as pd
import torch # Useful to potentially bring computations to CUDA
import pickle
from collections import Counter

'''
Numerical(numbers) = 0
Nominal(strings) / Categorical = 1
Target(satfin/satjob) = 3
'''
feature_data_type_lookup_table = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3]

# ------------------
# -- Main classes --
# ------------------

def selectDevice():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Device selected: {device}")
    
    return device

class Node():
  def __init__(self, feature_name=None, feature_id=None, value=None, left=None, right=None, leafValue=None):
    self.feature_name = feature_name
    self.feature = feature_id  # column
    self.value = value      # threshold for numerical, category title/value for categorical/nominal
    self.left = left        # left node 
    self.right = right      # right node
    self.leafValue = leafValue   # the final value for a leaf node
  
  def is_leaf_node(self):
    return self.leafValue is not None
  
  def describe_yourself(self, depth=0):
    # [RECURSIVE]
    indent = "    " * depth

    # Base case
    if self.is_leaf_node():
      return f"{indent}Leaf: {self.leafValue}\n"

    out = f"{indent}Current Node's value: {self.value}\n"

    # Left side
    out += f"{indent}Left side ({self.feature_name} <= {self.value})\n"
    if self.left is not None:
      out += self.left.describe_yourself(depth=(depth+1))
    else:
      out += "{indent}Empty\n"

    # Right side
    out += f"{indent}Right side ({self.feature_name} > {self.value})\n"
    if self.right is not None:
      out += self.right.describe_yourself(depth=(depth+1))
    else:
      out += f"{indent}Empty\n"

    return out

class DecisionTree():
  def __init__(self, device, min_samples_split=2, max_depth=100, num_features=None, feature_names=None):
    # Stopping criterias
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth

    # Other parameters
    self.num_features = num_features # Essential for random forest to use subsets of features
    self.device = device
    self.feature_names = feature_names

    # General
    self.root = None

  def fit(self, dataset, target):
    # X = dataset, y = target
    # _func_name = local function helper, not meant to be invoked by the user

    # Transform dataset and target into PyTorch tensors on the device(CUDA/CPU)
    '''
    # In the current implementation, main.py provides tensors already
    dataset = torch.tensor(dataset, dtype=torch.float32, device=self.device)
    target = torch.tensor(target, dtype=torch.long, device=self.device)
    '''
    
    # Ensure we have the correct number of features
    self.num_features = dataset.shape[1] if self.num_features is None else min(dataset.shape[1], self.num_features)

    # Get current root
    self.root = self._grow_tree(dataset, target)
   
  def predict(self, dataset): 
    return torch.tensor([self._traverse_tree(x, self.root) for x in dataset], dtype=torch.int64, device=self.device)

  # ----------------------
  # -- Static Functions --
  # ----------------------

  def _grow_tree(self, dataset, target, depth=0):
    # [RECURSIVE]

    # Get the parameters
    num_samples, total_num_features = dataset.shape
    num_target_types = len(torch.unique(target))

    # Check on the stopping criterias
    # The node must become a leaf if
      # The maximum depth is reached(based on max_depth threshold)
      # It contains only one type of targets(therefore, split is at its best performance)
      # There is too few samples to split (based on min_samples_split threshold)
    if( depth >= self.max_depth or num_target_types == 1 or num_samples < self.min_samples_split):
      leafValue = self._most_common_label(target)
      return Node(leafValue=leafValue)

    # If the stopping criterias are not met
    # Generate specific number(self.num_features) of random unique feature IDs
    feature_ids = torch.randperm(total_num_features, device=self.device)[:self.num_features]

    # Find the best split
    best_feature_name, best_feature_id, best_value, left_ids, right_ids = self._best_split(dataset, target, feature_ids)

    # If no valid split found, return a leaf node
    if (left_ids is None or right_ids is None 
        or len(left_ids) == 0 or len(right_ids) == 0):
        leafValue = self._most_common_label(target)
        return Node(leafValue=leafValue)
    
    # Recursive call to itself
    left = self._grow_tree(dataset[left_ids, :], target[left_ids], depth+1,)
    right = self._grow_tree(dataset[right_ids, :], target[right_ids], depth+1,)

    # Return the split info as a Node
    return Node(best_feature_name, best_feature_id, best_value, left, right)

  def _most_common_label(self, target):
    values, counts = torch.unique(target, return_counts=True)
    
    return values[torch.argmax(counts)].item()

  def _best_split(self, dataset, target, feature_ids):
    # Counter
    best_gain = torch.tensor(-1, dtype=torch.float32, device=self.device)

    # Final result holders
    split_feature_name, split_feature_id, split_values, left_ids, right_ids = None, None, None, None, None

    # Go over every feature (column) among the chosen subset
    for feature_id in feature_ids:
      X_column = dataset[:, feature_id]
      
      # Get unique values in the current feature (column)
      values = torch.unique(X_column)

      # Go over all of the possible thresholds to find the best split
      for value in values:
        gain, temp_left_ids, temp_right_ids = self._information_gain(target, X_column, value)

        # Update current best result
        if gain > best_gain:
          best_gain = gain
          split_feature_name = self.feature_names[feature_id]
          split_feature_id = feature_id
          split_values = value
          left_ids = temp_left_ids
          right_ids = temp_right_ids
      
    return split_feature_name, split_feature_id, split_values, left_ids, right_ids

  def _information_gain(self, target, X_column, value):
    # Parent entropy
    parent_entropy = self._entropy(target)

    # Get indices of children by splitting with the value as a threshold
    left_ids, right_ids = self._split(X_column, value)

    # In case the split resulted in an empty list, return 0 
    # since the split is practically impossible / meaningless
    n_left = len(left_ids)
    n_right = len(right_ids)
    if (n_left == 0) or (n_right == 0):
      return torch.tensor(-1.0, dtype=torch.float32, device=self.device), left_ids, right_ids

    # Prepare values to calculate the entropy for children
    n = len(target)
    entropy_left, entropy_right = self._entropy(target[left_ids]), self._entropy(target[right_ids])

    # Compute the weighte avg. entropy for children
    children_entropy = (n_left/n)*entropy_left + (n_right/n)*entropy_right

    # Calculate the information gain
    information_gain = parent_entropy - children_entropy 

    return information_gain, left_ids, right_ids 

  def _entropy(self, target):
    # Get the probability
    probs = torch.bincount(target) / len(target)

    # Apply the formula
    return -(torch.sum(torch.tensor([prob*torch.log2(prob) for prob in probs if prob > 0], dtype=torch.float32, device=self.device)))
  
  def _split(self, X_column, value):
    # Split by getting corresponding flat list of indices
    left_ids = torch.argwhere(X_column <= value).flatten()
    right_ids = torch.argwhere(X_column > value).flatten()

    return left_ids, right_ids
  
  def _traverse_tree(self, dataset, node):
    # [RECURSIVE]

    # Base case  
    if node.is_leaf_node():
      return node.leafValue
    
    # Traverse to the left or right
    if dataset[node.feature] <= node.value:
      return self._traverse_tree(dataset, node.left)
     
    return self._traverse_tree(dataset, node.right)

  def to_dict(self):
    # [RECURSIVE]
    
    def node_to_dict(node):
        # Base case
        if node.is_leaf_node():
            return {"leaf": int(node.leafValue), "leafValue": node.leafValue}
        
        return {
            "feature": int(node.feature),
            "feature_name": node.feature_name,
            "value": float(node.value),
            "left": node_to_dict(node.left),
            "right": node_to_dict(node.right)
        }
    
    return node_to_dict(self.root)
 

  def __repr__(self):
    # [RECURSIVE]

    # Ensure the tree has been generated already
    if self.root is None:
      return "DecisionTree(root=None)"
    
    return self.root.describe_yourself(depth=0)
    
  
  def _save(self, path: str):
        """Save trained tree to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
  
  @staticmethod
  def _load(path: str, device=None):
      """Load trained tree from disk."""
      with open(path, "rb") as f:
          tree: "DecisionTree" = pickle.load(f)
      if device is not None:
          tree.device = device
      return tree
