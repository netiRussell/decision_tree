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

# Question can be think of as a split node
class Question():
  def __init__(self, feature=None, value=None):
    assert self.feature <= (len(feature_data_type_lookup_table)-1), "feature is outside of the lookup table boundaries"

    feature_data_type = feature_data_type_lookup_table[self.feature]
    assert feature_data_type == 3, "Target feature is not supposed to be used in the split node"
    
    self.feature = feature
    self.value = value

    # Based on the data type, define the match and represent functions' behavior
    # Sample = a single row of data
    if( feature_data_type == 0 ):
      # Numerical
      self.match = lambda sample: sample[self.feature] >= self.value
      self.__repr__ = f"Question: {dataset.columns[self.feature]} >= {self.value}" 
    else:
      # Nominal / Categorical
      self.match = lambda sample: sample[self.feature] == self.value
      self.__repr__ = f"Question: {dataset.columns[self.feature]} == {self.value}" 


class Node():
  def __init__(self, feature=None, value=None, left=None, right=None, leafValue=None):
    self.feature = feature  # column
    self.value = value      # threshold for numerical, category title/value for categorical/nominal
    self.left = left        # left node 
    self.right = right      # right node
    self.leafValue = None   # the final value for a leaf node
  
  def is_leaf_node(self):
    return self.leafValue is None


class DesicitionTree():
  def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
    # Stopping criterias
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.num_features = num_features # Essential for random forest to use subsets of features

    # General
    self.root = None

  def fit(self, dataset, target):
    # X = dataset, y = target
    # _func_name = local function helper, not meant to be invoked by the user
    
    # Ensure we have the correct number of features
    self.num_features = dataset.shape[1] if self.num_features is None else min(dataset.shape[1], self.num_features)

    # Get current root
    self.root = self._grow_tree(dataset, target)

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
    feature_ids = torch.multinomial(total_num_features, self.num_features, replacement=True )

    # Find the best split
    best_feature, best_value, left_ids, right_ids = self._best_split(dataset, target, feature_ids)
    
    # Recursive call to itself
    left = self._grow_tree(dataset[left_ids, :], target[left_ids], depth+1,)
    right = self._grow_tree(dataset[right_ids, :], target[right_ids], depth+1,)

    # Return the split info as a Node
    return Node(best_feature, best_value, left, right)


  def _most_common_label(target):
    counter = Counter(target)

    # Get the most common element. Returns array of (element's value, frequency) tuple
    holder = counter.most_common(1)
    # Open up the array since we have only one tuple in it
    holder = holder[0]
    # Get the value
    holder = holder[0]

    return holder


  def _best_split(self, dataset, target, feature_ids):
    # Counter
    best_gain = -1

    # Final result holders
    split_ids, split_values, left_ids, right_ids = None, None, None, None

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
          split_ids = feature_id
          split_values = value
          left_ids = temp_left_ids
          right_ids = temp_right_ids
      
    return split_ids, split_values, left_ids, right_ids
  

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
      return 0

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
    return -(torch.sum([prob*torch.log2(prob) for prob in probs if prob > 0]))
  

  def _split(self, X_column, value):
    # Split by getting corresponding flat list of indices
    left_ids = torch.argwhere(X_column <= value).flatten()
    right_ids = torch.argwhere(X_column > value).flatten()

    return left_ids, right_ids



  def predict(self): 
    pass

if __name__ == "__main__":
  # Load in the dataset
  dataset = pd.read_parquet("/Users/ruslanabdulin/Desktop/CSUN/Fall25/COMP541/data/df_preprocessed.parquet")
  # dataset.info(verbose=True)
