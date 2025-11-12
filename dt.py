'''
Brief explanation of the Decision Tree's logic:
- The model learns a tree by splitting data according to mathematical rules
- The split choice is motivated by the most optimal information gain
- While the uncertainty(how mixed with different targets the result is) is measured with Gini Impurity
- The model must compare every possible split

- Information gain = E(parent) - (sum_of_every_E(child_i) * weights)
where E = entropy
- Gini Impurity

- Possible questions that define splits are any possible combinations column >= value

- DT is a greedy algorithm => no guarantee for the most optimal result
'''


import numpy as np
import pandas as pd
import torch # Useful to potentially bring computations to CUDA

'''
Numerical(numbers) = 0
Nominal(strings) / Categorical = 1
Target(satfin/satjob) = 3
'''
column_data_type_lookup_table = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3]

# Question can be think of as a split node
class Question():
  def __init__(self, column=None, value=None):
    assert self.column <= (len(column_data_type_lookup_table)-1), "Column is outside of the lookup table boundaries"

    column_data_type = column_data_type_lookup_table[self.column]
    assert column_data_type == 3, "Target column is not supposed to be used in the split node"
    
    self.column = column
    self.value = value

    # Based on the data type, define the match and represent functions' behavior
    # Sample = a single row of data
    if( column_data_type == 0 ):
      # Numerical
      self.match = lambda sample: sample[self.column] >= self.value
      self.__repr__ = f"Question: {dataset.columns[self.column]} >= {self.value}" 
    else:
      # Nominal / Categorical
      self.match = lambda sample: sample[self.column] == self.value
      self.__repr__ = f"Question: {dataset.columns[self.column]} == {self.value}" 


if __name__ == "__main__":
  # Load in the dataset
  dataset = pd.read_parquet("/Users/ruslanabdulin/Desktop/CSUN/Fall25/COMP541/data/df_preprocessed.parquet")
  # dataset.info(verbose=True)