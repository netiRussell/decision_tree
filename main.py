import pandas as pd
import torch
from dt import selectDevice
from rf import RandomForest

# Visualization logic
from graphviz import Digraph

def _add_nodes(dot, node, name="0"):
    # [RECURSIVE]

    # Base case
    if "leaf" in node:
        dot.node(name, f"Leaf: {node['leafValue']}", shape="box")
    # Internal node
    else:
        label = f"{node['feature_name']} <= {node['value']:.3f}"
        dot.node(name, label)
        left_name = name + "L"
        right_name = name + "R"
        dot.edge(name, left_name, label="True")
        dot.edge(name, right_name, label="False")
        _add_nodes(dot, node["left"], left_name)
        _add_nodes(dot, node["right"], right_name)

def tree_to_graphviz(tree_dict):
    dot = Digraph()
    _add_nodes(dot, tree_dict, "0")
    return dot

# Configurations holder
config = {
  "label_name": "satjob" # either satjob or satfin
}

# Main logic
if __name__ == "__main__":
  # Select device
  device = selectDevice()

  # Load in the dataset
  dataset = pd.read_parquet("/Users/ruslanabdulin/Desktop/CSUN/Fall25/COMP541/data/df_preprocessed.parquet")
  #dataset.info(verbose=True)

  # Define the model
  rf = RandomForest(device, num_trees=20, min_samples_split=6, max_depth=5, num_features=5, feature_names=dataset.columns.tolist())

  # Get the training samples out for the satjob:
    # All of the ones that have a non-NaN value for the label feature
  label_subset = dataset.loc[dataset[config['label_name']].notna()]

  # Build full X, y tensors
  X_all = torch.tensor(
      label_subset.drop(['satjob', 'satfin'], axis=1).to_numpy(dtype='float32'),
      dtype=torch.float32,
      device=device
  )
  y_all_label = torch.tensor(
      label_subset[config['label_name']].to_numpy(dtype='int64'),
      dtype=torch.int64,
      device=device
  )

  #  Train / test split 
  n_samples = X_all.shape[0]
  perm = torch.randperm(n_samples, device=device)  # shuffle indices
  
  # 80% train, 20% test
  train_size = int(0.8 * n_samples)
  train_idx = perm[:train_size]
  test_idx  = perm[train_size:]

  train_X, train_target = X_all[train_idx], y_all_label[train_idx]
  test_X,  test_target  = X_all[test_idx],  y_all_label[test_idx]

  # Train
  # print(f"Number of samples in X:{len(train_X)}, in Y:{len(train_target)}")
  print("Training has been started")
  rf.fit(train_X, train_target)

  # Test the accuracy for satjob
  print("Testing has been started")
  prediction = rf.predict(test_X)
  print(f"Accuracy: {torch.sum(prediction == test_target)/len(test_target)}")

  # Demonstrate the resulting trees
  # Reminder: I utilize the majority vote approach; hence, no explicit connection between the trees
  rf_dict = rf.to_dict()
  for i in range(rf.num_trees):
    tree = rf_dict["trees"][i]
    dot = tree_to_graphviz(tree)
    dot.render(f"./plots/{config['label_name']}/tree{i}", format="png")
  print("Decision Trees have been illustrated and saved")

  # Save
  rf._save(f"./{config['label_name']}/RandomForest.pkl")
  print("RF has been saved")