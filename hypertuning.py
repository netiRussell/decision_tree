import pandas as pd
import torch
import sys
import logging
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from datetime import datetime


from dt import selectDevice
from rf import RandomForest

# Optuna hypertuning function
def objective(trial):
  # Metric being evaluated
  accuracy = 0
  rf = None
  hyperparams = {
     "n_trees": 0,
     "min_samples_split": 0,
     "max_depth": 0,
     "num_features": 0
  }
  
  try:
    # Hyperparams and prints to keep track of the progress
    print(f"---- Current trial: {trial.number} ----")

    hyperparams["n_trees"] = trial.suggest_categorical("n_trees", [5, 10, 15, 20, 25, 30])
    hyperparams["min_samples_split"] = trial.suggest_categorical("min_samples_split", [2,3,4,5,6,7,8])
    hyperparams["max_depth"] = trial.suggest_categorical("max_depth", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    hyperparams["num_features"] = trial.suggest_categorical("num_features", [5, 10, 15, 20])

    print("----------------------------------------------------")
    print("🔧 Trial Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"\t• {key}: {value}")
    print("----------------------------------------------------\n\n")

    # Model
    rf = RandomForest(device, n_trees=hyperparams["n_trees"], min_samples_split=hyperparams["min_samples_split"], max_depth=hyperparams["max_depth"], num_features=hyperparams["num_features"])

    # Train
    rf.fit(satjob_train_X, satjob_train_target)

    # Test the accuracy for satjob
    prediction = rf.predict(satjob_test_X)
    accuracy = (prediction == satjob_test_target).float().mean().item()  
  
  # Free memory between each trial
  finally:
      del rf, hyperparams
      torch.cuda.empty_cache()



  return accuracy

if __name__ == "__main__":
  # Select device
  device = selectDevice()

  #  # # # # # # # # # # # # # #
  # --- Dataset Extraction --- #
  #  # # # # # # # # # # # # # #

  # Load in the dataset
  dataset = pd.read_parquet("/Users/ruslanabdulin/Desktop/CSUN/Fall25/COMP541/data/df_preprocessed.parquet")
  #dataset.info(verbose=True)

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


  #  # # # # # # # # # # #
  # ---- Hypertuning --- #
  #  # # # # # # # # # # #
  
  # Study name:
  study_name = "study_"+datetime.now().strftime("%m_%d_%y_%H_%M_%S")

  # Enabling print out
  optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

  # Enabling the reusability of the study
  storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)
      
  # Study definition
  study = optuna.create_study(
                              direction="maximize", 
                              sampler=TPESampler(multivariate=True, group=True, n_startup_trials=20),
                              pruner=None,
                              study_name=study_name,
                              storage=storage_name,
                              load_if_exists=True
                              )

  # Study with aggressive sampling
  study.optimize(objective, n_trials=250, timeout=None)


  #  # # # # # # # # # # # # # #
  # ---- Printing results ---- #
  #  # # # # # # # # # # # # # #

  complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

  print("Study statistics: ")
  print("  Number of finished trials: ", len(study.trials))
  print("  Number of complete trials: ", len(complete_trials))

  print("Best trial:")
  trial = study.best_trial

  print("\tValue: ", trial.value)

  print("\tParams: ")
  for key, value in trial.params.items():
      print(f"\t\t{key}: {value}")