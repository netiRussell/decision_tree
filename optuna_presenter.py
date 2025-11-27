import optuna
import logging
import sys
import pandas as pd

# Study name:
study_name = "study_11_25_25_18_07_46"

# Enabling print out
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Enabling the reusability of the study
storage_name = "sqlite:///saved_Optuna/{}.db".format(study_name)

# Loading the study
study = optuna.load_study(
    study_name=study_name,
    storage=storage_name
)

# Overview
print("Study name:", study.study_name)
print("Number of trials:", len(study.trials))
print("Best value:", study.best_value)
print("Best params:", study.best_params)