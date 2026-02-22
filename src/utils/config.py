# This file contains the setup configuration for the project, specifying dependencies and metadata.
# It contains the - 
# paths, thresholds, hyperparameters, model name, artifacts locations and other configurations required for the project.
# Steup.py will be divided into 4 parts - 
    # 1. Data Configuration
    # 2. Model Configuration  
    # 3. Training Configuration
    # 4. Inference Configuration

import os 

# ==============================
# PROJECT ROOT
# ==============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(BASE_DIR, "data")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "churn_data.csv")

MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "model", "tuned_xgboost_model.pkl")

# ==============================
# TRAINING CONFIG
# ==============================

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5

# ==============================
# MODEL CONFIG
# ==============================

MODEL_TYPE = "xgboost"  # logistic | xgboost | random_forest

# Logistic Regression Parameters
LOGISTIC_REGRESSION_PARAMS = {
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "max_iter": 1000
}

# XGBoost Parameters
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss"
}

# ==============================
# INFERENCE CONFIG
# ==============================

CHURN_THRESHOLD = 0.35