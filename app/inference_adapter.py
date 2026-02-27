# This file is responsible to -
# 1. load the model
# 2. Load the saved training metrics like median/mode
# 3. fills missing values using training knowledge
# 4. Calls prediction


import os 
import joblib
import pandas as pd


ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model", "tuned_xgboost_model.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.txt")
DEFAULTS_PATH = os.path.join(ARTIFACTS_DIR, "default_training_metrics.pkl")

class InferenceAdapterService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = float(f.read())
            
            # LOAD THE training metrics like median/mode
            self.default_values = joblib.load(DEFAULTS_PATH)

            # the most important columns
            self.expected_columns = self.model.feature_names_in_

    def predict(self, user_input:dict):
        
        # start with defaults 
        full_input_data = self.default_values.copy()

        # override input values with user input data
        for key, value in user_input.items():
            full_input_data[key] = value

        df = pd.DataFrame([full_input_data])

        df = df.reindex(columns=self.expected_columns)

        probs_res = self.model.predict_proba(df)[:,1]

        preds_res = (probs_res >= self.threshold).astype(int)

        return {
            "Churn_probability": float(probs_res[0]),
            "Churn_prediction": int(preds_res[0])
        }
        


