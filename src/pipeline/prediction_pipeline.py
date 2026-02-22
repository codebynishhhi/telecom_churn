# This file is responsible for:
# 1. Loading trained model artifact
# 2. Accepting input data
# 3. Producing probability
# 4. Applying threshold from config
# 5. Returning structured output

import os
import joblib
import pandas as pd
from typing import Union
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from src.utils.config import MODEL_SAVE_PATH, ARTIFACTS_DIR

class PredictionPipeline:
    
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        try:
            model = joblib.load(MODEL_SAVE_PATH)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def predict(self, input_data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
        try:
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()

            churn_probabilities = self.model.predict_proba(input_df)[:, 1]

            # Load optimized threshold
            with open(os.path.join(ARTIFACTS_DIR, "threshold.txt"), "r") as f:
                threshold = float(f.read())

            predictions = (churn_probabilities >= threshold).astype(int)

            result_dataframe = pd.DataFrame({
                "churn_probability": churn_probabilities,
                "churn_prediction": predictions
            })

            return result_dataframe

        except Exception as e:
            raise Exception(f"Error during prediction: {e}")