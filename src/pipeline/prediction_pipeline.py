# This file is responsible for:
# 1. Loading trained model artifact
# 2. Accepting input data
# 3. Producing probability
# 4. Applying threshold from config
# 5. Returning structured output

import joblib
import pandas as pd
from typing import Union
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from src.utils.config import MODEL_SAVE_PATH, CHURN_THRESHOLD

class PredictionPipeline:
    
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        try:
            model = joblib.load(MODEL_SAVE_PATH)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def predict(self, input_data : Union[pd.DataFrame,dict], y_test) -> pd.DataFrame:
        try:
            # convert dict input to dataframe if needed
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()

            # get probabilites of churn
            churn_probabilites = self.model.predict_proba(input_df)[:, 1]

            # apply threshold to get churn flag
            predictions = (churn_probabilites >= CHURN_THRESHOLD).astype(int)
            print(f"Churn probabilities: {churn_probabilites}")
            print(f"Predictions: {predictions}")
            print("=="*50)
            print(f"Recall at 0.35 threshold\n:", recall_score(y_test, predictions))
            print(f"Precision at 0.35 threshold\n:", precision_score(y_test, predictions))
            print(f"AUC-ROC at 0.35 threshold\n:", roc_auc_score(y_test, churn_probabilites))
            # Build results dataframe
            result_dataframe = pd.DataFrame({
                "churn_probability": churn_probabilites,
                "churn_prediction": predictions
            }) 
            print(f"Result DataFrame\n: {result_dataframe}")
            return result_dataframe
        except Exception as e:
            raise Exception(f"Error during prediction: {e}")
