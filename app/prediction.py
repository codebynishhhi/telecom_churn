import os
import joblib
import pandas as pd

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model", "tuned_xgboost_model.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.txt")

class PredictionService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = float(f.read())

        # Extract expected feature names from trained pipeline
        self.expected_columns = self.model.feature_names_in_

    def predict(self, data:dict):

        df = pd.DataFrame([data])
        # Reorder & align columns
        df = df.reindex(columns=self.expected_columns)
        probs = self.model.predict_proba(df)[:,1]
        preds = (probs >= self.threshold).astype(int)

        return {
            "churn probability": float(probs[0]),
            "churn predictions": int(preds[0])
        }

