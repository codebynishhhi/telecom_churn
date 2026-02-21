
import pandas as pd
class FeatureEngineering:

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # 1. New customer flag (tenure < 12 months)
            df["is_new_customer"] = (df["tenure"] < 12).astype(int)

            # 2. Electronic check flag
            df["is_electronic_check"] = (
                df["PaymentMethod"] == "Electronic check"
            ).astype(int)

            # 3. Count of missing service add-ons
            service_cols = [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies"
            ]

            df["num_missing_services"] = (
                df[service_cols] == "No"
            ).sum(axis=1)

            # 4. Drop customerID (not useful for model)
            if "customerID" in df.columns:
                df = df.drop("customerID", axis=1)

            return df

        except Exception as e:
            raise Exception(f"Error during feature engineering: {e}")
