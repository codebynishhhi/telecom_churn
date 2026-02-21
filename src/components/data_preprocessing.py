# This file is responsible to - 
# 1. clean the dataset - handle missigg values, handle outliers, handle duplicates (if any)
# 2. convert the TotalCharges to numeric column
# 3. drop the missing column with id 
# 4. convert target variable to binary format(0/1)
# 5. separate X and y (input and output features)
# 6. perform stratified train test split to maintain the distribution of target variable in both train and test set as the dataset is imbalanced.


import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import TEST_SIZE, RANDOM_STATE

class DataPreprocessing:

    def __init__(self):
        pass
    
    # clean the data and return the cleaned dataframe
    def clean_data(self, df):
        
        try:

            # convert TotalCharges to numeric column
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

            # handle missing values by dropping the rows with missing values
            df = df.dropna()
            
            # drop the missing column with id
            df = df.drop(columns=["customerID"])

            # convert target variable to binary format(0/1)
            df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

            return df
        except Exception as e:
            raise Exception(f"Error in data cleaning: {e}")
        
    # separated the X and y features & perform stratified train test split 
    def split_data(self, df):
        try:
            X = df.drop("Churn", axis = 1)
            y = df["Churn"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise Exception(f"Error in data splitting: {e}")