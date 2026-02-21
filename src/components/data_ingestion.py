import pandas as pd
from src.utils.config import RAW_DATA_PATH

class DataIngestion:
    
    # constructor to initialize the class with the raw data path
    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH

    # method to read the raw data and return a dataframe
    def load_data(self):
        try:
            df = pd.read_csv(self.raw_data_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")