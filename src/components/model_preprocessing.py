# This file is responsible to - 
# 1. separate numerical & categorical columns
# 2. Build the column transformers pipeline
# 3. return a reusable processing pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class ModelPreprocessing:
     
    def __init__(self):
        self.preprocessor = None

    def build_preprocessor(self, X):
        try:
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            # defining the transformation
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            # combining the transformations into a column transformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num_cols', numeric_transformer, numerical_cols),
                    ('cat_cols', categorical_transformer, categorical_cols)
                ]
            )

            return self.preprocessor
        except Exception as e:
            raise Exception(f"Error in identifying numerical and categorical columns: {e}")

