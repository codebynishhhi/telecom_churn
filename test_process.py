from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_preprocessing import ModelPreprocessing           
from src.components.model_training import ModelTraining
from src.pipeline.prediction_pipeline import PredictionPipeline

def main():
    # data ingestion
    data_ingestion = DataIngestion()
    df = data_ingestion.load_data()

    # data preprocessing
    data_preprocessing = DataPreprocessing()
    cleaned_df = data_preprocessing.clean_data(df)
    X_train, X_test, y_train, y_test = data_preprocessing.split_data(cleaned_df)

    # model preprocessing
    model_preprocessing = ModelPreprocessing()
    preprocessor = model_preprocessing.build_preprocessor(X_train)

    # model training
    model_training = ModelTraining()
    pipeline, roc_auc_res = model_training.train(X_train, y_train, X_test, y_test)

    # model prediction and evaluation
    
    prediction_pipeline = PredictionPipeline()
    predictions = prediction_pipeline.predict(X_test, y_test)

    print(f"Predictions: {predictions}")
    print(f"ROC AUC Score: {roc_auc_res}")
    print(f"Model pipeline: {pipeline}")
    print("Model training completed successfully.")


if __name__ == "__main__":
    main()