# This file is used to -
# 1. Build the model based on the configs
# 2. Build the preprocessing pipeline
# 3. Creates full sklearn pipeline by combining the preprocessing and model
# 4. Run stratified K-Fold cross validation to evaluate the model performance on training data 
# 5 log the metrics to mlflow
# 6. Fit the pipeline on the training data
# 7. Train the model
# 8. Evaluate the model 
# 9. Log to mlflow
# 10. Save the model to the artifacts folder


from typing import Tuple
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from src.components.model_preprocessing import ModelPreprocessing
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from xgboost import XGBClassifier
from src.utils.config import (
    MODEL_TYPE,
    LOGISTIC_REGRESSION_PARAMS,
    XGB_PARAMS,
    MODEL_SAVE_PATH,
    ARTIFACTS_DIR
)
class ModelTraining:

    def __init__(self):
        self.model = None
        self.pipeline = None

    def build_model(self):
        try:
            if MODEL_TYPE == "logistic":
                return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
            elif MODEL_TYPE == "xgboost":
                return XGBClassifier(**XGB_PARAMS)

            else:
                raise ValueError(f"Unsupported model type: {MODEL_TYPE}")
        except Exception as e:
            raise Exception(f"Error in building model: {e}")

    def train(self, X_train, y_train, X_test, y_test) -> Tuple[Pipeline, float]:
        try:
            mlflow.set_experiment('Telco Churn Prediction')

            with mlflow.start_run(run_name="Model Training"):

                # Build the preprocessing pipeline
                model_prep = ModelPreprocessing()
                preprocessor = model_prep.build_preprocessor(X_train)

                # build the model
                self.model = self.build_model()

                # create the full pipeline
                self.pipeline = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('model', self.model)
                    ]
                )

                # ==============================================
                # Cross-validation on training data to evaluate the model performance
                # ==============================================
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                cv_results = cross_validate(
                    self.pipeline, X_train, y_train, cv=skf, scoring={
                        'roc_auc': 'roc_auc',
                        'recall': 'recall',
                    } ,
                    return_train_score=False
                )

                cv_roc_auc_mean = np.mean(cv_results['test_roc_auc'])
                cv_recall_mean = np.mean(cv_results['test_recall'])

                cv_roc_auc_std = np.std(cv_results['test_roc_auc'])
                cv_recall_std = np.std(cv_results['test_recall'])
                
                print("Cross Validation Results")
                print(f"CV ROC-AUC Mean: {cv_roc_auc_mean}")
                print(f"CV ROC-AUC Std: {cv_roc_auc_std}")
                print(f"CV Recall Mean: {cv_recall_mean}")
                print(f"CV Recall Std: {cv_recall_std}")

                mlflow.log_metric("cv_roc_auc_mean", cv_roc_auc_mean)
                mlflow.log_metric("cv_roc_auc_std", cv_roc_auc_std)
                mlflow.log_metric("cv_recall_mean", cv_recall_mean)
                mlflow.log_metric("cv_recall_std", cv_recall_std)
                
                # fit the pipeline on training data
                self.pipeline.fit(X_train, y_train)

                # predict on test data at 0.5 threshold
                y_pred = self.pipeline.predict(X_test)
                y_pred_proba = self.pipeline.predict_proba(X_test)[:,1]

                # evaluate the model metrics
                roc_auc_res = roc_auc_score(y_test, y_pred_proba)
                recall_res = recall_score(y_test, y_pred)
                precision_res = precision_score(y_test, y_pred)

                # log the metrics to mlflow
                mlflow.log_metric("roc_auc", roc_auc_res)
                mlflow.log_metric("recall", recall_res)
                mlflow.log_metric("precision", precision_res)

                # log the model to mlflow
                mlflow.sklearn.log_model(self.pipeline, "model")

                # save the model to artifacts folder
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                joblib.dump(self.pipeline, MODEL_SAVE_PATH)

                print("Model training completed successfully with XgBoost at 0.5 Threshold!")
                print(f"ROC AUC: {roc_auc_res}, Recall: {recall_res}, Precision: {precision_res}")

                return self.pipeline, roc_auc_res
        except Exception as e:
            raise Exception(f"Error in model training: {e}")
