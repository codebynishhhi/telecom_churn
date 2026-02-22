# This file is used to -
# 1. Build the model based on the configs
# 2. Build the preprocessing pipeline
# 3. Creates full sklearn pipeline by combining the preprocessing and model
# 4. Run stratified K-Fold cross validation to evaluate the model performance on training data 
# 5. If enabled in config, run hyperparameter tuning using RandomizedSearchCV and log the best parameters and best score to mlflow
# 6. log the metrics to mlflow
# 7. Fit the pipeline on the training data
# 8. Train the model
# 9. Evaluate the model 
# 10. Log to mlflow
# 11. Save the model to the artifacts folder


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
from sklearn.model_selection import RandomizedSearchCV
from src.utils.config import (
    MODEL_TYPE,
    LOGISTIC_REGRESSION_PARAMS,
    XGB_PARAMS,
    MODEL_SAVE_PATH,
    ARTIFACTS_DIR,
    ENABLE_TUNING,
    N_ITER
    
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

    def train_model(self, X_train, y_train, X_test, y_test) -> Tuple[Pipeline,float]:
        try:
            mlflow.set_experiment("Telco Churn Prediction")
            with mlflow.start_run(run_name="Model Training"):

            # ======================================================================
            # Build preprocessor
            # ======================================================================
                model_preprocessing = ModelPreprocessing()
                preprocessor = model_preprocessing.build_preprocessor(X_train)

            # ======================================================================
            # Build base model
            # ======================================================================
                base_model = self.build_model()
            
            # ======================================================================
            # Build pipeline
            # ======================================================================
                self.pipeline = Pipeline(steps=[
                    ('preprocessor',preprocessor),
                    ('model', base_model)
                ])
            
            # ======================================================================
            # Hyperparameter tuning (if enabled in config)
            # ======================================================================
                if MODEL_TYPE == "xgboost" and ENABLE_TUNING:
                    print("Starting hyperparamter tuning...")
                    param_dist = {
                        'model__n_estimators': [200, 300, 500],
                        'model__max_depth': [3, 5, 7],
                        'model__learning_rate': [0.01, 0.05, 0.1],
                        'model__subsample': [0.8, 1.0],
                        'model__colsample_bytree': [0.8, 1.0]
                    }

                    # stratified k-fold cross validation
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                    # randomized search cv
                    random_search = RandomizedSearchCV(
                        estimator=self.pipeline,
                        param_distributions= param_dist,
                        n_iter=N_ITER,
                        cv = skf,
                        scoring='roc_auc',
                        verbose=1,
                        n_jobs=-1,
                        random_state=42
                    )

                    random_search.fit(X_train, y_train)

                    self.pipeline = random_search.best_estimator_
                    
                    print("Best pipeline updated contents:", self.pipeline)
                    print(f"Best hyperparameters: {random_search.best_params_}")
                    print(f"Best ROC AUC Score from tuning: {random_search.best_score_}")

                    mlflow.log_params(random_search.best_params_)
                    mlflow.log_metric("best_roc_auc_score", random_search.best_score_)

            # ======================================================================
            # Cross validation on training data
            # ======================================================================
                cv_results = cross_validate(self.pipeline, X_train, y_train, cv=skf, scoring={'roc_auc':'roc_auc', 'recall':'recall'}, return_train_score=True)

                print(f"Cross validation results: {cv_results}")
            # ======================================================================
                cv_roc_mean = np.mean(cv_results["test_roc_auc"])
                cv_roc_std = np.std(cv_results["test_roc_auc"])
                cv_recall_mean = np.mean(cv_results["test_recall"])
                cv_recall_std = np.std(cv_results["test_recall"])

                print("Cross Validation Results:")
                print(f"CV ROC-AUC Mean: {cv_roc_mean}")
                print(f"CV ROC-AUC Std: {cv_roc_std}")
                print(f"CV Recall Mean: {cv_recall_mean}")
                print(f"CV Recall Std: {cv_recall_std}")

                mlflow.log_metric("cv_roc_auc_mean", cv_roc_mean)
                mlflow.log_metric("cv_roc_auc_std", cv_roc_std)
                mlflow.log_metric("cv_recall_mean", cv_recall_mean)
                mlflow.log_metric("cv_recall_std", cv_recall_std)


            #=======================================================================
            # Fit the pipeline on the training data
            #=======================================================================
                self.pipeline.fit(X_train, y_train)

            # =======================================================================
            # Evaluate the model on test data
            # =======================================================================
                y_test_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
                y_test_pred = self.pipeline.predict(X_test)

                roc_auc_test_result = roc_auc_score(y_test, y_test_pred_proba)
                print(f"Test ROC AUC Score: {roc_auc_test_result}")

                mlflow.log_metric("test_roc_auc_score", roc_auc_test_result)

                recall_test_result = recall_score(y_test, y_test_pred)
                precision_test_result = precision_score(y_test, y_test_pred)
            
                print(f"Test Recall Score: {recall_test_result}")
                print(f"Test Precision Score: {precision_test_result}")
                mlflow.log_metric("test_recall_score", recall_test_result)
                mlflow.log_metric("test_precision_score", precision_test_result)

            # ======================================================================
            # log the model to mlflow
            # ======================================================================
                mlflow.sklearn.log_model(self.pipeline, artifact_path="model")

            # =======================================================================
            # Save the model to artifacts folder    
            # =======================================================================
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                joblib.dump(self.pipeline, MODEL_SAVE_PATH)
                print(f"Model saved at: {MODEL_SAVE_PATH}")

                print(f"Model training completed successfully with {MODEL_TYPE} model.")

                return self.pipeline, roc_auc_test_result
        except Exception as e:
            raise Exception(f"Error in model training: {e}")