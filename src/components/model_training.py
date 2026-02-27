"""
Model Training Module

Responsibilities:
1. Build preprocessing pipeline
2. Build model from config
3. Optional hyperparameter tuning (RandomizedSearchCV)
4. Cross-validation evaluation
5. Fit final model
6. Automatic threshold optimization (TRAIN data only)
7. Final evaluation on TEST data
8. Log metrics to MLflow
9. Save model + threshold
"""

from typing import Tuple
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV

from xgboost import XGBClassifier

from src.components.model_preprocessing import ModelPreprocessing
from src.utils.config import (
    MODEL_TYPE,
    LOGISTIC_REGRESSION_PARAMS,
    XGB_PARAMS,
    MODEL_SAVE_PATH,
    ARTIFACTS_DIR,
    ENABLE_TUNING,
    N_ITER,
    AUTO_THRESHOLD,
    RECALL_CONSTRAINT
)


class ModelTraining:

    def __init__(self):
        self.pipeline = None
        self.default_values = {}

    # ==========================================================
    # Build Model
    # ==========================================================
    def build_model(self):

        if MODEL_TYPE == "logistic":
            return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

        elif MODEL_TYPE == "xgboost":
            return XGBClassifier(**XGB_PARAMS)

        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    # ==========================================================
    # Train Model
    # ==========================================================
    def train_model(self, X_train, y_train, X_test, y_test) -> Tuple[Pipeline, float]:

        mlflow.set_experiment("Telco Churn Prediction")

        with mlflow.start_run(run_name=f"{MODEL_TYPE}_training"):

            # --------------------------------------------------
            # Build Preprocessor
            # --------------------------------------------------
            model_preprocessing = ModelPreprocessing()
            preprocessor = model_preprocessing.build_preprocessor(X_train)

            base_model = self.build_model()

            self.pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", base_model)
                ]
            )

            # --------------------------------------------------
            # Hyperparameter Tuning (XGBoost only)
            # --------------------------------------------------
            if MODEL_TYPE == "xgboost" and ENABLE_TUNING:

                print("Starting Hyperparameter Tuning...")

                param_dist = {
                    "model__n_estimators": [200, 300, 500],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0]
                }

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                random_search = RandomizedSearchCV(
                    estimator=self.pipeline,
                    param_distributions=param_dist,
                    n_iter=N_ITER,
                    scoring="roc_auc",
                    cv=skf,
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )

                random_search.fit(X_train, y_train)

                self.pipeline = random_search.best_estimator_

                print("Best Params:", random_search.best_params_)
                print("Best CV ROC-AUC:", random_search.best_score_)

                mlflow.log_params(random_search.best_params_)
                mlflow.log_metric("best_cv_roc_auc", random_search.best_score_)

            # --------------------------------------------------
            # Cross Validation (Training Data)
            # --------------------------------------------------
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            cv_results = cross_validate(
                self.pipeline,
                X_train,
                y_train,
                cv=skf,
                scoring={
                    "roc_auc": "roc_auc",
                    "recall": "recall"
                },
                return_train_score=False
            )

            cv_roc_mean = np.mean(cv_results["test_roc_auc"])
            cv_roc_std = np.std(cv_results["test_roc_auc"])
            cv_recall_mean = np.mean(cv_results["test_recall"])
            cv_recall_std = np.std(cv_results["test_recall"])

            print("\nCross Validation Results")
            print("CV ROC-AUC Mean:", cv_roc_mean)
            print("CV ROC-AUC Std:", cv_roc_std)
            print("CV Recall Mean:", cv_recall_mean)
            print("CV Recall Std:", cv_recall_std)

            mlflow.log_metric("cv_roc_auc_mean", cv_roc_mean)
            mlflow.log_metric("cv_roc_auc_std", cv_roc_std)
            mlflow.log_metric("cv_recall_mean", cv_recall_mean)
            mlflow.log_metric("cv_recall_std", cv_recall_std)

            # --------------------------------------------------
            # Fit Final Model
            # --------------------------------------------------
            self.pipeline.fit(X_train, y_train)

            # --------------------------------------------------
            # Automatic Threshold Optimization (TRAIN only)
            # --------------------------------------------------
            optimized_threshold = 0.5

            if AUTO_THRESHOLD:

                print("\nRunning Automatic Threshold Optimization (TRAIN)...")

                train_probs = self.pipeline.predict_proba(X_train)[:, 1]
                thresholds = np.arange(0.1, 0.91, 0.01)

                best_precision = 0

                for thresh in thresholds:
                    preds = (train_probs >= thresh).astype(int)
                    rec = recall_score(y_train, preds)
                    prec = precision_score(y_train, preds)

                    if rec >= RECALL_CONSTRAINT and prec > best_precision:
                        best_precision = prec
                        optimized_threshold = thresh

                print("Optimized Threshold:", optimized_threshold)
                print("Precision at Optimized Threshold:", best_precision)

                mlflow.log_metric("optimized_threshold", optimized_threshold)
                mlflow.log_metric("optimized_precision_train", best_precision)

                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                with open(os.path.join(ARTIFACTS_DIR, "threshold.txt"), "w") as f:
                    f.write(str(optimized_threshold))

            # --------------------------------------------------
            # Final Test Evaluation (Using Optimized Threshold)
            # --------------------------------------------------
            test_probs = self.pipeline.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= optimized_threshold).astype(int)

            test_roc = roc_auc_score(y_test, test_probs)
            test_recall = recall_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds)

            print("\nTest Results (Optimized Threshold)")
            print("Test ROC-AUC:", test_roc)
            print("Test Recall:", test_recall)
            print("Test Precision:", test_precision)

            mlflow.log_metric("test_roc_auc", test_roc)
            mlflow.log_metric("test_recall_optimized", test_recall)
            mlflow.log_metric("test_precision_optimized", test_precision)

            # --------------------------------------------------
            # Log Model
            # --------------------------------------------------
            mlflow.sklearn.log_model(self.pipeline, name="model")

            # --------------------------------------------------
            # Save Model
            # --------------------------------------------------
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            joblib.dump(self.pipeline, MODEL_SAVE_PATH)

            print(f"\nModel saved at: {MODEL_SAVE_PATH}")
            print("Training Completed Successfully")

            # Save default values for inference
            for col in X_train.columns:
                if np.issubdtype(X_train[col].dtype, np.number):
                    self.default_values[col] = float(X_train[col].median())
                else:
                    self.default_values[col] = X_train[col].mode()[0]

            joblib.dump(
                self.default_values,
                os.path.join(ARTIFACTS_DIR, "default_training_metrics.pkl"))
            
            return self.pipeline, test_roc