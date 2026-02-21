
## LEVEL 1 — Data Loader Module

Purpose:
Only loads raw dataset.

## LEVEL 2 — Preprocessing Module

Handles:
Type conversions
Missing values
Encoding
Scaling
ColumnTransformer

Returns:
preprocessor object

## LEVEL 3 — Feature Engineering Module

Handles:
is_new_customer
is_electronic_check
num_missing_services
tenure bins (if needed)

Returns:
transformed dataframe

## LEVEL 4 — Training Module

Handles:
Train/test split (stratified)
Cross-validation
Model training
Hyperparameter tuning

Saving model

## LEVEL 5 — Inference Module

Handles:
Load saved model
Take new input›
Output probability + churn flag