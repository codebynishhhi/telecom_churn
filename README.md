## Final Summary 

### Telecom Customer Churn Prediction

Built an end-to-end churn prediction system using Logistic Regression, Random Forest, and XGBoost on the Telco Customer dataset.

### Implemented full ML lifecycle:

EDA and feature engineering

Stratified 5-fold cross-validation

Hyperparameter tuning (GridSearchCV)

Threshold optimization for business objective

Model interpretability using coefficient analysis

### Final model:

ROC-AUC: 0.845 ± 0.0026

Recall optimized to 72% at threshold 0.35

Identified key churn drivers: Fiber optic users, high total charges, streaming services, electronic check payments, new customers.

### Business Impact:
Model enables proactive identification of 72% of churners before exit, supporting targeted retention campaigns.

### Steps taken to complete this project 
completed telecom churn ML pipeline with CV, tuning, threshold optimization & business insights

- Performed full EDA and data cleaning
- Engineered lifecycle and behavioral churn features
- Implemented Logistic Regression, Random Forest, XGBoost
- Applied Stratified 5-Fold Cross Validation
- Conducted hyperparameter tuning (GridSearchCV)
- Optimized decision threshold for recall-focused business objective
- Achieved ROC-AUC 0.845 ± 0.0026
- Final production threshold set to 0.35 (72% recall)
- Interpreted model coefficients for business impact
