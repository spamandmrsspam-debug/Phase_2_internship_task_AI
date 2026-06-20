# -*- coding: utf-8 -*-
"""
Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
Telco Customer Churn Prediction

Author: AI/ML Intern
Date: June 2026

This script builds a reusable ML pipeline that:
- Preprocesses numeric and categorical features
- Trains Logistic Regression and Random Forest classifiers
- Tunes hyperparameters using GridSearchCV
- Exports the final pipeline with joblib

Dataset: Telco Customer Churn (available at:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, precision_score, recall_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. Load and Explore Dataset
# ==============================
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# The 'TotalCharges' column contains some empty strings which we convert to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing TotalCharges (very few)
df = df.dropna(subset=['TotalCharges'])

# Drop customerID - not a predictive feature
df = df.drop('customerID', axis=1)

# Encode target: 'Yes' -> 1, 'No' -> 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\nTarget distribution:")
print(df['Churn'].value_counts(normalize=True).rename({0: 'No Churn', 1: 'Churn'}))

# ==============================
# 2. Split Data
# ==============================
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size:     {X_test.shape[0]}")

# ==============================
# 3. Define Preprocessing
# ==============================
# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Preprocessor: scale numeric, one-hot encode categorical
# drop='first' avoids dummy variable trap (multicollinearity in Logistic Regression)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'
)

# ==============================
# 4. Build Pipeline
# ==============================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Parameter grid for BOTH models in a single GridSearchCV
# This is the correct production approach - one search, one best result
param_grid = [
    {
        'classifier': [LogisticRegression(random_state=42, max_iter=1000)],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'saga']
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
]

# ==============================
# 5. Hyperparameter Tuning with GridSearchCV
# ==============================
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',          # optimize F1-score (dataset is imbalanced ~27% churn)
    n_jobs=-1,
    verbose=1
)

print("\nStarting GridSearchCV (both models tuned in a single search)...")
grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

# ==============================
# 6. Evaluate on Test Set
# ==============================
best_pipeline = grid_search.best_estimator_
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("TEST SET PERFORMANCE")
print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 7. Export the Complete Pipeline
# ==============================
model_filename = 'churn_pipeline.joblib'
joblib.dump(best_pipeline, model_filename)
print(f"\nPipeline exported to '{model_filename}'")
print("This file contains BOTH preprocessing + trained model — ready for production use.")

# Reload and run a prediction to prove reusability
loaded_pipeline = joblib.load(model_filename)
sample_pred  = loaded_pipeline.predict(X_test.iloc[[0]])[0]
sample_proba = loaded_pipeline.predict_proba(X_test.iloc[[0]])[0][1]
print(f"\nReload test -> Prediction: {'Churn' if sample_pred == 1 else 'No Churn'} "
      f"(churn probability: {sample_proba:.2%})")
print(f"Actual label:              {'Churn' if y_test.iloc[0] == 1 else 'No Churn'}")

# ==============================
# 8. Feature Importance (if Random Forest won)
# ==============================
if isinstance(best_pipeline.named_steps['classifier'], RandomForestClassifier):
    preprocessor_fitted = best_pipeline.named_steps['preprocessor']
    cat_encoder  = preprocessor_fitted.named_transformers_['cat']
    cat_columns  = cat_encoder.get_feature_names_out(categorical_cols)
    all_features = numeric_cols + list(cat_columns)

    importances = best_pipeline.named_steps['classifier'].feature_importances_
    feature_df  = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_df.head(10).to_string(index=False))

print("\nTask 2 completed successfully.")
