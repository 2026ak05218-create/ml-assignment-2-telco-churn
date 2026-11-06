"""
Helper functions shared by train_models.py and app.py.

Keeping the cleaning and metric code in one place makes sure the Streamlit app
treats the uploaded test file exactly the same way the training script did.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# File names inside the model/ folder, in the order they are shown in the app
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def check_columns(df):
    """Return the list of required columns that are missing from df."""
    return [col for col in FEATURES + [TARGET] if col not in df.columns]


def clean_data(df):
    """
    Clean a raw Telco churn dataframe and split it into X and y.

    - customerID is just an identifier, so it is dropped.
    - TotalCharges is stored as text in the original file. 11 rows have a blank
      value and all of them have tenure = 0 (new customers who have not been
      billed yet), so a blank is treated as 0 instead of being removed.
    - Churn is converted to 1 (Yes) / 0 (No). "Yes" is our positive class.
    """
    df = df.copy()

    # Strip stray spaces in text columns (e.g. " Yes") so they match training
    for col in CATEGORICAL_FEATURES + [TARGET]:
        df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_numeric = [c for c in NUMERIC_FEATURES if df[c].isna().any()]
    if bad_numeric:
        raise ValueError(f"Non-numeric or empty values found in: {', '.join(bad_numeric)}")

    target_map = {"Yes": 1, "No": 0, "1": 1, "0": 0}
    y = df[TARGET].map(target_map)
    if y.isna().any():
        raise ValueError("Churn column must only contain 'Yes'/'No' (or 1/0).")

    X = df[FEATURES]
    return X, y.astype(int)


def get_scores(model, X):
    """Score for the positive class, used for the AUC calculation."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    """Calculate the six metrics required by the assignment."""
    y_pred = model.predict(X)
    y_score = get_scores(model, X)

    # AUC is undefined if the file has only one class (e.g. only churners)
    auc = roc_auc_score(y, y_score) if y.nunique() == 2 else float("nan")

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return metrics, cm, y_pred
