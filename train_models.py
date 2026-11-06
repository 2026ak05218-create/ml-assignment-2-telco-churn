"""
ML Assignment 2 - Telco Customer Churn
Trains five classifiers on the same train/test split, evaluates them on the
held-out test set and saves everything the Streamlit app needs.

Run:  python train_models.py
"""

import json
import os
import platform
import time

# Avoids a harmless joblib warning on Windows about counting CPU cores
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from churn_utils import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MODEL_FILES,
    NUMERIC_FEATURES,
    TARGET,
    clean_data,
    evaluate,
)

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "model"
TEST_FILE = "test_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_preprocessor():
    # Scale the 4 numeric columns and one-hot encode the 15 text columns.
    # sparse_output=False because GaussianNB does not accept sparse input.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def get_models():
    """Each entry: (estimator, small hyper-parameter grid)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            # Without a depth limit the tree simply memorises the training data
            {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [10, 30, 50]},
        ),
        "KNN": (
            # kd_tree instead of the default brute-force search: the brute-force search
            # splits work across CPU threads and breaks tied distances differently
            # depending on the thread count, so results changed slightly between machines
            KNeighborsClassifier(algorithm="kd_tree"),
            {"clf__n_neighbors": [11, 21, 31, 41], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
        ),
        "Support Vector Machine": (
            # probability=False keeps training fast; AUC is computed from decision_function
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            {"clf__C": [0.1, 0.5, 1.0], "clf__gamma": ["scale", 0.01]},
        ),
    }


def main():
    print(f"Loading {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    print(f"Shape: {raw.shape}")

    X, y = clean_data(raw)
    print(f"Features used: {len(FEATURES)}  |  Churn rate: {y.mean():.2%}")

    # Stratified split so train and test keep the same ~26.5% churn ratio
    train_idx, test_idx = train_test_split(
        raw.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    print(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    # The test file keeps the original raw format (Yes/No target, TotalCharges as text)
    # so it can be uploaded to the app exactly like real data would be.
    raw.loc[test_idx].to_csv(TEST_FILE, index=False)
    print(f"Saved held-out test data to {TEST_FILE}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_params = {}

    for name, (estimator, grid) in get_models().items():
        start = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])

        # Hyper-parameters are picked with cross-validation on the training set only.
        # The test set is not touched until the final evaluation below.
        # n_jobs=1 on purpose: parallel workers crashed on Windows, and the dataset is small
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        metrics, cm, _ = evaluate(model, X_test, y_test)
        joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILES[name]))

        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        results.append({"ML Model Name": name, **metrics})

        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name}  ({time.time() - start:.1f}s)")
        print(f"  best params : {best_params[name]}")
        print(f"  CV AUC      : {search.best_score_:.4f}")
        print("  test        : " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print(f"  confusion   : TN={tn} FP={fp} FN={fn} TP={tp}")

    results_df = pd.DataFrame(results).round(4)
    results_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)

    # Small metadata file so it is clear how the saved models were produced
    metadata = {
        "dataset": "Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)",
        "target": TARGET,
        "positive_class": "Yes (churned) = 1",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_params": best_params,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\nComparison table (held-out test set):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
