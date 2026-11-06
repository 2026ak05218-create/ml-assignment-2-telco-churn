"""
Streamlit app for ML Assignment 2 - Telco Customer Churn.

The app only loads the models saved by train_models.py and evaluates them on
the uploaded test CSV. Nothing is retrained here.

Run locally:  streamlit run app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report

from churn_utils import MODEL_FILES, check_columns, clean_data, evaluate

MODEL_DIR = "model"
SAMPLE_TEST_FILE = "test_data.csv"

st.set_page_config(page_title="Telco Churn - Model Evaluation", layout="wide")


@st.cache_resource
def load_models():
    models = {}
    for name, file_name in MODEL_FILES.items():
        models[name] = joblib.load(os.path.join(MODEL_DIR, file_name))
    return models


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fmt(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"


# ---------------- Sidebar ----------------
st.sidebar.header("1. Test data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if os.path.exists(SAMPLE_TEST_FILE):
    with open(SAMPLE_TEST_FILE, "rb") as f:
        st.sidebar.download_button(
            "Download sample test_data.csv", f, file_name="test_data.csv", mime="text/csv"
        )
use_sample = st.sidebar.checkbox("Use the sample test_data.csv instead", value=False)

st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

st.sidebar.caption(
    "The CSV must contain all 19 input columns of the Telco dataset plus the "
    "'Churn' column (Yes/No). 'customerID' is optional and ignored."
)

# ---------------- Main page ----------------
st.title("Telco Customer Churn - Classification Models")
st.write(
    "Predicts whether a telecom customer will leave (**Churn = Yes**). "
    "Five models were trained offline and saved; this app evaluates the selected "
    "model on the test data you upload."
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    source = uploaded_file.name
elif use_sample and os.path.exists(SAMPLE_TEST_FILE):
    data = pd.read_csv(SAMPLE_TEST_FILE)
    source = SAMPLE_TEST_FILE
else:
    st.info("Upload a test CSV from the sidebar (or tick 'Use the sample test_data.csv') to begin.")
    st.stop()

if data.empty:
    st.error("The uploaded file is empty.")
    st.stop()

missing = check_columns(data)
if missing:
    st.error(f"The file is missing these required columns: {', '.join(missing)}")
    st.stop()

try:
    X, y = clean_data(data)
except ValueError as err:
    st.error(f"Could not read the file: {err}")
    st.stop()

with st.expander(f"Preview of {source}  ({len(data)} rows)"):
    st.dataframe(data.head(20), use_container_width=True)
    st.write(f"Churn = Yes: {int(y.sum())}  |  Churn = No: {int((y == 0).sum())}")

try:
    models = load_models()
except Exception as err:
    st.error(f"Could not load the saved models from '{MODEL_DIR}/': {err}")
    st.stop()

# ---------- Selected model ----------
st.subheader(f"Evaluation metrics - {model_name}")
metrics, cm, y_pred = evaluate(models[model_name], X, y)

cols = st.columns(6)
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, fmt(value))

if y.nunique() < 2:
    st.warning("The file contains only one class, so AUC cannot be calculated.")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion_matrix(cm, model_name))
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=["No churn", "Churn"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

# ---------- All models on the same file ----------
st.subheader("Comparison of all five models on this test data")
rows = []
for name, model in models.items():
    m, _, _ = evaluate(model, X, y)
    rows.append({"ML Model Name": name, **m})
comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)

st.dataframe(
    comparison.style.highlight_max(axis=0, color="#c6efce").format("{:.4f}"),
    use_container_width=True,
)
st.bar_chart(comparison[["Accuracy", "AUC", "F1", "MCC"]])

csv_bytes = comparison.to_csv().encode("utf-8")
st.download_button("Download comparison table (CSV)", csv_bytes, "model_comparison.csv", "text/csv")
