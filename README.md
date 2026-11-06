# Telco Customer Churn Prediction – ML Assignment 2

**M.Tech (AIML) – Machine Learning, Assignment 2**

- **GitHub repository:** `https://github.com/2026ak05218-create/ml-assignment-2-telco-churn.git`
- **Live Streamlit app:** `https://ml-assignment-2-telco-churn.streamlit.app/`

---

## a. Problem statement

A telecom company loses revenue every time a customer cancels their service ("churns"). If the company can spot customers who are likely to leave, it can contact them with a retention offer before they go.

The goal of this project is to build and compare five classification models that predict whether a customer will churn (**Churn = Yes / No**) from their account details, the services they use and their billing information. The trained models are then deployed in a Streamlit web app, where a test CSV can be uploaded and each model's performance checked.

This is a **binary classification** problem. "Yes" (the customer churned) is treated as the positive class.

---

## b. Dataset description

| Item | Details |
|---|---|
| Name | Telco Customer Churn (IBM sample dataset) |
| Source | Kaggle – https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Instances | **7,043** customers |
| Columns | 21: 1 ID, 19 input features, 1 target |
| Input features used | **19** (4 numeric + 15 categorical) |
| Target | `Churn` – Yes: 1,869 (26.5%), No: 5,174 (73.5%) |

**Features**

| Group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone services | `PhoneService`, `MultipleLines` |
| Internet services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

**Data preparation**

- `customerID` was dropped because it is only an identifier.
- `TotalCharges` is stored as text in the original file and has 11 blank values. All 11 belong to customers with `tenure = 0`, i.e. new customers who had not been billed yet, so the blanks were set to 0 instead of dropping the rows.
- `Churn` was mapped to 1 (Yes) and 0 (No).
- The 15 categorical columns were one-hot encoded, and the 4 numeric columns were standardised with `StandardScaler`. Both steps are inside each saved scikit-learn `Pipeline`, so the app applies exactly the same transformation to uploaded data.
- There were no other missing values and no duplicate rows.

**Train/test split:** stratified 80/20 split with `random_state = 42`.
Training set: 5,634 rows. Test set: **1,409 rows** (374 churners), saved as `test_data.csv`.
The test set was not used for training or tuning. Hyper-parameters were chosen with 5-fold stratified cross-validation (ROC AUC) on the training set only.

---

## c. GitHub repository link

`https://github.com/2026ak05218-create/ml-assignment-2-telco-churn.git`

**Repository structure**

```
ml-assignment-2-telco-churn/
├── app.py                 # Streamlit app (loads saved models, no retraining)
├── train_models.py        # Cleans data, trains + tunes the 5 models, saves them
├── churn_utils.py         # Shared cleaning and metric code used by both scripts
├── requirements.txt
├── README.md
├── test_data.csv          # Held-out test set (1,409 rows, all features + Churn)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv        # Comparison table produced by train_models.py
    └── metadata.json      # Features, split, best hyper-parameters, library versions
```

---

## d. Models used

All five models were trained on the same training data and evaluated on the same held-out test set.

| Model | Final settings (chosen by 5-fold CV on training data) |
|---|---|
| Logistic Regression | `C = 10` |
| Decision Tree | `max_depth = 8`, `min_samples_leaf = 50` |
| KNN | `n_neighbors = 31`, uniform weights |
| Naive Bayes (Gaussian) | `var_smoothing = 0.001` |
| Support Vector Machine (RBF kernel) | `C = 0.5`, `gamma = 0.01` |

### Comparison table (test set, 1,409 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8048 | 0.8412 | 0.6552 | 0.5588 | 0.6032 | 0.4775 |
| Decision Tree | 0.7913 | 0.8293 | 0.6361 | 0.5000 | 0.5599 | 0.4310 |
| KNN | 0.7935 | 0.8284 | 0.6131 | 0.6016 | 0.6073 | 0.4672 |
| Naive Bayes | 0.6948 | 0.8075 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Support Vector Machine | 0.7999 | 0.8315 | 0.6620 | 0.5027 | 0.5714 | 0.4512 |

Confusion matrices on the test set (374 actual churners, 1,035 non-churners):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 925 | 110 | 165 | 209 |
| Decision Tree | 928 | 107 | 187 | 187 |
| KNN | 893 | 142 | 149 | 225 |
| Naive Bayes | 666 | 369 | 61 | 313 |
| Support Vector Machine | 939 | 96 | 186 | 188 |

### Observations

Only about 26.5% of customers churn, so a model that always predicts "No" would already be about 73% accurate. For that reason I looked mainly at AUC, F1 and MCC, and not just accuracy.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: the highest accuracy (0.8048), AUC (0.8412) and MCC (0.4775) of the five, and an F1 (0.6032) almost equal to the best. Most of the churn signal is close to linear once the categories are one-hot encoded (for example month-to-month contract, short tenure, fibre optic internet), which suits this model. Its weak point is recall: it still misses 165 of the 374 churners at the default 0.5 threshold. |
| Decision Tree | The weakest of the "balanced" models: MCC 0.4310, and recall of only 0.50, so half of the churners are missed. The depth limit (8) and minimum leaf size (50) stopped it from overfitting, but a single tree splits the data into fixed boxes and its probability estimates are coarse, which is why its AUC (0.8293) is below Logistic Regression's. Its advantage is that the rules are easy to explain to a business user. |
| KNN | Best F1 (0.6073) and the most even trade-off between precision (0.6131) and recall (0.6016): it finds 225 churners, more than any model except Naive Bayes. It needed a fairly large k (31) to smooth out noise, and it depended on scaling the numeric features. AUC (0.8284) and MCC (0.4672) are slightly below Logistic Regression. It is also the slowest model at prediction time, because it compares every new customer with all 5,634 training customers. |
| Naive Bayes | Very different behaviour from the others: by far the highest recall (0.8369, it catches 313 of 374 churners) but the lowest precision (0.4589) and accuracy (0.6948), with 369 false alarms. Its "features are independent" assumption is clearly broken here (for example `TotalCharges` is roughly `tenure` × `MonthlyCharges`, and the "No internet service" values repeat across six columns), so its probabilities are pushed towards the churn side. It could still be useful if missing a churner is much more costly than sending an unnecessary retention offer. |
| Support Vector Machine | Highest precision (0.6620) and fewest false alarms (96), so when it predicts churn it is usually right, but its recall (0.5027) is close to the Decision Tree's. AUC (0.8315) is second best. Tuning mattered: with the default `gamma` its AUC was below 0.80, and a smaller `gamma` (0.01, a smoother boundary) improved it. It was also the slowest model to train (about 35 seconds for the grid search, compared with 1–9 seconds for the others). |
| **Overall winner** | **Logistic Regression.** It has the best accuracy, AUC and MCC and nearly the best F1, while being the fastest and simplest model to train and explain. KNN is a close second if F1 is the only criterion (0.6073 vs 0.6032), but the gap is tiny on a 1,409-row test set. If the business priority were to catch as many churners as possible, Naive Bayes (or Logistic Regression with a lower decision threshold) would be the better choice. |

---

## Streamlit app

The app (`app.py`) provides:

1. **CSV upload** for test data (a "Download sample test_data.csv" button and a "use sample" option are in the sidebar).
2. A **model selection dropdown** with all five models.
3. **Evaluation metrics** for the selected model: Accuracy, AUC, Precision, Recall, F1 and MCC.
4. The **confusion matrix** and **classification report** for the selected model.
5. A **comparison table and chart of all five models** on the uploaded test data.

The app only loads the saved `.joblib` pipelines from `model/` and calls `predict` / `predict_proba`. It never retrains on the uploaded data.
The uploaded CSV must contain the 19 input columns and the `Churn` column (Yes/No). `customerID` is optional and ignored. The app shows an error if columns are missing, the target has other values, or numeric columns contain text.

---

## How to run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

python train_models.py       # retrains and overwrites model/ and test_data.csv
streamlit run app.py
```

Training takes about one minute. The versions in `requirements.txt` are pinned because a model saved with one scikit-learn version may not load correctly with another.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app**, select this repository, branch `main` and file `app.py`.
4. Click **Deploy**.

## Notes and limitations

- The dataset is a fictional IBM sample dataset, so the results show how the methods compare rather than how a real telecom business would perform.
- All metrics use the default 0.5 decision threshold. Changing the threshold would trade precision for recall.
- KNN uses `algorithm="kd_tree"`. With the default brute-force search, tied distances were broken differently depending on the number of CPU threads, so the AUC changed slightly between machines.
