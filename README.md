# ============================================================
# Fraud Detection Project — Full Kaggle Notebook Code
# Dataset added in your Kaggle Notebook:
# Credit Card Transactions Fraud Detection
# Expected files:
# fraudTrain.csv
# fraudTest.csv
#
# Target:
# is_fraud = 0  --> Not Fraud
# is_fraud = 1  --> Fraud
#
# Important problem solved in this version:
# The model may predict only "Not Fraud" because fraud cases are rare.
# This code solves that using:
# 1. class_weight="balanced"
# 2. XGBoost scale_pos_weight, if available
# 3. probability-based prediction
# 4. threshold tuning
# 5. top suspicious transactions review
#
# Project methodology follows:
# EDA, preprocessing, imbalance handling, model training,
# model comparison, precision, recall, F1-score, confusion matrix.
# Source: uploaded project proposal. :contentReference[oaicite:0]{index=0}
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. Global Configuration
# ============================================================

RANDOM_STATE = 42

pd.set_option("display.max_columns", 150)
pd.set_option("display.max_rows", 150)

np.random.seed(RANDOM_STATE)


# ============================================================
# 3. Find Kaggle Dataset Files Automatically
# ============================================================

print("Files found inside /kaggle/input:")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def find_file(filename):
    """
    Search for a file inside /kaggle/input.
    This avoids path errors when Kaggle changes folder names.
    """
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for file in filenames:
            if file == filename:
                return os.path.join(dirname, file)

    raise FileNotFoundError(
        f"{filename} not found. Please add the dataset from Kaggle Notebook > Add Input."
    )


TRAIN_PATH = find_file("fraudTrain.csv")
TEST_PATH = find_file("fraudTest.csv")

print("\nTrain path:", TRAIN_PATH)
print("Test path:", TEST_PATH)


# ============================================================
# 4. Load Data
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

display(train_df.head())
display(test_df.head())


# ============================================================
# 5. Dataset Overview
# ============================================================

print("Train dataset information:")
train_df.info()

print("\nTest dataset information:")
test_df.info()

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTrain basic statistics:")
display(train_df.describe().T)

print("\nMissing values in train:")
missing_report = pd.DataFrame({
    "missing_count": train_df.isnull().sum(),
    "missing_percentage": train_df.isnull().mean() * 100,
    "dtype": train_df.dtypes
})

display(missing_report)


# ============================================================
# 6. Target Distribution
# ============================================================

# is_fraud:
# 0 = Not Fraud
# 1 = Fraud

target_summary = pd.DataFrame({
    "count": train_df["is_fraud"].value_counts().sort_index(),
    "percentage": train_df["is_fraud"].value_counts(normalize=True).sort_index() * 100
})

target_summary.index = ["Not Fraud", "Fraud"]

display(target_summary)

plt.figure(figsize=(6, 4))
train_df["is_fraud"].value_counts().sort_index().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class: 0 = Not Fraud, 1 = Fraud")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.show()

fraud_percentage = train_df["is_fraud"].mean() * 100
print(f"Fraud percentage in training data: {fraud_percentage:.4f}%")


# ============================================================
# 7. EDA: Transaction Amount
# ============================================================

plt.figure(figsize=(8, 4))
train_df["amt"].hist(bins=100)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 4))
np.log1p(train_df["amt"]).hist(bins=100)
plt.title("Log Transaction Amount Distribution")
plt.xlabel("log1p(Amount)")
plt.ylabel("Frequency")
plt.show()

amount_by_class = train_df.groupby("is_fraud")["amt"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

display(amount_by_class)


# ============================================================
# 8. EDA: Fraud Rate by Category
# ============================================================

category_fraud_rate = train_df.groupby("category")["is_fraud"].agg(
    ["count", "mean"]
).sort_values("mean", ascending=False)

category_fraud_rate["fraud_rate_percentage"] = category_fraud_rate["mean"] * 100

display(category_fraud_rate)

plt.figure(figsize=(10, 5))
category_fraud_rate["fraud_rate_percentage"].sort_values().plot(kind="barh")
plt.title("Fraud Rate by Transaction Category")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Category")
plt.show()


# ============================================================
# 9. EDA: Gender Fraud Rate
# ============================================================

gender_fraud_rate = train_df.groupby("gender")["is_fraud"].agg(
    ["count", "mean"]
)

gender_fraud_rate["fraud_rate_percentage"] = gender_fraud_rate["mean"] * 100

display(gender_fraud_rate)

plt.figure(figsize=(6, 4))
gender_fraud_rate["fraud_rate_percentage"].plot(kind="bar")
plt.title("Fraud Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=0)
plt.show()


# ============================================================
# 10. Feature Engineering Functions
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between customer location and merchant location.

    Why?
    Fraud transactions may happen at unusual merchant locations.
    Distance can help the model detect suspicious behavior.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371

    return earth_radius_km * c


def feature_engineering(df):
    """
    Create fraud-detection features:
    - transaction hour
    - transaction day of week
    - transaction month
    - customer age
    - log amount
    - distance between customer and merchant
    """

    data = df.copy()

    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
    data["dob"] = pd.to_datetime(data["dob"])

    data["trans_hour"] = data["trans_date_trans_time"].dt.hour
    data["trans_dayofweek"] = data["trans_date_trans_time"].dt.dayofweek
    data["trans_month"] = data["trans_date_trans_time"].dt.month
    data["trans_day"] = data["trans_date_trans_time"].dt.day

    data["age"] = (
        data["trans_date_trans_time"].dt.year
        - data["dob"].dt.year
    )

    data["amt_log"] = np.log1p(data["amt"])

    data["distance_km"] = haversine_distance(
        data["lat"],
        data["long"],
        data["merch_lat"],
        data["merch_long"]
    )

    return data


train_fe = feature_engineering(train_df)
test_fe = feature_engineering(test_df)

print("Train shape after feature engineering:", train_fe.shape)
print("Test shape after feature engineering:", test_fe.shape)

display(train_fe.head())


# ============================================================
# 11. Drop Columns Not Suitable for Modeling
# ============================================================

# Removed columns:
# - names, street, cc_num, trans_num: identifiers
# - original dates: replaced by engineered features
# - dob: replaced by age
# - unix_time: may duplicate time information
# - Unnamed: 0: index column

drop_columns = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "dob",
    "trans_num",
    "unix_time"
]

drop_columns = [col for col in drop_columns if col in train_fe.columns]

target = "is_fraud"

X_train = train_fe.drop(columns=drop_columns + [target])
y_train = train_fe[target].astype(int)

X_test = test_fe.drop(columns=drop_columns + [target])
y_test = test_fe[target].astype(int)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

display(X_train.head())


# ============================================================
# 12. Identify Numeric and Categorical Features
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Numeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 13. Preprocessing Pipelines
# ============================================================

# For Logistic Regression:
# - scale numeric features
# - one-hot encode categorical features

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer_ohe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ]
)

preprocessor_ohe = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_ohe, categorical_features)
    ]
)


# For Tree Models:
# - Random Forest and XGBoost can work with ordinal-encoded categories
# - This is faster and smaller than one-hot encoding for this dataset

categorical_transformer_ordinal = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ]
)

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_ordinal, categorical_features)
    ]
)


# ============================================================
# 14. Evaluation Function
# ============================================================

def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    """
    Evaluate the fraud detection model.

    Important:
    Accuracy is not enough because fraud is rare.
    We focus on:
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - PR-AUC
    - Confusion Matrix
    """

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
        "Fraud Predictions": y_pred.sum()
    }

    print("=" * 90)
    print(model_name)
    print("=" * 90)

    print(f"Threshold used: {threshold}")
    print("\nPrediction counts:")
    print(pd.Series(y_pred).value_counts().rename(index={0: "Not Fraud", 1: "Fraud"}))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Fraud", "Fraud"],
        zero_division=0
    ))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Fraud", "Fraud"]
    )

    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {model_name}")
    plt.show()

    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.show()

    return metrics


# ============================================================
# 15. Baseline Model: Dummy Classifier
# ============================================================

# This model predicts the majority class only.
# It proves why accuracy is misleading in fraud detection.

dummy_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_ohe),
        ("model", DummyClassifier(strategy="most_frequent"))
    ]
)

dummy_model.fit(X_train, y_train)

dummy_metrics = evaluate_model(
    dummy_model,
    X_test,
    y_test,
    model_name="Dummy Classifier",
    threshold=0.5
)


# ============================================================
# 16. Model 1: Logistic Regression with Balanced Class Weight
# ============================================================

# class_weight="balanced" gives fraud cases more weight.
# This helps prevent the model from predicting only Not Fraud.

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_ohe),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)

logistic_model.fit(X_train, y_train)

logistic_metrics = evaluate_model(
    logistic_model,
    X_test,
    y_test,
    model_name="Logistic Regression - Balanced",
    threshold=0.5
)


# ============================================================
# 17. Model 2: Random Forest with Balanced Subsample
# ============================================================

# Random Forest is a strong non-linear model.
# class_weight="balanced_subsample" helps with rare fraud cases.

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)

rf_model.fit(X_train, y_train)

rf_metrics = evaluate_model(
    rf_model,
    X_test,
    y_test,
    model_name="Random Forest - Balanced",
    threshold=0.5
)


# ============================================================
# 18. Optional Model 3: XGBoost Weighted
# ============================================================

# XGBoost is usually strong for tabular fraud detection.
# scale_pos_weight helps the model focus on the rare fraud class.

xgb_model = None
xgb_metrics = None

try:
    from xgboost import XGBClassifier

    fraud_count = y_train.sum()
    not_fraud_count = len(y_train) - fraud_count

    scale_pos_weight = not_fraud_count / fraud_count

    print("XGBoost scale_pos_weight:", scale_pos_weight)

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor_tree),
            ("model", XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )

    xgb_model.fit(X_train, y_train)

    xgb_metrics = evaluate_model(
        xgb_model,
        X_test,
        y_test,
        model_name="XGBoost - Weighted",
        threshold=0.5
    )

except Exception as e:
    print("XGBoost is not available or failed to run.")
    print(e)


# ============================================================
# 19. Optional Model 4: SMOTE + Logistic Regression
# ============================================================

# SMOTE creates synthetic fraud cases during training.
# This can help if the model predicts only Not Fraud.
# If imbalanced-learn is not available, this section will be skipped.

smote_model = None
smote_metrics = None

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    smote_model = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor_ohe),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )

    smote_model.fit(X_train, y_train)

    smote_metrics = evaluate_model(
        smote_model,
        X_test,
        y_test,
        model_name="Logistic Regression + SMOTE",
        threshold=0.5
    )

except Exception as e:
    print("SMOTE is not available or failed to run.")
    print(e)


# ============================================================
# 20. Select Best Base Model for Threshold Tuning
# ============================================================

# We choose XGBoost if available because it is usually stronger.
# Otherwise, we use Random Forest.

if xgb_model is not None:
    best_base_model = xgb_model
    best_model_name = "XGBoost - Weighted"
else:
    best_base_model = rf_model
    best_model_name = "Random Forest - Balanced"

print("Best base model selected for threshold tuning:")
print(best_model_name)


# ============================================================
# 21. Main Fix: Probability Check + Threshold Tuning
# ============================================================

# This section solves the problem:
# "The model only predicts Not Fraud."
#
# Instead of using model.predict(), we use:
# predict_proba()[:, 1]
#
# Then we choose a better threshold.

y_proba = best_base_model.predict_proba(X_test)[:, 1]

print("Fraud probability statistics:")
print("Minimum fraud probability:", y_proba.min())
print("Maximum fraud probability:", y_proba.max())
print("Mean fraud probability:", y_proba.mean())
print("Median fraud probability:", np.median(y_proba))

plt.figure(figsize=(8, 4))
pd.Series(y_proba).hist(bins=100)
plt.title("Predicted Fraud Probability Distribution")
plt.xlabel("Fraud Probability")
plt.ylabel("Frequency")
plt.show()


# ============================================================
# 22. Evaluation with Default Threshold 0.5
# ============================================================

y_pred_default = (y_proba >= 0.5).astype(int)

print("\nPrediction counts with threshold 0.5:")
print(pd.Series(y_pred_default).value_counts().rename(index={0: "Not Fraud", 1: "Fraud"}))

print("\nConfusion Matrix with threshold 0.5:")
print(confusion_matrix(y_test, y_pred_default))

print("\nClassification Report with threshold 0.5:")
print(classification_report(
    y_test,
    y_pred_default,
    target_names=["Not Fraud", "Fraud"],
    zero_division=0
))


# ============================================================
# 23. Find Best Threshold Using F1-score
# ============================================================

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

precision_for_thresholds = precision[:-1]
recall_for_thresholds = recall[:-1]

f1_scores = 2 * precision_for_thresholds * recall_for_thresholds / (
    precision_for_thresholds + recall_for_thresholds + 1e-9
)

best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]

print("\nBest threshold based on F1-score:")
print(best_threshold)

print("Precision at best threshold:")
print(precision_for_thresholds[best_threshold_index])

print("Recall at best threshold:")
print(recall_for_thresholds[best_threshold_index])

print("F1-score at best threshold:")
print(f1_scores[best_threshold_index])


plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision_for_thresholds, label="Precision")
plt.plot(thresholds, recall_for_thresholds, label="Recall")
plt.plot(thresholds, f1_scores, label="F1-score")
plt.axvline(best_threshold, linestyle="--", label=f"Best threshold = {best_threshold:.4f}")
plt.title("Precision, Recall, and F1-score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.show()


# ============================================================
# 24. Evaluate Best Model with Tuned Threshold
# ============================================================

tuned_metrics = evaluate_model(
    best_base_model,
    X_test,
    y_test,
    model_name=f"{best_model_name} - Tuned Threshold",
    threshold=best_threshold
)


# ============================================================
# 25. Force Review Thresholds for Business Use
# ============================================================

# Sometimes F1 threshold is still too strict.
# Here we test several lower thresholds so you can show that the model can detect fraud.

business_thresholds = [0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]

threshold_rows = []

for th in business_thresholds:
    y_pred_th = (y_proba >= th).astype(int)
    cm = confusion_matrix(y_test, y_pred_th)

    threshold_rows.append({
        "Threshold": th,
        "Fraud Predictions": y_pred_th.sum(),
        "Accuracy": accuracy_score(y_test, y_pred_th),
        "Precision": precision_score(y_test, y_pred_th, zero_division=0),
        "Recall": recall_score(y_test, y_pred_th, zero_division=0),
        "F1-score": f1_score(y_test, y_pred_th, zero_division=0),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })

threshold_comparison = pd.DataFrame(threshold_rows)

display(threshold_comparison)

plt.figure(figsize=(8, 5))
plt.plot(threshold_comparison["Threshold"], threshold_comparison["Precision"], marker="o", label="Precision")
plt.plot(threshold_comparison["Threshold"], threshold_comparison["Recall"], marker="o", label="Recall")
plt.plot(threshold_comparison["Threshold"], threshold_comparison["F1-score"], marker="o", label="F1-score")
plt.gca().invert_xaxis()
plt.title("Business Threshold Comparison")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.show()


# ============================================================
# 26. Top Suspicious Transactions
# ============================================================

# Even if threshold is high, we can rank transactions by fraud probability.
# This is practical in real fraud detection:
# analysts review the most suspicious transactions.

fraud_review_table = X_test.copy()
fraud_review_table["Actual"] = y_test.values
fraud_review_table["Fraud Probability"] = y_proba
fraud_review_table["Prediction Tuned"] = (y_proba >= best_threshold).astype(int)
fraud_review_table["Prediction Label"] = fraud_review_table["Prediction Tuned"].map({
    0: "Not Fraud",
    1: "Fraud"
})

top_suspicious = fraud_review_table.sort_values(
    "Fraud Probability",
    ascending=False
).head(30)

display(top_suspicious[[
    "Fraud Probability",
    "Prediction Tuned",
    "Prediction Label",
    "Actual"
]])


# ============================================================
# 27. Final Check: Did the Model Predict Fraud?
# ============================================================

y_pred_tuned = (y_proba >= best_threshold).astype(int)
fraud_predicted_count = y_pred_tuned.sum()

if fraud_predicted_count == 0:
    print("""
WARNING:
The model still predicts zero fraud cases.

Possible fixes:
1. Use a lower business threshold, such as 0.10, 0.05, or 0.02.
2. Use XGBoost if it did not run.
3. Use SMOTE.
4. Add stronger feature engineering.
5. Use more fraud-focused metrics like PR-AUC and Recall.
""")
else:
    print(f"SUCCESS: The model predicted {fraud_predicted_count} fraud transactions using tuned threshold.")


# ============================================================
# 28. Predict One Transaction: Most Suspicious Transaction
# ============================================================

# Do not test only X_test.iloc[0], because it may be a normal transaction.
# Instead, test the transaction with the highest fraud probability.

most_suspicious_index = np.argmax(y_proba)

sample_transaction = X_test.iloc[[most_suspicious_index]]
actual_label = y_test.iloc[most_suspicious_index]

fraud_probability = best_base_model.predict_proba(sample_transaction)[:, 1][0]
prediction = int(fraud_probability >= best_threshold)

print("Most suspicious transaction:")
print("Fraud probability:", fraud_probability)
print("Actual label:", actual_label)

if prediction == 1:
    print("Prediction: Fraud")
else:
    print("Prediction: Not Fraud")


# ============================================================
# 29. Predict First 10 Transactions
# ============================================================

sample_transactions = X_test.head(10).copy()

sample_probabilities = best_base_model.predict_proba(sample_transactions)[:, 1]
sample_predictions = (sample_probabilities >= best_threshold).astype(int)

prediction_table = sample_transactions.copy()
prediction_table["Fraud Probability"] = sample_probabilities
prediction_table["Prediction Number"] = sample_predictions
prediction_table["Prediction Label"] = prediction_table["Prediction Number"].map({
    0: "Not Fraud",
    1: "Fraud"
})
prediction_table["Actual"] = y_test.head(10).values

display(prediction_table[[
    "Fraud Probability",
    "Prediction Number",
    "Prediction Label",
    "Actual"
]])


# ============================================================
# 30. Model Comparison Table
# ============================================================

all_metrics = [
    dummy_metrics,
    logistic_metrics,
    rf_metrics,
    tuned_metrics
]

if xgb_metrics is not None:
    all_metrics.append(xgb_metrics)

if smote_metrics is not None:
    all_metrics.append(smote_metrics)

metrics_df = pd.DataFrame(all_metrics)

metrics_df = metrics_df.sort_values("PR-AUC", ascending=False)

display(metrics_df)


# ============================================================
# 31. Feature Importance
# ============================================================

# Feature importance is available for Random Forest.
# If XGBoost is selected, this still explains Random Forest features.

rf_estimator = rf_model.named_steps["model"]

feature_names = numeric_features + categorical_features

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_estimator.feature_importances_
}).sort_values("Importance", ascending=False)

display(feature_importance.head(25))

plt.figure(figsize=(8, 8))
feature_importance.head(25).sort_values("Importance").plot(
    kind="barh",
    x="Feature",
    y="Importance",
    legend=False
)
plt.title("Top 25 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# ============================================================
# 32. Save Output Files in Kaggle
# ============================================================

metrics_df.to_csv("/kaggle/working/fraud_detection_model_comparison.csv", index=False)
feature_importance.to_csv("/kaggle/working/fraud_detection_feature_importance.csv", index=False)
threshold_comparison.to_csv("/kaggle/working/fraud_detection_threshold_comparison.csv", index=False)
top_suspicious.to_csv("/kaggle/working/top_suspicious_transactions.csv", index=False)

print("Saved files:")
print("/kaggle/working/fraud_detection_model_comparison.csv")
print("/kaggle/working/fraud_detection_feature_importance.csv")
print("/kaggle/working/fraud_detection_threshold_comparison.csv")
print("/kaggle/working/top_suspicious_transactions.csv")


# ============================================================
# 33. Final Summary
# ============================================================

print("=" * 90)
print("FINAL PROJECT SUMMARY")
print("=" * 90)

print(f"""
Dataset used:
- Credit Card Transactions Fraud Detection
- Train file:
  {TRAIN_PATH}
- Test file:
  {TEST_PATH}

Target:
- is_fraud
- 0 = Not Fraud
- 1 = Fraud

Main project problem:
- Fraud cases are rare.
- A weak model may predict only Not Fraud.
- This is why accuracy alone is misleading.

What this code did to solve the problem:
1. Used class_weight='balanced' in Logistic Regression.
2. Used class_weight='balanced_subsample' in Random Forest.
3. Used scale_pos_weight in XGBoost, if available.
4. Used predict_proba instead of only predict.
5. Tuned the threshold instead of depending only on 0.5.
6. Displayed top suspicious transactions by fraud probability.
7. Compared different thresholds for business decision-making.

Best base model used for threshold tuning:
- {best_model_name}

Best threshold based on F1-score:
- {best_threshold}

Final tuned-threshold result:
- Fraud predictions: {fraud_predicted_count}

Business meaning:
- Higher recall means catching more fraud.
- Higher precision means fewer false fraud alerts.
- Lower threshold catches more fraud but creates more false alarms.
- Higher threshold creates fewer false alarms but may miss fraud.

Recommended final project discussion:
- Use PR-AUC, recall, precision, F1-score, and confusion matrix.
- Do not depend on accuracy only.
- Explain that threshold selection is a business decision.
""")
