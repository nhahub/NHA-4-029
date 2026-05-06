# Credit Card Fraud Detection
# ===========================

# Overview
# --------
# This project focuses on developing a machine learning solution to detect fraudulent credit card transactions.
# The system classifies each transaction as either Not Fraud or Fraud using transaction details, customer
# information, merchant information, and engineered features.
#
# The main challenge in this project is class imbalance, where most transactions are normal and only a small
# percentage are fraudulent. To solve this problem, the project uses imbalance handling techniques,
# probability-based prediction, and threshold tuning.


# Features
# --------
# - Fraud Classification Model:
#   Detects whether a transaction is Fraud or Not Fraud.
#
# - Imbalance Handling:
#   Uses class_weight, scale_pos_weight, and optional SMOTE.
#
# - Feature Engineering:
#   Creates features such as transaction hour, customer age, log amount, and distance between customer and merchant.
#
# - Threshold Tuning:
#   Adjusts the fraud decision threshold to avoid predicting only Not Fraud.
#
# - Suspicious Transaction Ranking:
#   Ranks transactions by fraud probability for review.
#
# - Model Comparison:
#   Compares Logistic Regression, Random Forest, XGBoost, and SMOTE-based models.


# Model Performance
# -----------------
# Best Model: XGBoost Weighted
# Maximum Fraud Probability: 0.999595
# Mean Fraud Probability: 0.024993
# Median Fraud Probability: 0.001150


# Top Suspicious Transactions
# ---------------------------
# Fraud Probability    Actual
# 0.999595             1
# 0.999588             1
# 0.999579             1
# 0.999564             1
# 0.999563             1
#
# The top suspicious transactions were actual fraud cases.
# This confirms that the model is able to detect real fraudulent transactions and is not predicting only
# the majority class.


# Installation
# ------------

# 1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install the required libraries:
pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn


# Usage
# -----

# 1. Open the Kaggle Notebook.
# 2. Add the dataset from Kaggle:
#    Credit Card Transactions Fraud Detection
# 3. Make sure the dataset contains:
#    fraudTrain.csv
#    fraudTest.csv
# 4. Run all notebook cells from top to bottom.
# 5. The model will train and generate:
#    - prediction results
#    - model comparison tables
#    - threshold analysis
#    - suspicious transaction rankings


# Dataset
# -------

# Dataset name:
# Credit Card Transactions Fraud Detection
#
# Dataset link:
# https://www.kaggle.com/datasets/kartik2112/fraud-detection
#
# Main files:
# fraudTrain.csv
# fraudTest.csv
#
# Target column:
# is_fraud
#
# Target meaning:
# 0 = Not Fraud
# 1 = Fraud


# Feature Engineering
# -------------------

# The following features were created to improve model performance:
#
# trans_hour       : Transaction hour
# trans_dayofweek  : Transaction day of week
# trans_month      : Transaction month
# trans_day        : Transaction day
# age              : Customer age
# amt_log          : Log transformation of transaction amount
# distance_km      : Distance between customer and merchant


# Models
# ------

# The project trains and compares the following models:
#
# - Dummy Classifier
# - Logistic Regression
# - Random Forest
# - XGBoost
# - SMOTE + Logistic Regression


# Results
# -------

# The best model was the XGBoost Weighted Model.
# It successfully assigned high fraud probabilities to real fraudulent transactions.
#
# Example:
# Fraud Probability: 0.999595
# Actual: 1
#
# This proves that the model can identify fraud transactions with high confidence.


# Output Files
# ------------

# The notebook saves the following files:
#
# fraud_detection_model_comparison.csv
# fraud_detection_feature_importance.csv
# fraud_detection_threshold_comparison.csv
# top_suspicious_transactions.csv
#
# Saved location in Kaggle:
# /kaggle/working/


# Future Enhancements
# -------------------

# - Use LightGBM and compare it with XGBoost.
# - Apply advanced hyperparameter tuning.
# - Add SHAP explainability for model interpretation.
# - Build a Streamlit web app for fraud prediction.
# - Deploy the model as a real-time API.
# - Add customer transaction history and velocity-based features.
# - Use business cost-based threshold optimization.


# Contributing
# ------------

# Contributions are welcome.
# Feel free to fork the repository and submit a pull request.


# Contact
# -------

# For any inquiries, reach out to:
# kimobadrsaber@gmail.com
