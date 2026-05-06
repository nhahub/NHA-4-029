# Credit Card Fraud Detection
=============================

# Overview
--------

This project focuses on developing a machine learning solution to detect fraudulent credit card transactions. The system classifies each transaction as **Not Fraud** or **Fraud** using transaction details, customer information, merchant information, and engineered features.

The main challenge in this project is the severe class imbalance, where most transactions are normal and only a small percentage are fraudulent. To solve this problem, the project uses imbalance handling techniques, probability-based prediction, and threshold tuning.

# Features
--------

* **Fraud Classification Model**: Detects whether a transaction is **Fraud** or **Not Fraud**.
* **Imbalance Handling**: Uses `class_weight`, `scale_pos_weight`, and optional `SMOTE` to improve fraud detection.
* **Feature Engineering**: Creates useful features such as transaction hour, customer age, log amount, and distance between customer and merchant.
* **Threshold Tuning**: Adjusts the fraud decision threshold to avoid predicting only **Not Fraud**.
* **Suspicious Transaction Ranking**: Ranks transactions by fraud probability for investigation and review.
* **Model Comparison**: Compares Logistic Regression, Random Forest, XGBoost, and SMOTE-based models.

# Model Performance
-----------------

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost Weighted |
| **Maximum Fraud Probability** | 0.999595 |
| **Mean Fraud Probability** | 0.024993 |
| **Median Fraud Probability** | 0.001150 |

## Top Suspicious Transactions

| Fraud Probability | Actual |
|-------------------|--------|
| 0.999595 | 1 |
| 0.999588 | 1 |
| 0.999579 | 1 |
| 0.999564 | 1 |
| 0.999563 | 1 |

The top suspicious transactions were actual fraud cases, which confirms that the model is able to detect real fraudulent transactions and is not predicting only the majority class.

# Installation
------------

1. Clone the repository:

    ```bash
    git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the required libraries:

    ```bash
    pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn
    ```

# Usage
-----

1. Open the Kaggle Notebook.

2. Add the dataset from Kaggle:

    ```text
    Credit Card Transactions Fraud Detection
    ```

3. Make sure the dataset contains:

    ```text
    fraudTrain.csv
    fraudTest.csv
    ```

4. Run all notebook cells from top to bottom.

5. The model will train and generate prediction results, model comparison tables, threshold analysis, and suspicious transaction rankings.

# Dataset
-------

The dataset used in this project is **Credit Card Transactions Fraud Detection** from Kaggle.

Dataset link:

```text
https://www.kaggle.com/datasets/kartik2112/fraud-detection
