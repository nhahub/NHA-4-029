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



# Feature Engineering
-------------------

The following features were created to improve model performance:

* `trans_hour`: Transaction hour extracted from the transaction date and time.
* `trans_dayofweek`: Transaction day of week to help detect weekly fraud patterns.
* `trans_month`: Transaction month to capture monthly transaction behavior.
* `trans_day`: Transaction day of the month.
* `age`: Customer age calculated from date of birth.
* `amt_log`: Log transformation of transaction amount to reduce skewness.
* `distance_km`: Distance between customer location and merchant location.

# Models
------

The project trains and compares the following models:

* Dummy Classifier: Used as a baseline model.
* Logistic Regression: Used with balanced class weights.
* Random Forest: Used as a strong non-linear model.
* XGBoost: Used as the main weighted boosting model.
* SMOTE + Logistic Regression: Used to test oversampling for fraud detection.

# Imbalance Handling
------------------

The dataset is highly imbalanced because most transactions are normal and only a small percentage are fraudulent.

To solve this problem, the project uses:

* `class_weight="balanced"` in Logistic Regression.
* `class_weight="balanced_subsample"` in Random Forest.
* `scale_pos_weight` in XGBoost.
* Optional `SMOTE` oversampling.
* Probability-based prediction using `predict_proba`.
* Decision threshold tuning.

# Threshold Tuning
----------------

The default classification threshold is usually `0.5`.

In fraud detection, this threshold may be too high because fraud cases are rare. This can make the model predict mostly **Not Fraud**.

This project uses predicted fraud probabilities and selects a better threshold based on model performance.



# Results
-------

The best model was the XGBoost Weighted Model.

* It successfully assigned high fraud probabilities to real fraudulent transactions.
* It helped solve the problem of predicting only Not Fraud.
* It ranked suspicious transactions based on fraud probability.

Example result:

| Fraud Probability | Actual |
|-------------------|--------|
| 0.999595 | 1 |

This proves that the model can identify fraud transactions with high confidence.

# Output Files
------------

The notebook saves the following files:

* `fraud_detection_model_comparison.csv`
* `fraud_detection_feature_importance.csv`
* `fraud_detection_threshold_comparison.csv`
* `top_suspicious_transactions.csv`



# Technologies Used
-----------------

The project uses the following tools and libraries:

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Kaggle Notebook

# Future Enhancements
-------------------

Future improvements can include:

* Use LightGBM and compare it with XGBoost.
* Apply advanced hyperparameter tuning.
* Add SHAP explainability for model interpretation.
* Build a Streamlit web app for fraud prediction.
* Deploy the model as a real-time API.
* Add customer transaction history and velocity-based features.
* Use business cost-based threshold optimization.

# Contributing
------------

Contributions are welcome.

* Feel free to fork the repository.
* Submit a pull request.
* Suggest improvements or report issues.

# Contact
-------

For any inquiries, reach out to:

```text
kimo badr saber@gmail.com




