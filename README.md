# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red?style=for-the-badge&logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Fraud%20Detection-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge)

## 🚀 Live Demo

🔗 [Open the Streamlit App](https://fraud-detection-mw9qdcgvjmoazcdg2lyeqc.streamlit.app/)

---

## 📌 Overview

This project is a machine learning application for detecting fraudulent credit card transactions.

The model predicts whether a transaction is:

- ✅ Not Fraud
- 🚨 Fraud

The main challenge is the highly imbalanced dataset, where fraud cases are much fewer than normal transactions.  
To improve detection, the project uses feature engineering, imbalance handling, probability prediction, and threshold tuning.

---

## ✨ Key Features

- 🚨 Fraud / Not Fraud classification
- 📊 Fraud probability prediction
- ⚖️ Imbalanced data handling
- 🎯 Threshold tuning
- 🔎 Suspicious transaction ranking
- 🌐 Interactive Streamlit web app
- 🧠 Model comparison between Logistic Regression, Random Forest, XGBoost, and SMOTE-based models

---

## 🧠 Best Model

The best-performing model is:

```text
XGBoost Weighted Model
```

### Model Probability Results

| Metric | Value |
|---|---|
| Maximum Fraud Probability | 0.999595 |
| Mean Fraud Probability | 0.024993 |
| Median Fraud Probability | 0.001150 |

The top suspicious transactions were correctly ranked as real fraud cases, showing that the model can detect fraud instead of predicting only the majority class.

---

## 🗂️ Dataset

Dataset used:

**Credit Card Transactions Fraud Detection**

Kaggle Dataset:  
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Main files:

```text
fraudTrain.csv
fraudTest.csv
```

Target column:

```text
is_fraud
```

| Value | Meaning |
|---|---|
| 0 | Not Fraud |
| 1 | Fraud |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Streamlit
- Kaggle Notebook

---

## ⚙️ How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Install requirements:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn streamlit
```

Run the app:

```bash
streamlit run app.py
```

---

## 📈 Project Outputs

The project generates:

```text
fraud_detection_model_comparison.csv
fraud_detection_feature_importance.csv
fraud_detection_threshold_comparison.csv
top_suspicious_transactions.csv
```

---

## 🚀 Future Improvements

- Add LightGBM model
- Add SHAP explainability
- Improve Streamlit UI
- Add batch upload prediction
- Deploy as a real-time API
- Add model monitoring

---

## ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub.

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only and should not be used as a production fraud detection system without further validation.
