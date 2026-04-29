# Credit Card Fraud Detection System

## Overview

This project builds an end-to-end Machine Learning system to detect fraudulent credit card transactions.

It combines data preprocessing, class balancing, feature selection, and multiple ML models, then deploys the best model in an interactive Streamlit dashboard for real-time prediction.

---

## Dataset

* **Total Transactions:** 284,807
* **Fraud Cases:** Highly imbalanced dataset
* **Features:** 30 anonymized features (V1–V28 + Time + Amount)

---

## Objective

To accurately classify transactions as:

* **0 → Normal**
* **1 → Fraud**

While focusing on maximizing **recall** to avoid missing fraudulent transactions.

---

## Data Preprocessing

* Handling class imbalance using **SMOTE**
* Feature scaling using **StandardScaler**
* Feature selection using **SelectKBest**
* Splitting data:

  * 80% Training
  * 20% Testing

---

## Exploratory Data Analysis (EDA)

Key visualizations include:

* Class distribution (Fraud vs Normal)
* Correlation heatmap
* Feature distributions
* Impact of selected features

---

## Machine Learning Models

The following models were trained and evaluated:

* Logistic Regression
* Random Forest
* XGBoost

---

## Model Performance

| Model               | Accuracy | Recall | F1-score |
| ------------------- | -------- | ------ | -------- |
| Logistic Regression | 0.986    | 0.91   | 0.19     |
| Random Forest       | 0.999    | 0.84   | 0.78     |
| XGBoost             | 0.998    | 0.88   | 0.61     |

**Best Model: Random Forest**
Provides the best balance between precision and recall.

---

## Hyperparameter Tuning

Random Forest was optimized using **RandomizedSearchCV**.

**Best Parameters:**

* `n_estimators = 50`
* `max_depth = 15`
* `min_samples_split = 5`

---

## Final Model Results (After Tuning)

* **Accuracy:** 0.9988
* **Recall:** 0.87
* **F1-score:** 0.72

High recall ensures most fraud cases are detected.
Balanced performance improves reliability.

---

## Streamlit Dashboard

An interactive web app was built using **Streamlit**.

### Features:

* Dashboard overview with key metrics
* Interactive visualizations (Plotly)
* Model comparison
* Real-time fraud prediction
* Upload CSV for batch prediction
* Fraud probability and risk analysis
* Download prediction results

---

## Key Insights

* Fraud detection requires focusing on **recall over accuracy**
* Feature selection significantly improved performance
* Random Forest achieved the best overall balance
* The system can be used as a real-time fraud detection tool

---

## Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Plotly
* Scikit-learn
* XGBoost
* Streamlit

---

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run credit_card_app
```

---

---

## Live Demo

You can access the deployed Streamlit app here:

🔗 https://creditcard-mauuwr8v5smyrqvppf3vvs.streamlit.app/
