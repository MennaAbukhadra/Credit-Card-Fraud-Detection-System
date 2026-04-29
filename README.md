# Credit Card Fraud Detection System

## Project Overview

The system uses a trained Random Forest model with a saved preprocessing pipeline to classify transactions as:

* `0` - Normal
* `1` - Fraud

The dashboard is designed as a fraud intelligence cockpit where users can inspect transaction patterns, compare model metrics, score new transactions, and review high-risk cases.

## Dataset

The original credit card fraud dataset contains:

* **284,807 transactions**
* **30 anonymized features**: `Time`, `Amount`, and `V1` to `V28`
* **Target column**: `Class`
* A highly imbalanced fraud distribution

Because the full `creditcard.csv` file is larger than GitHub's 100MB file limit, it is not included in the repository. The deployed app uses `creditcard_sample.csv`, a smaller sample dataset, so the dashboard can load quickly on Streamlit Cloud.

For full-data testing, upload the original `creditcard.csv` from the Streamlit sidebar.

## Streamlit Dashboard Features

* Overview page with transaction, fraud, normal, and fraud-rate metrics
* Model Lab for comparing Accuracy, Recall, and F1-score
* Interactive Data Explorer with histograms, boxplots, violin plots, and correlation heatmap
* CSV batch prediction with fraud probability and downloadable results
* Manual transaction scoring using model features
* Risk Monitor for reviewing suspicious transactions after batch scoring

## Machine Learning Pipeline

The app loads these saved artifacts:

* `fraud_model.pkl` - trained fraud detection model
* `scaler.pkl` - feature scaler
* `selector.pkl` - feature selector
* `features.pkl` - expected model feature order
* `labels.pkl` - prediction label mapping

## Model Performance

| Model | Accuracy | Recall | F1-score |
| --- | --- | --- | --- |
| Random Forest | 0.999192 | 0.846939 | 0.783019 |
| XGBoost | 0.998086 | 0.887755 | 0.614841 |
| Logistic Regression | 0.986833 | 0.918367 | 0.193548 |

**Selected model:** Random Forest, because it provides the strongest F1-score while keeping accuracy high on an imbalanced fraud dataset.

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Joblib
* Matplotlib
* Seaborn

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

🔗 https://credit-card-fraud-detection-system-qfbegfz9rqt9nji5kau7t4.streamlit.app/
