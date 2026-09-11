# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a subscription customer is likely to churn, built with pandas, seaborn, scikit-learn, and deployed as an interactive Gradio app.

## Overview

This project walks through the full data science lifecycle on a large (440K+ row) customer churn dataset: cleaning and inspecting the raw data, exploring it to understand what actually drives churn, encoding it for a tree-based model, training and evaluating a Decision Tree Classifier, and wrapping the final model in a Gradio interface so a prediction can be generated from a simple form — no code required.

**Test accuracy: 99.73%** on an 88,167-row held-out test set.

## Dataset

Source: [Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?select=customer_churn_dataset-training-master.csv) by Muhammad Shahid Azeem, on Kaggle (`customer_churn_dataset-training-master.csv`).

- **Rows:** 440,833
- **Columns:** 12
- **Target:** `Churn` (1 = churned, 0 = retained)

### Data Dictionary

| Column | Type | Description |
|---|---|---|
| `CustomerID` | Numeric (ID) | Unique customer identifier. Dropped before modeling. |
| `Age` | Numeric (years) | Customer age, 18–65. |
| `Gender` | Categorical | Male / Female. Label-encoded for modeling. |
| `Tenure` | Numeric (months) | Months the customer has held their subscription, 1–60. |
| `Usage Frequency` | Numeric (count) | How often the customer used the service, 1–30. |
| `Support Calls` | Numeric (count) | Number of support calls placed, 0–10. Strongest churn predictor. |
| `Payment Delay` | Numeric (days) | Average payment delay, 0–30 days. |
| `Subscription Type` | Categorical | Basic / Standard / Premium. One-hot encoded. |
| `Contract Length` | Categorical | Monthly / Quarterly / Annual. One-hot encoded. |
| `Total Spend` | Numeric (currency) | Total spend, $100–$1,000. |
| `Last Interaction` | Numeric (days) | Days since last interaction, 1–30. |
| `Churn` | Binary target | 1 = churned, 0 = retained. |

## Key Insights

- **Support Calls** is the dominant churn driver — correlation of +0.57 with churn, and ~37% of the trained tree's total feature importance.
- **Payment Delay** (+0.31 correlation) is the second strongest positive signal.
- **Total Spend** is protective (-0.43 correlation) — higher spenders churn less.
- **Age** matters a lot: the 50–65 age group churns at **88.8%**, versus 43.4% for ages 30–49.
- **Gender** shows a real gap: women churn at 66.7% vs 49.1% for men.
- **Tenure** barely matters (-0.05 correlation) — long-standing customers churn at nearly the same rate as new ones.
- **Subscription Type** has almost no effect on churn rate or average spend.

## Preprocessing Decisions

- Dropped `CustomerID` (no predictive value, risk of overfitting to an arbitrary ID).
- Dropped one row with data missing across every column (0.0002% of the data).
- No outlier removal — all numeric ranges checked and found realistic.
- One-hot encoded `Subscription Type` and `Contract Length` (unordered categories).
- Label-encoded `Gender` (binary category).
- No resampling — target imbalance (56.7% / 43.3%) was mild enough that both classes had ample data.
- 80/20 train/test split, **stratified** on `Churn`, `random_state=42` for reproducibility.

## Model

- **Algorithm:** Decision Tree Classifier (`max_depth=10`, `random_state=42`)
- **Why:** No scaling needed for a mix of numeric/one-hot features, and importances are directly interpretable — useful for explaining *why* a customer is flagged.

### Results (test set, n=88,167)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Retained) | 0.99 | 1.00 | 1.00 | 38,167 |
| 1 (Churn) | 1.00 | 1.00 | 1.00 | 50,000 |
| **Accuracy** | | | **1.00** | 88,167 |

Confusion matrix: only 239 misclassifications out of 88,167 predictions.

> **Note:** This accuracy is unusually high for a real-world churn problem, most likely because `Support Calls` and `Payment Delay` act as strong, near-deterministic proxies for churn within this dataset. Treat this as a demonstration of the full ML workflow rather than a benchmark for a production system.

## Gradio App

The trained model (`churn_model.pkl`) is served through a Gradio interface with the following inputs:

| Input | Widget | Range / Options |
|---|---|---|
| Age | Slider | 18 – 65 |
| Tenure (Months) | Slider | 1 – 60 |
| Usage Frequency | Slider | 1 – 30 |
| Support Calls | Slider | 0 – 10 |
| Payment Delay (Days) | Slider | 0 – 30 |
| Total Spend ($) | Slider | 100 – 1,000 |
| Last Interaction (Days ago) | Slider | 1 – 30 |
| Contract Type | Radio | Monthly / Annual / Quarterly |
| Gender | Radio | Male / Female |

Output: `"Customer Will Churn"` or `"Customer Will Stay"`.

## Project Structure

```
.
├── project.ipynb           # Full notebook: EDA, preprocessing, training, evaluation, Gradio app
├── churn_model.pkl         # Serialized trained Decision Tree model
├── README.md
└── requirements.txt
```

## Getting Started

### Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
joblib
gradio
```

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook project.ipynb
```

### Launch the app

Run the final cell of the notebook, or extract it into a script and run:

```bash
python app.py
```

The app will start locally (e.g. `http://127.0.0.1:7860`) and open an interactive prediction form in your browser.

## Author

**Umair**
BS Data Science Student
