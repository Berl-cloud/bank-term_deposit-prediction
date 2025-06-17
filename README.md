# Bank Term Deposit Subscription Prediction

This project aims to build a predictive model that determines whether a client will subscribe to a term deposit product based on historical bank marketing campaign data. It involves in-depth data analysis, statistical testing, feature engineering, machine learning modeling, and interactive model deployment using Streamlit.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Objectives](#project-objectives)
- [Key Insights](#key-insights)
- [Recommendations](#recommendations)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Deployment](#model-deployment)
- [Results](#results)
- [References](#references)

---

## Project Overview

This project uses data from a Portuguese bank's direct marketing campaign to predict client subscription to term deposit products. By leveraging machine learning and statistical techniques, we aim to help the marketing team optimize contact strategies and increase subscription rates.

---

## Dataset Description

- **Source**: [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Records**: 45,211
- **Features**: 16 independent variables (categorical & numerical), and 1 binary target variable (`y`)
- **Target Variable**: `y` (1 = client subscribed, 0 = not subscribed)

---

## Project Objectives

- Understand data patterns through statistical analysis and hypothesis testing
- Engineer meaningful features to enhance predictive power
- Evaluate multiple classification algorithms (linear, nonlinear, ensemble)
- Select the best-performing model through cross-validation and hyperparameter tuning
- Deploy the final model using Streamlit for real-time predictions

---

## Key Insights

- Campaigns conducted in **May**, **August**, and **October** yielded higher subscription rates.
- Clients with **successful previous campaign outcomes** were significantly more likely to subscribe again.
- **Call duration** is the strongest predictor of subscription.
- Clients contacted via **cellular networks** had higher conversion rates than those contacted via telephone lines.
- **Job type**, **education level**, and **contact frequency** significantly impact the likelihood of subscription.

---

## Recommendations

1. **Focus Campaigns in High-Conversion Months**  
   Optimize marketing schedules to align with months historically showing strong performance (e.g., May, August, October).

2. **Retarget Clients with Past Positive Responses**  
   Prioritize follow-up with clients who responded positively in prior campaigns to increase conversion rates.

3. **Train Reps to Increase Engagement Duration**  
   Emphasize deeper, longer conversations during calls as duration positively correlates with conversions.

4. **Prefer Mobile Channels Over Landlines**  
   Shift contact strategy to favor cellular outreach, which yielded better results in past campaigns.

---

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebooks
- Streamlit (for deployment)
- Yellowbrick (for model diagnostics)
- XGBoost, LightGBM
- SciPy (for statistical tests)

---

## Project Structure



---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bank-term-prediction.git
cd bank-term-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

---

## Model Deployment

The Streamlit app allows users to input client features and receive a real-time prediction on whether the client is likely to subscribe.
 [Live Demo](https://your-streamlit-app-url)

---

 ##  Results

- Best Model: Gradient Boosting Classifier (after hyperparameter tuning)

- Accuracy: 92.4%

- ROC-AUC Score: 0.93

- Precision: 0.87

- Recall: 0.89

All metrics were validated using cross-validation and a separate test set to ensure generalization.

---

## References

* [Scikit Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
* [Machine Learning Mastery](https://machinelearningmastery.com/extra-trees-ensemble-with-python/)
* [Yellowbrick Library](https://www.scikit-yb.org/en/latest/)
* [Data Science Infinity DS Templates](https://data-science-infinity.teachable.com/)
* [Data Science Blog](https://www.reneshbedre.com/blog/anova.html)