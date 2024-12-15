# Customer Churn Analytics Framework for Telecommunications

## Project Overview

In the competitive telecommunications industry, retaining customers is paramount. This project addresses the challenge of customer churn by developing a robust **Churn Prediction System** using advanced Machine Learning (ML) techniques. Our solution predicts the likelihood of customer churn and identifies key factors driving it, enabling telecom companies to implement effective retention strategies.

## Key Objectives

- Predict customer churn with high accuracy.
- Identify the most influential factors contributing to churn.
- Provide actionable insights to improve customer retention.

## Features

- **Data Preprocessing**: Handling missing values, outlier detection, normalization, and categorical encoding.
- **Exploratory Data Analysis (EDA)**: Insightful visualizations to understand churn patterns.
- **Feature Engineering**: Selection and transformation of features to optimize model performance.
- **Model Development**: Implementation of multiple machine learning algorithms.
- **Evaluation**: Comprehensive model assessment using metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

## Dataset

The project uses the **Telco Customer Churn Dataset** from Kaggle, provided by IBM.

### Dataset Composition

- **Samples**: 7,043 customer records.
- **Features**: 21 attributes, including demographics, account details, and service usage.
- **Target Variable**: `Churn` (Yes/No).

### Key Features

1. **Demographic Information**
   - Gender, Senior Citizen, Partner, Dependents

2. **Account Information**
   - Tenure, Contract, Payment Method, Monthly Charges, Total Charges

3. **Service Information**
   - Phone Service, Internet Service, Tech Support, Streaming Services

## Methodology

### 1. Data Processing

- Imputed missing values in `Total Charges`.
- Detected and handled outliers using Interquartile Range (IQR).
- Normalized numerical features.
- Encoded categorical variables using Label Encoding and One-Hot Encoding.

### 2. Exploratory Data Analysis (EDA)

- **Box Plots**: Analyzed charge distributions.
- **Bar Charts**: Visualized churn rates by contract type.
- **Correlation Matrix**: Identified strong relationships between features and churn.
- **Geographical Analysis**: Highlighted regions with high churn rates.

### 3. Feature Engineering

- Removed irrelevant columns (e.g., Customer ID, ZipCode).
- Balanced the dataset using SMOTE (Synthetic Minority Oversampling Technique).

### 4. Model Development

Implemented and compared the following models:

- **Logistic Regression**: A baseline model for binary classification.
- **Random Forest**: An ensemble model for better accuracy.
- **XGBoost**: A powerful gradient boosting algorithm.

### 5. Model Tuning

- Applied Grid Search and RandomizedSearchCV for hyperparameter tuning.
- Used k-fold cross-validation to validate model performance.

### 6. Evaluation Metrics

- **Accuracy**: 85.03%
- **F1-Score**: 84.51%
- **Precision**: 87.54%
- **Recall**: 81.68%
- **AUC-ROC**: 0.93

### 7. Error Analysis

- Analyzed False Positives and False Negatives to refine predictions.

## Results

- **Best Model**: XGBoost
- **Key Insights**:
  - Month-to-month contracts and lack of tech support are strong churn indicators.
  - Geographic patterns reveal regions with high churn rates.

## Implementation

### Dependencies

- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost


## Contributions

- **Denish Asodariya**: Data Processing, Exploratory Data Analysis
- **Prince Rajodiya**: Model Development, Evaluation, and Tuning

## References

1. [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [XGBoost Documentation](https://xgboost.readthedocs.io)
4. [SMOTE in Imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

This project was developed as part of the **CS484: Introduction to Machine Learning** course at Illinois Institute of Technology.
