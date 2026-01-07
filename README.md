Customer Churn Prediction (Telecom Company)
Project Overview

This project focuses on predicting customer churn for a telecom company using structured data. Customer churn—when a customer stops using a company’s services—is a critical metric for subscription-based businesses. By identifying customers likely to churn, businesses can take proactive measures to retain high-risk customers and maximize revenue.

This project is part of a data science mentorship program and emphasizes hands-on analysis, feature engineering, modeling, and business insights.

Dataset

Source: Telecom company dataset (fictitious but realistic)

Features: Customer demographics, account information, subscription details, and services usage.

Target Variable: Churn (Yes/No)

Key Columns:

gender, SeniorCitizen, Partner, Dependents – demographic features

tenure, MonthlyCharges, TotalCharges – account and usage metrics

PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc. – service usage indicators

Contract, PaperlessBilling, PaymentMethod – subscription and billing details

### Project Workflow
1. Exploratory Data Analysis (EDA)

Examined distributions of categorical and numerical features.

Visualized customer churn rates and MRR churn rates across demographics, services, and contract types.

Key insights:

High-risk demographics: Senior citizens, customers without partners or dependents.

High-risk services: Fiber optic internet, monthly contracts, lack of online security or tech support.

Churn trends: Customers in their first year with the service are more likely to churn.

2. Feature Engineering

Created Customer Churn Rate and MRR Churn Rate features.

Handled data type mismatches (TotalCharges converted to numeric).

Selected top categorical features using Chi-Square test.

Evaluated numerical features via correlation analysis with the target.

3. Modeling

Baseline Model: Logistic Regression

Accuracy: 0.788

ROC-AUC: 0.838

Key coefficients interpreted for business context.

Advanced Model: XGBoost Classifier

Accuracy: 0.807

ROC-AUC: 0.836

Feature importance analyzed and visualized.

Model Comparison:

Logistic Regression performs slightly better in ROC-AUC, while XGBoost shows higher overall accuracy.

Both models are capable of identifying high-risk customers, but XGBoost provides more nuanced feature importance insights.

4. Evaluation Metrics

Accuracy

ROC-AUC

Precision, Recall, F1-Score

Confusion Matrix

Feature Importance (for model interpretability)

Key Business Insights

Senior citizens and first-year customers are the highest churn risk segments.

Monthly contracts, lack of tech support, and lack of online security correlate with higher churn.

Fiber optic users show higher churn, suggesting potential service dissatisfaction.

MRR Churn analysis highlights revenue loss concentrated among certain contract types and services.

Recommendations:

Implement targeted retention campaigns for high-risk groups (e.g., incentives for first-year customers and senior citizens).

Offer add-on services like online security and tech support to reduce churn.

Reevaluate fiber optic service offerings or customer support for this segment.

### Folder Structure
DataScience_MentorshipProject/
│
├── README.md
├── Customer_Churn_Project/
│   ├── data/
│   │   └── telco.csv
│   ├── notebooks/
│   │   ├── churn_eda.ipynb
│   │   └── churn_modeling.ipynb
│   └── outputs/
│       └── charts/

### Tools & Libraries

Python 3.x

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn (Logistic Regression, train-test split, K-Folds, evaluation metrics)

Imbalanced-learn (SMOTE/ADASYN)

XGBoost

