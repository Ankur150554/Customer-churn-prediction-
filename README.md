**📊 Customer Churn Prediction with Machine Learning**

Author: Kristina Kunceviciute
Program: Data Part-Time, Barcelona
Date: December 2019

**📌 Project Overview**

Customer acquisition is significantly more expensive than customer retention. However, focusing on every customer individually is inefficient and costly.

This project builds a machine learning–based churn prediction system to identify customers who are likely to churn after their first month, allowing businesses to optimize retention strategies and marketing spend.

**🎯 Problem Statement**

Predict whether a customer will churn after the first month of subscription and identify the key factors influencing churn behavior.

**💡 Solution Approach**

Analyze customer subscription data

Engineer meaningful features

Train and evaluate multiple machine learning models

Identify the best-performing model using F1 score

Provide actionable business recommendations

**📁 Datasets**
File	Description
raw_encrypted.csv	Raw customer data (encrypted for confidentiality)
population_region.csv	Population statistics by region
income_region.csv	Average income by region
clean_data.csv	Cleaned dataset with engineered features
encoded_data.csv	Fully encoded dataset ready for modeling

**Dataset Summary**

Time period: 2017–2019

Total customers: 26,799

One row per customer

Entry-level subscription data

**📓 Notebook Structure**

All analysis is contained in one combined Python notebook, covering:

Data Cleaning & Validation

Exploratory Data Analysis (EDA)

Feature Engineering

Encoding Categorical Variables

Label Encoding

One-Hot Encoding

Model Training & Evaluation

Feature Importance Analysis

Hyperparameter Tuning

Dimensionality Reduction (PCA & UMAP)

Final Model Selection & Recommendations

**🤖 Machine Learning Models Used**

The following five models were implemented and evaluated:

RandomForestClassifier

XGBoost Classifier

LightGBM Classifier

Logistic Regression

Linear Support Vector Classifier (LinearSVC)

**⚙️ Model Evaluation**

Evaluation Metric: F1 Score

Cross-validation: Stratified K-Fold

Comparison between:

Default model parameters

Tuned hyperparameters

Feature importance extracted using Random Forest

**📚 Libraries Used**

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

category_encoders

imbalanced-learn

LightGBM

XGBoost

umap-learn

**📈 Key Insights**
Exploratory Data Analysis

Target variable is slightly imbalanced:

54% churned

46% retained

Higher churn observed for:

Certain acquisition channels (notably Channel F)

Customers subscribed to newsletters

Customers with more interaction touchpoints tend to churn less

No strong seasonal churn pattern

Important Features

Gender

Master channel

Newsletter subscription

Pick-up point usage

Lead source

**⚠️ Dataset shows low variability (≈95% joined via promotion), limiting churn pattern discovery.**

**🧠 Model Performance Summary**

Random Forest Classifier performed best overall

Achieved:

**F1 Score ≈ 69%**

Highest true positives

Lowest false negatives

Data balancing techniques did not improve results

Scaling observations:

Random Forest: Not required

LinearSVC: Slight improvement with StandardScaler

**🔬 Dimensionality Reduction**

PCA and UMAP reveal structural patterns

No clear separation based on churn label

Indicates churn decisions are influenced by complex or external factors

**✅ Final Conclusions**

Random Forest Classifier is the most effective model

Best performance achieved using:

Full feature set

No scaling or balancing

Churn prediction accuracy is limited by lack of behavioral data

**🚀 Recommendations**

Collect richer behavioral data

Customer service interactions

Website/app engagement

Customer feedback or surveys

Diversify marketing strategies

Adjust pricing per acquisition channel

Re-evaluate high-churn channels

Predict long-term churn

Extend prediction window to 3+ months

Use early engagement signals to improve accuracy

**📌 Key Takeaway**

Early churn prediction is feasible, but improving accuracy requires richer behavioral data and more diverse customer acquisition strategies.
