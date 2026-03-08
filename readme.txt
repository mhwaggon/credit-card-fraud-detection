# Credit Card Fraud Detection with XGBoost

## Overview

This project builds a machine learning pipeline to detect fraudulent financial transactions using the IEEE-CIS Fraud Detection dataset from Kaggle. The workflow focuses on fraud detection in a highly imbalanced setting, using preprocessing, class imbalance handling, XGBoost modeling, threshold tuning, and model interpretation visualizations.

The goal is to identify fraudulent transactions while minimizing false positives that could incorrectly block legitimate purchases.

## Dataset

This project uses the IEEE-CIS Fraud Detection dataset from Kaggle:

[IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The dataset includes:

- Transaction-level features
- Identity-related features
- High dimensionality
- Missing values
- A binary fraud target column: `isFraud`

Files used in this project:

- `train_transaction.csv`
- `train_identity.csv`

## Project Objectives

- Build a fraud detection model for highly imbalanced classification
- Compare fraud risk using predicted probabilities instead of only hard labels
- Tune the classification threshold to prioritize recall
- Visualize fraud patterns and model performance
- Interpret important drivers of fraud predictions

## Methods

The pipeline includes:

- Merging transaction and identity data
- Missing value imputation
- Numeric scaling
- One-hot encoding for categorical features
- Stratified train/test split
- XGBoost classification with `scale_pos_weight` for imbalance handling
- Threshold tuning based on recall targets
- Performance evaluation using:
  - Precision
  - Recall
  - ROC AUC
  - Precision-Recall AUC
  - Confusion Matrix

## Visualizations Produced

The code creates the following visualizations:

- Class balance plot
- Fraud rate plot
- Missingness plot for top variables
- Precision-Recall curve
- ROC curve
- Confusion matrix at a tuned classification threshold
- Feature contribution / importance style plots

## File Structure

Example project structure:

```text
project_folder/
│
├── train_transaction.csv
├── train_identity.csv
├── fraud_detection.ipynb
└── README.md
