# Credit Card Fraud Detection with XGBoost

Machine learning pipeline for detecting fraudulent credit card transactions using the IEEE-CIS Fraud Detection dataset.  
This project focuses on handling highly imbalanced financial transaction data using feature preprocessing, gradient boosting models, threshold tuning, and interpretability techniques.

---

## Project Overview

Financial institutions lose billions of dollars each year to credit card fraud. Fraud detection systems must balance two competing goals:

- Detect as many fraudulent transactions as possible
- Minimize false positives that incorrectly block legitimate purchases

This project builds a machine learning pipeline to predict fraudulent transactions and evaluate how well gradient boosting models perform in this environment.

The project includes data preprocessing, model training, threshold optimization, and performance visualizations.

---

## Dataset

Dataset used:

**IEEE-CIS Fraud Detection Dataset (Kaggle)**  
https://www.kaggle.com/competitions/ieee-fraud-detection/data

The dataset contains two main tables:

| File | Description |
|-----|-------------|
| `train_transaction.csv` | Transaction level data |
| `train_identity.csv` | Identity and device information |

Key characteristics of the dataset:

- High dimensionality (hundreds of features)
- Large number of missing values
- Highly imbalanced target variable
- Synthetic but designed to mimic real financial systems

Target variable: 0 legitimate, 1 fraud



---

## Project Pipeline

The workflow follows a standard machine learning pipeline:

### 1. Data Loading
Transaction and identity tables are merged on `TransactionID`.

### 2. Data Preprocessing
- Missing value imputation
- Numeric scaling
- One-hot encoding of categorical variables
- Feature engineering through preprocessing pipelines

### 3. Train/Test Split
A **stratified split** preserves the fraud ratio in both training and test datasets.

### 4. Model Training
The model used in this project:

**XGBoost Classifier**

Why XGBoost?

- Strong performance on tabular datasets
- Handles missing values well
- Captures nonlinear feature interactions
- Performs well in imbalanced classification tasks

### 5. Imbalance Handling

Fraud datasets are extremely imbalanced. The model handles this using: scale_pos_weight




which adjusts the model’s sensitivity to fraudulent cases.

### 6. Threshold Optimization

Instead of using the default classification threshold (0.50), the project tunes the threshold to achieve higher recall while controlling precision.

This reflects real-world fraud detection priorities where missing fraud is often more costly than reviewing legitimate transactions.

---

## Model Evaluation

Model performance is evaluated using metrics appropriate for imbalanced classification:

- Precision
- Recall
- F1 Score
- ROC AUC
- Precision-Recall AUC
- Confusion Matrix

Precision-Recall curves are particularly important because fraud events are rare.

---

## Visualizations

The code produces several visualizations to analyze the dataset and model performance.

### Class Distribution
Shows the imbalance between legitimate and fraudulent transactions.

### Fraud Rate
Displays the percentage of fraud cases in the dataset.

### Missing Data Analysis
Top features with the highest percentage of missing values.

### Precision-Recall Curve
Evaluates model performance in the imbalanced classification setting.

### ROC Curve
Shows tradeoff between true positive and false positive rates.

### Confusion Matrix
Displays classification results after threshold tuning.

### Feature Contribution Visualization
Interprets which variables most strongly influence fraud predictions.


