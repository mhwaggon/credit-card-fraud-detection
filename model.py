import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Try to import XGBoost; raise a clear error if it's not installed
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost is not installed. Install with: pip install xgboost") from e

# Try to import SHAP for feature importance explanations
try:
    import shap
except Exception as e:
    raise ImportError("shap is not installed. Install with: pip install shap") from e

# Paths to the input CSV files
DATA_DIR = "."
TX_PATH = os.path.join(DATA_DIR, "train_transaction.csv")
ID_PATH = os.path.join(DATA_DIR, "train_identity.csv")

# Reproducibility seed and train/test split ratio
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_ieee_cis(tx_path: str, id_path: str) -> pd.DataFrame:
    # Load the transaction data
    tx = pd.read_csv(tx_path)

    # If an identity file exists, merge it onto the transactions by ID
    if os.path.exists(id_path):
        ident = pd.read_csv(id_path)
        df = tx.merge(ident, on="TransactionID", how="left")
    else:
        df = tx

    return df


def build_preprocessor(num_cols, cat_cols):
    # For numeric columns: fill missing values with the median, then scale
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    # For categorical columns: fill missing with the most common value, then one-hot encode
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    # Combine both pipelines into a single preprocessor
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return pre


def evaluate_binary(y_true, y_proba, threshold=0.5, label="model"):
    # Convert probabilities to binary predictions using the given threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Compute PR AUC and ROC AUC scores
    ap = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)

    # Break the confusion matrix into its four components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Print all metrics
    print(f"\n[{label}] threshold={threshold:.6f}")
    print(f"PR AUC (Average Precision): {ap:.6f}")
    print(f"ROC AUC: {roc:.6f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(y_true, y_pred, digits=4))

    return {"pr_auc": ap, "roc_auc": roc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def pick_threshold_for_recall(y_true, y_proba, min_recall=0.80):
    # Get precision/recall values across all thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Drop the last point
    precision = precision[:-1]
    recall = recall[:-1]

    # Find all thresholds that meet the minimum recall requirement
    ok = np.where(recall >= min_recall)[0]

    # If no threshold meets the recall floor, just pick the one with the highest recall
    if len(ok) == 0:
        best_i = int(np.argmax(recall))
        return float(thresholds[best_i]), float(precision[best_i]), float(recall[best_i])

    # Among valid thresholds, pick the one with the highest precision
    best_i = ok[int(np.argmax(precision[ok]))]
    return float(thresholds[best_i]), float(precision[best_i]), float(recall[best_i])


def plot_curves(y_true, y_proba, title_prefix=""):
    # Compute curve data
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.grid(True)
    plt.show()


# Load the data
df = load_ieee_cis(TX_PATH, ID_PATH)

# Make sure the fraud label column exists
if "isFraud" not in df.columns:
    raise ValueError("Expected target column 'isFraud' not found. Check your CSVs.")

# Split into features and target
y = df["isFraud"].astype(int)
X = df.drop(columns=["isFraud"])

# Hold out 20% for testing, then carve a validation set out of the training portion
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_full
)

# Identify numeric and categorical columns from training data only
num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

# Build the feature preprocessor
pre = build_preprocessor(num_cols, cat_cols)

# Compute class imbalance ratio to help XGBoost handle the rare fraud class
pos = int(y_train.sum())
neg = int((y_train == 0).sum())
scale_pos_weight = float(neg / max(pos, 1))

# Define the XGBoost classifier
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50,  # Stop early if val score plateaus
)

# Chain the preprocessor and classifier into a single pipeline
pipe = Pipeline(steps=[("pre", pre), ("clf", xgb)])

# Preprocess the validation set so we can pass it to XGBoost for early stopping.
pre.fit(X_train, y_train)
X_val_trans = pre.transform(X_val)

# Train the model pass the preprocessed val set for early stopping
pipe.fit(
    X_train,
    y_train,
    clf__eval_set=[(X_val_trans, y_val)],
    clf__verbose=50,  # Print eval score every 50 rounds
)

print(f"\nBest iteration: {xgb.best_iteration} (out of 600 max)")

# Get predicted fraud probabilities on the test set
proba = pipe.predict_proba(X_test)[:, 1]

# Evaluate at the default 0.5 threshold
_ = evaluate_binary(y_test.values, proba, threshold=0.5, label="xgboost_default")

# Find a better threshold that ensures at least 80% recall, then re-evaluate
thr, p_at, r_at = pick_threshold_for_recall(y_test.values, proba, min_recall=0.80)
_ = evaluate_binary(y_test.values, proba, threshold=thr, label="xgboost_tuned_for_recall>=0.80")

# Plot PR and ROC curves
plot_curves(y_test.values, proba, title_prefix="xgboost")

# Extract the fitted preprocessor to transform the test set for SHAP
pre_fitted = pipe.named_steps["pre"]
X_test_trans = pre_fitted.transform(X_test)

# Get the feature names after preprocessing (one-hot encoding expands columns)
try:
    feature_names = pre_fitted.get_feature_names_out()
except Exception:
    feature_names = np.array([f"f{i}" for i in range(X_test_trans.shape[1])], dtype=object)

# Sample up to 5000 rows to keep SHAP computation fast
n_shap = min(5000, X_test_trans.shape[0])
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(X_test_trans.shape[0], size=n_shap, replace=False)
X_shap = X_test_trans[idx]

# Compute SHAP values using the trained XGBoost model
explainer = shap.TreeExplainer(pipe.named_steps["clf"])
shap_values = explainer.shap_values(X_shap)

# Bar chart: mean absolute SHAP value per feature (overall importance)
plt.figure()
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False, plot_type="bar", max_display=25)
plt.tight_layout()
plt.show()

# Dot plot: shows direction and magnitude of each feature's impact per sample
plt.figure()
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False, max_display=25)
plt.tight_layout()
plt.show()
