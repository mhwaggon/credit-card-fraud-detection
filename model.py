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

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost is not installed. Install with: pip install xgboost") from e

try:
    import shap
except Exception as e:
    raise ImportError("shap is not installed. Install with: pip install shap") from e


DATA_DIR = "."
TX_PATH = os.path.join(DATA_DIR, "train_transaction.csv")
ID_PATH = os.path.join(DATA_DIR, "train_identity.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_ieee_cis(tx_path: str, id_path: str) -> pd.DataFrame:
    tx = pd.read_csv(tx_path)
    if os.path.exists(id_path):
        ident = pd.read_csv(id_path)
        df = tx.merge(ident, on="TransactionID", how="left")
    else:
        df = tx
    return df


def build_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ]
    )

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
    y_pred = (y_proba >= threshold).astype(int)

    ap = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n[{label}] threshold={threshold:.6f}")
    print(f"PR AUC (Average Precision): {ap:.6f}")
    print(f"ROC AUC: {roc:.6f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(y_true, y_pred, digits=4))
    return {"pr_auc": ap, "roc_auc": roc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def pick_threshold_for_recall(y_true, y_proba, min_recall=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision = precision[:-1]
    recall = recall[:-1]

    ok = np.where(recall >= min_recall)[0]
    if len(ok) == 0:
        best_i = int(np.argmax(recall))
        return float(thresholds[best_i]), float(precision[best_i]), float(recall[best_i])

    best_i = ok[int(np.argmax(precision[ok]))]
    return float(thresholds[best_i]), float(precision[best_i]), float(recall[best_i])


def plot_curves(y_true, y_proba, title_prefix=""):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.grid(True)
    plt.show()


df = load_ieee_cis(TX_PATH, ID_PATH)

if "isFraud" not in df.columns:
    raise ValueError("Expected target column 'isFraud' not found. Check your CSVs.")

y = df["isFraud"].astype(int)
X = df.drop(columns=["isFraud"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

pre = build_preprocessor(df, target_col="isFraud")

pos = int(y_train.sum())
neg = int((y_train == 0).sum())
scale_pos_weight = float(neg / max(pos, 1))

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
)

pipe = Pipeline(steps=[("pre", pre), ("clf", xgb)])
pipe.fit(X_train, y_train)

proba = pipe.predict_proba(X_test)[:, 1]

_ = evaluate_binary(y_test.values, proba, threshold=0.5, label="xgboost_default")

thr, p_at, r_at = pick_threshold_for_recall(y_test.values, proba, min_recall=0.80)
_ = evaluate_binary(y_test.values, proba, threshold=thr, label="xgboost_tuned_for_recall>=0.80")

plot_curves(y_test.values, proba, title_prefix="xgboost")

pre_fitted = pipe.named_steps["pre"]
X_test_trans = pre_fitted.transform(X_test)

try:
    feature_names = pre_fitted.get_feature_names_out()
except Exception:
    feature_names = np.array([f"f{i}" for i in range(X_test_trans.shape[1])], dtype=object)

n_shap = min(5000, X_test_trans.shape[0])
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(X_test_trans.shape[0], size=n_shap, replace=False)
X_shap = X_test_trans[idx]

explainer = shap.TreeExplainer(pipe.named_steps["clf"])
shap_values = explainer.shap_values(X_shap)

plt.figure()
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False, plot_type="bar", max_display=25)
plt.tight_layout()
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False, max_display=25)
plt.tight_layout()
plt.show()
