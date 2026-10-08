import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/processed/"
    "combined_purchase_cart_search_features.parquet"
)

MODEL_COMPARISON_PATH = (
    "data/processed/model_comparison.csv"
)

SHAP_IMPORTANCE_PATH = (
    "data/processed/shap_feature_importance.csv"
)

OUTPUT_DIR = "data/processed/evaluation"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FINAL_THRESHOLD = 0.40

CUTOFF_DATE = "2022-10-09 00:09:15"
CHURN_WINDOW_DAYS = 60


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    # Purchase features
    "purchase_count",
    "distinct_purchase_days",
    "unique_product_purchased",
    "days_since_last_purchase",
    "avg_days_between_purchases",
    "purchases_last_30d",
    "purchases_previous_30d",
    "purchase_trend_ratio",
    "purchase_per_active_day",
    "purchase_per_lifetime_day",

    # Cart features
    "add_to_cart_count",
    "distinct_add_to_cart_days",
    "days_since_last_add_to_cart",
    "avg_days_between_adds",
    "adds_last_30d",
    "remove_from_cart_count",
    "days_since_last_remove",
    "remove_to_add_ratio",
    "net_cart_activity",
    "has_add_activity",
    "has_remove_activity",

    # Search features
    "search_count",
    "distinct_search_days",
    "days_since_last_search",
    "searches_last_30d",
    "searches_previous_30d",
    "search_trend_ratio",
    "searches_per_lifetime_day",
    "has_search_activity",
]

TARGET = "churn"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

print("Dataset shape:", df.shape)
print("Features:", len(FEATURES))
print("Cutoff date:", CUTOFF_DATE)
print("Churn window:", CHURN_WINDOW_DAYS, "days")


# ============================================================
# SAME TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("CREATING TEST SET")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)

X_train_imp = imputer.fit_transform(
    X_train
)

X_test_imp = imputer.transform(
    X_test
)


# ============================================================
# TRAIN BASELINE MODELS
# ============================================================

print("\n" + "=" * 80)
print("TRAINING MODELS FOR EVALUATION")
print("=" * 80)


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_imp,
    y_train
)

logistic_prob = (
    logistic_model.predict_proba(
        X_test_imp
    )[:, 1]
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

random_forest_model.fit(
    X_train_imp,
    y_train
)

random_forest_prob = (
    random_forest_model.predict_proba(
        X_test_imp
    )[:, 1]
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgboost_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgboost_model.fit(
    X_train_imp,
    y_train
)

xgboost_prob = (
    xgboost_model.predict_proba(
        X_test_imp
    )[:, 1]
)

print("All models trained.")


# ============================================================
# FINAL XGBOOST PREDICTIONS
# ============================================================

xgboost_pred = (
    xgboost_prob >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    xgboost_pred
)

precision = precision_score(
    y_test,
    xgboost_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    xgboost_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    xgboost_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    xgboost_prob
)

pr_auc = average_precision_score(
    y_test,
    xgboost_prob
)


# ============================================================
# PRINT FINAL METRICS
# ============================================================

print("\n" + "=" * 80)
print("FINAL XGBOOST RESULTS")
print("=" * 80)

print("Threshold :", FINAL_THRESHOLD)
print("Accuracy  :", round(accuracy, 6))
print("Precision :", round(precision, 6))
print("Recall    :", round(recall, 6))
print("F1 Score  :", round(f1, 6))
print("ROC-AUC   :", round(roc_auc, 6))
print("PR-AUC    :", round(pr_auc, 6))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    xgboost_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "XGBoost Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["Retained", "Churned"]
)

plt.yticks(
    [0, 1],
    ["Retained", "Churned"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    random_forest_prob
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgboost_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label="Logistic Regression"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label="Random Forest"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label="XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PRECISION RECALL CURVE
# ============================================================

logistic_precision, logistic_recall, _ = (
    precision_recall_curve(
        y_test,
        logistic_prob
    )
)

rf_precision, rf_recall, _ = (
    precision_recall_curve(
        y_test,
        random_forest_prob
    )
)

xgb_precision, xgb_recall, _ = (
    precision_recall_curve(
        y_test,
        xgboost_prob
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_recall,
    logistic_precision,
    label="Logistic Regression"
)

plt.plot(
    rf_recall,
    rf_precision,
    label="Random Forest"
)

plt.plot(
    xgb_recall,
    xgb_precision,
    label="XGBoost"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "precision_recall_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

threshold_results = []

for threshold in thresholds:

    predictions = (
        xgboost_prob >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_accuracy = accuracy_score(
        y_test,
        predictions
    )

    threshold_results.append({
        "threshold": threshold,
        "accuracy": threshold_accuracy,
        "precision": threshold_precision,
        "recall": threshold_recall,
        "f1": threshold_f1
    })


threshold_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

print(
    "Best F1 threshold:",
    round(best_row["threshold"], 2)
)

print(
    "Best F1:",
    round(best_row["f1"], 6)
)

print(
    "Precision:",
    round(best_row["precision"], 6)
)

print(
    "Recall:",
    round(best_row["recall"], 6)
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "threshold_analysis.csv"
    ),
    index=False
)


# ============================================================
# THRESHOLD F1 PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    label="F1"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    label="Precision"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    label="Recall"
)

plt.axvline(
    FINAL_THRESHOLD,
    linestyle="--",
    label="Final threshold = 0.40"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "XGBoost Threshold Analysis"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "threshold_f1_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# MODEL COMPARISON PLOT
# ============================================================

if os.path.exists(
    MODEL_COMPARISON_PATH
):

    comparison = pd.read_csv(
        MODEL_COMPARISON_PATH
    )

    plt.figure(figsize=(10, 6))

    x = np.arange(
        len(comparison)
    )

    width = 0.25

    plt.bar(
        x - width,
        comparison["precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x,
        comparison["recall"],
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        comparison["f1"],
        width,
        label="F1"
    )

    plt.xticks(
        x,
        comparison["model"],
        rotation=20
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Model Performance Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# SHAP IMPORTANCE PLOT
# ============================================================

if os.path.exists(
    SHAP_IMPORTANCE_PATH
):

    shap_df = pd.read_csv(
        SHAP_IMPORTANCE_PATH
    )

    top_shap = (
        shap_df
        .head(15)
        .sort_values(
            "mean_abs_shap"
        )
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        top_shap["feature"],
        top_shap["mean_abs_shap"]
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 15 Churn Drivers - SHAP"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "shap_importance.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL TEXT REPORT
# ============================================================

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "final_evaluation_report.txt"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("CUSTOMER CHURN PREDICTION - FINAL EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 80 + "\n")
    f.write(
        "Dataset rows: "
        + str(len(df))
        + "\n"
    )

    f.write(
        "Features: "
        + str(len(FEATURES))
        + "\n"
    )

    f.write(
        "Observation cutoff: "
        + CUTOFF_DATE
        + "\n"
    )

    f.write(
        "Churn window: "
        + str(CHURN_WINDOW_DAYS)
        + " days\n"
    )

    f.write(
        "Churn rate: "
        + str(round(y.mean(), 6))
        + "\n\n"
    )

    f.write("FINAL MODEL\n")
    f.write("-" * 80 + "\n")
    f.write("Model: XGBoost\n")
    f.write(
        "Threshold: "
        + str(FINAL_THRESHOLD)
        + "\n\n"
    )

    f.write("METRICS\n")
    f.write("-" * 80 + "\n")

    f.write(
        "Accuracy: "
        + str(round(accuracy, 6))
        + "\n"
    )

    f.write(
        "Precision: "
        + str(round(precision, 6))
        + "\n"
    )

    f.write(
        "Recall: "
        + str(round(recall, 6))
        + "\n"
    )

    f.write(
        "F1 Score: "
        + str(round(f1, 6))
        + "\n"
    )

    f.write(
        "ROC-AUC: "
        + str(round(roc_auc, 6))
        + "\n"
    )

    f.write(
        "PR-AUC: "
        + str(round(pr_auc, 6))
        + "\n\n"
    )

    f.write("CONFUSION MATRIX\n")
    f.write("-" * 80 + "\n")

    f.write(
        "True Negatives: "
        + str(tn)
        + "\n"
    )

    f.write(
        "False Positives: "
        + str(fp)
        + "\n"
    )

    f.write(
        "False Negatives: "
        + str(fn)
        + "\n"
    )

    f.write(
        "True Positives: "
        + str(tp)
        + "\n\n"
    )

    f.write("INTERPRETATION\n")
    f.write("-" * 80 + "\n")

    f.write(
        "The final XGBoost model was selected after "
        "comparison with Logistic Regression and Random Forest.\n\n"
    )

    f.write(
        "The classification threshold was reduced from "
        "0.50 to 0.40 to improve churn recall and F1 score.\n\n"
    )

    f.write(
        "The final model achieved a recall of "
        + str(round(recall * 100, 2))
        + "%, meaning it identified approximately "
        + str(round(recall * 100, 2))
        + " out of every 100 actual churned customers "
        "in the test set.\n\n"
    )

    f.write(
        "The main behavioral signals identified by SHAP "
        "include purchase frequency, search recency, "
        "cart recency, product diversity, and recent "
        "customer activity.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)

print("Output directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print("confusion_matrix.png")
print("roc_curve.png")
print("precision_recall_curve.png")
print("model_comparison.png")
print("threshold_f1_curve.png")
print("shap_importance.png")
print("threshold_analysis.csv")
print("final_evaluation_report.txt")

print("\nDone.")