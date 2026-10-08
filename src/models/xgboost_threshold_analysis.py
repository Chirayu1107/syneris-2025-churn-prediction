import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/combined_purchase_cart_search_features.parquet"

RESULTS_PATH = "data/processed/xgboost_threshold_results.csv"

RANDOM_STATE = 42


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
    "has_search_activity"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# PREPARE DATA
# ============================================================

print("\n" + "=" * 80)
print("PREPARING DATA")
print("=" * 80)

X = df[FEATURES].copy()
y = df["churn"].copy()

print("Number of features:", len(FEATURES))
print("Total rows:", len(df))


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("CREATING TRAIN TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# BUILD XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("BUILDING XGBOOST MODEL")
print("=" * 80)

xgb_model = XGBClassifier(
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


# ============================================================
# PIPELINE
# ============================================================

model_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),
    (
        "model",
        xgb_model
    )
])


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

model_pipeline.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# GET CHURN PROBABILITIES
# ============================================================

print("\n" + "=" * 80)
print("GENERATING CHURN PROBABILITIES")
print("=" * 80)

y_prob = model_pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("ROC-AUC:", round(roc_auc, 6))
print("PR-AUC :", round(pr_auc, 6))


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = [
    round(x, 2)
    for x in list(
        __import__("numpy").arange(0.10, 0.91, 0.05)
    )
]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    })


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nThreshold comparison:")

print(
    results_df[
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 80)
print("BEST THRESHOLD BY F1")
print("=" * 80)

print(
    "Threshold:",
    best_f1_row["threshold"]
)

print(
    "Accuracy :",
    round(best_f1_row["accuracy"], 6)
)

print(
    "Precision:",
    round(best_f1_row["precision"], 6)
)

print(
    "Recall   :",
    round(best_f1_row["recall"], 6)
)

print(
    "F1 Score :",
    round(best_f1_row["f1"], 6)
)


# ============================================================
# BEST THRESHOLD BY PRECISION
# ============================================================

best_precision_row = results_df.loc[
    results_df["precision"].idxmax()
]

print("\n" + "=" * 80)
print("BEST THRESHOLD BY PRECISION")
print("=" * 80)

print(
    "Threshold:",
    best_precision_row["threshold"]
)

print(
    "Precision:",
    round(best_precision_row["precision"], 6)
)

print(
    "Recall   :",
    round(best_precision_row["recall"], 6)
)

print(
    "F1 Score :",
    round(best_precision_row["f1"], 6)
)


# ============================================================
# BEST THRESHOLD BY RECALL
# ============================================================

best_recall_row = results_df.loc[
    results_df["recall"].idxmax()
]

print("\n" + "=" * 80)
print("BEST THRESHOLD BY RECALL")
print("=" * 80)

print(
    "Threshold:",
    best_recall_row["threshold"]
)

print(
    "Precision:",
    round(best_recall_row["precision"], 6)
)

print(
    "Recall   :",
    round(best_recall_row["recall"], 6)
)

print(
    "F1 Score :",
    round(best_recall_row["f1"], 6)
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(RESULTS_PATH),
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print("Saved to:", RESULTS_PATH)