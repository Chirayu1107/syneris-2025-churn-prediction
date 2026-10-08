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
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/combined_purchase_cart_search_features.parquet"
RESULTS_PATH = "data/processed/model_results_xgboost.csv"

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
# VALIDATE DATA
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING INPUT DATA")
print("=" * 80)

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    print("ERROR: Missing features:")
    for feature in missing_features:
        print(" -", feature)
    raise ValueError("Required features are missing.")

if "churn" not in df.columns:
    raise ValueError("ERROR: churn column not found.")

print("All required features are present.")

print("Number of features:", len(FEATURES))


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[FEATURES].copy()
y = df["churn"].copy()

print("\nTarget distribution:")
print(y.value_counts())
print("\nTarget percentage:")
print((y.value_counts(normalize=True) * 100).round(2))


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

print("\nTraining churn distribution:")
print(y_train.value_counts())

print("\nTesting churn distribution:")
print(y_test.value_counts())


# ============================================================
# XGBOOST MODEL
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
# TRAIN
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

model_pipeline.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)


print("\n" + "=" * 80)
print("XGBOOST RESULTS")
print("=" * 80)

print("Accuracy :", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall   :", round(recall, 6))
print("F1 Score :", round(f1, 6))
print("ROC-AUC  :", round(roc_auc, 6))
print("PR-AUC   :", round(pr_auc, 6))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "model": ["XGBoost"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1": [f1],
    "roc_auc": [roc_auc],
    "pr_auc": [pr_auc]
})

os.makedirs(
    os.path.dirname(RESULTS_PATH),
    exist_ok=True
)

results.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print("Saved to:", RESULTS_PATH)