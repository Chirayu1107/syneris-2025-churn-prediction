import pandas as pd
import numpy as np

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
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/combined_purchase_cart_search_features.parquet"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Final threshold selected from previous analysis
THRESHOLD = 0.40

CUTOFF_DATE = pd.Timestamp("2022-10-09 00:09:15")


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
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)
print("Cutoff date:", CUTOFF_DATE)
print("Target column:", TARGET)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING DATA")
print("=" * 80)

missing_features = [f for f in FEATURES if f not in df.columns]

if missing_features:
    raise ValueError(
        "Missing features:\n" + "\n".join(missing_features)
    )

if TARGET not in df.columns:
    raise ValueError("Churn target column not found.")

print("All expected features are present.")
print("Number of features:", len(FEATURES))

print("\nChurn distribution:")
print(df[TARGET].value_counts())

print("\nChurn percentage:")
print(df[TARGET].value_counts(normalize=True) * 100)


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].astype(int)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
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

print("\nTraining churn rate:")
print(y_train.mean())

print("\nTesting churn rate:")
print(y_test.mean())


# ============================================================
# IMPUTATION
# ============================================================

print("\n" + "=" * 80)
print("IMPUTING MISSING VALUES")
print("=" * 80)

imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

print("Original feature count:", X_train.shape[1])
print("Feature count after imputation:", X_train_imp.shape[1])


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

model = XGBClassifier(
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

model.fit(
    X_train_imp,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

y_probability = model.predict_proba(X_test_imp)[:, 1]

# Use previously selected threshold
y_pred = (y_probability >= THRESHOLD).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)
pr_auc = average_precision_score(y_test, y_probability)


print("\nThreshold:", THRESHOLD)

print("\nAccuracy :", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall   :", round(recall, 6))
print("F1 Score :", round(f1, 6))
print("ROC-AUC  :", round(roc_auc, 6))
print("PR-AUC   :", round(pr_auc, 6))


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Retained", "Churned"],
    zero_division=0
))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

importance = model.feature_importances_

# Get the feature names generated by SimpleImputer
# This includes the original features + missing-value indicators.
feature_names = list(FEATURES)

missing_count = X_train.isna().sum()

for feature in FEATURES:
    if missing_count[feature] > 0:
        feature_names.append(feature + "_missing")

# Safety check
if len(feature_names) != len(importance):
    print("ERROR: Feature name count does not match importance count.")
    print("Feature names:", len(feature_names))
    print("Importances  :", len(importance))
    raise ValueError("Feature importance length mismatch.")

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# TOP 20 FEATURES
# ============================================================

print("\nTop 20 features:\n")

print(
    feature_importance.head(20).to_string(index=False)
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

IMPORTANCE_PATH = "data/processed/xgboost_feature_importance.csv"

feature_importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)

print("\nFeature importance saved to:")
print(IMPORTANCE_PATH)


# ============================================================
# TOP 10 FEATURES
# ============================================================

print("\n" + "=" * 80)
print("TOP 10 CHURN FEATURES")
print("=" * 80)

for i, row in feature_importance.head(10).iterrows():

    print(
        str(i + 1)
        + ". "
        + str(row["feature"])
        + " : "
        + str(round(row["importance"], 6))
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL SUMMARY")
print("=" * 80)

print("Model       : XGBoost")
print("Cutoff date : " + str(CUTOFF_DATE))
print("Churn window: 60 days")
print("Threshold   : " + str(THRESHOLD))

print("\nF1 Score    :", round(f1, 4))
print("Recall      :", round(recall, 4))
print("Precision   :", round(precision, 4))
print("ROC-AUC     :", round(roc_auc, 4))
print("PR-AUC      :", round(pr_auc, 4))

print("\nDone.")