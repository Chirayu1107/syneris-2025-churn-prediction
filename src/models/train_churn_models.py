import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = (
    "data/processed/"
    "combined_purchase_cart_search_features.parquet"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nChurn distribution:")
print(df["churn"].value_counts())

print(
    f"\nChurn rate: "
    f"{df['churn'].mean() * 100:.2f}%"
)


# ============================================================
# 2. REMOVE ID
# ============================================================

# client_id is an identifier, NOT a predictive feature.

df = df.drop(
    columns=["client_id"]
)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

# We intentionally remove some highly redundant features
# identified during the correlation audit.

FEATURES = [

    # --------------------------------------------------------
    # PURCHASE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CART
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

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
# 4. CHECK FEATURES
# ============================================================

missing_features = [
    col
    for col in FEATURES
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

print("\nNumber of selected features:")
print(len(FEATURES))


# ============================================================
# 5. CREATE X AND y
# ============================================================

X = df[FEATURES].copy()

y = df["churn"].copy()


print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    f"Training samples: "
    f"{len(X_train):,}"
)

print(
    f"Testing samples : "
    f"{len(X_test):,}"
)

print(
    f"\nTraining churn rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Testing churn rate : "
    f"{y_test.mean() * 100:.2f}%"
)


# ============================================================
# 7. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability of churn
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 80)

logistic_model = Pipeline([
    
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])


logistic_model.fit(
    X_train,
    y_train
)


logistic_result = evaluate_model(
    "LOGISTIC REGRESSION",
    logistic_model,
    X_test,
    y_test
)


# ============================================================
# 9. RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

# Random Forest doesn't require scaling.

rf_model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),

    (
        "model",
        RandomForestClassifier(
            n_estimators=300,

            max_depth=15,

            min_samples_leaf=5,

            class_weight="balanced",

            n_jobs=-1,

            random_state=RANDOM_STATE
        )
    )
])


rf_model.fit(
    X_train,
    y_train
)


rf_result = evaluate_model(
    "RANDOM FOREST",
    rf_model,
    X_test,
    y_test
)


# ============================================================
# 10. RESULTS
# ============================================================

results = pd.DataFrame([
    logistic_result,
    rf_result
])


print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results
    .sort_values(
        "pr_auc",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

OUTPUT_PATH = (
    "data/processed/"
    "model_results_purchase_cart_search.csv"
)

Path(
    OUTPUT_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\n Results saved to:\n"
    f"{OUTPUT_PATH}"
)


print("\n" + "=" * 80)
print("BASELINE ML PIPELINE COMPLETE")
print("=" * 80)