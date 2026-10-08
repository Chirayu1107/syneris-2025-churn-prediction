import os
import joblib
import pandas as pd

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
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/processed/"
    "combined_purchase_cart_search_features.parquet"
)

MODEL_DIR = "models"
RESULTS_DIR = "data/processed"

RANDOM_STATE = 42
TEST_SIZE = 0.20

DEFAULT_THRESHOLD = 0.50
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
# HELPER FUNCTION
# ============================================================

def calculate_metrics(y_true, probabilities, threshold=0.50):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities
        )
    }


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)
print("Number of features:", len(FEATURES))
print("Cutoff date:", CUTOFF_DATE)
print("Churn window:", CHURN_WINDOW_DAYS, "days")


# ============================================================
# VALIDATE DATA
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING DATA")
print("=" * 80)

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print("Missing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "Required features are missing."
    )


if TARGET not in df.columns:

    raise ValueError(
        "Target column 'churn' not found."
    )


print("All features found.")
print("Target column found.")


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

print("\nChurn distribution:")
print(y.value_counts())

print("\nChurn rate:")
print(round(y.mean(), 6))


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

print(
    "Training churn rate:",
    round(y_train.mean(), 6)
)

print(
    "Testing churn rate :",
    round(y_test.mean(), 6)
)


# ============================================================
# IMPUTER
# ============================================================

print("\n" + "=" * 80)
print("FITTING IMPUTER")
print("=" * 80)

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

print(
    "Original features:",
    X_train.shape[1]
)

print(
    "Features after imputation:",
    X_train_imp.shape[1]
)


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
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

logistic_probabilities = (
    logistic_model.predict_proba(X_test)[:, 1]
)

logistic_metrics = calculate_metrics(
    y_test,
    logistic_probabilities,
    DEFAULT_THRESHOLD
)

print("Logistic Regression completed.")


# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

random_forest_model = Pipeline([
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

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_probabilities = (
    random_forest_model.predict_proba(X_test)[:, 1]
)

random_forest_metrics = calculate_metrics(
    y_test,
    random_forest_probabilities,
    DEFAULT_THRESHOLD
)

print("Random Forest completed.")


# ============================================================
# MODEL 3: XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

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

xgboost_probabilities = (
    xgboost_model.predict_proba(
        X_test_imp
    )[:, 1]
)

xgboost_metrics = calculate_metrics(
    y_test,
    xgboost_probabilities,
    DEFAULT_THRESHOLD
)

print("XGBoost completed.")


# ============================================================
# XGBOOST FINAL THRESHOLD
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST THRESHOLD OPTIMIZATION")
print("=" * 80)

xgboost_final_metrics = calculate_metrics(
    y_test,
    xgboost_probabilities,
    FINAL_THRESHOLD
)

print(
    "Default threshold:",
    DEFAULT_THRESHOLD
)

print(
    "Final threshold:",
    FINAL_THRESHOLD
)


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "threshold": DEFAULT_THRESHOLD,
        **logistic_metrics
    },

    {
        "model": "Random Forest",
        "threshold": DEFAULT_THRESHOLD,
        **random_forest_metrics
    },

    {
        "model": "XGBoost",
        "threshold": DEFAULT_THRESHOLD,
        **xgboost_metrics
    },

    {
        "model": "XGBoost Final",
        "threshold": FINAL_THRESHOLD,
        **xgboost_final_metrics
    }
])


# ============================================================
# PRINT MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

comparison.to_csv(
    COMPARISON_PATH,
    index=False
)

print("\nModel comparison saved:")
print(COMPARISON_PATH)


# ============================================================
# SAVE LOGISTIC REGRESSION
# ============================================================

LOGISTIC_PATH = os.path.join(
    MODEL_DIR,
    "churn_logistic_regression.joblib"
)

joblib.dump(
    logistic_model,
    LOGISTIC_PATH
)

print("\nSaved:")
print(LOGISTIC_PATH)


# ============================================================
# SAVE RANDOM FOREST
# ============================================================

RANDOM_FOREST_PATH = os.path.join(
    MODEL_DIR,
    "churn_random_forest.joblib"
)

joblib.dump(
    random_forest_model,
    RANDOM_FOREST_PATH
)

print("Saved:")
print(RANDOM_FOREST_PATH)


# ============================================================
# SAVE XGBOOST BASE MODEL
# ============================================================

XGBOOST_PATH = os.path.join(
    MODEL_DIR,
    "churn_xgboost.joblib"
)

joblib.dump(
    xgboost_model,
    XGBOOST_PATH
)

print("Saved:")
print(XGBOOST_PATH)


# ============================================================
# SAVE FINAL XGBOOST PACKAGE
# ============================================================

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_xgboost_final.joblib"
)

final_model_package = {
    "model": xgboost_model,
    "imputer": imputer,
    "features": FEATURES,
    "threshold": FINAL_THRESHOLD,
    "cutoff_date": CUTOFF_DATE,
    "churn_window_days": CHURN_WINDOW_DAYS,
    "random_state": RANDOM_STATE,
    "metrics": xgboost_final_metrics
}

joblib.dump(
    final_model_package,
    FINAL_MODEL_PATH
)

print("Saved:")
print(FINAL_MODEL_PATH)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "churn_features.txt"
)

with open(
    FEATURE_PATH,
    "w",
    encoding="utf-8"
) as f:

    for feature in FEATURES:
        f.write(feature + "\n")

print("Saved:")
print(FEATURE_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL TRAINING SUMMARY")
print("=" * 80)

print("Models trained:")
print("1. Logistic Regression")
print("2. Random Forest")
print("3. XGBoost")

print("\nSelected final model: XGBoost")

print(
    "Final threshold:",
    FINAL_THRESHOLD
)

print(
    "Final F1:",
    round(
        xgboost_final_metrics["f1"],
        6
    )
)

print(
    "Final Recall:",
    round(
        xgboost_final_metrics["recall"],
        6
    )
)

print(
    "Final Precision:",
    round(
        xgboost_final_metrics["precision"],
        6
    )
)

print(
    "Final ROC-AUC:",
    round(
        xgboost_final_metrics["roc_auc"],
        6
    )
)

print(
    "Final PR-AUC:",
    round(
        xgboost_final_metrics["pr_auc"],
        6
    )
)

print("\nAll models and results saved successfully.")
print("Done.")