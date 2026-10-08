import os
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/churn_xgboost_final.joblib"

OLD_THRESHOLD = 0.40
NEW_THRESHOLD = 0.38

CUTOFF_DATE = "2022-10-09 00:09:15"
CHURN_WINDOW_DAYS = 60


# ============================================================
# LOAD EXISTING MODEL PACKAGE
# ============================================================

print("=" * 80)
print("LOADING FINAL XGBOOST MODEL")
print("=" * 80)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Final model not found: " + MODEL_PATH
    )

model_package = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# CHECK CURRENT CONFIGURATION
# ============================================================

print("\n" + "=" * 80)
print("CURRENT CONFIGURATION")
print("=" * 80)

current_threshold = model_package.get(
    "threshold",
    None
)

print(
    "Current threshold:",
    current_threshold
)

print(
    "New threshold:",
    NEW_THRESHOLD
)


# ============================================================
# UPDATE THRESHOLD
# ============================================================

model_package["threshold"] = NEW_THRESHOLD

model_package["cutoff_date"] = CUTOFF_DATE

model_package["churn_window_days"] = (
    CHURN_WINDOW_DAYS
)


# ============================================================
# UPDATE METRICS
# ============================================================

model_package["metrics"] = {
    "threshold": NEW_THRESHOLD,
    "f1": 0.801341,
    "precision": 0.689456,
    "recall": 0.956572,
    "accuracy": None,
    "roc_auc": 0.712113,
    "pr_auc": 0.793771
}


# ============================================================
# SAVE UPDATED MODEL
# ============================================================

joblib.dump(
    model_package,
    MODEL_PATH
)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING UPDATED MODEL")
print("=" * 80)

updated_package = joblib.load(
    MODEL_PATH
)

print(
    "Saved threshold:",
    updated_package["threshold"]
)

print(
    "Cutoff date:",
    updated_package["cutoff_date"]
)

print(
    "Churn window:",
    updated_package["churn_window_days"],
    "days"
)

print(
    "F1:",
    updated_package["metrics"]["f1"]
)

print(
    "Precision:",
    updated_package["metrics"]["precision"]
)

print(
    "Recall:",
    updated_package["metrics"]["recall"]
)

print(
    "ROC-AUC:",
    updated_package["metrics"]["roc_auc"]
)

print(
    "PR-AUC:",
    updated_package["metrics"]["pr_auc"]
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL UPDATED")
print("=" * 80)

print("Model:", MODEL_PATH)
print("Threshold: 0.38")
print("F1: 0.801341")
print("Recall: 0.956572")
print("Precision: 0.689456")
print("ROC-AUC: 0.712113")
print("PR-AUC: 0.793771")

print("\nDone.")