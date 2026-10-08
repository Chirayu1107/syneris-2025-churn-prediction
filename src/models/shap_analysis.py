import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/combined_purchase_cart_search_features.parquet"

RANDOM_STATE = 42
TEST_SIZE = 0.20

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


# ============================================================
# PREPARE DATA
# ============================================================

print("\n" + "=" * 80)
print("PREPARING DATA")
print("=" * 80)

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

print("Number of features:", len(FEATURES))
print("Number of samples:", len(X))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

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

print("\n" + "=" * 80)
print("IMPUTING MISSING VALUES")
print("=" * 80)

imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)


# ============================================================
# CREATE FEATURE NAMES
# ============================================================

feature_names = list(FEATURES)

missing_features = []

for feature in FEATURES:

    if X_train[feature].isna().sum() > 0:

        missing_features.append(feature)

        feature_names.append(
            feature + "_missing"
        )

print("Original features:", len(FEATURES))
print("Features after imputation:", len(feature_names))

print("\nMissing-value indicators added:")

for feature in missing_features:
    print(" -", feature)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

X_train_imp_df = pd.DataFrame(
    X_train_imp,
    columns=feature_names,
    index=X_train.index
)

X_test_imp_df = pd.DataFrame(
    X_test_imp,
    columns=feature_names,
    index=X_test.index
)


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
    X_train_imp_df,
    y_train
)

print("XGBoost training completed.")


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 80)
print("CREATING SHAP EXPLAINER")
print("=" * 80)

explainer = shap.TreeExplainer(model)

# Use test data for explanation
# Limit to 10,000 samples to keep SHAP fast
SHAP_SAMPLE_SIZE = min(10000, len(X_test_imp_df))

X_shap = X_test_imp_df.sample(
    n=SHAP_SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

print("SHAP samples:", len(X_shap))

shap_values = explainer.shap_values(X_shap)

print("SHAP calculation completed.")


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL SHAP IMPORTANCE")
print("=" * 80)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
).reset_index(drop=True)


# ============================================================
# DISPLAY TOP 20
# ============================================================

print("\nTop 20 features by SHAP importance:\n")

print(
    shap_importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE SHAP IMPORTANCE
# ============================================================

SHAP_IMPORTANCE_PATH = (
    "data/processed/shap_feature_importance.csv"
)

shap_importance.to_csv(
    SHAP_IMPORTANCE_PATH,
    index=False
)

print("\nSHAP importance saved to:")
print(SHAP_IMPORTANCE_PATH)


# ============================================================
# SHAP SUMMARY BAR PLOT
# ============================================================

print("\n" + "=" * 80)
print("GENERATING SHAP BAR PLOT")
print("=" * 80)

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.title("Global SHAP Feature Importance")

plt.tight_layout()

BAR_PLOT_PATH = (
    "data/processed/shap_summary_bar.png"
)

plt.savefig(
    BAR_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(BAR_PLOT_PATH)


# ============================================================
# SHAP BEESWARM PLOT
# ============================================================

print("\n" + "=" * 80)
print("GENERATING SHAP SUMMARY PLOT")
print("=" * 80)

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    max_display=20,
    show=False
)

plt.title("SHAP Summary - Churn Prediction")

plt.tight_layout()

SUMMARY_PLOT_PATH = (
    "data/processed/shap_summary.png"
)

plt.savefig(
    SUMMARY_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(SUMMARY_PLOT_PATH)


# ============================================================
# TOP 10
# ============================================================

print("\n" + "=" * 80)
print("TOP 10 CHURN DRIVERS")
print("=" * 80)

for i, row in shap_importance.head(10).iterrows():

    print(
        str(i + 1)
        + ". "
        + str(row["feature"])
        + " : "
        + str(round(row["mean_abs_shap"], 6))
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SHAP ANALYSIS COMPLETED")
print("=" * 80)

print("Model: XGBoost")
print("Cutoff date:", CUTOFF_DATE)
print("SHAP samples:", SHAP_SAMPLE_SIZE)

print("\nGenerated files:")

print(SHAP_IMPORTANCE_PATH)
print(BAR_PLOT_PATH)
print(SUMMARY_PLOT_PATH)

print("\nDone.")