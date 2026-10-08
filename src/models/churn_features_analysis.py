import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = (
    "data/processed/"
    "combined_purchase_cart_search_features.parquet"
)

OUTPUT_DIR = Path("data/processed/model_analysis")

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

df = df.drop(
    columns=["client_id"]
)


# ============================================================
# 3. FEATURES
# ============================================================

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


print(
    f"\nNumber of features: {len(FEATURES)}"
)


# ============================================================
# 5. CREATE X AND y
# ============================================================

X = df[FEATURES].copy()

y = df["churn"].copy()


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
    f"Training samples: {len(X_train):,}"
)

print(
    f"Testing samples : {len(X_test):,}"
)


# ============================================================
# 7. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


# ============================================================
# 9. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION FEATURE ANALYSIS")
print("=" * 80)


# ------------------------------------------------------------
# Get fitted imputer
# ------------------------------------------------------------

imputer = logistic_model.named_steps["imputer"]

model = logistic_model.named_steps["model"]


# ------------------------------------------------------------
# Get feature names after imputation
# ------------------------------------------------------------

feature_names_after_imputation = (
    imputer.get_feature_names_out(FEATURES)
)


# ------------------------------------------------------------
# Get coefficients
# ------------------------------------------------------------

coefficients = model.coef_[0]


logistic_importance = pd.DataFrame({

    "feature": feature_names_after_imputation,

    "coefficient": coefficients,

    "absolute_coefficient": np.abs(
        coefficients
    )

})


# Sort by absolute coefficient

logistic_importance = (
    logistic_importance
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
)


print("\nTop 15 Logistic Regression features:")

print(
    logistic_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 10. SAVE LOGISTIC RESULTS
# ============================================================

logistic_output = (
    OUTPUT_DIR /
    "logistic_feature_importance.csv"
)


logistic_importance.to_csv(
    logistic_output,
    index=False
)


print(
    f"\n Saved:\n{logistic_output}"
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)


rf_imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)


X_train_rf = rf_imputer.fit_transform(
    X_train
)


X_test_rf = rf_imputer.transform(
    X_test
)


rf_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    min_samples_leaf=5,

    class_weight="balanced",

    n_jobs=-1,

    random_state=RANDOM_STATE
)


rf_model.fit(
    X_train_rf,
    y_train
)


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE ANALYSIS")
print("=" * 80)


rf_feature_names = (
    rf_imputer
    .get_feature_names_out(FEATURES)
)


rf_importance = pd.DataFrame({

    "feature": rf_feature_names,

    "importance": rf_model.feature_importances_

})


rf_importance = (
    rf_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\nTop 15 Random Forest features:")

print(
    rf_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 13. SAVE RANDOM FOREST RESULTS
# ============================================================

rf_output = (
    OUTPUT_DIR /
    "random_forest_feature_importance.csv"
)


rf_importance.to_csv(
    rf_output,
    index=False
)


print(
    f"\n Saved:\n{rf_output}"
)


# ============================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE CHART
# ============================================================

print("\nCreating Random Forest chart...")


top_rf = (
    rf_importance
    .head(15)
    .sort_values(
        "importance"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_rf["feature"],
    top_rf["importance"]
)


plt.xlabel(
    "Feature Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Top 15 Random Forest Features"
)


plt.tight_layout()


rf_chart = (
    OUTPUT_DIR /
    "random_forest_feature_importance.png"
)


plt.savefig(
    rf_chart,
    dpi=300
)


plt.close()


print(
    f" Chart saved:\n{rf_chart}"
)


# ============================================================
# 15. LOGISTIC REGRESSION COEFFICIENT CHART
# ============================================================

print("\nCreating Logistic Regression chart...")


top_logistic = (
    logistic_importance
    .head(15)
    .sort_values(
        "coefficient"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_logistic["feature"],
    top_logistic["coefficient"]
)


plt.axvline(
    0
)


plt.xlabel(
    "Coefficient"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Top 15 Logistic Regression Features"
)


plt.tight_layout()


logistic_chart = (
    OUTPUT_DIR /
    "logistic_regression_coefficients.png"
)


plt.savefig(
    logistic_chart,
    dpi=300
)


plt.close()


print(
    f" Chart saved:\n{logistic_chart}"
)


# ============================================================
# 16. FEATURE RANKING COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("FEATURE RANKING COMPARISON")
print("=" * 80)


# Random Forest ranking

rf_rank = rf_importance.copy()

rf_rank["rf_rank"] = (
    rf_rank["importance"]
    .rank(
        ascending=False,
        method="min"
    )
)


# Logistic ranking

log_rank = logistic_importance.copy()

log_rank["logistic_rank"] = (
    log_rank["absolute_coefficient"]
    .rank(
        ascending=False,
        method="min"
    )
)


# Merge

comparison = pd.merge(

    rf_rank[
        [
            "feature",
            "importance",
            "rf_rank"
        ]
    ],

    log_rank[
        [
            "feature",
            "absolute_coefficient",
            "logistic_rank"
        ]
    ],

    on="feature",

    how="outer"
)


comparison = comparison.sort_values(
    "rf_rank"
)


print(
    comparison
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 17. SAVE COMPARISON
# ============================================================

comparison_output = (
    OUTPUT_DIR /
    "feature_importance_comparison.csv"
)


comparison.to_csv(
    comparison_output,
    index=False
)


print(
    f"\n Comparison saved:\n"
    f"{comparison_output}"
)


# ============================================================
# 18. SIMPLE FEATURE GROUP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE GROUP ANALYSIS")
print("=" * 80)


def get_feature_group(feature):

    if (
        feature.startswith("purchase")
        or feature.startswith("purchases")
        or feature in [
            "distinct_purchase_days",
            "unique_product_purchased",
            "days_since_last_purchase",
            "avg_days_between_purchases",
            "purchase_trend_ratio",
            "purchase_per_active_day",
            "purchase_per_lifetime_day"
        ]
    ):
        return "Purchase"

    elif (
        feature.startswith("add")
        or feature.startswith("remove")
        or feature in [
            "distinct_add_to_cart_days",
            "days_since_last_add_to_cart",
            "avg_days_between_adds",
            "remove_to_add_ratio",
            "net_cart_activity",
            "has_add_activity",
            "has_remove_activity"
        ]
    ):
        return "Cart"

    elif (
        feature.startswith("search")
        or feature.startswith("searches")
        or feature in [
            "distinct_search_days",
            "days_since_last_search",
            "search_trend_ratio",
            "has_search_activity"
        ]
    ):
        return "Search"

    else:
        return "Other"


comparison["feature_group"] = (
    comparison["feature"]
    .apply(get_feature_group)
)


group_importance = (
    comparison
    .groupby("feature_group")["importance"]
    .sum()
    .sort_values(
        ascending=False
    )
)


print(
    "\nRandom Forest importance by feature group:"
)

print(
    group_importance
)


# ============================================================
# 19. SAVE GROUP ANALYSIS
# ============================================================

group_output = (
    OUTPUT_DIR /
    "feature_group_importance.csv"
)


group_importance.to_csv(
    group_output
)


print(
    f"\n Group analysis saved:\n"
    f"{group_output}"
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nAll outputs saved to:\n"
    f"{OUTPUT_DIR}"
)

print("\nGenerated files:")

print(
    "1. logistic_feature_importance.csv"
)

print(
    "2. random_forest_feature_importance.csv"
)

print(
    "3. feature_importance_comparison.csv"
)

print(
    "4. feature_group_importance.csv"
)

print(
    "5. random_forest_feature_importance.png"
)

print(
    "6. logistic_regression_coefficients.png"
)

print("\n" + "=" * 80)