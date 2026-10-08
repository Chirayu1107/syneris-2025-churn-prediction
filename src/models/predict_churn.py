import os
import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/churn_xgboost_final.joblib"


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print("=" * 80)
print("LOADING FINAL CHURN MODEL")
print("=" * 80)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found: " + MODEL_PATH
    )

model_package = joblib.load(
    MODEL_PATH
)

model = model_package["model"]
imputer = model_package["imputer"]
features = model_package["features"]
threshold = model_package["threshold"]

print("Model loaded successfully.")
print("Number of features:", len(features))
print("Threshold:", threshold)
print(
    "Cutoff date:",
    model_package["cutoff_date"]
)
print(
    "Churn window:",
    model_package["churn_window_days"],
    "days"
)


# ============================================================
# CUSTOMER INPUT
# ============================================================
#
# Replace these values with a real customer's features.
#
# IMPORTANT:
# The names must exactly match the features used during
# model training.
# ============================================================

customer = {

    # Purchase features
    "purchase_count": 4,
    "distinct_purchase_days": 2,
    "unique_product_purchased": 3,
    "days_since_last_purchase": 20.0,
    "avg_days_between_purchases": 15.0,
    "purchases_last_30d": 2,
    "purchases_previous_30d": 3,
    "purchase_trend_ratio": 0.67,
    "purchase_per_active_day": 0.20,
    "purchase_per_lifetime_day": 0.05,

    # Cart features
    "add_to_cart_count": 8,
    "distinct_add_to_cart_days": 3,
    "days_since_last_add_to_cart": 15.0,
    "avg_days_between_adds": 8.0,
    "adds_last_30d": 4,
    "remove_from_cart_count": 2,
    "days_since_last_remove": 18.0,
    "remove_to_add_ratio": 0.25,
    "net_cart_activity": 6,
    "has_add_activity": 1,
    "has_remove_activity": 1,

    # Search features
    "search_count": 6,
    "distinct_search_days": 3,
    "days_since_last_search": 12.0,
    "searches_last_30d": 4,
    "searches_previous_30d": 2,
    "search_trend_ratio": 2.0,
    "searches_per_lifetime_day": 0.08,
    "has_search_activity": 1
}


# ============================================================
# CONVERT CUSTOMER TO DATAFRAME
# ============================================================

customer_df = pd.DataFrame(
    [customer],
    columns=features
)


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in customer_df.columns
]

if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "Customer input is missing required features."
    )


# ============================================================
# APPLY SAME IMPUTER
# ============================================================

customer_imputed = imputer.transform(
    customer_df
)


# ============================================================
# PREDICT CHURN PROBABILITY
# ============================================================

churn_probability = model.predict_proba(
    customer_imputed
)[0, 1]


# ============================================================
# APPLY FINAL THRESHOLD
# ============================================================

prediction = int(
    churn_probability >= threshold
)


# ============================================================
# DETERMINE RISK LEVEL
# ============================================================

if churn_probability >= 0.70:

    risk_level = "HIGH"

elif churn_probability >= 0.40:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 80)
print("CHURN PREDICTION")
print("=" * 80)

print(
    "Churn probability:",
    round(churn_probability, 4)
)

print(
    "Churn probability (%):",
    round(churn_probability * 100, 2)
)

print(
    "Decision threshold:",
    threshold
)

if prediction == 1:

    print("Prediction: CHURN")

else:

    print("Prediction: RETAINED")


print(
    "Risk level:",
    risk_level
)


# ============================================================
# DISPLAY CUSTOMER INPUT
# ============================================================

print("\n" + "=" * 80)
print("CUSTOMER FEATURES")
print("=" * 80)

for feature in features:

    value = customer_df.iloc[0][feature]

    print(
        feature + ":",
        value
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION COMPLETED")
print("=" * 80)

print(
    "Final model: XGBoost"
)

print(
    "Threshold:",
    threshold
)

print(
    "Prediction:",
    "CHURN" if prediction == 1 else "RETAINED"
)

print(
    "Risk:",
    risk_level
)

print("\nDone.")