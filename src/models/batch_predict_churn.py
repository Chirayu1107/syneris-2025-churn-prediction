import os
import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/processed/"
    "combined_purchase_cart_search_features.parquet"
)

MODEL_PATH = (
    "models/"
    "churn_xgboost_final.joblib"
)

OUTPUT_PATH = (
    "data/processed/"
    "customer_churn_predictions.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING CUSTOMER DATA")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING FINAL MODEL")
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
print("Features:", len(features))
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
# VALIDATE DATA
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING INPUT DATA")
print("=" * 80)

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("Missing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "Required features are missing."
    )


if "client_id" not in df.columns:

    raise ValueError(
        "client_id column not found."
    )


print("All required features found.")


# ============================================================
# CHECK CLIENT IDs
# ============================================================

duplicate_clients = df["client_id"].duplicated().sum()

print(
    "Duplicate client IDs:",
    duplicate_clients
)

if duplicate_clients > 0:

    raise ValueError(
        "Duplicate client IDs found."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\n" + "=" * 80)
print("PREPARING FEATURES")
print("=" * 80)

X = df[features].copy()

print("Customers:", len(X))
print("Features:", len(X.columns))


# ============================================================
# APPLY SAVED IMPUTER
# ============================================================

print("\n" + "=" * 80)
print("APPLYING SAVED IMPUTER")
print("=" * 80)

X_imputed = imputer.transform(
    X
)

print(
    "Input feature count:",
    len(features)
)

print(
    "Model feature count:",
    X_imputed.shape[1]
)


# ============================================================
# GENERATE CHURN PROBABILITIES
# ============================================================

print("\n" + "=" * 80)
print("GENERATING CHURN PREDICTIONS")
print("=" * 80)

churn_probability = model.predict_proba(
    X_imputed
)[:, 1]

print("Predictions generated.")


# ============================================================
# APPLY FINAL THRESHOLD
# ============================================================

churn_prediction = (
    churn_probability >= threshold
).astype(int)


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

predictions = pd.DataFrame({

    "client_id":
        df["client_id"].values,

    "churn_probability":
        churn_probability,

    "churn_prediction":
        churn_prediction
})


# ============================================================
# ADD RISK CATEGORY
# ============================================================
#
# These categories are operational labels, not model-learned
# probabilities.
#
# HIGH   : probability >= 0.70
# MEDIUM : probability >= 0.38 and < 0.70
# LOW    : probability < 0.38
# ============================================================

predictions["risk_level"] = "LOW"

predictions.loc[
    predictions["churn_probability"] >= 0.38,
    "risk_level"
] = "MEDIUM"

predictions.loc[
    predictions["churn_probability"] >= 0.70,
    "risk_level"
] = "HIGH"


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("SAVING PREDICTIONS")
print("=" * 80)

predictions.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    "Saved:",
    OUTPUT_PATH
)


# ============================================================
# PREDICTION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION SUMMARY")
print("=" * 80)

total_customers = len(predictions)

predicted_churn = (
    predictions["churn_prediction"] == 1
).sum()

predicted_retained = (
    predictions["churn_prediction"] == 0
).sum()

high_risk = (
    predictions["risk_level"] == "HIGH"
).sum()

medium_risk = (
    predictions["risk_level"] == "MEDIUM"
).sum()

low_risk = (
    predictions["risk_level"] == "LOW"
).sum()


print(
    "Total customers:",
    total_customers
)

print(
    "Predicted churn:",
    predicted_churn
)

print(
    "Predicted retained:",
    predicted_retained
)

print(
    "Predicted churn rate:",
    round(
        predicted_churn / total_customers * 100,
        2
    )
)

print("\nRisk distribution:")

print(
    "HIGH   :",
    high_risk
)

print(
    "MEDIUM :",
    medium_risk
)

print(
    "LOW    :",
    low_risk
)


# ============================================================
# PROBABILITY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("CHURN PROBABILITY STATISTICS")
print("=" * 80)

print(
    "Mean:",
    round(
        predictions["churn_probability"].mean(),
        4
    )
)

print(
    "Median:",
    round(
        predictions["churn_probability"].median(),
        4
    )
)

print(
    "Minimum:",
    round(
        predictions["churn_probability"].min(),
        4
    )
)

print(
    "Maximum:",
    round(
        predictions["churn_probability"].max(),
        4
    )
)


# ============================================================
# TOP 20 HIGHEST-RISK CUSTOMERS
# ============================================================

print("\n" + "=" * 80)
print("TOP 20 HIGHEST-RISK CUSTOMERS")
print("=" * 80)

top_customers = (
    predictions
    .sort_values(
        "churn_probability",
        ascending=False
    )
    .head(20)
)

print(
    top_customers.to_string(
        index=False
    )
)


# ============================================================
# SAVE HIGH-RISK CUSTOMERS
# ============================================================

HIGH_RISK_PATH = (
    "data/processed/"
    "high_risk_customers.parquet"
)

high_risk_customers = (
    predictions[
        predictions["risk_level"] == "HIGH"
    ]
    .sort_values(
        "churn_probability",
        ascending=False
    )
)

high_risk_customers.to_parquet(
    HIGH_RISK_PATH,
    index=False
)

print("\nHigh-risk customers saved:")
print(HIGH_RISK_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BATCH PREDICTION COMPLETED")
print("=" * 80)

print(
    "Final model: XGBoost"
)

print(
    "Threshold:",
    threshold
)

print(
    "Customers processed:",
    total_customers
)

print(
    "Predicted churn:",
    predicted_churn
)

print(
    "Predicted retained:",
    predicted_retained
)

print(
    "Output:",
    OUTPUT_PATH
)

print("\nDone.")