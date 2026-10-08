from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/processed")

df = pd.read_parquet(
    DATA_DIR / "purchase_features.parquet"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in ["client_id", "churn"]
]


# ============================================================
# GROUP STATISTICS
# ============================================================

print("=" * 80)
print("PURCHASE FEATURE ANALYSIS")
print("=" * 80)


summary = (
    df
    .groupby("churn")[feature_columns]
    .median()
    .T
)


summary.columns = [
    "Retained" if column == 0 else "Churned"
    for column in summary.columns
]


print("\nMedian feature values:")
print(summary)


# ============================================================
# MEAN VALUES
# ============================================================

mean_summary = (
    df
    .groupby("churn")[feature_columns]
    .mean()
    .T
)


mean_summary.columns = [
    "Retained" if column == 0 else "Churned"
    for column in mean_summary.columns
]


print("\nMean feature values:")
print(mean_summary)


# ============================================================
# MISSING VALUES BY CLASS
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES BY CLASS")
print("=" * 80)


missing_by_class = (
    df
    .groupby("churn")[feature_columns]
    .apply(lambda x: x.isna().sum())
    .T
)


missing_by_class.columns = [
    "Retained" if column == 0 else "Churned"
    for column in missing_by_class.columns
]


print(missing_by_class)


# ============================================================
# SAVE SUMMARY
# ============================================================

output = Path(
    "data/processed/purchase_feature_analysis.csv"
)

analysis_table = summary.copy()

analysis_table["Difference"] = (
    analysis_table["Churned"]
    -
    analysis_table["Retained"]
)


analysis_table.to_csv(output)

print("\nSaved:")
print(output)