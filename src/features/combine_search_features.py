import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

COMBINED_PATH = "data/processed/combined_features.parquet"
SEARCH_PATH = "data/processed/search_features.parquet"

OUTPUT_PATH = "data/processed/combined_purchase_cart_search_features.parquet"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

combined = pd.read_parquet(COMBINED_PATH)
search = pd.read_parquet(SEARCH_PATH)

print(f"Purchase + Cart dataset : {combined.shape}")
print(f"Search feature dataset  : {search.shape}")


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING INPUT DATA")
print("=" * 80)

# Client ID uniqueness
assert combined["client_id"].is_unique, \
    "ERROR: Duplicate client_id in combined dataset."

assert search["client_id"].is_unique, \
    "ERROR: Duplicate client_id in search dataset."

# Churn must exist in combined
assert "churn" in combined.columns, \
    "ERROR: churn column missing from combined dataset."

print(" Combined client IDs are unique")
print(" Search client IDs are unique")
print(" Churn column exists")


# ============================================================
# 3. SEARCH FEATURES TO KEEP
# ============================================================

search_features = [
    "client_id",

    "search_count",
    "distinct_search_days",
    "unique_search_queries",

    "days_since_last_search",

    "searches_last_30d",
    "searches_previous_30d",

    "search_trend_ratio",

    "searches_per_lifetime_day",

    "avg_days_between_searches"
]

# Check all required columns exist
missing_columns = [
    col for col in search_features
    if col not in search.columns
]

if missing_columns:
    raise ValueError(
        f"Missing search columns: {missing_columns}"
    )

search_selected = search[search_features].copy()

print("\nSearch features selected:")
for col in search_features[1:]:
    print(f"  - {col}")


# ============================================================
# 4. CHECK CUSTOMER OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("CHECKING CUSTOMER OVERLAP")
print("=" * 80)

combined_ids = set(combined["client_id"])
search_ids = set(search["client_id"])

common_ids = combined_ids.intersection(search_ids)

print(f"Modeling customers       : {len(combined_ids):,}")
print(f"Customers with search    : {len(search_ids):,}")
print(f"Common customers         : {len(common_ids):,}")
print(f"Customers without search : {len(combined_ids - search_ids):,}")


# ============================================================
# 5. MERGE
# ============================================================

print("\n" + "=" * 80)
print("MERGING SEARCH FEATURES")
print("=" * 80)

# IMPORTANT:
# left join keeps the original modeling population unchanged.

merged = combined.merge(
    search_selected,
    on="client_id",
    how="left",
    validate="one_to_one"
)

print(f"Shape after merge: {merged.shape}")


# ============================================================
# 6. CREATE SEARCH ACTIVITY INDICATOR
# ============================================================

# Search feature rows exist only for customers who performed
# at least one search.

merged["has_search_activity"] = (
    merged["search_count"].notna().astype("int8")
)

print("\nSearch activity:")
print(
    merged["has_search_activity"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. HANDLE SEARCH MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("HANDLING SEARCH MISSING VALUES")
print("=" * 80)

# ------------------------------------------------------------
# Count / frequency features
# ------------------------------------------------------------
# No search activity = 0 searches.

zero_fill_columns = [
    "search_count",
    "distinct_search_days",
    "unique_search_queries",
    "searches_last_30d",
    "searches_previous_30d",
    "searches_per_lifetime_day"
]

for col in zero_fill_columns:
    merged[col] = merged[col].fillna(0)


# ------------------------------------------------------------
# Behavioral features
# ------------------------------------------------------------
# These remain NaN when there is no search activity because
# "no search" is different from a numerical value such as 0.
#
# days_since_last_search
# avg_days_between_searches
# search_trend_ratio
#
# These will be handled later during model preprocessing.

print("\nMissing values intentionally retained for:")
for col in [
    "days_since_last_search",
    "avg_days_between_searches",
    "search_trend_ratio"
]:
    print(
        f"  {col}: "
        f"{merged[col].isna().sum():,}"
    )


# ============================================================
# 8. VERIFY CHURN LABEL
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING CHURN LABEL")
print("=" * 80)

original_churn = combined[
    ["client_id", "churn"]
].copy()

merged_churn = merged[
    ["client_id", "churn"]
].copy()

churn_check = original_churn.merge(
    merged_churn,
    on="client_id",
    suffixes=("_original", "_merged"),
    validate="one_to_one"
)

churn_mismatches = (
    churn_check["churn_original"]
    != churn_check["churn_merged"]
).sum()

print(f"Churn label mismatches: {churn_mismatches}")

assert churn_mismatches == 0, \
    "ERROR: Churn labels changed during merge."

print(" Churn labels preserved")


# ============================================================
# 9. VERIFY ROW COUNT
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING DATASET INTEGRITY")
print("=" * 80)

assert len(merged) == len(combined), \
    "ERROR: Row count changed during merge."

assert merged["client_id"].is_unique, \
    "ERROR: Duplicate clients created."

assert merged["client_id"].notna().all(), \
    "ERROR: Missing client IDs."

print(f" Row count unchanged: {len(merged):,}")
print(" No duplicate customers")
print(" No missing client IDs")


# ============================================================
# 10. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(f"Rows    : {merged.shape[0]:,}")
print(f"Columns : {merged.shape[1]}")

print("\nChurn distribution:")
print(
    merged["churn"]
    .value_counts()
    .sort_index()
)

print("\nChurn rate:")
print(
    f"{merged['churn'].mean() * 100:.2f}%"
)


# ============================================================
# 11. SEARCH FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SEARCH FEATURE SUMMARY")
print("=" * 80)

search_summary = merged[
    [
        "search_count",
        "distinct_search_days",
        "unique_search_queries",
        "days_since_last_search",
        "searches_last_30d",
        "searches_previous_30d",
        "search_trend_ratio",
        "searches_per_lifetime_day",
        "avg_days_between_searches",
        "has_search_activity"
    ]
].describe().T

print(search_summary)


# ============================================================
# 12. MISSING VALUE AUDIT
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUE AUDIT")
print("=" * 80)

missing = merged.isna().sum()

missing = missing[missing > 0].sort_values(
    ascending=False
)

if len(missing) == 0:
    print(" No missing values")
else:
    for col, count in missing.items():
        percentage = count / len(merged) * 100

        print(
            f"{col:35s} "
            f"{count:10,} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# 13. FEATURE ↔ CHURN CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE <-> CHURN CORRELATION")
print("=" * 80)

numeric_columns = merged.select_dtypes(
    include=[np.number]
).columns

correlations = (
    merged[numeric_columns]
    .corr()["churn"]
    .drop("churn")
    .sort_values()
)

print("\nMost negative correlations:")
print(correlations.head(15))

print("\nMost positive correlations:")
print(correlations.tail(15))


# ============================================================
# 14. FEATURE-FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("HIGH FEATURE-FEATURE CORRELATIONS")
print("=" * 80)

feature_columns = [
    col for col in numeric_columns
    if col != "churn"
]

corr_matrix = merged[
    feature_columns
].corr()

# Only upper triangle
upper = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

high_correlations = (
    upper
    .stack()
    .reset_index()
)

high_correlations.columns = [
    "feature_1",
    "feature_2",
    "correlation"
]

high_correlations["abs_correlation"] = (
    high_correlations["correlation"].abs()
)

high_correlations = (
    high_correlations
    .sort_values(
        "abs_correlation",
        ascending=False
    )
)

high_correlations = high_correlations[
    high_correlations["abs_correlation"] >= 0.80
]

if len(high_correlations) == 0:
    print("No feature pairs with correlation >= 0.80")
else:
    print(high_correlations.to_string(index=False))


# ============================================================
# 15. SAVE
# ============================================================

print("\n" + "=" * 80)
print("SAVING DATASET")
print("=" * 80)

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(f" Saved to:")
print(f"  {OUTPUT_PATH}")

print("\n" + "=" * 80)
print("COMBINATION COMPLETE")
print("=" * 80)