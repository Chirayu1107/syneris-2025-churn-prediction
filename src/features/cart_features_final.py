from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

Data_dir = Path("data/raw")
Processed_dir = Path("data/processed")

cutoff_date = pd.Timestamp("2022-10-09 00:09:15")
label_end = cutoff_date + pd.Timedelta(days=60)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FINAL CART FEATURE ENGINEERING")
print("=" * 80)


# ============================================================
# LOAD CART FEATURES
# ============================================================

print("\nLoading cart features...")

cart = pd.read_parquet(
    Processed_dir / "cart_features.parquet"
)

print("Cart customers:", len(cart))
print("Cart columns:", len(cart.columns))


# ============================================================
# SELECT FINAL CART FEATURES
# ============================================================

selected_features = [
    "client_id",

    # Add behavior
    "add_to_cart_count",
    "unique_products_added",
    "distinct_add_to_cart_days",
    "days_since_last_add_to_cart",
    "avg_days_between_adds",
    "adds_last_30d",

    # Remove behavior
    "remove_from_cart_count",
    "unique_products_removed",
    "days_since_last_remove",

    # Combined behavior
    "remove_to_add_ratio",
    "net_cart_activity"
]


cart = cart[selected_features].copy()


print("\nSelected cart features:")

for column in cart.columns:
    print(" -", column)


# ============================================================
# LOAD PURCHASE DATA
# ============================================================

print("\nLoading purchase data...")

buy = pd.read_parquet(
    Data_dir / "product_buy.parquet"
)

buy["timestamp"] = pd.to_datetime(
    buy["timestamp"]
)

print("Total purchase events:", len(buy))


# ============================================================
# HISTORICAL PURCHASE DATA
# ============================================================

print("\nCreating historical purchase population...")

historical = buy[
    buy["timestamp"] <= cutoff_date
].copy()

historical["purchase_date"] = (
    historical["timestamp"].dt.date
)


# ============================================================
# CREATE MODELING POPULATION
# ============================================================

print("\nCreating modeling population...")

purchase_population = (
    historical
    .groupby("client_id")
    .agg(
        purchase_count=("timestamp", "count"),
        distinct_purchase_days=("purchase_date", "nunique")
    )
    .reset_index()
)


eligible_customers = purchase_population[
    purchase_population["distinct_purchase_days"] >= 2
].copy()


print(
    "Eligible customers:",
    len(eligible_customers)
)


# ============================================================
# CREATE CHURN LABEL
# ============================================================

print("\nCreating 60-day churn labels...")

future = buy[
    (buy["timestamp"] > cutoff_date)
    &
    (buy["timestamp"] <= label_end)
]

future_clients = set(
    future["client_id"].unique()
)


eligible_customers["churn"] = (
    ~eligible_customers["client_id"]
    .isin(future_clients)
).astype(int)


print("\nChurn distribution:")

print(
    eligible_customers["churn"]
    .value_counts()
)


# ============================================================
# KEEP ONLY MODELING CUSTOMERS
# ============================================================

labels = eligible_customers[
    ["client_id", "churn"]
].copy()


# ============================================================
# MERGE CART FEATURES WITH MODELING POPULATION
# ============================================================

print("\nMerging cart features with modeling population...")

final_cart = labels.merge(
    cart,
    on="client_id",
    how="left"
)


print(
    "Final cart dataset rows:",
    len(final_cart)
)


# ============================================================
# CART ACTIVITY INDICATORS
# ============================================================

print("\nCreating cart activity indicators...")

final_cart["has_add_activity"] = (
    final_cart["add_to_cart_count"]
    .fillna(0)
    > 0
).astype(int)


final_cart["has_remove_activity"] = (
    final_cart["remove_from_cart_count"]
    .fillna(0)
    > 0
).astype(int)


# ============================================================
# HANDLE CUSTOMERS WITH NO CART ACTIVITY
# ============================================================

print("\nHandling missing cart activity...")

count_columns = [
    "add_to_cart_count",
    "unique_products_added",
    "distinct_add_to_cart_days",
    "adds_last_30d",

    "remove_from_cart_count",
    "unique_products_removed",

    "net_cart_activity"
]


for column in count_columns:

    final_cart[column] = (
        final_cart[column]
        .fillna(0)
    )


# ============================================================
# RECENCY FEATURES
# ============================================================

# Customers with no add/remove activity do not have a
# meaningful recency value.
#
# We keep NaN here and provide activity indicators so that
# the model can distinguish:
#
#   "no activity"
#
# from:
#
#   "activity exists but value is missing/undefined"


# ============================================================
# RATIO HANDLING
# ============================================================

# remove_to_add_ratio is undefined when add_to_cart_count = 0.
#
# We keep it as NaN for now.
#
# This will be handled during the model preprocessing stage.

print(
    "\nUndefined remove_to_add_ratio:",
    final_cart["remove_to_add_ratio"].isna().sum()
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL CART DATASET SUMMARY")
print("=" * 80)


print(
    "\nRows:",
    len(final_cart)
)


print(
    "Columns:",
    len(final_cart.columns)
)


print("\nColumns:")

for column in final_cart.columns:
    print(" -", column)


print("\nChurn distribution:")

print(
    final_cart["churn"]
    .value_counts()
)


print("\nChurn rate:")

print(
    round(
        final_cart["churn"].mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

missing = (
    final_cart
    .isnull()
    .sum()
)

print(missing)


# ============================================================
# DUPLICATE CUSTOMER CHECK
# ============================================================

print("\nDuplicate customers:")

print(
    final_cart["client_id"]
    .duplicated()
    .sum()
)


# ============================================================
# CART ACTIVITY COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("CART ACTIVITY COVERAGE")
print("=" * 80)


print(
    "\nCustomers with add activity:"
)

print(
    final_cart["has_add_activity"]
    .value_counts()
)


print(
    "\nCustomers with remove activity:"
)

print(
    final_cart["has_remove_activity"]
    .value_counts()
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    Processed_dir /
    "final_cart_features.parquet"
)


final_cart.to_parquet(
    output_path,
    index=False
)


print("\nSaved:")

print(output_path)


print("\n" + "=" * 80)
print("FINAL CART FEATURE ENGINEERING COMPLETE")
print("=" * 80)