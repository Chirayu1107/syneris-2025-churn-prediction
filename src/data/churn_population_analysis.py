from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# --------------------------------------------------
# LOAD PURCHASE DATA
# --------------------------------------------------

buy = pd.read_parquet(DATA_DIR / "product_buy.parquet")

buy["timestamp"] = pd.to_datetime(buy["timestamp"])

buy = buy.sort_values(["client_id", "timestamp"])

# --------------------------------------------------
# CUSTOMER PURCHASE SUMMARY
# --------------------------------------------------

customer_summary = buy.groupby("client_id").agg(
    first_purchase=("timestamp", "min"),
    last_purchase=("timestamp", "max"),
    purchase_count=("timestamp", "count")
).reset_index()

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

dataset_start = buy["timestamp"].min()
dataset_end = buy["timestamp"].max()

print("=" * 60)
print("CHURN POPULATION ANALYSIS")
print("=" * 60)

print("\nDataset start:")
print(dataset_start)

print("\nDataset end:")
print(dataset_end)

print("\nTotal customers:")
print(len(customer_summary))

# --------------------------------------------------
# CUSTOMER TYPES
# --------------------------------------------------

one_time = customer_summary[
    customer_summary["purchase_count"] == 1
]

returning = customer_summary[
    customer_summary["purchase_count"] >= 2
]

print("\nOne-time customers:")
print(len(one_time))

print("\nReturning customers:")
print(len(returning))

print("\nReturning customer percentage:")
print(
    len(returning) / len(customer_summary) * 100
)

# --------------------------------------------------
# 60-DAY CHURN WINDOW
# --------------------------------------------------

CHURN_DAYS = 60

latest_usable_cutoff = (
    dataset_end - pd.Timedelta(days=CHURN_DAYS)
)

print("\nLatest usable observation cutoff:")
print(latest_usable_cutoff)

# Customers whose last purchase occurred
# before the latest usable cutoff can potentially
# receive a full 60-day future observation window.

eligible = customer_summary[
    customer_summary["last_purchase"] <= latest_usable_cutoff
]

print("\nCustomers with complete 60-day follow-up:")
print(len(eligible))

# --------------------------------------------------
# RETURNING CUSTOMERS WITH COMPLETE FOLLOW-UP
# --------------------------------------------------

eligible_returning = eligible[
    eligible["purchase_count"] >= 2
]

print("\nReturning customers with complete 60-day follow-up:")
print(len(eligible_returning))

# --------------------------------------------------
# LAST PURCHASE DISTRIBUTION
# --------------------------------------------------

print("\nLast purchase date percentiles:")

print(
    customer_summary["last_purchase"]
    .quantile([0.50, 0.75, 0.90, 0.95, 0.99])
)

# --------------------------------------------------
# PURCHASE COUNT DISTRIBUTION
# --------------------------------------------------

print("\nPurchase count distribution:")

print(
    customer_summary["purchase_count"].describe()
)

# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------

eligible_returning.to_parquet(
    DATA_DIR / "eligible_returning_customers.parquet",
    index=False
)

print("\nSaved:")
print(
    "data/raw/eligible_returning_customers.parquet"
)