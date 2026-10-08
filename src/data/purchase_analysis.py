from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

buy = pd.read_parquet(DATA_DIR / "product_buy.parquet")

buy["timestamp"] = pd.to_datetime(buy["timestamp"])

# Sort customer purchases chronologically
buy = buy.sort_values(["client_id", "timestamp"])

# Time between consecutive purchases for each customer
buy["days_since_previous_purchase"] = (
    buy.groupby("client_id")["timestamp"]
    .diff()
    .dt.total_seconds()
    / (24 * 60 * 60)
)

intervals = buy["days_since_previous_purchase"].dropna()

print("=" * 60)
print("PURCHASE INTERVAL ANALYSIS")
print("=" * 60)

print("\nNumber of intervals:", len(intervals))

print("\nPercentiles:")
print(intervals.quantile([
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99
]))

print("\nMean:", intervals.mean())
print("Median:", intervals.median())
print("Maximum:", intervals.max())

purchase_counts = buy.groupby("client_id").size()

print("\nPurchase frequency:")
print(purchase_counts.describe())

print(
    "\nCustomers with exactly 1 purchase:",
    (purchase_counts == 1).sum()
)

print(
    "Customers with 2+ purchases:",
    (purchase_counts >= 2).sum()
)