from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

CUTOFF_DATE = pd.Timestamp("2022-10-09 00:09:15")

buy = pd.read_parquet(
    DATA_DIR / "product_buy.parquet"
)

buy["timestamp"] = pd.to_datetime(buy["timestamp"])

# Only historical purchases
historical = buy[
    buy["timestamp"] <= CUTOFF_DATE
].copy()

# Calendar date of purchase
historical["purchase_date"] = (
    historical["timestamp"].dt.date
)

# Number of distinct purchasing days per customer
purchase_days = (
    historical
    .groupby("client_id")["purchase_date"]
    .nunique()
)

print("=" * 60)
print("REPEAT CUSTOMER ANALYSIS")
print("=" * 60)

print("\nTotal customers:", len(purchase_days))

print(
    "\nCustomers with 1 purchase event:",
    (historical.groupby("client_id").size() == 1).sum()
)

print(
    "\nCustomers with 2+ purchase events:",
    (historical.groupby("client_id").size() >= 2).sum()
)

print(
    "\nCustomers with 2+ distinct purchase days:",
    (purchase_days >= 2).sum()
)

print(
    "\nCustomers with 3+ distinct purchase days:",
    (purchase_days >= 3).sum()
)

print(
    "\nCustomers with 5+ distinct purchase days:",
    (purchase_days >= 5).sum()
)

print("\nPurchase-day distribution:")
print(purchase_days.describe())

print("\nPurchase-day percentiles:")
print(
    purchase_days.quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99]
    )
)