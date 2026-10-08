from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

CUTOFF_DATE = pd.Timestamp("2022-10-09 00:09:15")
CHURN_WINDOW_DAYS = 60

LABEL_END = (
    CUTOFF_DATE +
    pd.Timedelta(days=CHURN_WINDOW_DAYS)
)

# ============================================================
# LOAD PURCHASE DATA
# ============================================================

buy = pd.read_parquet(
    DATA_DIR / "product_buy.parquet"
)

buy["timestamp"] = pd.to_datetime(
    buy["timestamp"]
)

# ============================================================
# SPLIT HISTORY / FUTURE
# ============================================================

historical = buy[
    buy["timestamp"] <= CUTOFF_DATE
].copy()

future = buy[
    (buy["timestamp"] > CUTOFF_DATE) &
    (buy["timestamp"] <= LABEL_END)
].copy()

# ============================================================
# DISTINCT PURCHASE DAYS
# ============================================================

historical["purchase_date"] = (
    historical["timestamp"].dt.date
)

purchase_days = (
    historical
    .groupby("client_id")["purchase_date"]
    .nunique()
)

future_clients = set(
    future["client_id"].unique()
)

print("=" * 70)
print("REPEAT CUSTOMER CHURN ANALYSIS")
print("=" * 70)

print("\nCutoff:", CUTOFF_DATE)
print("Label end:", LABEL_END)
print("Total historical customers:", len(purchase_days))

# ============================================================
# TEST POPULATIONS
# ============================================================

thresholds = [2, 3, 5]

results = []

for threshold in thresholds:

    eligible = purchase_days[
        purchase_days >= threshold
    ]

    total = len(eligible)

    retained = sum(
        client in future_clients
        for client in eligible.index
    )

    churned = total - retained

    churn_rate = (
        churned / total * 100
    )

    results.append({
        "minimum_purchase_days": threshold,
        "customers": total,
        "churned": churned,
        "retained": retained,
        "churn_rate": churn_rate
    })

# ============================================================
# DISPLAY
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print(
    results_df.to_string(index=False)
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for _, row in results_df.iterrows():

    print(
        f"\n{int(row['minimum_purchase_days'])}+ "
        f"distinct purchase days:"
    )

    print(
        f"  Customers: {int(row['customers']):,}"
    )

    print(
        f"  Churned: {int(row['churned']):,}"
    )

    print(
        f"  Retained: {int(row['retained']):,}"
    )

    print(
        f"  Churn rate: {row['churn_rate']:.2f}%"
    )