from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

buy = pd.read_parquet(
    DATA_DIR / "product_buy.parquet"
)

buy["timestamp"] = pd.to_datetime(buy["timestamp"])

buy = buy.sort_values(["client_id", "timestamp"])

# ------------------------------------------------------------
# FIXED OBSERVATION CUTOFF
# ------------------------------------------------------------

CUTOFF_DATE = pd.Timestamp("2022-10-09 00:09:15")

historical = buy[
    buy["timestamp"] <= CUTOFF_DATE
]

# Customers with at least 2 historical purchases
historical_counts = (
    historical
    .groupby("client_id")
    .size()
)

returning_clients = historical_counts[
    historical_counts >= 2
].index

print("=" * 70)
print("CHURN WINDOW COMPARISON")
print("=" * 70)

print("\nObservation cutoff:", CUTOFF_DATE)
print("Returning customers:", len(returning_clients))

# ------------------------------------------------------------
# TEST DIFFERENT WINDOWS
# ------------------------------------------------------------

windows = [30, 45, 60, 90]

results = []

for days in windows:

    label_end = (
        CUTOFF_DATE +
        pd.Timedelta(days=days)
    )

    future = buy[
        (buy["timestamp"] > CUTOFF_DATE) &
        (buy["timestamp"] <= label_end)
    ]

    future_clients = set(
        future["client_id"].unique()
    )

    churned = sum(
        client not in future_clients
        for client in returning_clients
    )

    retained = (
        len(returning_clients) - churned
    )

    churn_rate = (
        churned / len(returning_clients)
    )

    results.append({
        "churn_window_days": days,
        "churned": churned,
        "retained": retained,
        "churn_rate": churn_rate * 100
    })

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

for _, row in results_df.iterrows():

    print(
        f"\n{int(row['churn_window_days'])}-day window:"
    )

    print(
        f"  Churn rate: {row['churn_rate']:.2f}%"
    )

    print(
        f"  Churned: {int(row['churned']):,}"
    )

    print(
        f"  Retained: {int(row['retained']):,}"
    )