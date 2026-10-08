from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

Data_dir = Path("data/raw")
Processed_dir = Path("data/processed")
Analysis_dir = Path("data/analysis")

Analysis_dir.mkdir(parents=True, exist_ok=True)

cutoff_date = pd.Timestamp("2022-10-09 00:09:15")
label_end = cutoff_date + pd.Timedelta(days=60)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("CART FEATURE ANALYSIS")
print("=" * 80)


# ============================================================
# LOAD CART FEATURES
# ============================================================

print("\nLoading cart features...")

cart = pd.read_parquet(
    Processed_dir / "cart_features.parquet"
)

print("Cart customers:", len(cart))
print("Cart features:", len(cart.columns))


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
# CREATE HISTORICAL PURCHASE DATA
# ============================================================

print("\nCreating historical purchase population...")

historical_buy = buy[
    buy["timestamp"] <= cutoff_date
].copy()

historical_buy["purchase_date"] = (
    historical_buy["timestamp"].dt.date
)


# ============================================================
# CREATE PURCHASE POPULATION
# ============================================================

purchase_customers = (
    historical_buy
    .groupby("client_id")
    .agg(
        purchase_count=("timestamp", "count"),
        distinct_purchase_days=("purchase_date", "nunique")
    )
    .reset_index()
)


# ============================================================
# MODELING POPULATION
# ============================================================

print("\nCreating modeling population...")

eligible_customers = purchase_customers[
    purchase_customers["distinct_purchase_days"] >= 2
].copy()

print(
    "Eligible customers:",
    len(eligible_customers)
)


# ============================================================
# CREATE CHURN LABEL
# ============================================================

print("\nCreating 60-day churn labels...")

future_buy = buy[
    (buy["timestamp"] > cutoff_date)
    &
    (buy["timestamp"] <= label_end)
]

future_clients = set(
    future_buy["client_id"].unique()
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
# MERGE CART FEATURES WITH LABELS
# ============================================================

print("\nMerging cart features with modeling population...")

dataset = eligible_customers[
    ["client_id", "churn"]
].merge(
    cart,
    on="client_id",
    how="left"
)

print("Final analysis rows:", len(dataset))


# ============================================================
# CART FEATURE LIST
# ============================================================

cart_features = [
    "add_to_cart_count",
    "distinct_add_to_cart_days",
    "unique_products_added",
    "days_since_last_add_to_cart",
    "days_since_first_add_to_cart",
    "avg_days_between_adds",
    "adds_last_7d",
    "adds_last_14d",
    "adds_last_30d",
    "adds_previous_30d",
    "add_to_cart_trend_ratio",
    "add_per_active_day",
    "add_per_lifetime_day",

    "remove_from_cart_count",
    "distinct_remove_days",
    "unique_products_removed",
    "days_since_last_remove",
    "days_since_first_remove",
    "remove_per_active_day",
    "removes_last_7d",
    "removes_last_14d",
    "removes_last_30d",
    "removes_previous_30d",

    "remove_to_add_ratio",
    "net_cart_activity"
]


# ============================================================
# COVERAGE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CART ACTIVITY COVERAGE")
print("=" * 80)

dataset["has_add_activity"] = (
    dataset["add_to_cart_count"].fillna(0) > 0
).astype(int)

dataset["has_remove_activity"] = (
    dataset["remove_from_cart_count"].fillna(0) > 0
).astype(int)


print(
    "\nCustomers with add-to-cart activity:"
)

print(
    dataset["has_add_activity"]
    .value_counts()
)


print(
    "\nCustomers with remove-from-cart activity:"
)

print(
    dataset["has_remove_activity"]
    .value_counts()
)


# ============================================================
# FILL ONLY BASIC COUNT FEATURES
# ============================================================

count_features = [
    "add_to_cart_count",
    "distinct_add_to_cart_days",
    "unique_products_added",
    "adds_last_7d",
    "adds_last_14d",
    "adds_last_30d",
    "adds_previous_30d",

    "remove_from_cart_count",
    "distinct_remove_days",
    "unique_products_removed",
    "removes_last_7d",
    "removes_last_14d",
    "removes_last_30d",
    "removes_previous_30d",

    "net_cart_activity"
]

for column in count_features:

    if column in dataset.columns:
        dataset[column] = (
            dataset[column]
            .fillna(0)
        )


# ============================================================
# MEDIAN ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("MEDIAN FEATURE ANALYSIS")
print("=" * 80)

median_analysis = (
    dataset
    .groupby("churn")[cart_features]
    .median()
    .T
)

median_analysis.columns = [
    "Retained",
    "Churned"
]

median_analysis["Difference"] = (
    median_analysis["Churned"]
    -
    median_analysis["Retained"]
)

print("\n")

print(
    median_analysis.to_string()
)


# ============================================================
# MEAN ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("MEAN FEATURE ANALYSIS")
print("=" * 80)

mean_analysis = (
    dataset
    .groupby("churn")[cart_features]
    .mean()
    .T
)

mean_analysis.columns = [
    "Retained",
    "Churned"
]

mean_analysis["Difference"] = (
    mean_analysis["Churned"]
    -
    mean_analysis["Retained"]
)

print("\n")

print(
    mean_analysis.to_string()
)


# ============================================================
# RELATIVE DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("RELATIVE DIFFERENCE")
print("=" * 80)

relative_analysis = pd.DataFrame(index=cart_features)

relative_analysis["retained_median"] = (
    dataset[dataset["churn"] == 0][cart_features]
    .median()
)

relative_analysis["churned_median"] = (
    dataset[dataset["churn"] == 1][cart_features]
    .median()
)

relative_analysis["absolute_difference"] = (
    relative_analysis["churned_median"]
    -
    relative_analysis["retained_median"]
)

relative_analysis["relative_difference_pct"] = (
    (
        relative_analysis["churned_median"]
        -
        relative_analysis["retained_median"]
    )
    /
    relative_analysis["retained_median"].replace(0, np.nan)
    * 100
)

relative_analysis = relative_analysis.sort_values(
    "relative_difference_pct"
)

print(
    relative_analysis.to_string()
)


# ============================================================
# CART ACTIVITY BY CHURN
# ============================================================

print("\n" + "=" * 80)
print("CART ACTIVITY BY CHURN")
print("=" * 80)

activity_summary = (
    dataset
    .groupby("churn")
    .agg(
        customers=("client_id", "count"),

        avg_adds=(
            "add_to_cart_count",
            "mean"
        ),

        median_adds=(
            "add_to_cart_count",
            "median"
        ),

        avg_removes=(
            "remove_from_cart_count",
            "mean"
        ),

        median_removes=(
            "remove_from_cart_count",
            "median"
        ),

        avg_net_cart=(
            "net_cart_activity",
            "mean"
        ),

        median_net_cart=(
            "net_cart_activity",
            "median"
        )
    )
)

print(
    activity_summary.to_string()
)


# ============================================================
# ADD ACTIVITY CHURN RATE
# ============================================================

print("\n" + "=" * 80)
print("CHURN RATE BY ADD-TO-CART ACTIVITY")
print("=" * 80)

dataset["add_activity_group"] = pd.cut(
    dataset["add_to_cart_count"],
    bins=[
        -1,
        0,
        1,
        3,
        5,
        10,
        20,
        np.inf
    ],
    labels=[
        "0",
        "1",
        "2-3",
        "4-5",
        "6-10",
        "11-20",
        "20+"
    ]
)

add_churn = (
    dataset
    .groupby("add_activity_group", observed=False)
    .agg(
        customers=("client_id", "count"),
        churned=("churn", "sum"),
        churn_rate=("churn", "mean")
    )
)

add_churn["churn_rate"] *= 100

print(
    add_churn.to_string()
)


# ============================================================
# REMOVE ACTIVITY CHURN RATE
# ============================================================

print("\n" + "=" * 80)
print("CHURN RATE BY REMOVE-FROM-CART ACTIVITY")
print("=" * 80)

dataset["remove_activity_group"] = pd.cut(
    dataset["remove_from_cart_count"],
    bins=[
        -1,
        0,
        1,
        3,
        5,
        10,
        20,
        np.inf
    ],
    labels=[
        "0",
        "1",
        "2-3",
        "4-5",
        "6-10",
        "11-20",
        "20+"
    ]
)

remove_churn = (
    dataset
    .groupby("remove_activity_group", observed=False)
    .agg(
        customers=("client_id", "count"),
        churned=("churn", "sum"),
        churn_rate=("churn", "mean")
    )
)

remove_churn["churn_rate"] *= 100

print(
    remove_churn.to_string()
)


# ============================================================
# RECENT ACTIVITY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("RECENT CART ACTIVITY")
print("=" * 80)

recent_features = [
    "adds_last_7d",
    "adds_last_14d",
    "adds_last_30d",
    "removes_last_7d",
    "removes_last_14d",
    "removes_last_30d"
]

recent_summary = (
    dataset
    .groupby("churn")[recent_features]
    .median()
    .T
)

recent_summary.columns = [
    "Retained",
    "Churned"
]

print(
    recent_summary.to_string()
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CORRELATION")
print("=" * 80)

correlation = (
    dataset[cart_features]
    .corr()
)

# Find highly correlated feature pairs

correlation_pairs = []

for i in range(len(correlation.columns)):

    for j in range(i + 1, len(correlation.columns)):

        value = correlation.iloc[i, j]

        if abs(value) >= 0.80:

            correlation_pairs.append(
                (
                    correlation.columns[i],
                    correlation.columns[j],
                    value
                )
            )

correlation_pairs = sorted(
    correlation_pairs,
    key=lambda x: abs(x[2]),
    reverse=True
)

print("\nHighly correlated pairs (|r| >= 0.80):")

if len(correlation_pairs) == 0:

    print("None found.")

else:

    for feature1, feature2, value in correlation_pairs:

        print(
            f"{feature1:35s} "
            f"{feature2:35s} "
            f"{value:.3f}"
        )


# ============================================================
# SAVE ANALYSIS RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SAVING ANALYSIS RESULTS")
print("=" * 80)


median_analysis.to_csv(
    Analysis_dir / "cart_median_analysis.csv"
)

mean_analysis.to_csv(
    Analysis_dir / "cart_mean_analysis.csv"
)

relative_analysis.to_csv(
    Analysis_dir / "cart_relative_analysis.csv"
)

activity_summary.to_csv(
    Analysis_dir / "cart_activity_summary.csv"
)

add_churn.to_csv(
    Analysis_dir / "cart_add_churn_rate.csv"
)

remove_churn.to_csv(
    Analysis_dir / "cart_remove_churn_rate.csv"
)

recent_summary.to_csv(
    Analysis_dir / "cart_recent_activity.csv"
)

print("\nSaved analysis files to:")

print(Analysis_dir)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CART FEATURE ANALYSIS COMPLETE")
print("=" * 80)

print("\nModeling customers:", len(dataset))

print(
    "\nChurned:",
    dataset["churn"].sum()
)

print(
    "Retained:",
    (dataset["churn"] == 0).sum()
)

print(
    "\nOverall churn rate:",
    round(dataset["churn"].mean() * 100, 2),
    "%"
)

print("\nAnalysis complete.")
