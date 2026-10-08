from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

Data_dir = Path("data/raw")
Processed_dir = Path("data/processed")
Analysis_dir = Path("data/analysis")

Analysis_dir.mkdir(
    parents=True,
    exist_ok=True
)

cutoff_date = pd.Timestamp(
    "2022-10-09 00:09:15"
)

label_end = (
    cutoff_date
    +
    pd.Timedelta(days=60)
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("SEARCH FEATURE ANALYSIS")
print("=" * 80)


# ============================================================
# LOAD SEARCH FEATURES
# ============================================================

print("\nLoading search features...")

search = pd.read_parquet(
    Processed_dir /
    "search_features.parquet"
)

print(
    "Search customers:",
    len(search)
)

print(
    "Search features:",
    len(search.columns)
)


# ============================================================
# LOAD PURCHASE DATA
# ============================================================

print("\nLoading purchase data...")

buy = pd.read_parquet(
    Data_dir /
    "product_buy.parquet"
)

buy["timestamp"] = pd.to_datetime(
    buy["timestamp"]
)

print(
    "Total purchase events:",
    len(buy)
)


# ============================================================
# HISTORICAL PURCHASE POPULATION
# ============================================================

print(
    "\nCreating historical purchase population..."
)

historical = buy[
    buy["timestamp"] <= cutoff_date
].copy()

historical["purchase_date"] = (
    historical["timestamp"].dt.date
)


purchase_population = (
    historical
    .groupby("client_id")
    .agg(
        purchase_count=(
            "timestamp",
            "count"
        ),

        distinct_purchase_days=(
            "purchase_date",
            "nunique"
        )
    )
    .reset_index()
)


# ============================================================
# MODELING POPULATION
# ============================================================

print(
    "\nCreating modeling population..."
)

eligible_customers = purchase_population[
    purchase_population[
        "distinct_purchase_days"
    ] >= 2
].copy()


print(
    "Eligible customers:",
    len(eligible_customers)
)


# ============================================================
# CHURN LABEL
# ============================================================

print(
    "\nCreating 60-day churn labels..."
)

future = buy[
    (buy["timestamp"] > cutoff_date)
    &
    (buy["timestamp"] <= label_end)
]

future_clients = set(
    future["client_id"].unique()
)


eligible_customers["churn"] = (
    ~eligible_customers[
        "client_id"
    ].isin(future_clients)
).astype(int)


print(
    "\nChurn distribution:"
)

print(
    eligible_customers[
        "churn"
    ].value_counts()
)


# ============================================================
# MERGE SEARCH FEATURES
# ============================================================

print(
    "\nMerging search features..."
)

dataset = eligible_customers[
    ["client_id", "churn"]
].merge(
    search,
    on="client_id",
    how="left"
)


print(
    "Final analysis rows:",
    len(dataset)
)


# ============================================================
# SEARCH FEATURE LIST
# ============================================================

search_features = [
    "search_count",
    "unique_search_queries",
    "distinct_search_days",
    "days_since_last_search",
    "days_since_first_search",
    "avg_days_between_searches",
    "searches_last_7d",
    "searches_last_14d",
    "searches_last_30d",
    "searches_previous_30d",
    "search_trend_ratio",
    "searches_per_active_day",
    "searches_per_lifetime_day",
    "has_search_activity"
]


# ============================================================
# SEARCH ACTIVITY COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("SEARCH ACTIVITY COVERAGE")
print("=" * 80)


dataset["has_search"] = (
    dataset["search_count"]
    .fillna(0)
    > 0
).astype(int)


print(
    "\nCustomers with search activity:"
)

print(
    dataset[
        "has_search"
    ].value_counts()
)


# ============================================================
# FILL COUNT FEATURES
# ============================================================

count_features = [
    "search_count",
    "unique_search_queries",
    "distinct_search_days",
    "searches_last_7d",
    "searches_last_14d",
    "searches_last_30d",
    "searches_previous_30d"
]


for column in count_features:

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
    .groupby("churn")[
        search_features
    ]
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
    .groupby("churn")[
        search_features
    ]
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


print(
    mean_analysis.to_string()
)


# ============================================================
# RELATIVE DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("RELATIVE DIFFERENCE")
print("=" * 80)


relative_analysis = pd.DataFrame(
    index=search_features
)


relative_analysis[
    "retained_median"
] = (
    dataset[
        dataset["churn"] == 0
    ][search_features]
    .median()
)


relative_analysis[
    "churned_median"
] = (
    dataset[
        dataset["churn"] == 1
    ][search_features]
    .median()
)


relative_analysis[
    "absolute_difference"
] = (
    relative_analysis[
        "churned_median"
    ]
    -
    relative_analysis[
        "retained_median"
    ]
)


relative_analysis[
    "relative_difference_pct"
] = (
    (
        relative_analysis[
            "churned_median"
        ]
        -
        relative_analysis[
            "retained_median"
        ]
    )
    /
    relative_analysis[
        "retained_median"
    ].replace(0, np.nan)
    *
    100
)


relative_analysis = (
    relative_analysis
    .sort_values(
        "relative_difference_pct"
    )
)


print(
    relative_analysis.to_string()
)


# ============================================================
# SEARCH ACTIVITY CHURN RATE
# ============================================================

print("\n" + "=" * 80)
print("CHURN RATE BY SEARCH ACTIVITY")
print("=" * 80)


dataset["search_activity_group"] = pd.cut(
    dataset["search_count"],
    bins=[
        -1,
        0,
        1,
        3,
        5,
        10,
        20,
        50,
        np.inf
    ],
    labels=[
        "0",
        "1",
        "2-3",
        "4-5",
        "6-10",
        "11-20",
        "21-50",
        "50+"
    ]
)


search_churn = (
    dataset
    .groupby(
        "search_activity_group",
        observed=False
    )
    .agg(
        customers=(
            "client_id",
            "count"
        ),

        churned=(
            "churn",
            "sum"
        ),

        churn_rate=(
            "churn",
            "mean"
        )
    )
)


search_churn[
    "churn_rate"
] *= 100


print(
    search_churn.to_string()
)


# ============================================================
# RECENT SEARCH ACTIVITY
# ============================================================

print("\n" + "=" * 80)
print("RECENT SEARCH ACTIVITY")
print("=" * 80)


recent_features = [
    "searches_last_7d",
    "searches_last_14d",
    "searches_last_30d",
    "searches_previous_30d"
]


recent_summary = (
    dataset
    .groupby("churn")[
        recent_features
    ]
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
# SEARCH DIVERSITY
# ============================================================

print("\n" + "=" * 80)
print("SEARCH DIVERSITY")
print("=" * 80)


diversity_features = [
    "search_count",
    "unique_search_queries"
]


diversity_summary = (
    dataset
    .groupby("churn")[
        diversity_features
    ]
    .agg(
        [
            "mean",
            "median"
        ]
    )
)


print(
    diversity_summary.to_string()
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CORRELATION")
print("=" * 80)


numeric_features = (
    dataset[
        search_features
    ]
    .select_dtypes(
        include=np.number
    )
)


correlation = (
    numeric_features
    .corr()
)


correlation_pairs = []


for i in range(
    len(correlation.columns)
):

    for j in range(
        i + 1,
        len(correlation.columns)
    ):

        value = correlation.iloc[
            i,
            j
        ]

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


print(
    "\nHighly correlated pairs "
    "(|r| >= 0.80):"
)


if not correlation_pairs:

    print("None found.")

else:

    for (
        feature1,
        feature2,
        value
    ) in correlation_pairs:

        print(
            f"{feature1:40s} "
            f"{feature2:40s} "
            f"{value:.3f}"
        )


# ============================================================
# SEARCH FEATURE ↔ CHURN CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("SEARCH FEATURE <-> CHURN CORRELATION")
print("=" * 80)


churn_correlation = (
    numeric_features
    .corrwith(
        dataset["churn"]
    )
    .sort_values()
)


print(
    "\nMost negative:"
)

print(
    churn_correlation
    .head(10)
    .to_string()
)


print(
    "\nMost positive:"
)

print(
    churn_correlation
    .tail(10)
    .to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SAVING SEARCH ANALYSIS")
print("=" * 80)


median_analysis.to_csv(
    Analysis_dir /
    "search_median_analysis.csv"
)


mean_analysis.to_csv(
    Analysis_dir /
    "search_mean_analysis.csv"
)


relative_analysis.to_csv(
    Analysis_dir /
    "search_relative_analysis.csv"
)


search_churn.to_csv(
    Analysis_dir /
    "search_churn_rate.csv"
)


recent_summary.to_csv(
    Analysis_dir /
    "search_recent_activity.csv"
)


print(
    "\nAnalysis files saved to:"
)

print(
    Analysis_dir
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SEARCH FEATURE ANALYSIS COMPLETE")
print("=" * 80)


print(
    "\nModeling customers:",
    len(dataset)
)


print(
    "Churned:",
    dataset["churn"].sum()
)


print(
    "Retained:",
    (
        dataset["churn"] == 0
    ).sum()
)


print(
    "Overall churn rate:",
    round(
        dataset["churn"].mean() * 100,
        2
    ),
    "%"
)

print("\nAnalysis complete.")