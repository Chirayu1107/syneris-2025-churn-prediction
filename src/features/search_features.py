from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

Data_dir = Path("data/raw")
output_dir = Path("data/processed")

cutoff_date = pd.Timestamp("2022-10-09 00:09:15")

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SEARCH QUERY FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# LOAD SEARCH DATA
# ============================================================

print("\nLoading search query data...")

search = pd.read_parquet(
    Data_dir / "search_query.parquet"
)

print(
    "Total search events:",
    len(search)
)


# ============================================================
# DATETIME
# ============================================================

print("\nConverting timestamp...")

search["timestamp"] = pd.to_datetime(
    search["timestamp"]
)


# ============================================================
# HISTORICAL DATA
# ============================================================

print("\nFiltering historical search events...")

search = search[
    search["timestamp"] <= cutoff_date
].copy()

print(
    "Historical search events:",
    len(search)
)


# ============================================================
# SEARCH DATE
# ============================================================

search["search_date"] = (
    search["timestamp"].dt.date
)


# ============================================================
# BASIC SEARCH FEATURES
# ============================================================

print("\nCreating basic search features...")

search_features = (
    search
    .groupby("client_id")
    .agg(
        search_count=("timestamp", "count"),

        unique_search_queries=(
            "query",
            "nunique"
        ),

        distinct_search_days=(
            "search_date",
            "nunique"
        ),

        first_search=(
            "timestamp",
            "min"
        ),

        last_search=(
            "timestamp",
            "max"
        )
    )
    .reset_index()
)


# ============================================================
# RECENCY
# ============================================================

print("\nCreating search recency features...")

search_features[
    "days_since_last_search"
] = (
    (
        cutoff_date
        -
        search_features["last_search"]
    )
    .dt.total_seconds()
    /
    (60 * 60 * 24)
)


search_features[
    "days_since_first_search"
] = (
    (
        cutoff_date
        -
        search_features["first_search"]
    )
    .dt.total_seconds()
    /
    (60 * 60 * 24)
)


# ============================================================
# SEARCH GAP
# ============================================================

print("\nCreating search gap feature...")

search = search.sort_values(
    ["client_id", "timestamp"]
)


search["search_gap_days"] = (
    search
    .groupby("client_id")["timestamp"]
    .diff()
    .dt.total_seconds()
    /
    (60 * 60 * 24)
)


average_search_gap = (
    search
    .groupby("client_id")[
        "search_gap_days"
    ]
    .mean()
    .rename(
        "avg_days_between_searches"
    )
)


search_features = search_features.merge(
    average_search_gap,
    on="client_id",
    how="left"
)


# ============================================================
# RECENT SEARCH ACTIVITY
# ============================================================

print(
    "\nCreating recent search activity..."
)


def search_in_period(days):

    start_date = (
        cutoff_date
        -
        pd.Timedelta(days=days)
    )

    mask = (
        (search["timestamp"] > start_date)
        &
        (search["timestamp"] <= cutoff_date)
    )

    counts = (
        search.loc[mask]
        .groupby("client_id")
        .size()
        .rename(
            f"searches_last_{days}d"
        )
    )

    return counts


for days in [7, 14, 30]:

    counts = search_in_period(days)

    search_features = search_features.merge(
        counts,
        on="client_id",
        how="left"
    )


# ============================================================
# PREVIOUS 30 DAYS
# ============================================================

print(
    "Creating search trend features..."
)

recent_start = (
    cutoff_date
    -
    pd.Timedelta(days=30)
)

previous_start = (
    cutoff_date
    -
    pd.Timedelta(days=60)
)


previous_mask = (
    (search["timestamp"] > previous_start)
    &
    (search["timestamp"] <= recent_start)
)


previous_counts = (
    search.loc[previous_mask]
    .groupby("client_id")
    .size()
    .rename(
        "searches_previous_30d"
    )
)


search_features = search_features.merge(
    previous_counts,
    on="client_id",
    how="left"
)


# ============================================================
# FILL RECENT ACTIVITY
# ============================================================

activity_columns = [
    "searches_last_7d",
    "searches_last_14d",
    "searches_last_30d",
    "searches_previous_30d"
]


for column in activity_columns:

    search_features[column] = (
        search_features[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# SEARCH TREND
# ============================================================

print("\nCreating search trend ratio...")

search_features[
    "search_trend_ratio"
] = (
    search_features["searches_last_30d"]
    /
    search_features[
        "searches_previous_30d"
    ].replace(0, np.nan)
)


# ============================================================
# SEARCH FREQUENCY
# ============================================================

print(
    "\nCreating search frequency features..."
)


search_features[
    "searches_per_active_day"
] = (
    search_features["search_count"]
    /
    search_features[
        "distinct_search_days"
    ]
)


search_features[
    "searches_per_lifetime_day"
] = (
    search_features["search_count"]
    /
    search_features[
        "days_since_first_search"
    ].clip(lower=1)
)


# ============================================================
# ACTIVITY INDICATOR
# ============================================================

search_features[
    "has_search_activity"
] = 1


# ============================================================
# REMOVE RAW TIMESTAMP COLUMNS
# ============================================================

search_features = search_features.drop(
    columns=[
        "first_search",
        "last_search"
    ]
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SEARCH FEATURE DATASET SUMMARY")
print("=" * 80)


print(
    "\nRows:",
    len(search_features)
)


print(
    "Columns:",
    len(search_features.columns)
)


print("\nColumns:")

for column in search_features.columns:

    print(
        " -",
        column
    )


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    search_features.isnull().sum()
)


# ============================================================
# DUPLICATE CUSTOMERS
# ============================================================

print("\nDuplicate customers:")

print(
    search_features[
        "client_id"
    ].duplicated().sum()
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    output_dir /
    "search_features.parquet"
)


search_features.to_parquet(
    output_path,
    index=False
)


print("\nSaved:")

print(output_path)


print("\n" + "=" * 80)
print("SEARCH FEATURE ENGINEERING COMPLETE")
print("=" * 80)