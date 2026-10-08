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
# ADD TO CART FEATURE ENGINEERING
# ============================================================

print("=" * 70)
print("Cart Feature Engineering")
print("=" * 70)

print("\nLoading add_to_cart data...")

add_cart = pd.read_parquet(Data_dir / "add_to_cart.parquet")

add_cart["timestamp"] = pd.to_datetime(add_cart["timestamp"])

print("Total add_to_cart events:", len(add_cart))


# Keep only historical data
add_cart = add_cart[
    add_cart["timestamp"] <= cutoff_date
].copy()

print("Historical add_to_cart events:", len(add_cart))


# Add event date
add_cart["cart_date"] = add_cart["timestamp"].dt.date


# ============================================================
# BASIC ADD TO CART FEATURES
# ============================================================

print("\nCreating basic add_to_cart features...")

add_features = (
    add_cart.groupby("client_id")
    .agg(
        add_to_cart_count=("timestamp", "count"),
        distinct_add_to_cart_days=("cart_date", "nunique"),
        unique_products_added=("sku", "nunique"),
        first_add_to_cart=("timestamp", "min"),
        last_add_to_cart=("timestamp", "max"),
    )
    .reset_index()
)


# ============================================================
# ADD TO CART RECENCY
# ============================================================

print("\nCreating add_to_cart recency features...")

add_features["days_since_last_add_to_cart"] = (
    (
        cutoff_date - add_features["last_add_to_cart"]
    ).dt.total_seconds()
    / (60 * 60 * 24)
)

add_features["days_since_first_add_to_cart"] = (
    (
        cutoff_date - add_features["first_add_to_cart"]
    ).dt.total_seconds()
    / (60 * 60 * 24)
)


# ============================================================
# ADD TO CART GAP
# ============================================================

print("\nCreating add_to_cart gap feature...")

add_cart = add_cart.sort_values(
    ["client_id", "timestamp"]
)

add_cart["add_cart_gap_days"] = (
    add_cart.groupby("client_id")["timestamp"]
    .diff()
    .dt.total_seconds()
    / (60 * 60 * 24)
)

average_add_gap = (
    add_cart.groupby("client_id")["add_cart_gap_days"]
    .mean()
    .rename("avg_days_between_adds")
)

add_features = add_features.merge(
    average_add_gap,
    on="client_id",
    how="left"
)


# ============================================================
# RECENT ADD TO CART ACTIVITY
# ============================================================

print("\nCreating recent add_to_cart activity features...")


def add_cart_in_period(days):

    start_date = cutoff_date - pd.Timedelta(days=days)

    mask = (
        (add_cart["timestamp"] > start_date)
        &
        (add_cart["timestamp"] <= cutoff_date)
    )

    counts = (
        add_cart.loc[mask]
        .groupby("client_id")
        .size()
        .rename(f"adds_last_{days}d")
    )

    return counts


for days in [7, 14, 30]:

    counts = add_cart_in_period(days)

    add_features = add_features.merge(
        counts,
        on="client_id",
        how="left"
    )


# ============================================================
# PREVIOUS 30 DAYS
# ============================================================

print("Creating add_to_cart trend features...")

recent_start = cutoff_date - pd.Timedelta(days=30)

previous_start = cutoff_date - pd.Timedelta(days=60)

previous_mask = (
    (add_cart["timestamp"] > previous_start)
    &
    (add_cart["timestamp"] <= recent_start)
)

previous_counts = (
    add_cart.loc[previous_mask]
    .groupby("client_id")
    .size()
    .rename("adds_previous_30d")
)

add_features = add_features.merge(
    previous_counts,
    on="client_id",
    how="left"
)


# ============================================================
# FILL RECENT ACTIVITY
# ============================================================

add_activity_columns = [
    "adds_last_7d",
    "adds_last_14d",
    "adds_last_30d",
    "adds_previous_30d"
]

for column in add_activity_columns:

    add_features[column] = (
        add_features[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# ADD TO CART TREND
# ============================================================

add_features["add_to_cart_trend_ratio"] = (
    add_features["adds_last_30d"]
    /
    add_features["adds_previous_30d"].replace(0, np.nan)
)


# ============================================================
# ADD TO CART FREQUENCY
# ============================================================

add_features["add_per_active_day"] = (
    add_features["add_to_cart_count"]
    /
    add_features["distinct_add_to_cart_days"]
)


add_features["add_per_lifetime_day"] = (
    add_features["add_to_cart_count"]
    /
    add_features["days_since_first_add_to_cart"].clip(lower=1)
)


# ============================================================
# REMOVE FROM CART
# ============================================================

print("\nLoading remove_from_cart data...")

remove_cart = pd.read_parquet(
    Data_dir / "remove_from_cart.parquet"
)

remove_cart["timestamp"] = pd.to_datetime(
    remove_cart["timestamp"]
)

print("Total remove_from_cart events:", len(remove_cart))


# Keep only historical data
remove_cart = remove_cart[
    remove_cart["timestamp"] <= cutoff_date
].copy()

print(
    "Historical remove_from_cart events:",
    len(remove_cart)
)


# Add date
remove_cart["remove_date"] = (
    remove_cart["timestamp"].dt.date
)


# ============================================================
# BASIC REMOVE FEATURES
# ============================================================

print("\nCreating remove_from_cart features...")

remove_features = (
    remove_cart.groupby("client_id")
    .agg(
        remove_from_cart_count=("timestamp", "count"),
        distinct_remove_days=("remove_date", "nunique"),
        unique_products_removed=("sku", "nunique"),
        first_remove_from_cart=("timestamp", "min"),
        last_remove_from_cart=("timestamp", "max"),
    )
    .reset_index()
)


# ============================================================
# REMOVE RECENCY
# ============================================================

print("\nCreating remove_from_cart recency features...")

remove_features["days_since_last_remove"] = (
    (
        cutoff_date
        - remove_features["last_remove_from_cart"]
    ).dt.total_seconds()
    / (60 * 60 * 24)
)

remove_features["days_since_first_remove"] = (
    (
        cutoff_date
        - remove_features["first_remove_from_cart"]
    ).dt.total_seconds()
    / (60 * 60 * 24)
)


# ============================================================
# REMOVE FREQUENCY
# ============================================================

remove_features["remove_per_active_day"] = (
    remove_features["remove_from_cart_count"]
    /
    remove_features["distinct_remove_days"]
)


# ============================================================
# RECENT REMOVE ACTIVITY
# ============================================================

print("\nCreating recent remove activity features...")


def remove_in_period(days):

    start_date = cutoff_date - pd.Timedelta(days=days)

    mask = (
        (remove_cart["timestamp"] > start_date)
        &
        (remove_cart["timestamp"] <= cutoff_date)
    )

    counts = (
        remove_cart.loc[mask]
        .groupby("client_id")
        .size()
        .rename(f"removes_last_{days}d")
    )

    return counts


for days in [7, 14, 30]:

    counts = remove_in_period(days)

    remove_features = remove_features.merge(
        counts,
        on="client_id",
        how="left"
    )


# ============================================================
# PREVIOUS 30 DAYS REMOVE ACTIVITY
# ============================================================

previous_remove_counts = (
    remove_cart.loc[
        (
            remove_cart["timestamp"] > previous_start
        )
        &
        (
            remove_cart["timestamp"] <= recent_start
        )
    ]
    .groupby("client_id")
    .size()
    .rename("removes_previous_30d")
)

remove_features = remove_features.merge(
    previous_remove_counts,
    on="client_id",
    how="left"
)


# ============================================================
# FILL REMOVE ACTIVITY
# ============================================================

remove_activity_columns = [
    "removes_last_7d",
    "removes_last_14d",
    "removes_last_30d",
    "removes_previous_30d"
]

for column in remove_activity_columns:

    remove_features[column] = (
        remove_features[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# MERGE ADD + REMOVE
# ============================================================

print("\nCombining add and remove features...")

cart_features = add_features.merge(
    remove_features,
    on="client_id",
    how="outer"
)


# ============================================================
# FILL CUSTOMERS WITH NO REMOVE ACTIVITY
# ============================================================

remove_numeric_columns = [
    "remove_from_cart_count",
    "distinct_remove_days",
    "unique_products_removed",
    "days_since_last_remove",
    "days_since_first_remove",
    "remove_per_active_day",
    "removes_last_7d",
    "removes_last_14d",
    "removes_last_30d",
    "removes_previous_30d"
]

for column in remove_numeric_columns:

    cart_features[column] = (
        cart_features[column]
        .fillna(0)
    )


# ============================================================
# COMBINED CART FEATURES
# ============================================================

print("\nCreating combined cart features...")

cart_features["remove_to_add_ratio"] = (
    cart_features["remove_from_cart_count"]
    /
    cart_features["add_to_cart_count"].replace(0, np.nan)
)


cart_features["net_cart_activity"] = (
    cart_features["add_to_cart_count"]
    -
    cart_features["remove_from_cart_count"]
)


# ============================================================
# REMOVE RAW TIMESTAMP COLUMNS
# ============================================================

cart_features = cart_features.drop(
    columns=[
        "first_add_to_cart",
        "last_add_to_cart",
        "first_remove_from_cart",
        "last_remove_from_cart"
    ]
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("Cart Feature Dataset Summary")
print("=" * 80)

print("\nRows:", len(cart_features))

print(
    "Columns:",
    len(cart_features.columns)
)

print("\nColumns:")

for column in cart_features.columns:
    print(" -", column)


print("\nMissing Values:")

print(
    cart_features.isnull().sum()
)


print("\nDuplicate Customers:")

print(
    cart_features["client_id"].duplicated().sum()
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    output_dir / "cart_features.parquet"
)

cart_features.to_parquet(
    output_path,
    index=False
)

print("\nSaved:")

print(output_path)

print("\nCart feature engineering complete.")