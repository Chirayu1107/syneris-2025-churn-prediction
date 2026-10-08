import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "data/raw/page_visit.parquet"
OUTPUT_PATH = "data/processed/page_visit_features.parquet"

CUTOFF_DATE = pd.Timestamp("2022-10-09 00:09:15")

CHUNK_SIZE = 1_000_000


# ============================================================
# 1. LOAD DATASET SCHEMA
# ============================================================

print("=" * 80)
print("PAGE VISIT FEATURE ENGINEERING")
print("=" * 80)

dataset = ds.dataset(INPUT_PATH, format="parquet")

print("\nDataset schema:")
print(dataset.schema)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

available_columns = [
    field.name for field in dataset.schema
]

required_columns = [
    "client_id",
    "timestamp"
]

missing = [
    col for col in required_columns
    if col not in available_columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("\n client_id found")
print(" timestamp found")


# ============================================================
# 3. ACCUMULATORS
# ============================================================

# Customer-level aggregates
stats = defaultdict(lambda: {
    "visit_count": 0,
    "first_visit": None,
    "last_visit": None,
    "distinct_days": set(),

    "visits_last_7d": 0,
    "visits_last_14d": 0,
    "visits_last_30d": 0,
    "visits_previous_30d": 0,
})


# ============================================================
# 4. SCAN PARQUET IN BATCHES
# ============================================================

print("\n" + "=" * 80)
print("SCANNING PAGE VISITS")
print("=" * 80)

scanner = dataset.scanner(
    columns=["client_id", "timestamp"],
    batch_size=CHUNK_SIZE
)

batch_number = 0
total_rows = 0
historical_rows = 0

for batch in scanner.to_batches():

    batch_number += 1

    df = batch.to_pandas()

    total_rows += len(df)

    # --------------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["client_id", "timestamp"]
    )

    # Only information available before prediction cutoff
    df = df[
        df["timestamp"] <= CUTOFF_DATE
    ]

    if df.empty:
        continue

    historical_rows += len(df)

    # --------------------------------------------------------
    # Convert client IDs
    # --------------------------------------------------------

    grouped = df.groupby("client_id")

    # --------------------------------------------------------
    # Process each customer
    # --------------------------------------------------------

    for client_id, group in grouped:

        s = stats[client_id]

        timestamps = group["timestamp"]

        # Basic count
        s["visit_count"] += len(group)

        # First visit
        first_visit = timestamps.min()

        if (
            s["first_visit"] is None
            or first_visit < s["first_visit"]
        ):
            s["first_visit"] = first_visit

        # Last visit
        last_visit = timestamps.max()

        if (
            s["last_visit"] is None
            or last_visit > s["last_visit"]
        ):
            s["last_visit"] = last_visit

        # Distinct visit days
        visit_days = (
            timestamps
            .dt.normalize()
            .unique()
        )

        s["distinct_days"].update(
            visit_days
        )

        # ----------------------------------------------------
        # Time-window activity
        # ----------------------------------------------------

        s["visits_last_7d"] += (
            (timestamps >
             CUTOFF_DATE - pd.Timedelta(days=7))
            &
            (timestamps <= CUTOFF_DATE)
        ).sum()

        s["visits_last_14d"] += (
            (timestamps >
             CUTOFF_DATE - pd.Timedelta(days=14))
            &
            (timestamps <= CUTOFF_DATE)
        ).sum()

        s["visits_last_30d"] += (
            (timestamps >
             CUTOFF_DATE - pd.Timedelta(days=30))
            &
            (timestamps <= CUTOFF_DATE)
        ).sum()

        s["visits_previous_30d"] += (
            (timestamps >
             CUTOFF_DATE - pd.Timedelta(days=60))
            &
            (timestamps <=
             CUTOFF_DATE - pd.Timedelta(days=30))
        ).sum()

    if batch_number % 10 == 0:
        print(
            f"Processed batches: {batch_number:4d} | "
            f"Rows scanned: {total_rows:,} | "
            f"Historical rows: {historical_rows:,} | "
            f"Customers: {len(stats):,}"
        )


# ============================================================
# 5. CONVERT AGGREGATES TO DATAFRAME
# ============================================================

print("\n" + "=" * 80)
print("CREATING CUSTOMER FEATURES")
print("=" * 80)

records = []

for client_id, s in stats.items():

    first_visit = s["first_visit"]
    last_visit = s["last_visit"]

    if first_visit is not None:
        days_since_first = (
            CUTOFF_DATE - first_visit
        ).total_seconds() / 86400

        days_since_last = (
            CUTOFF_DATE - last_visit
        ).total_seconds() / 86400

        lifetime_days = max(
            days_since_first,
            1
        )

        visits_per_lifetime_day = (
            s["visit_count"] /
            lifetime_days
        )

    else:
        days_since_first = np.nan
        days_since_last = np.nan
        visits_per_lifetime_day = np.nan

    active_days = max(
        len(s["distinct_days"]),
        1
    )

    visits_per_active_day = (
        s["visit_count"] /
        active_days
    )

    # --------------------------------------------------------
    # Visit trend
    # --------------------------------------------------------

    recent = s["visits_last_30d"]
    previous = s["visits_previous_30d"]

    if previous > 0:
        visit_trend_ratio = (
            recent / previous
        )
    else:
        visit_trend_ratio = np.nan

    records.append({
        "client_id": client_id,

        "page_visit_count":
            s["visit_count"],

        "distinct_visit_days":
            len(s["distinct_days"]),

        "days_since_last_visit":
            days_since_last,

        "days_since_first_visit":
            days_since_first,

        "visits_last_7d":
            s["visits_last_7d"],

        "visits_last_14d":
            s["visits_last_14d"],

        "visits_last_30d":
            s["visits_last_30d"],

        "visits_previous_30d":
            s["visits_previous_30d"],

        "visit_trend_ratio":
            visit_trend_ratio,

        "visits_per_active_day":
            visits_per_active_day,

        "visits_per_lifetime_day":
            visits_per_lifetime_day,

        "has_page_visit_activity":
            1
    })


features = pd.DataFrame(records)


# ============================================================
# 6. REMOVE DUPLICATES / SORT
# ============================================================

features = (
    features
    .drop_duplicates("client_id")
    .sort_values("client_id")
    .reset_index(drop=True)
)


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PAGE VISIT FEATURE SUMMARY")
print("=" * 80)

print(
    f"Total rows scanned      : {total_rows:,}"
)

print(
    f"Historical rows used    : {historical_rows:,}"
)

print(
    f"Unique customers        : {len(features):,}"
)

print(
    f"Features created        : {len(features.columns)}"
)

print("\nFeatures:")

for col in features.columns:
    print(f"  - {col}")


# ============================================================
# 8. FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE STATISTICS")
print("=" * 80)

print(
    features
    .drop(columns=["client_id"])
    .describe()
    .T
)


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = features.isna().sum()

missing = missing[
    missing > 0
].sort_values(
    ascending=False
)

if len(missing) == 0:
    print(" No missing values")
else:
    for col, count in missing.items():

        percentage = (
            count /
            len(features) *
            100
        )

        print(
            f"{col:35s}"
            f"{count:10,}"
            f" ({percentage:6.2f}%)"
        )


# ============================================================
# 10. SAVE
# ============================================================

print("\n" + "=" * 80)
print("SAVING FEATURES")
print("=" * 80)

Path(
    OUTPUT_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True
)

features.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    f"✓ Saved to:\n"
    f"  {OUTPUT_PATH}"
)

print("\n" + "=" * 80)
print("PAGE VISIT FEATURE ENGINEERING COMPLETE")
print("=" * 80)