from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

Processed_dir = Path("data/processed")

purchase_path = Processed_dir / "purchase_features.parquet"
cart_path = Processed_dir / "final_cart_features.parquet"

output_path = Processed_dir / "combined_features.parquet"


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("COMBINING PURCHASE + CART FEATURES")
print("=" * 80)


# ============================================================
# LOAD PURCHASE FEATURES
# ============================================================

print("\nLoading purchase features...")

purchase = pd.read_parquet(purchase_path)

print("Purchase rows:", len(purchase))
print("Purchase columns:", len(purchase.columns))

print("\nPurchase columns:")

for column in purchase.columns:
    print(" -", column)


# ============================================================
# LOAD CART FEATURES
# ============================================================

print("\nLoading final cart features...")

cart = pd.read_parquet(cart_path)

print("Cart rows:", len(cart))
print("Cart columns:", len(cart.columns))

print("\nCart columns:")

for column in cart.columns:
    print(" -", column)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("BASIC VALIDATION")
print("=" * 80)


# Check client ID uniqueness

print("\nPurchase duplicate clients:")

print(
    purchase["client_id"].duplicated().sum()
)


print("\nCart duplicate clients:")

print(
    cart["client_id"].duplicated().sum()
)


# ============================================================
# CUSTOMER OVERLAP
# ============================================================

print("\nChecking customer overlap...")

purchase_clients = set(
    purchase["client_id"]
)

cart_clients = set(
    cart["client_id"]
)


common_clients = (
    purchase_clients
    & cart_clients
)

purchase_only = (
    purchase_clients
    - cart_clients
)

cart_only = (
    cart_clients
    - purchase_clients
)


print(
    "Purchase customers:",
    len(purchase_clients)
)

print(
    "Cart customers:",
    len(cart_clients)
)

print(
    "Common customers:",
    len(common_clients)
)

print(
    "Purchase-only customers:",
    len(purchase_only)
)

print(
    "Cart-only customers:",
    len(cart_only)
)


# ============================================================
# CHURN LABEL VALIDATION
# ============================================================

print("\nChecking churn labels...")

purchase_labels = (
    purchase[
        ["client_id", "churn"]
    ]
    .set_index("client_id")
)

cart_labels = (
    cart[
        ["client_id", "churn"]
    ]
    .set_index("client_id")
)


common_label_index = (
    purchase_labels.index
    .intersection(cart_labels.index)
)


label_comparison = (
    purchase_labels.loc[common_label_index, "churn"]
    ==
    cart_labels.loc[common_label_index, "churn"]
)


label_mismatches = (
    (~label_comparison)
    .sum()
)


print(
    "Common customers checked:",
    len(common_label_index)
)

print(
    "Churn label mismatches:",
    label_mismatches
)


if label_mismatches > 0:

    print(
        "\nWARNING: Churn labels do not match!"
    )

else:

    print(
        "\nChurn labels match perfectly."
    )


# ============================================================
# FIND OVERLAPPING COLUMN NAMES
# ============================================================

print("\nChecking overlapping columns...")

purchase_columns = set(
    purchase.columns
)

cart_columns = set(
    cart.columns
)

overlapping_columns = (
    purchase_columns
    & cart_columns
)

print("\nOverlapping columns:")

for column in sorted(overlapping_columns):
    print(" -", column)


# client_id and churn are expected overlaps.
unexpected_overlap = (
    overlapping_columns
    - {"client_id", "churn"}
)


if len(unexpected_overlap) > 0:

    print(
        "\nWARNING: Unexpected overlapping columns:"
    )

    for column in sorted(unexpected_overlap):
        print(" -", column)

else:

    print(
        "\nNo unexpected feature-name collisions."
    )


# ============================================================
# MERGE
# ============================================================

print("\n" + "=" * 80)
print("MERGING DATASETS")
print("=" * 80)


# Keep only common modeling population.
combined = purchase.merge(
    cart.drop(columns=["churn"]),
    on="client_id",
    how="inner",
    validate="one_to_one"
)


print(
    "\nCombined rows:",
    len(combined)
)

print(
    "Combined columns:",
    len(combined.columns)
)


# ============================================================
# VERIFY MERGE
# ============================================================

print("\nVerifying combined dataset...")


print(
    "\nDuplicate clients:"
)

print(
    combined["client_id"]
    .duplicated()
    .sum()
)


print(
    "\nMissing client IDs:"
)

print(
    combined["client_id"].isna().sum()
)


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

print("\nChurn distribution:")

print(
    combined["churn"]
    .value_counts()
)


print("\nChurn percentage:")

print(
    (
        combined["churn"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# FEATURE LIST
# ============================================================

print("\n" + "=" * 80)
print("COMBINED FEATURE LIST")
print("=" * 80)


feature_columns = [
    column
    for column in combined.columns
    if column not in ["client_id", "churn"]
]


print(
    "\nTotal modeling features:",
    len(feature_columns)
)


for i, column in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:02d}. {column}"
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUE ANALYSIS")
print("=" * 80)


missing = (
    combined[feature_columns]
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)


missing_percentage = (
    missing
    /
    len(combined)
    * 100
)


missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percentage
})


print(
    "\nFeatures with missing values:"
)

print(
    missing_table[
        missing_table["missing_count"] > 0
    ].to_string()
)


# ============================================================
# CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("CONSTANT FEATURE CHECK")
print("=" * 80)


unique_counts = (
    combined[feature_columns]
    .nunique()
    .sort_values()
)


constant_features = (
    unique_counts[
        unique_counts <= 1
    ]
)


if len(constant_features) == 0:

    print(
        "\nNo constant features found."
    )

else:

    print(
        "\nConstant features:"
    )

    print(
        constant_features.to_string()
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 80)


numeric_features = (
    combined[feature_columns]
    .select_dtypes(
        include=np.number
    )
)


correlation = (
    numeric_features
    .corr()
)


# Find highly correlated pairs

correlation_pairs = []


for i in range(
    len(correlation.columns)
):

    for j in range(
        i + 1,
        len(correlation.columns)
    ):

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


print(
    "\nHighly correlated pairs "
    "(|r| >= 0.80):"
)


if len(correlation_pairs) == 0:

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
# CORRELATION WITH CHURN
# ============================================================

print("\n" + "=" * 80)
print("FEATURE <-> CHURN CORRELATION")
print("=" * 80)


churn_correlation = (
    numeric_features
    .corrwith(
        combined["churn"]
    )
    .sort_values()
)


print(
    "\nMost negative relationships:"
)

print(
    churn_correlation.head(10).to_string()
)


print(
    "\nMost positive relationships:"
)

print(
    churn_correlation.tail(10).to_string()
)


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)


print(
    combined.dtypes.to_string()
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL COMBINED DATASET SUMMARY")
print("=" * 80)


print(
    "\nRows:",
    len(combined)
)

print(
    "Columns:",
    len(combined.columns)
)

print(
    "Modeling features:",
    len(feature_columns)
)

print(
    "Churned:",
    combined["churn"].sum()
)

print(
    "Retained:",
    (combined["churn"] == 0).sum()
)

print(
    "Churn rate:",
    round(
        combined["churn"].mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving combined dataset...")

combined.to_parquet(
    output_path,
    index=False
)


print(
    "\nSaved:"
)

print(
    output_path
)


print("\n" + "=" * 80)
print("COMBINATION COMPLETE")
print("=" * 80)