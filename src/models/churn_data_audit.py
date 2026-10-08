import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = (
    "data/processed/"
    "combined_purchase_cart_search_features.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/data_audit"
)

CORRELATION_THRESHOLD = 0.90

NEAR_CONSTANT_THRESHOLD = 0.01


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    # --------------------------------------------------------
    # PURCHASE
    # --------------------------------------------------------

    "purchase_count",
    "distinct_purchase_days",
    "unique_product_purchased",

    "days_since_last_purchase",

    "avg_days_between_purchases",

    "purchases_last_30d",
    "purchases_previous_30d",

    "purchase_trend_ratio",

    "purchase_per_active_day",
    "purchase_per_lifetime_day",


    # --------------------------------------------------------
    # CART
    # --------------------------------------------------------

    "add_to_cart_count",
    "distinct_add_to_cart_days",

    "days_since_last_add_to_cart",

    "avg_days_between_adds",

    "adds_last_30d",

    "remove_from_cart_count",

    "days_since_last_remove",

    "remove_to_add_ratio",

    "net_cart_activity",

    "has_add_activity",
    "has_remove_activity",


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    "search_count",
    "distinct_search_days",

    "days_since_last_search",

    "searches_last_30d",
    "searches_previous_30d",

    "search_trend_ratio",

    "searches_per_lifetime_day",

    "has_search_activity"
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_parquet(DATA_PATH)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("BASIC DATA INFORMATION")
print("=" * 80)

print("\nColumns:")

for column in df.columns:
    print(
        f"  - {column}"
    )


print(
    f"\nTotal rows: {len(df):,}"
)

print(
    f"Total columns: {len(df.columns)}"
)


# ============================================================
# 3. TARGET CHECK
# ============================================================

print("\n" + "=" * 80)
print("TARGET CHECK")
print("=" * 80)

if "churn" not in df.columns:

    raise ValueError(
        "Target column 'churn' not found."
    )


print(
    "\nChurn distribution:"
)

print(
    df["churn"].value_counts()
)


print(
    f"\nChurn rate: "
    f"{df['churn'].mean() * 100:.2f}%"
)


# ============================================================
# 4. FEATURE EXISTENCE CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE EXISTENCE CHECK")
print("=" * 80)


missing_features = [

    feature

    for feature in FEATURES

    if feature not in df.columns

]


if missing_features:

    print(
        "\nMissing features:"
    )

    for feature in missing_features:
        print(
            f"  - {feature}"
        )

    raise ValueError(
        "Some expected features are missing."
    )


print(
    f"\n All {len(FEATURES)} expected features exist."
)


# ============================================================
# 5. DUPLICATE ROW CHECK
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE ROW CHECK")
print("=" * 80)


duplicate_rows = df.duplicated().sum()


print(
    f"\nDuplicate rows: "
    f"{duplicate_rows:,}"
)


if duplicate_rows == 0:

    print(
        " No duplicate rows found."
    )

else:

    print(
        " Duplicate rows found."
    )


# ============================================================
# 6. DUPLICATE CLIENT CHECK
# ============================================================

print("\n" + "=" * 80)
print("CLIENT ID UNIQUENESS CHECK")
print("=" * 80)


if "client_id" in df.columns:

    unique_clients = (
        df["client_id"]
        .nunique()
    )

    total_rows = len(df)

    print(
        f"\nTotal rows: {total_rows:,}"
    )

    print(
        f"Unique clients: {unique_clients:,}"
    )

    if unique_clients == total_rows:

        print(
            " Each client appears exactly once."
        )

    else:

        print(
            " Some clients appear multiple times."
        )

else:

    print(
        "client_id not available."
    )


# ============================================================
# 7. MISSING VALUE AUDIT
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUE AUDIT")
print("=" * 80)


missing_count = (
    df[FEATURES]
    .isna()
    .sum()
)


missing_percentage = (
    missing_count /
    len(df) *
    100
)


missing_report = pd.DataFrame({

    "feature": FEATURES,

    "missing_count":
        missing_count.values,

    "missing_percentage":
        missing_percentage.values

})


missing_report = (
    missing_report
    .sort_values(
        "missing_count",
        ascending=False
    )
)


print(
    "\nFeatures with missing values:"
)


print(
    missing_report[
        missing_report["missing_count"] > 0
    ]
    .to_string(index=False)
)


total_missing = (
    missing_count.sum()
)


print(
    f"\nTotal missing feature values: "
    f"{total_missing:,}"
)


if total_missing == 0:

    print(
        " No missing values found."
    )

else:

    print(
        " Missing values exist."
    )


missing_output = (
    OUTPUT_DIR /
    "missing_value_report.csv"
)


missing_report.to_csv(
    missing_output,
    index=False
)


# ============================================================
# 8. INFINITE VALUE CHECK
# ============================================================

print("\n" + "=" * 80)
print("INFINITE VALUE CHECK")
print("=" * 80)


numeric_features = (
    df[FEATURES]
    .select_dtypes(
        include=np.number
    )
    .columns
)


infinite_report = []


for feature in numeric_features:

    positive_inf = np.isposinf(
        df[feature]
    ).sum()

    negative_inf = np.isneginf(
        df[feature]
    ).sum()

    total_inf = (
        positive_inf +
        negative_inf
    )

    infinite_report.append({

        "feature": feature,

        "positive_infinity":
            positive_inf,

        "negative_infinity":
            negative_inf,

        "total_infinity":
            total_inf

    })


infinite_report = pd.DataFrame(
    infinite_report
)


total_infinite = (
    infinite_report["total_infinity"]
    .sum()
)


print(
    f"\nTotal infinite values: "
    f"{total_infinite:,}"
)


if total_infinite == 0:

    print(
        " No infinite values found."
    )

else:

    print(
        " Infinite values found!"
    )

    print(
        infinite_report[
            infinite_report[
                "total_infinity"
            ] > 0
        ]
        .to_string(index=False)
    )


infinite_output = (
    OUTPUT_DIR /
    "infinite_value_report.csv"
)


infinite_report.to_csv(
    infinite_output,
    index=False
)


# ============================================================
# 9. CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("CONSTANT FEATURE CHECK")
print("=" * 80)


unique_counts = (
    df[FEATURES]
    .nunique(
        dropna=False
    )
)


constant_features = (
    unique_counts[
        unique_counts <= 1
    ]
)


if len(constant_features) == 0:

    print(
        "\n No constant features found."
    )

else:

    print(
        "\n Constant features:"
    )

    print(
        constant_features
    )


# ============================================================
# 10. NEAR-CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("NEAR-CONSTANT FEATURE CHECK")
print("=" * 80)


near_constant_rows = []


for feature in FEATURES:

    value_counts = (
        df[feature]
        .value_counts(
            normalize=True,
            dropna=False
        )
    )

    if len(value_counts) == 0:
        continue

    dominant_percentage = (
        value_counts.iloc[0]
    )

    if (
        dominant_percentage
        >= 1 - NEAR_CONSTANT_THRESHOLD
    ):

        near_constant_rows.append({

            "feature": feature,

            "dominant_value":
                value_counts.index[0],

            "dominant_percentage":
                dominant_percentage * 100

        })


near_constant_report = pd.DataFrame(
    near_constant_rows
)


if len(near_constant_report) == 0:

    print(
        "\n No near-constant features found."
    )

else:

    print(
        "\nPotential near-constant features:"
    )

    print(
        near_constant_report
        .to_string(index=False)
    )


# ============================================================
# 11. FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION SUMMARY")
print("=" * 80)


feature_statistics = (
    df[FEATURES]
    .describe()
    .T
)


feature_statistics[
    "missing_count"
] = (
    df[FEATURES]
    .isna()
    .sum()
)


feature_statistics[
    "unique_values"
] = (
    df[FEATURES]
    .nunique()
)


print(
    feature_statistics[
        [
            "count",
            "unique_values",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "missing_count"
        ]
    ]
    .to_string()
)


statistics_output = (
    OUTPUT_DIR /
    "feature_statistics.csv"
)


feature_statistics.to_csv(
    statistics_output
)


# ============================================================
# 12. OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("OUTLIER ANALYSIS")
print("=" * 80)


outlier_rows = []


for feature in numeric_features:

    series = df[feature].dropna()

    if len(series) == 0:
        continue

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outliers = (
        (series < lower_bound)
        |
        (series > upper_bound)
    )

    outlier_count = (
        outliers.sum()
    )

    outlier_percentage = (
        outlier_count /
        len(series) *
        100
    )

    outlier_rows.append({

        "feature": feature,

        "q1": q1,

        "q3": q3,

        "iqr": iqr,

        "lower_bound":
            lower_bound,

        "upper_bound":
            upper_bound,

        "outlier_count":
            outlier_count,

        "outlier_percentage":
            outlier_percentage

    })


outlier_report = pd.DataFrame(
    outlier_rows
)


outlier_report = (
    outlier_report
    .sort_values(
        "outlier_percentage",
        ascending=False
    )
)


print(
    "\nTop features by outlier percentage:"
)


print(
    outlier_report
    .head(15)
    .to_string(index=False)
)


outlier_output = (
    OUTPUT_DIR /
    "outlier_report.csv"
)


outlier_report.to_csv(
    outlier_output,
    index=False
)


# ============================================================
# 13. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)


correlation_matrix = (
    df[FEATURES]
    .corr()
)


correlation_output = (
    OUTPUT_DIR /
    "feature_correlation_matrix.csv"
)


correlation_matrix.to_csv(
    correlation_output
)


# ------------------------------------------------------------
# Find highly correlated pairs
# ------------------------------------------------------------

correlation_pairs = []


for i in range(
    len(FEATURES)
):

    for j in range(
        i + 1,
        len(FEATURES)
    ):

        feature_1 = FEATURES[i]

        feature_2 = FEATURES[j]

        correlation = (
            correlation_matrix
            .loc[
                feature_1,
                feature_2
            ]
        )

        if (
            abs(correlation)
            >= CORRELATION_THRESHOLD
        ):

            correlation_pairs.append({

                "feature_1":
                    feature_1,

                "feature_2":
                    feature_2,

                "correlation":
                    correlation,

                "absolute_correlation":
                    abs(correlation)

            })


high_correlation_report = (
    pd.DataFrame(
        correlation_pairs
    )
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
)


if len(high_correlation_report) == 0:

    print(
        "\n No highly correlated feature pairs "
        f"above {CORRELATION_THRESHOLD:.2f}"
    )

else:

    print(
        "\nHighly correlated feature pairs:"
    )

    print(
        high_correlation_report
        .to_string(index=False)
    )


high_correlation_output = (
    OUTPUT_DIR /
    "high_correlation_features.csv"
)


high_correlation_report.to_csv(
    high_correlation_output,
    index=False
)


# ============================================================
# 14. CORRELATION HEATMAP
# ============================================================

print(
    "\nCreating correlation heatmap..."
)


plt.figure(
    figsize=(18, 15)
)


plt.imshow(
    correlation_matrix,
    aspect="auto"
)


plt.colorbar(
    label="Correlation"
)


plt.xticks(
    range(len(FEATURES)),
    FEATURES,
    rotation=90,
    fontsize=7
)


plt.yticks(
    range(len(FEATURES)),
    FEATURES,
    fontsize=7
)


plt.title(
    "Feature Correlation Matrix"
)


plt.tight_layout()


heatmap_output = (
    OUTPUT_DIR /
    "feature_correlation_heatmap.png"
)


plt.savefig(
    heatmap_output,
    dpi=300
)


plt.close()


print(
    f" Heatmap saved:\n"
    f"{heatmap_output}"
)


# ============================================================
# 15. CHURN VS FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CHURN VS FEATURE ANALYSIS")
print("=" * 80)


churn_comparison = []


for feature in numeric_features:

    retained = (
        df.loc[
            df["churn"] == 0,
            feature
        ]
    )

    churned = (
        df.loc[
            df["churn"] == 1,
            feature
        ]
    )

    retained_median = (
        retained.median()
    )

    churned_median = (
        churned.median()
    )

    retained_mean = (
        retained.mean()
    )

    churned_mean = (
        churned.mean()
    )

    churn_comparison.append({

        "feature": feature,

        "retained_mean":
            retained_mean,

        "churned_mean":
            churned_mean,

        "retained_median":
            retained_median,

        "churned_median":
            churned_median,

        "median_difference":
            churned_median -
            retained_median

    })


churn_comparison = pd.DataFrame(
    churn_comparison
)


churn_comparison = (
    churn_comparison
    .sort_values(
        "median_difference",
        key=abs,
        ascending=False
    )
)


print(
    "\nFeatures with largest median differences:"
)


print(
    churn_comparison
    .head(20)
    .to_string(index=False)
)


churn_output = (
    OUTPUT_DIR /
    "churn_feature_comparison.csv"
)


churn_comparison.to_csv(
    churn_output,
    index=False
)


# ============================================================
# 16. FEATURE-TARGET CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE-TARGET CORRELATION")
print("=" * 80)


target_correlation = (
    df[FEATURES + ["churn"]]
    .corr()["churn"]
    .drop("churn")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print(
    "\nFeature correlation with churn:"
)


print(
    target_correlation
    .to_string()
)


target_output = (
    OUTPUT_DIR /
    "feature_target_correlation.csv"
)


target_correlation.to_csv(
    target_output
)


# ============================================================
# 17. POTENTIAL LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("POTENTIAL LEAKAGE CHECK")
print("=" * 80)


print(
    """
This is an automated warning check.

Features with extremely high correlation
with churn may require investigation.

Correlation does NOT automatically mean
data leakage.
"""
)


potential_leakage = (
    target_correlation[
        abs(target_correlation)
        >= 0.80
    ]
)


if len(potential_leakage) == 0:

    print(
        " No feature has target correlation "
        "above 0.80."
    )

else:

    print(
        " Potentially suspicious features:"
    )

    print(
        potential_leakage
    )


# ============================================================
# 18. SUMMARY REPORT
# ============================================================

print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)


print(
    f"\nRows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Features analyzed: {len(FEATURES)}"
)

print(
    f"Duplicate rows: {duplicate_rows:,}"
)

print(
    f"Missing values: {total_missing:,}"
)

print(
    f"Infinite values: {total_infinite:,}"
)

print(
    f"Constant features: "
    f"{len(constant_features)}"
)

print(
    f"Highly correlated pairs: "
    f"{len(high_correlation_report)}"
)

print(
    f"Potential leakage features: "
    f"{len(potential_leakage)}"
)


# ============================================================
# 19. FINAL OUTPUTS
# ============================================================

print("\n" + "=" * 80)
print("DATA AUDIT COMPLETE")
print("=" * 80)


print(
    f"\nAll audit files saved to:\n"
    f"{OUTPUT_DIR}"
)


print(
    """
Generated files:

1. missing_value_report.csv
2. infinite_value_report.csv
3. feature_statistics.csv
4. outlier_report.csv
5. feature_correlation_matrix.csv
6. high_correlation_features.csv
7. churn_feature_comparison.csv
8. feature_target_correlation.csv
9. feature_correlation_heatmap.png
"""
)


print("\n" + "=" * 80)