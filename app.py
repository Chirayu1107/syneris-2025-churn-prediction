import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_PATH = (
    "data/processed/customer_churn_predictions.parquet"
)

FEATURES_PATH = (
    "data/processed/combined_purchase_cart_search_features.parquet"
)

MODEL_PATH = (
    "models/churn_xgboost_final.joblib"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_predictions():

    df = pd.read_parquet(PREDICTIONS_PATH)

    return df


@st.cache_data
def load_features():

    df = pd.read_parquet(FEATURES_PATH)

    return df


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


predictions = load_predictions()
features = load_features()
model_package = load_model()


# ============================================================
# MERGE DATA
# ============================================================

@st.cache_data
def prepare_customer_data(predictions, features):

    feature_columns = [
        "client_id",
        "churn",
        "purchase_count",
        "distinct_purchase_days",
        "unique_product_purchased",
        "days_since_last_purchase",
        "avg_days_between_purchases",
        "purchases_last_30d",
        "purchases_previous_30d",
        "purchase_trend_ratio",
        "add_to_cart_count",
        "distinct_add_to_cart_days",
        "days_since_last_add_to_cart",
        "avg_days_between_adds",
        "adds_last_30d",
        "remove_from_cart_count",
        "days_since_last_remove",
        "remove_to_add_ratio",
        "net_cart_activity",
        "search_count",
        "distinct_search_days",
        "days_since_last_search",
        "searches_last_30d",
        "searches_previous_30d",
        "search_trend_ratio"
    ]

    available_columns = [
        column
        for column in feature_columns
        if column in features.columns
    ]

    feature_data = features[available_columns].copy()

    merged = predictions.merge(
        feature_data,
        on="client_id",
        how="left"
    )

    return merged


customers = prepare_customer_data(
    predictions,
    features
)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk(probability):

    if probability >= 0.70:
        return "HIGH"

    elif probability >= 0.38:
        return "MEDIUM"

    else:
        return "LOW"


customers["risk"] = customers[
    "churn_probability"
].apply(get_risk)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Churn Intelligence")

st.sidebar.markdown(
    "Customer churn prediction and risk analysis"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Customer Search",
        "Risk Analysis"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("Churn Intelligence Dashboard")

    st.markdown(
        "ML-powered customer churn prediction dashboard"
    )

    st.divider()

    total_customers = len(customers)

    predicted_churn = (
        customers["churn_prediction"] == 1
    ).sum()

    high_risk = (
        customers["risk"] == "HIGH"
    ).sum()

    medium_risk = (
        customers["risk"] == "MEDIUM"
    ).sum()

    low_risk = (
        customers["risk"] == "LOW"
    ).sum()

    average_probability = (
        customers["churn_probability"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Predicted Churn",
            f"{predicted_churn:,}"
        )

    with col3:
        st.metric(
            "High Risk",
            f"{high_risk:,}"
        )

    with col4:
        st.metric(
            "Avg Churn Probability",
            f"{average_probability:.1%}"
        )

    st.divider()

    st.subheader("Risk Distribution")

    risk_counts = (
        customers["risk"]
        .value_counts()
        .reindex(
            ["HIGH", "MEDIUM", "LOW"],
            fill_value=0
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            risk_counts
        )

    with col2:

        st.dataframe(
            pd.DataFrame(
                {
                    "Risk Level": risk_counts.index,
                    "Customers": risk_counts.values,
                    "Percentage": (
                        risk_counts.values /
                        total_customers *
                        100
                    ).round(2)
                }
            ),
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    st.subheader("Highest Risk Customers")

    top_customers = (
        customers[
            [
                "client_id",
                "churn_probability",
                "risk"
            ]
        ]
        .sort_values(
            "churn_probability",
            ascending=False
        )
        .head(20)
    )

    top_customers = top_customers.copy()

    top_customers["churn_probability"] = (
        top_customers["churn_probability"]
        .map(lambda x: f"{x:.2%}")
    )

    st.dataframe(
        top_customers,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# CUSTOMER SEARCH
# ============================================================

elif page == "Customer Search":

    st.title("Customer Search")

    st.write(
        "Search for an individual customer using client ID."
    )

    client_id = st.number_input(
        "Enter Client ID",
        min_value=0,
        step=1
    )

    if st.button("Search Customer"):

        customer = customers[
            customers["client_id"] == client_id
        ]

        if customer.empty:

            st.error(
                "Customer ID not found."
            )

        else:

            customer = customer.iloc[0]

            probability = (
                customer["churn_probability"]
            )

            prediction = (
                "CHURN"
                if customer["churn_prediction"] == 1
                else "RETAINED"
            )

            risk = customer["risk"]

            st.divider()

            st.subheader(
                f"Customer {int(client_id)}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}"
                )

            with col2:

                st.metric(
                    "Prediction",
                    prediction
                )

            with col3:

                st.metric(
                    "Risk Level",
                    risk
                )

            st.divider()

            st.subheader(
                "Customer Behavior"
            )

            behavior_columns = [
                "purchase_count",
                "distinct_purchase_days",
                "unique_product_purchased",
                "days_since_last_purchase",
                "avg_days_between_purchases",
                "purchases_last_30d",
                "add_to_cart_count",
                "distinct_add_to_cart_days",
                "days_since_last_add_to_cart",
                "adds_last_30d",
                "remove_from_cart_count",
                "days_since_last_remove",
                "search_count",
                "distinct_search_days",
                "days_since_last_search",
                "searches_last_30d"
            ]

            behavior_data = {}

            for column in behavior_columns:

                if column in customer.index:

                    value = customer[column]

                    if pd.isna(value):

                        value = "N/A"

                    elif isinstance(
                        value,
                        float
                    ):

                        value = round(
                            value,
                            3
                        )

                    behavior_data[column] = value

            behavior_df = pd.DataFrame(
                behavior_data.items(),
                columns=[
                    "Feature",
                    "Value"
                ]
            )

            st.dataframe(
                behavior_df,
                hide_index=True,
                use_container_width=True
            )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.title("Risk Analysis")

    st.subheader(
    "Churn Probability Distribution"
)

# Create probability ranges
probability_bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

probability_labels = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%"
]

probability_distribution = pd.cut(
    customers["churn_probability"],
    bins=probability_bins,
    labels=probability_labels,
    include_lowest=True
)

probability_counts = (
    probability_distribution
    .value_counts()
    .reindex(
        probability_labels,
        fill_value=0
    )
)

probability_chart = pd.DataFrame(
    {
        "Customers": probability_counts.values
    },
    index=probability_labels
)

probability_chart.index.name = "Probability Range"

st.bar_chart(
    probability_chart
)

st.divider()

st.subheader(
        "High-Risk Customers"
    )

high_risk_customers = (
        customers[
            customers["risk"] == "HIGH"
        ]
        .sort_values(
            "churn_probability",
            ascending=False
        )
    )

display_columns = [
        "client_id",
        "churn_probability",
        "churn_prediction",
        "purchase_count",
        "days_since_last_purchase",
        "add_to_cart_count",
        "days_since_last_add_to_cart",
        "search_count",
        "days_since_last_search"
    ]

display_columns = [
        column
        for column in display_columns
        if column in high_risk_customers.columns
    ]

st.dataframe(
        high_risk_customers[
            display_columns
        ].head(100),
        hide_index=True,
        use_container_width=True
    )

st.write(
        "Showing top 100 high-risk customers."
    )