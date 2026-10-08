from pathlib import Path
import pandas as pd
import numpy as np

#configuration
Data_dir=Path("data/raw")
output_dir=Path("data/processed")

cutoff_date=pd.Timestamp("2022-10-09 00:09:15")

#create output directory
output_dir.mkdir(parents=True,exist_ok=True)

#load purchase data

print("="*70)
print("Purchase feature Engineering")
print("="*70)

print("\nLoading purchase data...")

buy=pd.read_parquet(Data_dir/"product_buy.parquet")

buy["timestamp"] = pd.to_datetime(buy["timestamp"])
print("Total purchase events:",len(buy))

#Keep only history data
historical=buy[buy["timestamp"]<=cutoff_date].copy()
print("Historical purchase events:",len(historical))

#add purchase date
historical["purchase_date"]=(historical["timestamp"].dt.date)

#Basic customer features
print("\nCreating basic purchase features...")

customer_features=(
    historical.groupby("client_id").agg(
        purchase_count=("timestamp","count"),
        distinct_purchase_days=("purchase_date","nunique"),
        unique_product_purchased=("sku","nunique"),
        first_purchase=("timestamp","min"),
        last_purchase=("timestamp","max"),
    ).reset_index()
)

#Recency features
print("\nCreating recency features...")

customer_features["days_since_last_purchase"]=(
    (cutoff_date-customer_features["last_purchase"]).dt.total_seconds()/(60*60*24)
)

customer_features["days_since_first_purchase"]=(
    (cutoff_date-customer_features["first_purchase"]).dt.total_seconds()/(60*60*24)
)

#Average purchase gap
print("\nCreating purchase gap feature...")

historical= historical.sort_values(
    ["client_id","timestamp"]
)

historical["purchase_gap_days"]=(
    historical.groupby("client_id")["timestamp"].diff().dt.total_seconds()/(60*60*24)
)

average_gap=(
    historical.groupby("client_id")["purchase_gap_days"].mean().rename("avg_days_between_purchases")
)

customer_features=customer_features.merge(average_gap,on="client_id",how="left")

#Recent purchase counts
print("\nCreating recent activity features...")

def purchase_in_period(days):
    start_date=(cutoff_date-pd.Timedelta(days=days))
    mask=(
        (historical["timestamp"]>start_date) &
        (historical["timestamp"]<=cutoff_date)
    )
    counts=(historical.loc[mask].groupby("client_id").size().rename(f"purchases_last_{days}d"))
    return counts

for days in [7,14,30]:
    counts=purchase_in_period(days)
    customer_features=customer_features.merge(counts,on="client_id",how="left")

#Previous 30-days period
print("Creating purchase trend features...")

recent_start=(cutoff_date-pd.Timedelta(days=30))
previous_start=(cutoff_date-pd.Timedelta(days=60))
previous_mask=(
    (historical["timestamp"]>previous_start) &
    (historical["timestamp"]<=recent_start)
)
previous_counts=(historical.loc[previous_mask].groupby("client_id").size().rename("purchases_previous_30d"))
customer_features=customer_features.merge(previous_counts,on="client_id",how="left")

#Filling recent activity

activity_columns=[
    "purchases_last_7d",
    "purchases_last_14d",
    "purchases_last_30d",
    "purchases_previous_30d"
]

for column in activity_columns:
    customer_features[column]=customer_features[column].fillna(0).astype(int)

#Purchase Trend
customer_features["purchase_trend_ratio"]=(
    customer_features["purchases_last_30d"]/customer_features["purchases_previous_30d"].replace(0,np.nan)
) # if their is no previous purchase, then setting it NaN instead of zero.

#Purchase Frequency
customer_features["purchase_per_active_day"]=(
    customer_features["purchase_count"]/customer_features["distinct_purchase_days"]
)

#customer lifetime activity
customer_features["purchase_per_lifetime_day"]=(
    customer_features["purchase_count"]/customer_features["days_since_first_purchase"].clip(lower=1)
)

#Keeping only modeling population
print("\nFiltering modeling population")

eligible_customers=(
    customer_features[customer_features["distinct_purchase_days"]>=2].copy()
)

print("Eligible customers:",len(eligible_customers))

#Load churn labels
print("Loading churn labels...")
labels=pd.read_parquet(Data_dir/"churn_labels.parquet")

#Creating final churn label due previous was for >= purchase-event
print("Creating final churn labels...")

Label_end=(cutoff_date+pd.Timedelta(days=60))
future=buy[(buy["timestamp"]>cutoff_date)&(buy["timestamp"]<=Label_end)]
future_clients=set(future["client_id"].unique())

eligible_customers["churn"]=(
    ~eligible_customers["client_id"].isin(future_clients)
).astype(int)

#Remove raw timestamp columns
customer_features=eligible_customers.drop(columns=["first_purchase","last_purchase"])

#Dataset
print("\n"+"="*80)
print("Feature Dataset Summary")

print("\nRows:",len(customer_features))
print("Columns:",len(customer_features.columns))

print("\nColumns:")
for column in customer_features.columns:
    print(" -",column)

print("\nChurn distribution:")
print(customer_features["churn"].value_counts())

print("\nMissing Values:")
print(customer_features.isnull().sum())

print("\nDuplicate Values:")
print(customer_features["client_id"].duplicated().sum())

#Saves
output_path=output_dir/"purchase_features.parquet"

customer_features.to_parquet(output_path,index=False)
print("\nSaved:")
print(output_path)
print("\nPurchase feature engineering complete.")