from pathlib import Path
import pandas as pd

Data_dir=Path("data/raw")

FILES=[
    "product_buy.parquet",
    "add_to_cart.parquet",
    "remove_from_cart.parquet",
]

for filename in FILES:
    path = Data_dir/filename

    print("\n"+"="*80)
    print(filename)

    df=pd.read_parquet(path)

    print("Rows:",len(df))
    print("Unique Clients:", df["client_id"].nunique())
    print("Unique products:", df["sku"].nunique())

    df["timestamp"]=pd.to_datetime(df["timestamp"])

    print("start:", df["timestamp"].min())
    print("end:",df["timestamp"].max())

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())