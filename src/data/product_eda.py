from pathlib import Path
import pandas as pd

data_dir=Path("data/raw")

products= pd.read_parquet(data_dir/"product_properties.parquet")

print("Rows:",len(products))
print("Columns:", products.columns.tolist())

print("\nData types:\n",products.dtypes)
print("\nMissing Values:\n",products.isna().sum())

print("\nUnique SKUs:",products["sku"].nunique())
print("Unique categories:", products["category"].nunique())
print("Unique Price:", products["price"].nunique())

print("\nSample:")
print(products.head())