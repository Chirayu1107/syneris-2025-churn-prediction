from pathlib import Path
import pandas as pd

Data_dir = Path("data/raw")

print("=" * 70)
print("SEARCH QUERY SCHEMA CHECK")
print("=" * 70)

path = Data_dir / "search_query.parquet"

print("\nLoading search_query.parquet...")

search = pd.read_parquet(path)

print("\nRows:", len(search))
print("Columns:", len(search.columns))

print("\nColumn names:")

for column in search.columns:
    print(" -", column)

print("\nData types:")

print(search.dtypes)

print("\nFirst 10 rows:")

print(search.head(10))

print("\nMissing values:")

print(search.isnull().sum())

print("\nUnique values:")

for column in search.columns:

    if search[column].dtype == "object":

        print(
            f"{column}:",
            search[column].nunique()
        )

print("\n" + "=" * 70)
print("SCHEMA CHECK COMPLETE")
print("=" * 70)