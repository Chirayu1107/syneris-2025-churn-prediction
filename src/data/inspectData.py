from pathlib import Path
import pyarrow.parquet as pq

Data_dir=Path("data/raw")
for file in Data_dir.glob("*.parquet"):
    par_file=pq.ParquetFile(file)

    size_gb=file.stat().st_size/(1024**3)
    print(f"{file.name:35} {size_gb:.2f} GB \t {par_file.metadata.num_rows} \t{par_file.num_row_groups}")

    for column in par_file.schema.names:
        print(f" - {column}")