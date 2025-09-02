import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data_raw")
PROC = Path("data_proc")
PROC.mkdir(exist_ok=True, parents=True)

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def convert_csv_to_parquet(name):
    df = pd.read_csv(RAW/f"{name}.csv")
    df = downcast_df(df)
    # S_2 是日期列：转为 datetime 或保留原始字符串都可
    if "S_2" in df.columns:
        df["S_2"] = pd.to_datetime(df["S_2"])
    df.to_parquet(PROC/f"{name}.parquet", index=False)

if __name__ == "__main__":
    # 根据你下载的官方文件名改这两个
    convert_csv_to_parquet("train_data")
    convert_csv_to_parquet("test_data")
