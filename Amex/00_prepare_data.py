# 00_prepare_data.py
import pandas as pd
from pathlib import Path

RAW = Path("data_raw")
PROC = Path("data_proc"); PROC.mkdir(exist_ok=True, parents=True)

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

if __name__ == "__main__":
    # 读入三份原始CSV（按你的文件名放在 data_raw/ 下）
    train_data  = pd.read_csv(RAW/"train_data.csv")
    train_label = pd.read_csv(RAW/"train_labels.csv")
    test_data   = pd.read_csv(RAW/"test_data.csv")

    # 重要：转日期（若已是字符串也可，但建议转为 datetime）
    if "S_2" in train_data.columns:
        train_data["S_2"] = pd.to_datetime(train_data["S_2"])
    if "S_2" in test_data.columns:
        test_data["S_2"] = pd.to_datetime(test_data["S_2"])

    # 合并标签（假设 train_label = [customer_ID, target]）
    train = train_data.merge(train_label, on="customer_ID", how="left")
    # 类型压缩
    train = downcast_df(train)
    test  = downcast_df(test_data)

    # 存成 Parquet 供下游快速读取
    train.to_parquet(PROC/"train.parquet", index=False)
    test.to_parquet(PROC/"test.parquet",  index=False)

    print("Saved to data_proc/train.parquet and data_proc/test.parquet")
