# 10_make_features.py
# Purpose: Aggregate row-level data to the customer level and output train/test customer-level feature tables (V1 final version)
# Features:
#  - Fix: correct multi-function aggregation → prevents duplicate column names and parquet save errors
#  - Performance: vectorized top1freq; optional FAST mode; column selection; type downcasting; sampling
#  - Progress: stage logging, simple progress display, runtime/memory usage reporting

import os
import time
import gc
import psutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# -------- 配置与常量 --------
PROC = Path("data_proc")
FEAT = Path("features"); FEAT.mkdir(exist_ok=True, parents=True)

# 社区共识的类别白名单（V1 兜底）
CAT_WHITELIST = {
    "B_30","B_38",
    "D_63","D_64","D_66","D_68",
    "D_114","D_116","D_117","D_120","D_126"
}
EXCLUDE = {"customer_ID","S_2","target"}

# -------- 小工具 --------
def tprint(msg: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def mem_gb() -> float:
    return psutil.virtual_memory().used / (1024**3)

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    # 降精度：float64->float32, int64->int32
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        if c not in ("customer_ID",):  # 防止ID列被意外降为整数产生问题
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

def detect_categorical_columns(df: pd.DataFrame, max_unique:int=20, min_non_null:int=1000):
    # 自动识别：唯一值 <= max_unique 且 非空数量 >= min_non_null
    cats = []
    for c in df.columns:
        if c in EXCLUDE:
            continue
        nun = df[c].nunique(dropna=True)
        if nun == 0:
            continue
        nnz = df[c].notna().sum()
        if nun <= max_unique and nnz >= min_non_null:
            cats.append(c)
    return set(cats)

def split_num_cat_columns(df: pd.DataFrame):
    auto_cats = detect_categorical_columns(df)
    cat_cols = (auto_cats | CAT_WHITELIST) - EXCLUDE
    num_cols = [c for c in df.columns if c not in cat_cols and c not in EXCLUDE]
    return num_cols, list(cat_cols)

def compute_slope_by_days(sub_series: pd.Series, dates: pd.Series) -> float:
    # 用真实天数作为自变量，更稳健；仅在 --use-slope 打开时计算
    mask = sub_series.notna() & dates.notna()
    y = sub_series[mask].astype(float)
    if len(y) <= 1:
        return 0.0
    t = (dates[mask] - dates[mask].iloc[0]).dt.days.astype(float)
    if t.var() == 0:
        return 0.0
    t_c = t - t.mean()
    y_c = y - y.mean()
    beta = (t_c * y_c).sum() / (t_c**2).sum()
    return float(beta)

def aggregate_customer(df: pd.DataFrame,
                       use_slope: bool = False,
                       fast_mode: bool = False,
                       min_non_null_ratio: float = 0.7) -> pd.DataFrame:
    """
    将行级流水聚合到客户级。
    - 数值列：mean/std/min/max + last (+ 可选 slope)
    - 类别列：last/top1freq (+ 可选 nunique)
    - fast_mode=True: 数值只保留 mean/std/last；类别只保留 last/top1freq
    """

    tprint("Sorting by customer_ID, S_2 ...")
    df = df.sort_values(["customer_ID", "S_2"]).reset_index(drop=True)

    # 列划分
    num_cols, cat_cols = split_num_cat_columns(df)
    tprint(f"Detected numeric cols: {len(num_cols)}, categorical cols: {len(cat_cols)}")

    # -------- 数值聚合：mean/std/(min/max 可选) --------
    tprint("Aggregating numeric statistics ...")
    if fast_mode:
        aggs = {c: ["mean", "std"] for c in num_cols}
    else:
        aggs = {c: ["mean", "std", "min", "max"] for c in num_cols}

    if len(aggs) > 0:
        g_num = df.groupby("customer_ID").agg(aggs)
        # 扁平化列名（修复：确保使用 MultiIndex 正确展开）
        g_num.columns = [f"{col}__{stat}" for col, stat in g_num.columns.to_flat_index()]
    else:
        g_num = pd.DataFrame(index=df["customer_ID"].drop_duplicates())

    # -------- 数值：last --------
    tprint("Aggregating numeric last ...")
    if len(num_cols) > 0:
        g_last_num = df.groupby("customer_ID", sort=False)[num_cols].last()
        g_last_num.columns = [f"{c}__last" for c in g_last_num.columns]
    else:
        g_last_num = pd.DataFrame(index=g_num.index)

    # -------- 数值：slope（可选，较慢） --------
    if use_slope and len(num_cols) > 0:
        tprint("Computing numeric slopes (this may take a while) ...")
        # 仅选择必要列，减少传输体积
        need = ["customer_ID", "S_2"] + num_cols
        df_need = df[need]
        # apply 子函数：对每个客户计算所有数值列的 slope
        def slope_block(sub: pd.DataFrame):
            out = {}
            dates = sub["S_2"]
            for c in num_cols:
                out[f"{c}__slope"] = compute_slope_by_days(sub[c], dates)
            return pd.Series(out, dtype="float32")
        g_slope = (
            df_need.groupby("customer_ID", include_groups=False, sort=False)
                  .apply(slope_block)
        )
    else:
        g_slope = pd.DataFrame(index=g_num.index)

    gc.collect()

    # -------- 类别：last --------
    tprint("Aggregating categorical last ...")
    if len(cat_cols) > 0:
        g_cat_last = df.groupby("customer_ID", sort=False)[cat_cols].last()
        g_cat_last.columns = [f"{c}__last" for c in g_cat_last.columns]
    else:
        g_cat_last = pd.DataFrame(index=g_num.index)

    # -------- 类别：top1freq（向量化） --------
    tprint("Computing categorical top1freq ...")
    if len(cat_cols) > 0:
        # 对每个客户、每个类别列：取 value_counts(normalize=True) 的最大值
        # 通过 agg(lambda s: ...) 向量化，一次性完成
        g_cat_top1 = df.groupby("customer_ID", sort=False)[cat_cols].agg(
            lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 0.0
        )
        g_cat_top1.columns = [f"{c}__top1freq" for c in g_cat_top1.columns]
    else:
        g_cat_top1 = pd.DataFrame(index=g_num.index)

    # （非 fast 模式可加上类别 nunique）
    if not fast_mode and len(cat_cols) > 0:
        tprint("Computing categorical nunique ...")
        g_cat_nu = df.groupby("customer_ID", sort=False)[cat_cols].nunique(dropna=True)
        g_cat_nu.columns = [f"{c}__nuniq" for c in g_cat_nu.columns]
    else:
        g_cat_nu = pd.DataFrame(index=g_num.index)

    # -------- 合并所有块 --------
    tprint("Concatenating all feature blocks ...")
    parts = [x for x in [g_num, g_last_num, g_slope, g_cat_last, g_cat_top1, g_cat_nu] if len(x) > 0]
    g_all = pd.concat(parts, axis=1)

    # -------- 去常数列 / 非空率过滤 / 降精度 --------
    tprint("Removing constant / sparse columns and downcasting ...")
    nun = g_all.nunique()
    g_all = g_all.loc[:, nun > 1]  # 去常数列

    non_null_ratio = 1.0 - (g_all.isna().sum() / len(g_all))
    g_all = g_all.loc[:, non_null_ratio >= min_non_null_ratio]

    g_all = downcast_df(g_all)

    # -------- 回填 target（仅训练集时）--------
    if "target" in df.columns:
        y = df.groupby("customer_ID", sort=False)["target"].first().astype("int8")
        g_all["target"] = y

    tprint(f"Final features shape: {g_all.shape} | mem used ~ {mem_gb():.1f} GB")
    return g_all

def maybe_sample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    tprint(f"DEBUG_FRAC enabled -> sampling {frac:.2%} customers ...")
    cust = df["customer_ID"].drop_duplicates()
    sample_ids = cust.sample(frac=frac, random_state=42)
    out = df[df["customer_ID"].isin(sample_ids)]
    tprint(f"Sampled rows: {len(out):,} / {len(df):,}")
    return out

# -------- 主入口 --------
def main():
    parser = argparse.ArgumentParser(description="Aggregate row-level AMEX data to customer-level features (V1 final).")
    parser.add_argument("--debug-frac", type=float, default=float(os.getenv("DEBUG_FRAC", "1.0")),
                        help="按客户抽样比例（默认1.0=全量；调试可设0.05/0.1）")
    parser.add_argument("--fast", action="store_true", help="极速模式：数值仅 mean/std/last；类别仅 last/top1freq（默认关）")
    parser.add_argument("--use-slope", action="store_true", help="是否计算 slope（较慢，默认关）")
    parser.add_argument("--min-non-null", type=float, default=0.7, help="最小非空率阈值（默认0.7）")
    args = parser.parse_args()

    start_time = time.time()
    tprint(f"Args: debug_frac={args.debug_frac} fast={args.fast} use_slope={args.use_slope} min_non_null={args.min_non_null}")
    tprint(f"Starting... current mem used ~ {mem_gb():.1f} GB")

    # 读取 parquet
    train_pq = PROC/"train.parquet"
    test_pq  = PROC/"test.parquet"
    if not train_pq.exists() or not test_pq.exists():
        raise FileNotFoundError("Missing data_proc/train.parquet or data_proc/test.parquet. Run 00_prepare_data.py first.")

    tprint("Loading train.parquet ...")
    train = pd.read_parquet(train_pq)
    tprint(f"Train rows: {len(train):,} | mem ~ {mem_gb():.1f} GB")

    tprint("Loading test.parquet ...")
    test = pd.read_parquet(test_pq)
    tprint(f"Test rows: {len(test):,} | mem ~ {mem_gb():.1f} GB")

    # 小样本抽样（按客户）
    if args.debug_frac < 1.0:
        train = maybe_sample(train, args.debug_frac)
        test  = maybe_sample(test,  args.debug_frac)

    # 类型降精度（先做一遍，减内存）
    tprint("Downcasting dtypes ...")
    for df in (train, test):
        if "S_2" in df.columns and not np.issubdtype(df["S_2"].dtype, np.datetime64):
            df["S_2"] = pd.to_datetime(df["S_2"])
        downcast_df(df)
    tprint(f"After downcast mem ~ {mem_gb():.1f} GB")

    # 训练集聚合
    tprint("Aggregating TRAIN features ...")
    train_agg = aggregate_customer(train,
                                   use_slope=args.use_slope,
                                   fast_mode=args.fast,
                                   min_non_null_ratio=args.min_non_null)
    tprint("Saving TRAIN features ...")
    train_agg.to_parquet(FEAT/"train_customer_v1.parquet")

    del train; gc.collect()

    # 测试集聚合
    tprint("Aggregating TEST features ...")
    test_agg = aggregate_customer(test,
                                  use_slope=args.use_slope,
                                  fast_mode=args.fast,
                                  min_non_null_ratio=args.min_non_null)
    tprint("Saving TEST features ...")
    test_agg.to_parquet(FEAT/"test_customer_v1.parquet")

    # 输出列名清单（用于推理时对齐）
    tprint("Writing features_list.txt ...")
    feat_cols = [c for c in train_agg.columns if c != "target"]
    with open("features_list.txt", "w", encoding="utf-8") as f:
        for c in feat_cols:
            f.write(f"{c}\n")

    tprint(f"Done. Total time: {(time.time()-start_time)/60:.1f} min | mem ~ {mem_gb():.1f} GB")

if __name__ == "__main__":
    main()
