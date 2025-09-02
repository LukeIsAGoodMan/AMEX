import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path("data_proc")
FEAT = Path("features")
FEAT.mkdir(exist_ok=True, parents=True)

CAT_COLS = set(["B_30","B_38","D_63","D_64","D_66","D_68","D_114","D_116","D_117","D_120","D_126"])  # 可再补充
EXCLUDE = set(["customer_ID","S_2","target"])

def compute_slope(x: np.ndarray) -> float:
    # x 是按时间排序的一列数值，缺失先填充为上一个观测或均值
    if len(x) <= 1:
        return 0.0
    idx = np.arange(len(x))
    # 简单缺失处理（V1）：前向填充再均值补
    s = pd.Series(x).ffill().fillna(pd.Series(x).mean()).values
    slope = np.polyfit(idx, s, 1)[0]
    return slope

def aggregate_customer(df: pd.DataFrame, is_train=True):
    cols = [c for c in df.columns if c not in EXCLUDE]
    num_cols = [c for c in cols if c not in CAT_COLS]
    cat_cols = [c for c in cols if c in CAT_COLS]

    # 先按时间排序，便于取 last 和计算斜率
    df = df.sort_values(["customer_ID","S_2"])

    # 数值聚合
    aggs = {}
    for fun in ["mean","std","min","max"]:
        aggs.update({c:fun for c in num_cols})
    g_num = df.groupby("customer_ID").agg(aggs)
    # 改列名
    g_num.columns = [f"{c[0]}__{c[1]}" for c in g_num.columns.to_flat_index()]

    # last
    g_last = df.groupby("customer_ID")[num_cols].tail(1).set_index("customer_ID", drop=True) if "customer_ID" in df.columns else None
    # 更通用写法：
    g_last = df.groupby("customer_ID")[num_cols].last()
    g_last.columns = [f"{c}__last" for c in g_last.columns]

    # slope（对每个数值列逐客户计算）
    def slope_block(sub: pd.DataFrame):
        out = {}
        for c in num_cols:
            out[f"{c}__slope"] = compute_slope(sub[c].values)
        return pd.Series(out)

    g_slope = df.groupby("customer_ID").apply(slope_block)

    # 类别聚合：last / nunique / top1_freq
    g_cat_last = df.groupby("customer_ID")[cat_cols].last()
    g_cat_last.columns = [f"{c}__last" for c in g_cat_last.columns]

    g_cat_nu = df.groupby("customer_ID")[cat_cols].nunique()
    g_cat_nu.columns = [f"{c}__nuniq" for c in g_cat_nu.columns]

    def top1freq_block(sub: pd.DataFrame):
        out = {}
        for c in cat_cols:
            vc = sub[c].value_counts(dropna=True)
            top1 = (vc.iloc[0] / vc.sum()) if len(vc) else 0.0
            out[f"{c}__top1freq"] = top1
        return pd.Series(out)
    g_cat_top1 = df.groupby("customer_ID").apply(top1freq_block)

    # 合并
    g_all = pd.concat([g_num, g_last, g_slope, g_cat_last, g_cat_nu, g_cat_top1], axis=1)
    # 可选：去掉全空或方差为0的列
    nunique = g_all.nunique()
    keep = nunique[nunique > 1].index
    g_all = g_all[keep]

    if is_train and "target" in df.columns:
        y = df.groupby("customer_ID")["target"].first()
        g_all["target"] = y
    return g_all

if __name__ == "__main__":
    train = pd.read_parquet(PROC/"train_data.parquet")
    test  = pd.read_parquet(PROC/"test_data.parquet")

    train_agg = aggregate_customer(train, is_train=True)
    test_agg  = aggregate_customer(test,  is_train=False)

    train_agg.to_parquet(FEAT/"train_customer_v1.parquet")
    test_agg.to_parquet(FEAT/"test_customer_v1.parquet")
