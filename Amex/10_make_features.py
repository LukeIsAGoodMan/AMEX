# 10_make_features.py
# Purpose: Aggregate row-level AMEX data to customer-level features (V1.1).
# Add minimal, targeted time-series features (trend block) on selected columns,
# keep stable, memory-aware implementation compatible with pandas 2.3+.
#
# Key features:
# - Numeric: mean/std/(min/max optional via --fast off) + last (+ optional slope)
# - Categorical: last, top1freq (+ nunique if --fast off)
# - Trend block (selected columns only): prev_last, delta, delta_pct, rolling_mean_3, rolling_std_3, ewm3, slope(optional)
# - Flags: --debug-frac, --fast, --use-slope, --trend, --trend-cols-file
# - Progress logs, memory printouts, dtype downcasting
#
# Example runs:
#   # Quick smoke (10% + FAST) to validate pipeline end-to-end
#   python 10_make_features.py --debug-frac 0.1 --fast
#
#   # Full run (no slope by default)
#   python 10_make_features.py
#
#   # Full run with slope on selected TREND_COLS (slower)
#   python 10_make_features.py --use-slope
#
# Outputs:
#   features/train_customer_v1.parquet
#   features/test_customer_v1.parquet
#   features_list.txt  (feature order for training/inference alignment)

import os
import gc
import time
import psutil
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ---------------- Paths ----------------
PROC = Path("data_proc")
FEAT = Path("features"); FEAT.mkdir(exist_ok=True, parents=True)


# ---------------- Defaults ----------------
# Categorical whitelist used widely in the AMEX competition community.
CAT_WHITELIST = {
    "B_30","B_38",
    "D_63","D_64","D_66","D_68",
    "D_114","D_116","D_117","D_120","D_126"
}
EXCLUDE = {"customer_ID","S_2","target"}

# Selected columns for trend features. Keep this list short (15–25) to control memory/time.
DEFAULT_TREND_COLS = [
    "P_2","B_1","B_2","B_9","R_1",
    "D_39","D_41","D_44","D_48","B_11",
    "B_3","R_2","B_4","S_3","P_3",
    "B_5","R_3","D_50","B_6","R_4"
]


# ---------------- Logging helpers ----------------
def tprint(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def mem_gb() -> float:
    return psutil.virtual_memory().used / (1024**3)


# ---------------- Dtype helpers ----------------
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64->float32, int64->int32 to reduce memory footprint."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        if c != "customer_ID":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


# ---------------- Column detection ----------------
def detect_categorical_columns(df: pd.DataFrame, max_unique: int = 20, min_non_null: int = 1000) -> set:
    """Heuristic: low-unique, sufficiently non-null columns as categorical candidates."""
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
    """Split columns into numeric vs categorical with a whitelist override."""
    auto_cats = detect_categorical_columns(df)
    cat_cols = (auto_cats | CAT_WHITELIST) - EXCLUDE
    num_cols = [c for c in df.columns if c not in cat_cols and c not in EXCLUDE]
    return num_cols, list(cat_cols)


# ---------------- Trend features (V1.1) ----------------
def build_trend_features(df: pd.DataFrame, cols: List[str], compute_slope: bool = False) -> pd.DataFrame:
    """
    Create trend features per customer for selected numeric columns:
      - prev_last: last value from previous row (lag1 at row level), aggregated by 'last'
      - delta: last - prev_last (row level), aggregated by 'last'
      - delta_pct: (last - prev_last) / (abs(prev_last)+eps), aggregated by 'last'
      - rolling_mean_3 / rolling_std_3: 3-step rolling stats (row level), aggregated by 'last'
      - ewm3: exponentially weighted mean (alpha=0.6), aggregated by 'last'
      - slope: least squares slope over time (days) for selected base columns (optional)
    Returns a customer-level DataFrame with all derived features.
    Notes:
      - This function assumes df contains ["customer_ID","S_2"] + selected base columns.
      - It is intentionally limited to a small set of columns to control memory/time cost.
    """
    eps = 1e-6
    keep_cols = ["customer_ID", "S_2"] + [c for c in cols if c in df.columns]
    tmp = df[keep_cols].copy()
    tmp = tmp.sort_values(["customer_ID", "S_2"]).reset_index(drop=True)

    g = tmp.groupby("customer_ID", sort=False, group_keys=False)

    # Row-level derivations, then we'll aggregate "last" per customer.
    for c in cols:
        if c not in tmp.columns:
            continue

        # lag1 -> prev_last at row level
        tmp[f"{c}__prev"] = g[c].shift(1)

        # deltas at row level
        tmp[f"{c}__delta"] = tmp[c] - tmp[f"{c}__prev"]
        tmp[f"{c}__delta_pct"] = tmp[f"{c}__delta"] / (tmp[f"{c}__prev"].abs() + eps)

        # rolling(3) stats at row level
        # Note: groupby.rolling yields a MultiIndex; reset_index(drop=True) aligns back to row order.
        tmp[f"{c}__roll3_mean"] = g[c].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        tmp[f"{c}__roll3_std"]  = g[c].rolling(3, min_periods=1).std().reset_index(level=0, drop=True)

        # EWMA (recency emphasis), alpha ~ 0.6 for last-3 emphasis
        tmp[f"{c}__ewm3"] = g[c].apply(lambda s: s.ewm(alpha=0.6, adjust=False).mean())

    # Aggregate only derived columns (exclude base columns) by 'last'
    DERIV_SUFFIXES = ["__prev", "__delta", "__delta_pct", "__roll3_mean", "__roll3_std", "__ewm3"]
    deriv_cols = [c for c in tmp.columns
                if any(c.endswith(suf) for suf in DERIV_SUFFIXES)]

    if len(deriv_cols) > 0:
        g_last_deriv = g[deriv_cols].last()
        # append '__last' to indicate customer-level aggregation of row-level features
        g_last_deriv.columns = [f"{c}__last" for c in g_last_deriv.columns]
    else:
        # empty frame with correct index
        g_last_deriv = pd.DataFrame(index=tmp["customer_ID"].drop_duplicates())


    # Optional: slope on base columns (few selected only)
    if compute_slope and any(col in tmp.columns for col in cols):
        def slope_block(sub: pd.DataFrame) -> pd.Series:
            out = {}
            dates = sub["S_2"]
            for c in cols:
                if c not in sub.columns:
                    continue
                mask = sub[c].notna() & dates.notna()
                y = sub.loc[mask, c].astype(float)
                if len(y) <= 1:
                    out[f"{c}__slope"] = 0.0
                    continue
                t = (dates[mask] - dates[mask].iloc[0]).dt.days.astype(float)
                if t.var() == 0:
                    out[f"{c}__slope"] = 0.0
                    continue
                t_c = t - t.mean()
                y_c = y - y.mean()
                beta = (t_c * y_c).sum() / (t_c**2).sum()
                out[f"{c}__slope"] = float(beta)
            return pd.Series(out, dtype="float32")

        g_slope = (
            tmp.groupby("customer_ID", include_groups=False, sort=False)
               .apply(slope_block)
        )
    else:
        g_slope = pd.DataFrame(index=g_last_deriv.index)

    out = pd.concat([g_last_deriv, g_slope], axis=1)
    return out


# ---------------- Core aggregation ----------------
def aggregate_customer(df: pd.DataFrame,
                       use_slope_all: bool = False,
                       fast_mode: bool = False,
                       min_non_null_ratio: float = 0.7,
                       enable_trend: bool = True,
                       trend_cols: List[str] = None,
                       trend_slope: bool = False) -> pd.DataFrame:
    """
    Aggregate row-level statements to customer-level table.
    - Numeric stats: mean/std/(min/max if not fast) + last (+ optional slope across ALL numeric columns)
    - Categorical stats: last/top1freq (+ nunique if not fast)
    - Trend block (selected columns): prev_last/delta/delta_pct/roll3_mean/roll3_std/ewm3 (+ optional slope)
    """
    tprint("Sorting by customer_ID, S_2 ...")
    df = df.sort_values(["customer_ID", "S_2"]).reset_index(drop=True)

    # Column split
    num_cols, cat_cols = split_num_cat_columns(df)
    tprint(f"Detected numeric cols: {len(num_cols)}, categorical cols: {len(cat_cols)}")

    # ---------- Numeric aggregations ----------
    tprint("Aggregating numeric statistics ...")
    if fast_mode:
        aggs = {c: ["mean", "std"] for c in num_cols}
    else:
        aggs = {c: ["mean", "std", "min", "max"] for c in num_cols}

    if len(aggs) > 0:
        g_num = df.groupby("customer_ID").agg(aggs)
        g_num.columns = [f"{col}__{stat}" for col, stat in g_num.columns.to_flat_index()]
    else:
        g_num = pd.DataFrame(index=df["customer_ID"].drop_duplicates())

    # Numeric last
    tprint("Aggregating numeric last ...")
    if len(num_cols) > 0:
        g_last_num = df.groupby("customer_ID", sort=False)[num_cols].last()
        g_last_num.columns = [f"{c}__last" for c in g_last_num.columns]
    else:
        g_last_num = pd.DataFrame(index=g_num.index)

    # Optional slope across ALL numeric columns (expensive; disabled by default)
    if use_slope_all and len(num_cols) > 0:
        tprint("Computing numeric slopes for ALL numeric columns (slow) ...")
        need = ["customer_ID", "S_2"] + num_cols
        df_need = df[need]
        def slope_block(sub: pd.DataFrame) -> pd.Series:
            out = {}
            dates = sub["S_2"]
            for c in num_cols:
                mask = sub[c].notna() & dates.notna()
                y = sub.loc[mask, c].astype(float)
                if len(y) <= 1:
                    out[f"{c}__slope"] = 0.0
                    continue
                t = (dates[mask] - dates[mask].iloc[0]).dt.days.astype(float)
                if t.var() == 0:
                    out[f"{c}__slope"] = 0.0
                    continue
                t_c = t - t.mean()
                y_c = y - y.mean()
                beta = (t_c * y_c).sum() / (t_c**2).sum()
                out[f"{c}__slope"] = float(beta)
            return pd.Series(out, dtype="float32")
        g_slope_all = (
            df_need.groupby("customer_ID", include_groups=False, sort=False)
                   .apply(slope_block)
        )
    else:
        g_slope_all = pd.DataFrame(index=g_num.index)

    gc.collect()

    # ---------- Categorical aggregations ----------
    tprint("Aggregating categorical last ...")
    if len(cat_cols) > 0:
        g_cat_last = df.groupby("customer_ID", sort=False)[cat_cols].last()
        g_cat_last.columns = [f"{c}__last" for c in g_cat_last.columns]
    else:
        g_cat_last = pd.DataFrame(index=g_num.index)

    tprint("Computing categorical top1freq ...")
    if len(cat_cols) > 0:
        g_cat_top1 = df.groupby("customer_ID", sort=False)[cat_cols].agg(
            lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 0.0
        )
        g_cat_top1.columns = [f"{c}__top1freq" for c in g_cat_top1.columns]
    else:
        g_cat_top1 = pd.DataFrame(index=g_num.index)

    if not fast_mode and len(cat_cols) > 0:
        tprint("Computing categorical nunique ...")
        g_cat_nu = df.groupby("customer_ID", sort=False)[cat_cols].nunique(dropna=True)
        g_cat_nu.columns = [f"{c}__nuniq" for c in g_cat_nu.columns]
    else:
        g_cat_nu = pd.DataFrame(index=g_num.index)

    # ---------- Trend block (selected numeric columns only) ----------
    if enable_trend:
        tprint("Building trend features for selected columns ...")
        trend_cols = trend_cols or DEFAULT_TREND_COLS
        # Intersect with available numeric base columns
        trend_cols_avail = [c for c in trend_cols if c in num_cols]
        if len(trend_cols_avail) == 0:
            tprint("[WARN] No selected trend columns are available in numeric columns; skipping trend block.")
            g_trend = pd.DataFrame(index=g_num.index)
        else:
            g_trend = build_trend_features(
                df[["customer_ID", "S_2"] + trend_cols_avail],
                cols=trend_cols_avail,
                compute_slope=trend_slope
            )
    else:
        g_trend = pd.DataFrame(index=g_num.index)

    # ---------- Concat all blocks ----------
    tprint("Concatenating all feature blocks ...")
    parts = [x for x in [g_num, g_last_num, g_slope_all, g_cat_last, g_cat_top1, g_cat_nu, g_trend] if len(x) > 0]
    g_all = pd.concat(parts, axis=1)

    # ---------- Drop constant / sparse cols; downcast ----------
    tprint("Removing constant / sparse columns and downcasting ...")
    nun = g_all.nunique()
    g_all = g_all.loc[:, nun > 1]  # remove constant columns

    non_null_ratio = 1.0 - (g_all.isna().sum() / len(g_all))
    g_all = g_all.loc[:, non_null_ratio >= min_non_null_ratio]

    g_all = downcast_df(g_all)

    # ---------- Attach target (train only) ----------
    if "target" in df.columns:
        y = df.groupby("customer_ID", sort=False)["target"].first().astype("int8")
        g_all["target"] = y

    tprint(f"Final features shape: {g_all.shape} | mem used ~ {mem_gb():.1f} GB")
    return g_all


# ---------------- Sampling ----------------
def maybe_sample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    tprint(f"DEBUG_FRAC enabled -> sampling {frac:.2%} customers ...")
    cust = df["customer_ID"].drop_duplicates()
    sample_ids = cust.sample(frac=frac, random_state=42)
    out = df[df["customer_ID"].isin(sample_ids)]
    tprint(f"Sampled rows: {len(out):,} / {len(df):,}")
    return out


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate row-level AMEX data to customer-level features (V1.1).")
    parser.add_argument("--debug-frac", type=float, default=float(os.getenv("DEBUG_FRAC", "1.0")),
                        help="Customer-level sampling ratio (default 1.0 for full data).")
    parser.add_argument("--fast", action="store_true", help="FAST mode: numeric mean/std/last; categorical last/top1freq.")
    parser.add_argument("--use-slope", action="store_true",
                        help="Compute slope across ALL numeric columns (slow; trend block slope is separate).")
    parser.add_argument("--min-non-null", type=float, default=0.7, help="Min non-null ratio to keep a column.")
    parser.add_argument("--trend", action="store_true", default=True,
                        help="Enable trend features block on selected columns (default True).")
    parser.add_argument("--no-trend", dest="trend", action="store_false",
                        help="Disable trend features block.")
    parser.add_argument("--trend-slope", action="store_true",
                        help="Compute slope inside trend block for selected columns only.")
    parser.add_argument("--trend-cols-file", type=str, default="",
                        help="Optional text file containing one base column per line for trend features.")
    args = parser.parse_args()

    start_time = time.time()
    tprint(f"Args: debug_frac={args.debug_frac} fast={args.fast} use_slope_all={args.use_slope} "
           f"trend={args.trend} trend_slope={args.trend_slope} min_non_null={args.min_non_null}")
    tprint(f"Starting... current mem used ~ {mem_gb():.1f} GB")

    # Load preprocessed parquet
    train_pq = PROC / "train.parquet"
    test_pq  = PROC / "test.parquet"
    if not train_pq.exists() or not test_pq.exists():
        raise FileNotFoundError("Missing data_proc/train.parquet or data_proc/test.parquet. Run 00_prepare_data.py first.")

    tprint("Loading train.parquet ...")
    train = pd.read_parquet(train_pq)
    tprint(f"Train rows: {len(train):,} | mem ~ {mem_gb():.1f} GB")

    tprint("Loading test.parquet ...")
    test = pd.read_parquet(test_pq)
    tprint(f"Test rows: {len(test):,} | mem ~ {mem_gb():.1f} GB")

    # Ensure S_2 is datetime
    for df in (train, test):
        if "S_2" in df.columns and not np.issubdtype(df["S_2"].dtype, np.datetime64):
            df["S_2"] = pd.to_datetime(df["S_2"])

    # Optional sampling by customer
    if args.debug_frac < 1.0:
        train = maybe_sample(train, args.debug_frac)
        test  = maybe_sample(test,  args.debug_frac)

    # Downcast early to save memory
    tprint("Downcasting dtypes ...")
    for df in (train, test):
        downcast_df(df)
    tprint(f"After downcast mem ~ {mem_gb():.1f} GB")

    # Optional trend column list from file
    trend_cols = None
    if args.trend and args.trend_cols_file:
        path = Path(args.trend_cols_file)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                trend_cols = [line.strip() for line in f if line.strip()]
            tprint(f"Loaded {len(trend_cols)} trend columns from file: {args.trend_cols_file}")
        else:
            tprint(f"[WARN] trend-cols-file not found: {args.trend_cols_file}. Using default list.")

    # Aggregate train
    tprint("Aggregating TRAIN features ...")
    train_agg = aggregate_customer(
        train,
        use_slope_all=args.use_slope,
        fast_mode=args.fast,
        min_non_null_ratio=args.min_non_null,
        enable_trend=args.trend,
        trend_cols=trend_cols,
        trend_slope=args.trend_slope
    )
    tprint("Saving TRAIN features ...")
    FEAT.mkdir(exist_ok=True, parents=True)
    train_agg.to_parquet(FEAT / "train_customer_v1.parquet")

    del train; gc.collect()

    # Aggregate test
    tprint("Aggregating TEST features ...")
    test_agg = aggregate_customer(
        test,
        use_slope_all=args.use_slope,
        fast_mode=args.fast,
        min_non_null_ratio=args.min_non_null,
        enable_trend=args.trend,
        trend_cols=trend_cols,
        trend_slope=args.trend_slope
    )
    tprint("Saving TEST features ...")
    test_agg.to_parquet(FEAT / "test_customer_v1.parquet")

    # Emit features_list.txt (used by 20/30 to align columns)
    tprint("Writing features_list.txt ...")
    feat_cols = [c for c in train_agg.columns if c != "target"]
    with open("features_list.txt", "w", encoding="utf-8") as f:
        for c in feat_cols:
            f.write(f"{c}\n")

    tprint(f"Done. Total time: {(time.time()-start_time)/60:.1f} min | mem ~ {mem_gb():.1f} GB")


if __name__ == "__main__":
    main()
