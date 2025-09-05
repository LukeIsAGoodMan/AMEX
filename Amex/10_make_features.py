# 10_make_features.py
# Purpose: Aggregate row-level AMEX data to customer-level features (V1.1).
# Adds targeted time-series features (trend block) on selected columns,
# optional slope, memory-aware implementation, unified file logging.
#
# Outputs:
#   features/train_customer_v1.parquet
#   features/test_customer_v1.parquet
#   features_list.txt

import os
import gc
import time
import psutil
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---------------- Paths ----------------
PROC = Path("data_proc")
FEAT = Path("features"); FEAT.mkdir(exist_ok=True, parents=True)

# ---------------- Defaults ----------------
CAT_WHITELIST = {
    "B_30","B_38",
    "D_63","D_64","D_66","D_68",
    "D_114","D_116","D_117","D_120","D_126"
}
EXCLUDE = {"customer_ID","S_2","target"}
DEFAULT_TREND_COLS = [
    "P_2","B_1","B_2","B_9","R_1",
    "D_39","D_41","D_44","D_48","B_11",
    "B_3","R_2","B_4","S_3","P_3",
    "B_5","R_3","D_50","B_6","R_4"
]

# ---------------- Logging helpers ----------------
def mem_gb() -> float:
    return psutil.virtual_memory().used / (1024**3)

def tprint(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def _default_log_path(script_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("logs").mkdir(parents=True, exist_ok=True)
    return f"logs/{script_name}_{ts}.log"

class RunLogger:
    """Context manager that writes both to stdout and a log file; captures duration & exceptions."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._fh = None
        self._t0 = None
        self.status = "OK"

    def __enter__(self):
        self._fh = open(self.log_path, "a", encoding="utf-8")
        self._t0 = time.time()
        self.log(f"==> START (pid={os.getpid()}) mem~{mem_gb():.1f} GB")
        return self

    def __exit__(self, etype, e, tb):
        if etype is not None:
            self.status = "ERROR"
            self.log("!! EXCEPTION RAISED !!")
            self.log("".join(traceback.format_exception(etype, e, tb)).rstrip())
        elapsed = time.time() - self._t0
        self.log(f"<== END status={self.status} elapsed={elapsed/60:.2f} min mem~{mem_gb():.1f} GB")
        if self._fh:
            self._fh.flush()
            self._fh.close()

    def log(self, msg: str):
        now = time.strftime("%H:%M:%S")
        line = f"[{now}] {msg}"
        print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

def attach_logger_to_tprint(logger: RunLogger):
    """Redirect tprint() to also write into the file logger."""
    global tprint
    _old = tprint
    def _wrapper(msg: str):
        _old(msg)
        logger.log(msg)
    tprint = _wrapper

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
      - prev_last, delta, delta_pct (row-level then aggregated by 'last')
      - rolling_mean_3, rolling_std_3 (row-level then aggregated by 'last')
      - ewm3 (row-level then aggregated by 'last')
      - slope (least squares over days) on base columns (optional)
    Returns a customer-level DataFrame.
    """
    eps = 1e-6
    keep_cols = ["customer_ID", "S_2"] + [c for c in cols if c in df.columns]
    tmp = df[keep_cols].copy().sort_values(["customer_ID", "S_2"]).reset_index(drop=True)
    g = tmp.groupby("customer_ID", sort=False, group_keys=False)

    for c in cols:
        if c not in tmp.columns:
            continue
        tmp[f"{c}__prev"] = g[c].shift(1)
        tmp[f"{c}__delta"] = tmp[c] - tmp[f"{c}__prev"]
        tmp[f"{c}__delta_pct"] = tmp[f"{c}__delta"] / (tmp[f"{c}__prev"].abs() + eps)
        tmp[f"{c}__roll3_mean"] = g[c].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        tmp[f"{c}__roll3_std"]  = g[c].rolling(3, min_periods=1).std().reset_index(level=0, drop=True)
        tmp[f"{c}__ewm3"] = g[c].apply(lambda s: s.ewm(alpha=0.6, adjust=False).mean())

    # Aggregate only derived columns by 'last' (avoid duplicating base "__last")
    DERIV_SUFFIXES = ["__prev", "__delta", "__delta_pct", "__roll3_mean", "__roll3_std", "__ewm3"]
    deriv_cols = [c for c in tmp.columns if any(c.endswith(suf) for suf in DERIV_SUFFIXES)]
    if len(deriv_cols) > 0:
        g_last_deriv = g[deriv_cols].last()
        g_last_deriv.columns = [f"{c}__last" for c in g_last_deriv.columns]
    else:
        g_last_deriv = pd.DataFrame(index=tmp["customer_ID"].drop_duplicates())

    # Optional slope on base columns
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
                t_c = t - t.mean(); y_c = y - y.mean()
                beta = (t_c * y_c).sum() / (t_c**2).sum()
                out[f"{c}__slope"] = float(beta)
            return pd.Series(out, dtype="float32")
        g_slope = tmp.groupby("customer_ID", include_groups=False, sort=False).apply(slope_block)
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
    """Main aggregation pipeline."""
    tprint("Sorting by customer_ID, S_2 ...")
    df = df.sort_values(["customer_ID", "S_2"]).reset_index(drop=True)

    num_cols, cat_cols = split_num_cat_columns(df)
    tprint(f"Detected numeric cols: {len(num_cols)}, categorical cols: {len(cat_cols)}")

    # Numeric stats
    tprint("Aggregating numeric statistics ...")
    if fast_mode:
        aggs = {c: ["mean", "std"] for c in num_cols}
    else:
        aggs = {c: ["mean", "std", "min", "max"] for c in num_cols}
    g_num = df.groupby("customer_ID").agg(aggs) if aggs else pd.DataFrame(index=df["customer_ID"].drop_duplicates())
    if len(g_num) > 0:
        g_num.columns = [f"{col}__{stat}" for col, stat in g_num.columns.to_flat_index()]

    # Numeric last
    tprint("Aggregating numeric last ...")
    g_last_num = df.groupby("customer_ID", sort=False)[num_cols].last() if len(num_cols) > 0 else pd.DataFrame(index=g_num.index)
    if len(g_last_num) > 0:
        g_last_num.columns = [f"{c}__last" for c in g_last_num.columns]

    # Optional slope across ALL numeric columns
    if use_slope_all and len(num_cols) > 0:
        tprint("Computing numeric slopes for ALL numeric columns (slow) ...")
        df_need = df[["customer_ID", "S_2"] + num_cols]
        def slope_block(sub: pd.DataFrame) -> pd.Series:
            out = {}
            dates = sub["S_2"]
            for c in num_cols:
                mask = sub[c].notna() & dates.notna()
                y = sub.loc[mask, c].astype(float)
                if len(y) <= 1:
                    out[f"{c}__slope"] = 0.0; continue
                t = (dates[mask] - dates[mask].iloc[0]).dt.days.astype(float)
                if t.var() == 0:
                    out[f"{c}__slope"] = 0.0; continue
                t_c = t - t.mean(); y_c = y - y.mean()
                beta = (t_c * y_c).sum() / (t_c**2).sum()
                out[f"{c}__slope"] = float(beta)
            return pd.Series(out, dtype="float32")
        g_slope_all = df_need.groupby("customer_ID", include_groups=False, sort=False).apply(slope_block)
    else:
        g_slope_all = pd.DataFrame(index=g_num.index)

    gc.collect()

    # Categorical
    tprint("Aggregating categorical last ...")
    g_cat_last = df.groupby("customer_ID", sort=False)[cat_cols].last() if len(cat_cols) > 0 else pd.DataFrame(index=g_num.index)
    if len(g_cat_last) > 0:
        g_cat_last.columns = [f"{c}__last" for c in g_cat_last.columns]

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

    # Trend block
    if enable_trend:
        tprint("Building trend features for selected columns ...")
        trend_cols = trend_cols or DEFAULT_TREND_COLS
        trend_cols_avail = [c for c in trend_cols if c in num_cols]
        if len(trend_cols_avail) == 0:
            tprint("[WARN] No selected trend columns are available in numeric columns; skipping trend block.")
            g_trend = pd.DataFrame(index=g_num.index)
        else:
            g_trend = build_trend_features(
                df[["customer_ID", "S_2"] + trend_cols_avail], cols=trend_cols_avail, compute_slope=trend_slope
            )
    else:
        g_trend = pd.DataFrame(index=g_num.index)

    # Concat blocks
    tprint("Concatenating all feature blocks ...")
    parts = [x for x in [g_num, g_last_num, g_slope_all, g_cat_last, g_cat_top1, g_cat_nu, g_trend] if len(x) > 0]
    g_all = pd.concat(parts, axis=1)
    # Safety: drop any duplicated columns (should be none after fix)
    g_all = g_all.loc[:, ~g_all.columns.duplicated()]

    # Drop constant / sparse; downcast
    tprint("Removing constant / sparse columns and downcasting ...")
    nun = g_all.nunique()
    g_all = g_all.loc[:, nun > 1]
    non_null_ratio = 1.0 - (g_all.isna().sum() / len(g_all))
    g_all = g_all.loc[:, non_null_ratio >= min_non_null_ratio]
    g_all = downcast_df(g_all)

    # Attach target for train
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
                        help="Customer-level sampling ratio (default 1.0).")
    parser.add_argument("--fast", action="store_true", help="FAST mode: numeric mean/std/last; categorical last/top1freq.")
    parser.add_argument("--use-slope", action="store_true",
                        help="Compute slope across ALL numeric columns (slow; trend block slope is separate).")
    parser.add_argument("--min-non-null", type=float, default=0.7, help="Min non-null ratio to keep a column.")
    parser.add_argument("--trend", action="store_true", default=True, help="Enable trend features block (default True).")
    parser.add_argument("--no-trend", dest="trend", action="store_false", help="Disable trend features block.")
    parser.add_argument("--trend-slope", action="store_true", help="Compute slope inside trend block.")
    parser.add_argument("--trend-cols-file", type=str, default="", help="Optional text file for trend base columns.")
    parser.add_argument("--log-file", type=str, default="", help="Optional path to write a run log.")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    log_path = args.log_file or _default_log_path(script_name)
    with RunLogger(log_path) as RLOG:
        attach_logger_to_tprint(RLOG)
        RLOG.log(f"Args: {vars(args)}")

        tprint(f"Starting... mem ~ {mem_gb():.1f} GB")

        # Load parquet
        train_pq = PROC / "train.parquet"
        test_pq  = PROC / "test.parquet"
        if not train_pq.exists() or not test_pq.exists():
            raise FileNotFoundError("Missing data_proc/train.parquet or data_proc/test.parquet. Run 00_prepare_data.py.")

        tprint("Loading train.parquet ...")
        train = pd.read_parquet(train_pq); tprint(f"Train rows: {len(train):,} | mem ~ {mem_gb():.1f} GB")
        tprint("Loading test.parquet ...")
        test = pd.read_parquet(test_pq);  tprint(f"Test rows: {len(test):,} | mem ~ {mem_gb():.1f} GB")

        # Ensure S_2 datetime
        for df in (train, test):
            if "S_2" in df.columns and not np.issubdtype(df["S_2"].dtype, np.datetime64):
                df["S_2"] = pd.to_datetime(df["S_2"])

        # Optional sampling
        if args.debug_frac < 1.0:
            train = maybe_sample(train, args.debug_frac)
            test  = maybe_sample(test,  args.debug_frac)

        # Downcast early
        tprint("Downcasting dtypes ...")
        for df in (train, test):
            downcast_df(df)
        tprint(f"After downcast mem ~ {mem_gb():.1f} GB")

        # Optional external trend list
        trend_cols = None
        if args.trend and args.trend_cols_file:
            path = Path(args.trend_cols_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    trend_cols = [line.strip() for line in f if line.strip()]
                tprint(f"Loaded {len(trend_cols)} trend columns from file: {args.trend_cols_file}")
            else:
                tprint(f"[WARN] trend-cols-file not found: {args.trend_cols_file}. Using default list.")

        # Aggregate TRAIN
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
        RLOG.log(f"[SUMMARY] TRAIN shape={train_agg.shape}")

        del train; gc.collect()

        # Aggregate TEST
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
        RLOG.log(f"[SUMMARY] TEST shape={test_agg.shape}")

        # features_list.txt
        tprint("Writing features_list.txt ...")
        feat_cols = [c for c in train_agg.columns if c != "target"]
        with open("features_list.txt", "w", encoding="utf-8") as f:
            for c in feat_cols: f.write(f"{c}\n")
        RLOG.log(f"[SUMMARY] feature_count={len(feat_cols)}")
        tprint(f"Done. mem ~ {mem_gb():.1f} GB")

if __name__ == "__main__":
    main()
