# 20_train_cv.py
# Purpose: Train baseline models (LightGBM / XGBoost) with GroupKFold CV,
#          log progress/timings/memory, and persist OOF predictions & feature importances.
# Notes:
#   - Comments are in English only (per project convention).
#   - This script is compatible with feature tables produced by 10_make_features.py.
#   - Metrics: ROC-AUC, PR-AUC, and a simplified AMEX-M (for sanity checks).

import os
import time
import json
import psutil
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import lightgbm as lgb
import xgboost as xgb


# ------------ Paths & constants ------------
FEAT = Path("features")
OOF_DIR = Path("oof"); OOF_DIR.mkdir(exist_ok=True, parents=True)
MODELS = Path("models"); MODELS.mkdir(exist_ok=True, parents=True)
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True, parents=True)

SEED = int(os.getenv("SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))

FEATURES_LIST_TXT = Path("features_list.txt")  # optional column alignment file


# ------------ Lightweight logging helpers ------------
def tprint(msg: str) -> None:
    """Timestamped print with flush for real-time progress."""
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def mem_used_gb() -> float:
    """Return current used system memory in GB."""
    return psutil.virtual_memory().used / (1024**3)

class Timer:
    """Context timer for stage-level elapsed time logs."""
    def __init__(self, label: str):
        self.label = label
        self.start = None

    def __enter__(self):
        self.start = time.time()
        tprint(f"Start: {self.label} (mem ~ {mem_used_gb():.1f} GB)")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        tprint(f"End:   {self.label} | elapsed {elapsed/60:.2f} min (mem ~ {mem_used_gb():.1f} GB)")


# ------------ Metrics ------------
def amex_metric(y_true, y_pred):
    """
    Simplified AMEX-like metric:
      0.5 * (weighted_gini + top_4pct_captured)
    (No class weights here; suitable for internal sanity checks.)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Top 4% captured (by count, unweighted)
    n = max(1, int(0.04 * len(y_pred)))
    idx = np.argsort(-y_pred)[:n]
    top4 = y_true[idx].sum()
    total_pos = y_true.sum()
    top4_captured = (top4 / total_pos) if total_pos > 0 else 0.0

    # "Weighted" gini (actually unweighted here)
    order = np.argsort(-y_pred, kind="mergesort")
    y = y_true[order]
    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos
    cum_pos = np.cumsum(y)
    cum_neg = np.cumsum(1 - y)
    if total_pos > 0 and total_neg > 0:
        lorentz = cum_pos / total_pos
        g = np.sum(lorentz - (cum_neg / total_neg))
        g /= len(y_true)
    else:
        g = 0.0

    return 0.5 * (g + top4_captured)

def evaluate_dict(y, oof):
    """Return a dict of evaluation metrics."""
    return {
        "auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "amex_m": float(amex_metric(y, oof)),
    }


# ------------ Data loading & categorical handling ------------
def load_train_tables():
    """
    Load customer-level features from parquet.
    Align columns with features_list.txt if present (excluding 'target').
    Return:
      X_raw (DataFrame),
      y (Series[int]),
      groups (Index),
      feature_columns (list[str])
    """
    with Timer("Load train_customer_v1.parquet"):
        df = pd.read_parquet(FEAT / "train_customer_v1.parquet")

    # Ensure index is customer_ID (if not already)
    if "customer_ID" in df.columns:
        df = df.set_index("customer_ID")

    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in train_customer_v1.parquet.")

    y = df["target"].astype(int)

    # Column alignment via features_list.txt (optional but recommended)
    if FEATURES_LIST_TXT.exists():
        with open(FEATURES_LIST_TXT, "r", encoding="utf-8") as f:
            feat_cols = [line.strip() for line in f if line.strip()]
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from train parquet: {missing[:10]} ...")
        X_raw = df[feat_cols].copy()
    else:
        # Fallback: use all non-target columns
        feat_cols = [c for c in df.columns if c != "target"]
        X_raw = df[feat_cols].copy()

    groups = X_raw.index  # GroupKFold by customer_ID
    tprint(f"[INFO] Train customers: {len(X_raw):,} | Features: {X_raw.shape[1]} | PosRate: {y.mean():.4f}")
    return X_raw, y, groups, feat_cols

def detect_categorical_columns(X: pd.DataFrame, max_unique: int = 64):
    """
    Detect columns that should be treated as categorical:
      - dtype is 'object' OR 'category'
      - and number of unique non-null values <= max_unique
    Return list of column names.
    """
    cat_cols = []
    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_categorical_dtype(dt):
            nun = X[c].nunique(dropna=True)
            if 0 < nun <= max_unique:
                cat_cols.append(c)
    return cat_cols

def prepare_X_for_models(X_raw: pd.DataFrame):
    """
    Prepare two versions of X:
      - X_lgb: object/category columns cast to 'category' dtype (for LightGBM)
      - X_xgb: same categorical columns converted to integer codes (for XGBoost)
    Also return the list of categorical feature names used for LGBM.
    """
    X_lgb = X_raw.copy()
    # Detect categorical candidates (object/category with limited uniques)
    cat_cols = detect_categorical_columns(X_lgb, max_unique=64)

    # Cast to pandas 'category' for LGBM
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(X_lgb[c].dtype):
            X_lgb[c] = X_lgb[c].astype("category")

    # Build X_xgb by converting the same categorical columns to integer codes
    X_xgb = X_lgb.copy()
    for c in cat_cols:
        # cat.codes returns -1 for NaN; keep as int32
        X_xgb[c] = X_xgb[c].cat.codes.astype("int32")

    tprint(f"[INFO] Detected categorical features: {len(cat_cols)}")
    if len(cat_cols) > 0:
        tprint(f"[INFO] Example cats: {cat_cols[: min(5, len(cat_cols))]}")

    return X_lgb, X_xgb, cat_cols


# ------------ LightGBM training with progress logs ------------
def train_lgbm(X_lgb, y, groups, feature_names, categorical_features):
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=5000,
        random_state=SEED,
        verbose=-1,
    )

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_lgb), dtype=float)
    feat_gain = pd.Series(0.0, index=feature_names, dtype=float)

    with Timer("Train LightGBM (GroupKFold)"):
        for fold, (tr, va) in enumerate(gkf.split(X_lgb, y, groups=groups), start=1):
            fold_label = f"LGBM fold {fold}/{N_SPLITS}"
            with Timer(fold_label):
                dtr = lgb.Dataset(
                    X_lgb.iloc[tr],
                    label=y.iloc[tr],
                    feature_name=feature_names,
                    categorical_feature=categorical_features or None,
                    free_raw_data=True,
                )
                dva = lgb.Dataset(
                    X_lgb.iloc[va],
                    label=y.iloc[va],
                    feature_name=feature_names,
                    categorical_feature=categorical_features or None,
                    free_raw_data=True,
                )

                model = lgb.train(
                    params,
                    dtr,
                    valid_sets=[dtr, dva],
                    valid_names=["train", "valid"],
                    num_boost_round=5000,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=200, verbose=False),
                        lgb.log_evaluation(period=100)
                    ],
                )

                best_iter = model.best_iteration
                tprint(f"[LGBM] best_iteration={best_iter}")

                # Predict validation fold
                pred = model.predict(X_lgb.iloc[va], num_iteration=best_iter)
                oof[va] = pred

                # Feature importance (gain)
                gain = pd.Series(model.feature_importance(importance_type="gain"), index=feature_names, dtype=float)
                feat_gain = feat_gain.add(gain, fill_value=0.0)

                # Persist fold model
                model_path = MODELS / f"lgbm_v1_fold{fold}.txt"
                model.save_model(str(model_path))
                tprint(f"[LGBM] Saved: {model_path}")

    # Evaluate OOF
    metrics = evaluate_dict(y, oof)
    tprint(f"[LGBM] OOF metrics: {json.dumps(metrics)}")

    # Save OOF and importance
    pd.DataFrame({"customer_ID": X_lgb.index, "oof": oof}).set_index("customer_ID").to_parquet(OOF_DIR / "lgbm_oof_v1.parquet")
    feat_gain.to_frame("gain").sort_values("gain", ascending=False).to_csv(REPORTS / "lgbm_featimp_v1.csv")

    return metrics


# ------------ XGBoost training with progress logs ------------
def train_xgb(X_xgb, y, groups, feature_names):
    """
    Train XGBoost with native API (DMatrix + xgb.train) for maximum version compatibility.
    Supports early stopping across XGBoost versions.
    """
    params = dict(
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",   # use "gpu_hist" if GPU is available
        random_state=SEED,
        eval_metric="auc",
        nthread=os.cpu_count(),
        verbosity=0,
    )
    num_boost_round = 5000
    early_stopping_rounds = 200

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_xgb), dtype=float)
    feat_gain = pd.Series(0.0, index=feature_names, dtype=float)

    with Timer("Train XGBoost (GroupKFold, native API)"):
        for fold, (tr, va) in enumerate(gkf.split(X_xgb, y, groups=groups), start=1):
            fold_label = f"XGB fold {fold}/{N_SPLITS}"
            with Timer(fold_label):
                # Build DMatrices with feature names so importance maps correctly
                dtr = xgb.DMatrix(
                    X_xgb.iloc[tr].values,
                    label=y.iloc[tr].values,
                    feature_names=feature_names
                )
                dva = xgb.DMatrix(
                    X_xgb.iloc[va].values,
                    label=y.iloc[va].values,
                    feature_names=feature_names
                )

                # Train with early stopping
                evals = [(dtr, "train"), (dva, "valid")]
                bst = xgb.train(
                    params=params,
                    dtrain=dtr,
                    num_boost_round=num_boost_round,
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False,
                )

                best_iter = getattr(bst, "best_iteration", None)
                tprint(f"[XGB] best_iteration={best_iter}")

                # Predict with best iteration (handle version differences)
                try:
                    # XGBoost >= 2.0 preferred
                    pred = bst.predict(dva, iteration_range=(0, (best_iter or 0) + 1))
                except TypeError:
                    # Older versions: use ntree_limit if available
                    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
                    if best_ntree_limit is not None:
                        pred = bst.predict(dva, ntree_limit=best_ntree_limit)
                    else:
                        pred = bst.predict(dva)

                oof[va] = pred

                # Feature importance by gain
                imp = bst.get_score(importance_type="gain")  # keys are feature names
                gain_series = pd.Series(imp, dtype=float)
                gain_series = gain_series.reindex(feature_names).fillna(0.0)
                feat_gain = feat_gain.add(gain_series, fill_value=0.0)

                # Persist fold model
                model_path = MODELS / f"xgb_v1_fold{fold}.json"
                bst.save_model(str(model_path))
                tprint(f"[XGB] Saved: {model_path}")

    # Evaluate OOF
    metrics = evaluate_dict(y, oof)
    tprint(f"[XGB] OOF metrics: {json.dumps(metrics)}")

    # Save OOF and importance
    pd.DataFrame({"customer_ID": X_xgb.index, "oof": oof}).set_index("customer_ID").to_parquet(OOF_DIR / "xgb_oof_v1.parquet")
    feat_gain.to_frame("gain").sort_values("gain", ascending=False).to_csv(REPORTS / "xgb_featimp_v1.csv")

    return metrics


# ------------ Main ------------
if __name__ == "__main__":
    overall = time.time()
    tprint(f"Starting 20_train_cv.py | N_SPLITS={N_SPLITS} | SEED={SEED} (mem ~ {mem_used_gb():.1f} GB)")

    # Load raw feature table
    X_raw, y, groups, feature_names = load_train_tables()

    # Optional debug fraction (subset) via env var, without changing code:
    debug_frac = float(os.getenv("DEBUG_FRAC", "1.0"))
    if 0.0 < debug_frac < 1.0:
        tprint(f"[DEBUG] Using DEBUG_FRAC={debug_frac:.2f} of data")
        n = int(len(X_raw) * debug_frac)
        X_raw = X_raw.iloc[:n].copy()
        y = y.iloc[:n].copy()
        groups = groups[:n]
        tprint(f"[DEBUG] Subset customers: {len(X_raw):,} | Features: {X_raw.shape[1]}")

    # Prepare categorical handling for both models
    with Timer("Prepare categorical features for LGB/XGB"):
        X_lgb, X_xgb, cat_cols = prepare_X_for_models(X_raw)

    # Train LGBM (pass categorical feature names)
    lgbm_metrics = train_lgbm(X_lgb, y, groups, feature_names, cat_cols)

    # Train XGB (already encoded to integer codes)
    xgb_metrics = train_xgb(X_xgb, y, groups, feature_names)

    # Save a small manifest for reproducibility
    manifest = {
        "n_customers": int(len(X_raw)),
        "n_features": int(X_raw.shape[1]),
        "n_splits": int(N_SPLITS),
        "seed": int(SEED),
        "metrics": {
            "lgbm": lgbm_metrics,
            "xgb": xgb_metrics,
        },
        "categorical_features": cat_cols,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (REPORTS / "train_manifest_v1.json").write_text(json.dumps(manifest, indent=2))

    tprint(f"Done. Total elapsed {(time.time()-overall)/60:.2f} min (mem ~ {mem_used_gb():.1f} GB)")
