# 30_infer.py
# Purpose: Inference script for test set using trained fold models (LGBM and XGB).
#          Ensures categorical handling matches training, aligns feature order,
#          logs progress and memory usage, and writes a submission CSV.
# Notes:
#   - Comments are in English only (per project convention).
#   - Requires artifacts from 10_make_features.py and 20_train_cv.py:
#       features/test_customer_v1.parquet
#       features_list.txt
#       models/lgbm_v1_fold*.txt
#       models/xgb_v1_fold*.json (optional if you want XGB inference)
#       reports/train_manifest_v1.json (to get categorical feature names)
import os
import json
import time
import psutil
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

# ---------- Paths ----------
FEAT = Path("features")
MODELS = Path("models")
REPORTS = Path("reports")
SUBMIT_NAME_LGB = "submission_v1_lgbm.csv"
SUBMIT_NAME_XGB = "submission_v1_xgb.csv"
SUBMIT_NAME_ENS = "submission_v1_ensemble.csv"
FEATURES_LIST_TXT = Path("features_list.txt")
MANIFEST_JSON = REPORTS / "train_manifest_v1.json"

# ---------- Lightweight logging ----------
def tprint(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def mem_gb() -> float:
    return psutil.virtual_memory().used / (1024**3)

# ---------- Load helpers ----------
def load_feature_order() -> list[str]:
    if not FEATURES_LIST_TXT.exists():
        raise FileNotFoundError("features_list.txt not found. Run 10_make_features.py first.")
    with open(FEATURES_LIST_TXT, "r", encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols

def load_categorical_feature_list() -> list[str]:
    """Read categorical feature names saved by 20_train_cv.py manifest; fallback to empty list."""
    if not MANIFEST_JSON.exists():
        tprint("[WARN] train_manifest_v1.json not found; assuming no categorical features.")
        return []
    data = json.loads(MANIFEST_JSON.read_text())
    cats = data.get("categorical_features", None)
    if cats is None:
        tprint("[WARN] 'categorical_features' key missing in manifest; assuming none.")
        return []
    # Keep only non-empty strings
    return [c for c in cats if isinstance(c, str) and c]

def load_test_features() -> pd.DataFrame:
    path = FEAT / "test_customer_v1.parquet"
    if not path.exists():
        raise FileNotFoundError("features/test_customer_v1.parquet not found. Run 10_make_features.py first.")
    df = pd.read_parquet(path)
    if "customer_ID" in df.columns:
        df = df.set_index("customer_ID")
    return df

# ---------- Preparation for model inputs ----------
def prepare_for_lgb(test_df: pd.DataFrame, feat_order: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Align columns and cast categorical columns to pandas 'category' dtype."""
    X = test_df.reindex(columns=feat_order).copy()
    # Cast only intersection to avoid KeyError if manifest cats contain filtered-out cols
    cast_cols = [c for c in cat_cols if c in X.columns]
    for c in cast_cols:
        if not pd.api.types.is_categorical_dtype(X[c].dtype):
            X[c] = X[c].astype("category")
    return X

def prepare_for_xgb(test_df: pd.DataFrame, feat_order: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Align columns and convert categorical columns to integer codes (matching training)."""
    X = test_df.reindex(columns=feat_order).copy()
    cast_cols = [c for c in cat_cols if c in X.columns]
    for c in cast_cols:
        if not pd.api.types.is_categorical_dtype(X[c].dtype):
            X[c] = X[c].astype("category")
        X[c] = X[c].cat.codes.astype("int32")  # -1 for NaN
    return X

# ---------- Inference: LightGBM ----------
def infer_with_lgbm() -> pd.Series:
    start = time.time()
    tprint("Loading artifacts for LGBM inference ...")
    feat_order = load_feature_order()
    cat_cols = load_categorical_feature_list()
    test = load_test_features()
    tprint(f"[INFO] Test customers: {len(test):,} | Raw columns: {test.shape[1]} (mem ~ {mem_gb():.1f} GB)")

    X = prepare_for_lgb(test, feat_order, cat_cols)
    tprint(f"[INFO] Aligned test shape: {X.shape} | casting {len(cat_cols)} categorical cols")

    # Collect all LGBM fold models
    model_files = sorted(MODELS.glob("lgbm_v1_fold*.txt"))
    if not model_files:
        raise FileNotFoundError("No LGBM model files found under models/ (expected lgbm_v1_fold*.txt)")

    preds = np.zeros(len(X), dtype=float)
    for i, mf in enumerate(model_files, start=1):
        with open(mf, "r"):
            pass  # ensure file exists / readable
        tprint(f"[LGBM] Loading {mf.name} ...")
        model = lgb.Booster(model_file=str(mf))

        # Predict; handle best_iteration if present
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None or best_iter <= 0:
            fold_pred = model.predict(X)  # fallback
        else:
            fold_pred = model.predict(X, num_iteration=best_iter)

        preds += fold_pred
        tprint(f"[LGBM] Fold {i} done. Partial avg mean={preds[:10].mean():.6f}")

    preds /= len(model_files)
    tprint(f"[LGBM] Finished. Inference elapsed {(time.time()-start)/60:.2f} min")

    # Return as Series aligned to index (customer_ID)
    return pd.Series(preds, index=X.index, name="prediction")

# ---------- Inference: XGBoost (native API) ----------
def infer_with_xgb() -> pd.Series:
    start = time.time()
    tprint("Loading artifacts for XGB inference ...")
    feat_order = load_feature_order()
    cat_cols = load_categorical_feature_list()
    test = load_test_features()
    tprint(f"[INFO] Test customers: {len(test):,} | Raw columns: {test.shape[1]} (mem ~ {mem_gb():.1f} GB)")

    X_codes = prepare_for_xgb(test, feat_order, cat_cols)
    tprint(f"[INFO] Aligned test shape: {X_codes.shape} | integer-coded {len(cat_cols)} categorical cols")

    model_files = sorted(MODELS.glob("xgb_v1_fold*.json"))
    if not model_files:
        raise FileNotFoundError("No XGB model files found under models/ (expected xgb_v1_fold*.json)")

    preds = np.zeros(len(X_codes), dtype=float)
    dtest = xgb.DMatrix(X_codes.values, feature_names=feat_order)

    for i, mf in enumerate(model_files, start=1):
        tprint(f"[XGB] Loading {mf.name} ...")
        bst = xgb.Booster()
        bst.load_model(str(mf))

        # Predict with best iteration if available (iteration_range preferred; fallback to ntree_limit/default)
        try:
            best_iter = getattr(bst, "best_iteration", None)
            if best_iter is not None and best_iter >= 0:
                fold_pred = bst.predict(dtest, iteration_range=(0, best_iter + 1))
            else:
                fold_pred = bst.predict(dtest)
        except TypeError:
            best_ntree_limit = getattr(bst, "best_ntree_limit", None)
            fold_pred = bst.predict(dtest, ntree_limit=best_ntree_limit) if best_ntree_limit else bst.predict(dtest)

        preds += fold_pred
        tprint(f"[XGB] Fold {i} done. Partial avg mean={preds[:10].mean():.6f}")

    preds /= len(model_files)
    tprint(f"[XGB] Finished. Inference elapsed {(time.time()-start)/60:.2f} min")

    return pd.Series(preds, index=X_codes.index, name="prediction")

# ---------- Main ----------
if __name__ == "__main__":
    overall = time.time()
    tprint(f"Starting 30_infer.py (mem ~ {mem_gb():.1f} GB)")

    # LGBM inference (default)
    p_lgb = infer_with_lgbm()
    sub_lgb = p_lgb.reset_index().rename(columns={"index": "customer_ID"})
    sub_lgb.to_csv(SUBMIT_NAME_LGB, index=False)
    tprint(f"[SAVE] {SUBMIT_NAME_LGB} | rows={len(sub_lgb):,}")

    # OPTIONAL: XGB inference
    # p_xgb = infer_with_xgb()
    # sub_xgb = p_xgb.reset_index().rename(columns={"index": "customer_ID"})
    # sub_xgb.to_csv(SUBMIT_NAME_XGB, index=False)
    # tprint(f"[SAVE] {SUBMIT_NAME_XGB} | rows={len(sub_xgb):,}")

    # OPTIONAL: simple ensemble
    # if 'p_xgb' in locals():
    #     p_ens = 0.5 * p_lgb.align(p_xgb, join="left")[0].fillna(0) + 0.5 * p_xgb
    #     sub_ens = p_ens.reset_index().rename(columns={"index": "customer_ID"})
    #     sub_ens.to_csv(SUBMIT_NAME_ENS, index=False)
    #     tprint(f"[SAVE] {SUBMIT_NAME_ENS} | rows={len(sub_ens):,}")

    tprint(f"Done. Total elapsed {(time.time()-overall)/60:.2f} min (mem ~ {mem_gb():.1f} GB)")
