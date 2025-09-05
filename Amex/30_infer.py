# 30_infer.py (v1.2)
# Purpose: Inference with LGBM and optional XGB; optional isotonic calibration and OOF-optimized ensemble.

import os
import json
import time
import psutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib

# ---------- Paths ----------
FEAT = Path("features")
MODELS = Path("models")
REPORTS = Path("reports")
CAL_DIR = Path("calibrators")
SUBMIT_NAME_LGB = "submission_v1_lgbm.csv"
SUBMIT_NAME_XGB = "submission_v1_xgb.csv"
SUBMIT_NAME_ENS = "submission_v1_ensemble.csv"
FEATURES_LIST_TXT = Path("features_list.txt")
MANIFEST_JSON = REPORTS / "train_manifest_v1.json"

# ---------- Logging ----------
def tprint(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def mem_gb() -> float:
    return psutil.virtual_memory().used / (1024**3)

def _default_log_path(script_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("logs").mkdir(parents=True, exist_ok=True)
    return f"logs/{script_name}_{ts}.log"

class RunLogger:
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
    global tprint
    _old = tprint
    def _wrapper(msg: str):
        _old(msg); logger.log(msg)
    tprint = _wrapper

# ---------- Load helpers ----------
def load_feature_order() -> list[str]:
    if not FEATURES_LIST_TXT.exists():
        raise FileNotFoundError("features_list.txt not found. Run 10_make_features.py first.")
    with open(FEATURES_LIST_TXT, "r", encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols

def load_categorical_feature_list() -> list[str]:
    if not MANIFEST_JSON.exists():
        tprint("[WARN] train_manifest_v1.json not found; assuming no categorical features.")
        return []
    data = json.loads(MANIFEST_JSON.read_text())
    cats = data.get("categorical_features", None)
    if cats is None:
        tprint("[WARN] 'categorical_features' missing; assuming none.")
        return []
    return [c for c in cats if isinstance(c, str) and c]

def load_test_features() -> pd.DataFrame:
    path = FEAT / "test_customer_v1.parquet"
    if not path.exists():
        raise FileNotFoundError("features/test_customer_v1.parquet not found. Run 10_make_features.py first.")
    df = pd.read_parquet(path)
    if "customer_ID" in df.columns:
        df = df.set_index("customer_ID")
    return df

# ---------- Preparation ----------
def prepare_for_lgb(test_df: pd.DataFrame, feat_order: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = test_df.reindex(columns=feat_order).copy()
    cast_cols = [c for c in cat_cols if c in X.columns]
    for c in cast_cols:
        if not pd.api.types.is_categorical_dtype(X[c].dtype):
            X[c] = X[c].astype("category")
    return X

def prepare_for_xgb(test_df: pd.DataFrame, feat_order: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = test_df.reindex(columns=feat_order).copy()
    cast_cols = [c for c in cat_cols if c in X.columns]
    for c in cast_cols:
        if not pd.api.types.is_categorical_dtype(X[c].dtype):
            X[c] = X[c].astype("category")
        X[c] = X[c].cat.codes.astype("int32")
    return X

# ---------- Inference: LGBM ----------
def infer_with_lgbm(apply_calibration: bool = False) -> pd.Series:
    start = time.time()
    tprint("Loading artifacts for LGBM inference ...")
    feat_order = load_feature_order()
    cat_cols = load_categorical_feature_list()
    test = load_test_features()
    tprint(f"[INFO] Test customers: {len(test):,} | Raw columns: {test.shape[1]} (mem ~ {mem_gb():.1f} GB)")
    X = prepare_for_lgb(test, feat_order, cat_cols)
    tprint(f"[INFO] Aligned test shape: {X.shape} | casting {len(cat_cols)} categorical cols")

    model_files = sorted(MODELS.glob("lgbm_v1_fold*.txt"))
    if not model_files:
        raise FileNotFoundError("No LGBM model files found under models/ (expected lgbm_v1_fold*.txt)")

    preds = np.zeros(len(X), dtype=float)
    for i, mf in enumerate(model_files, start=1):
        tprint(f"[LGBM] Loading {mf.name} ...")
        model = lgb.Booster(model_file=str(mf))
        best_iter = getattr(model, "best_iteration", None)
        fold_pred = model.predict(X, num_iteration=best_iter if best_iter and best_iter > 0 else None)
        preds += fold_pred
        tprint(f"[LGBM] Fold {i} done.")

    preds /= len(model_files)
    preds = np.clip(preds, 0.0, 1.0)

    if apply_calibration:
        cal_path = CAL_DIR / "lgbm_iso_v1.joblib"
        if cal_path.exists():
            iso = joblib.load(cal_path)
            preds = iso.predict(preds)
            preds = np.clip(preds, 0.0, 1.0)
            tprint("[CAL] applied isotonic calibration")
        else:
            tprint("[CAL] calibrator not found; skipping")

    tprint(f"[LGBM] Finished. Inference elapsed {(time.time()-start)/60:.2f} min")
    return pd.Series(preds, index=X.index, name="prediction")

# ---------- Inference: XGB ----------
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

    dtest = xgb.DMatrix(X_codes.values, feature_names=feat_order)
    preds = np.zeros(len(X_codes), dtype=float)
    for i, mf in enumerate(model_files, start=1):
        tprint(f"[XGB] Loading {mf.name} ...")
        bst = xgb.Booster(); bst.load_model(str(mf))
        try:
            best_iter = getattr(bst, "best_iteration", None)
            fold_pred = bst.predict(dtest, iteration_range=(0, best_iter + 1)) if best_iter is not None else bst.predict(dtest)
        except TypeError:
            best_ntree_limit = getattr(bst, "best_ntree_limit", None)
            fold_pred = bst.predict(dtest, ntree_limit=best_ntree_limit) if best_ntree_limit else bst.predict(dtest)
        preds += fold_pred
        tprint(f"[XGB] Fold {i} done.")

    preds /= len(model_files)
    preds = np.clip(preds, 0.0, 1.0)
    tprint(f"[XGB] Finished. Inference elapsed {(time.time()-start)/60:.2f} min")
    return pd.Series(preds, index=X_codes.index, name="prediction")

# ---------- Main ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference with LGB/XGB; optional calibration and ensemble-opt (v1.2).")
    parser.add_argument("--log-file", type=str, default="", help="Optional path to write a run log.")
    parser.add_argument("--xgb", action="store_true", help="Also run XGB inference and save an extra CSV.")
    parser.add_argument("--ensemble", action="store_true", help="If both LGB & XGB available, save 0.5/0.5 ensemble.")
    parser.add_argument("--ensemble-opt", action="store_true", help="Use optimized alpha from reports/ensemble_alpha_v12.txt.")
    parser.add_argument("--calibrate", action="store_true", help="Apply isotonic calibration to LGB predictions.")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    log_path = args.log_file or _default_log_path(script_name)
    with RunLogger(log_path) as RLOG:
        attach_logger_to_tprint(RLOG)
        RLOG.log(f"Args: {vars(args)}")

        tprint(f"Starting 30_infer.py (mem ~ {mem_gb():.1f} GB)")

        p_lgb = infer_with_lgbm(apply_calibration=args.calibrate)
        sub_lgb = p_lgb.reset_index().rename(columns={"index": "customer_ID"})
        sub_lgb.to_csv(SUBMIT_NAME_LGB, index=False)
        tprint(f"[SAVE] {SUBMIT_NAME_LGB} | rows={len(sub_lgb):,}")

        if args.xgb:
            p_xgb = infer_with_xgb()
            sub_xgb = p_xgb.reset_index().rename(columns={"index": "customer_ID"})
            sub_xgb.to_csv(SUBMIT_NAME_XGB, index=False)
            tprint(f"[SAVE] {SUBMIT_NAME_XGB} | rows={len(sub_xgb):,}")

            if args.ensemble:
                p_xgb = p_xgb.reindex(p_lgb.index)

                if args.ensemble_opt and (REPORTS / "ensemble_alpha_v12.txt").exists():
                    a = float((REPORTS / "ensemble_alpha_v12.txt").read_text().strip())
                    tprint(f"[ENSEMBLE] Using optimized alpha={a:.2f}")
                else:
                    a = 0.5
                    if args.ensemble_opt:
                        tprint("[ENSEMBLE] optimized alpha not found, fallback to 0.5")

                p_ens = a * p_lgb + (1 - a) * p_xgb
                sub_ens = p_ens.reset_index().rename(columns={"index": "customer_ID"})
                sub_ens.to_csv(SUBMIT_NAME_ENS, index=False)
                tprint(f"[SAVE] {SUBMIT_NAME_ENS} | rows={len(sub_ens):,}")

        tprint("Done.")
