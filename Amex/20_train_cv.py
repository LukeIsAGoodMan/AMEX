# 20_train_cv.py (v1.2)
# Purpose: Train LightGBM / XGBoost with GroupKFold, optional correlation-based pruning,
#          isotonic calibration on LGBM OOF, and OOF-based ensemble weight search.
# Outputs:
#   oof/*.parquet, reports/*.csv & train_manifest_v1.json, calibrators/lgbm_iso_v1.joblib, reports/ensemble_alpha_v12.txt

import os
import time
import json
import psutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
import joblib

import lightgbm as lgb
import xgboost as xgb

# ------------ Paths & constants ------------
FEAT = Path("features")
OOF_DIR = Path("oof"); OOF_DIR.mkdir(exist_ok=True, parents=True)
MODELS = Path("models"); MODELS.mkdir(exist_ok=True, parents=True)
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True, parents=True)
CAL_DIR = Path("calibrators"); CAL_DIR.mkdir(exist_ok=True, parents=True)

SEED = int(os.getenv("SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
FEATURES_LIST_TXT = Path("features_list.txt")

# ------------ Lightweight logging ------------
def tprint(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def mem_used_gb() -> float:
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
        self.log(f"==> START (pid={os.getpid()}) mem~{mem_used_gb():.1f} GB")
        return self

    def __exit__(self, etype, e, tb):
        if etype is not None:
            self.status = "ERROR"
            self.log("!! EXCEPTION RAISED !!")
            self.log("".join(traceback.format_exception(etype, e, tb)).rstrip())
        elapsed = time.time() - self._t0
        self.log(f"<== END status={self.status} elapsed={elapsed/60:.2f} min mem~{mem_used_gb():.1f} GB")
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

class Timer:
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
    """Simplified AMEX-M."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = max(1, int(0.04 * len(y_pred)))
    idx = np.argsort(-y_pred)[:n]
    top4 = y_true[idx].sum(); total_pos = y_true.sum()
    top4_captured = (top4 / total_pos) if total_pos > 0 else 0.0

    order = np.argsort(-y_pred, kind="mergesort")
    y = y_true[order]
    total_pos = y_true.sum(); total_neg = len(y_true) - total_pos
    cum_pos = np.cumsum(y); cum_neg = np.cumsum(1 - y)
    if total_pos > 0 and total_neg > 0:
        lorentz = cum_pos / total_pos
        g = np.sum(lorentz - (cum_neg / total_neg))
        g /= len(y_true)
    else:
        g = 0.0
    return 0.5 * (g + top4_captured)

def evaluate_dict(y, oof):
    return {
        "auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "amex_m": float(amex_metric(y, oof)),
    }

# ------------ Data & categoricals ------------
def load_train_tables():
    with Timer("Load train_customer_v1.parquet"):
        df = pd.read_parquet(FEAT / "train_customer_v1.parquet")
    if "customer_ID" in df.columns:
        df = df.set_index("customer_ID")
    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in train_customer_v1.parquet.")
    y = df["target"].astype(int)

    if FEATURES_LIST_TXT.exists():
        with open(FEATURES_LIST_TXT, "r", encoding="utf-8") as f:
            feat_cols = [line.strip() for line in f if line.strip()]
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from train parquet: {missing[:10]} ...")
        X_raw = df[feat_cols].copy()
    else:
        feat_cols = [c for c in df.columns if c != "target"]
        X_raw = df[feat_cols].copy()

    groups = X_raw.index
    tprint(f"[INFO] Train customers: {len(X_raw):,} | Features: {X_raw.shape[1]} | PosRate: {y.mean():.4f}")
    return X_raw, y, groups, feat_cols

def detect_categorical_columns(X: pd.DataFrame, max_unique: int = 64):
    cat_cols = []
    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_categorical_dtype(dt):
            nun = X[c].nunique(dropna=True)
            if 0 < nun <= max_unique:
                cat_cols.append(c)
    return cat_cols

def prepare_X_for_models(X_raw: pd.DataFrame):
    X_lgb = X_raw.copy()
    cat_cols = detect_categorical_columns(X_lgb, max_unique=64)
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(X_lgb[c].dtype):
            X_lgb[c] = X_lgb[c].astype("category")
    X_xgb = X_lgb.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].cat.codes.astype("int32")
    tprint(f"[INFO] Detected categorical features: {len(cat_cols)}")
    if len(cat_cols) > 0:
        tprint(f"[INFO] Example cats: {cat_cols[: min(5, len(cat_cols))]}")
    return X_lgb, X_xgb, cat_cols

# ------------ Optional pruning ------------
def corr_prune(X: pd.DataFrame, threshold: float = 0.995):
    """Drop highly correlated numeric features (|ρ| > threshold)."""
    tprint(f"[PRUNE] Computing correlation matrix (threshold={threshold}) ...")
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if (upper[c] > threshold).any()]
    if drop_cols:
        tprint(f"[PRUNE] dropping {len(drop_cols)} features (highly correlated)")
        X_new = X.drop(columns=drop_cols)
    else:
        X_new = X
    return X_new, drop_cols

# ------------ LGBM ------------
def train_lgbm(X_lgb, y, groups, feature_names, categorical_features):
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=96,
        feature_fraction=0.85,
        bagging_fraction=0.85,
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
            with Timer(f"LGBM fold {fold}/{N_SPLITS}"):
                dtr = lgb.Dataset(
                    X_lgb.iloc[tr], label=y.iloc[tr],
                    feature_name=feature_names,
                    categorical_feature=categorical_features or None,
                    free_raw_data=True,
                )
                dva = lgb.Dataset(
                    X_lgb.iloc[va], label=y.iloc[va],
                    feature_name=feature_names,
                    categorical_feature=categorical_features or None,
                    free_raw_data=True,
                )
                model = lgb.train(
                    params, dtr,
                    valid_sets=[dtr, dva],
                    valid_names=["train", "valid"],
                    num_boost_round=5000,
                    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                               lgb.log_evaluation(period=100)],
                )
                best_iter = model.best_iteration
                tprint(f"[LGBM] best_iteration={best_iter}")
                pred = model.predict(X_lgb.iloc[va], num_iteration=best_iter)
                oof[va] = pred
                gain = pd.Series(model.feature_importance(importance_type="gain"), index=feature_names, dtype=float)
                feat_gain = feat_gain.add(gain, fill_value=0.0)
                model_path = MODELS / f"lgbm_v1_fold{fold}.txt"
                model.save_model(str(model_path))
                tprint(f"[LGBM] Saved: {model_path}")

    oof = np.clip(oof, 0.0, 1.0)
    metrics = evaluate_dict(y, oof)
    tprint(f"[LGBM] OOF metrics: {json.dumps(metrics)}")
    pd.DataFrame({"customer_ID": X_lgb.index, "oof": oof}).set_index("customer_ID").to_parquet(OOF_DIR / "lgbm_oof_v1.parquet")
    feat_gain.to_frame("gain").sort_values("gain", ascending=False).to_csv(REPORTS / "lgbm_featimp_v1.csv")
    return metrics

# ------------ XGBoost ------------
def train_xgb(X_xgb, y, groups, feature_names):
    params = dict(
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
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
            with Timer(f"XGB fold {fold}/{N_SPLITS}"):
                dtr = xgb.DMatrix(X_xgb.iloc[tr].values, label=y.iloc[tr].values, feature_names=feature_names)
                dva = xgb.DMatrix(X_xgb.iloc[va].values, label=y.iloc[va].values, feature_names=feature_names)
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
                try:
                    pred = bst.predict(dva, iteration_range=(0, (best_iter or 0) + 1))
                except TypeError:
                    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
                    pred = bst.predict(dva, ntree_limit=best_ntree_limit) if best_ntree_limit else bst.predict(dva)
                oof[va] = pred

                imp = bst.get_score(importance_type="gain")
                gain_series = pd.Series(imp, dtype=float).reindex(feature_names).fillna(0.0)
                feat_gain = feat_gain.add(gain_series, fill_value=0.0)

                model_path = MODELS / f"xgb_v1_fold{fold}.json"
                bst.save_model(str(model_path))
                tprint(f"[XGB] Saved: {model_path}")

    oof = np.clip(oof, 0.0, 1.0)
    metrics = evaluate_dict(y, oof)
    tprint(f"[XGB] OOF metrics: {json.dumps(metrics)}")
    pd.DataFrame({"customer_ID": X_xgb.index, "oof": oof}).set_index("customer_ID").to_parquet(OOF_DIR / "xgb_oof_v1.parquet")
    feat_gain.to_frame("gain").sort_values("gain", ascending=False).to_csv(REPORTS / "xgb_featimp_v1.csv")
    return metrics

# ------------ Main ------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LGBM/XGB with GroupKFold, pruning & calibration (v1.2).")
    parser.add_argument("--log-file", type=str, default="", help="Optional path to write a run log.")
    parser.add_argument("--debug-frac", type=float, default=float(os.getenv("DEBUG_FRAC", "1.0")),
                        help="Subset fraction of customers for quick debug (default 1.0).")
    parser.add_argument("--corr-threshold", type=float, default=0.0,
                        help="If >0, drop numeric features with |ρ| > threshold (e.g., 0.995).")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    log_path = args.log_file or _default_log_path(script_name)
    with RunLogger(log_path) as RLOG:
        attach_logger_to_tprint(RLOG)
        RLOG.log(f"Args: {vars(args)}")

        tprint(f"Starting 20_train_cv.py | N_SPLITS={N_SPLITS} | SEED={SEED} (mem ~ {mem_used_gb():.1f} GB)")
        X_raw, y, groups, feature_names = load_train_tables()

        # Optional subset
        if 0.0 < args.debug_frac < 1.0:
            tprint(f"[DEBUG] Using DEBUG_FRAC={args.debug_frac:.2f} of data")
            n = int(len(X_raw) * args.debug_frac)
            X_raw = X_raw.iloc[:n].copy(); y = y.iloc[:n].copy(); groups = groups[:n]
            tprint(f"[DEBUG] Subset customers: {len(X_raw):,} | Features: {X_raw.shape[1]}")

        # Optional correlation pruning
        dropped = []
        if args.corr_threshold and args.corr_threshold > 0:
            X_raw, dropped = corr_prune(X_raw, threshold=args.corr_threshold)
            (REPORTS / "pruned_features_v12.txt").write_text("\n".join(dropped))
            feature_names = [c for c in feature_names if c not in dropped]
            tprint(f"[PRUNE] kept_features={len(feature_names)}")

        # Prepare categoricals
        with Timer("Prepare categorical features for LGB/XGB"):
            X_lgb, X_xgb, cat_cols = prepare_X_for_models(X_raw)

        # Train
        lgbm_metrics = train_lgbm(X_lgb, y, groups, feature_names, cat_cols)
        xgb_metrics  = train_xgb(X_xgb, y, groups, feature_names)

        # Save manifest
        manifest = {
            "n_customers": int(len(X_raw)),
            "n_features": int(X_raw.shape[1]),
            "n_splits": int(N_SPLITS),
            "seed": int(SEED),
            "metrics": {"lgbm": lgbm_metrics, "xgb": xgb_metrics},
            "categorical_features": cat_cols,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corr_pruned": len(dropped),
        }
        (REPORTS / "train_manifest_v1.json").write_text(json.dumps(manifest, indent=2))
        tprint(f"[SUMMARY] LGBM OOF: {json.dumps(lgbm_metrics)}")
        tprint(f"[SUMMARY] XGB  OOF: {json.dumps(xgb_metrics)}")

        # --- v1.2: Isotonic calibration on LGBM OOF ---
        tprint("[CAL] Fitting isotonic regression on LGBM OOF ...")
        lgb_oof = pd.read_parquet(OOF_DIR / "lgbm_oof_v1.parquet")["oof"].reindex(y.index).values
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(lgb_oof, y.values)
        joblib.dump(iso, CAL_DIR / "lgbm_iso_v1.joblib")
        tprint("[CAL] saved calibrator -> calibrators/lgbm_iso_v1.joblib")

        # --- v1.2: OOF-based ensemble weight search ---
        tprint("[ENSEMBLE] Searching best alpha (LGB vs XGB) by AMEX-M ...")
        xgb_oof = pd.read_parquet(OOF_DIR / "xgb_oof_v1.parquet")["oof"].reindex(y.index).values
        best_alpha, best_score = 0.5, -1.0
        for a in np.linspace(0, 1, 21):
            score = amex_metric(y.values, a * lgb_oof + (1 - a) * xgb_oof)
            if score > best_score:
                best_alpha, best_score = float(a), float(score)
        (REPORTS / "ensemble_alpha_v12.txt").write_text(f"{best_alpha}\n")
        tprint(f"[ENSEMBLE] best_alpha={best_alpha:.2f}, amex_m={best_score:.6f}")

        tprint(f"Done. mem ~ {mem_used_gb():.1f} GB")
