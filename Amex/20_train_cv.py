# 20_train_cv.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb

FEAT = Path("features")
OOF  = Path("oof");     OOF.mkdir(exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
REPORTS= Path("reports");REPORTS.mkdir(exist_ok=True)

SEED = 2025
N_SPLITS = 5

def amex_metric(y_true, y_pred):
    # 简化版AMEX-M（面试足够），用于参考
    def top_four_pct_captured(y_true, y_pred):
        n = int(0.04 * len(y_pred))
        idx = np.argsort(-y_pred)[:n]
        s = y_true[idx].sum()
        t = y_true.sum()
        return (s / t) if t > 0 else 0.0
    def weighted_gini(y_true, y_score):
        order = np.argsort(-y_score, kind="mergesort")
        y = y_true[order]
        cum_pos = np.cumsum(y)
        total_pos = y_true.sum()
        total_neg = len(y_true) - total_pos
        cum_neg = np.cumsum(1 - y)
        lorentz = cum_pos / total_pos if total_pos > 0 else cum_pos
        gini = np.sum(lorentz - (cum_neg / total_neg if total_neg > 0 else cum_neg))
        return gini / len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 0.5 * (weighted_gini(y_true, y_pred) + top_four_pct_captured(y_true, y_pred))

def evaluate(y, oof):
    return {
        "auc": roc_auc_score(y, oof),
        "pr_auc": average_precision_score(y, oof),
        "amex_m": amex_metric(y, oof)
    }

def load_train():
    df = pd.read_parquet(FEAT/"train_customer_v1.parquet")
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    X.index.name = "customer_ID"
    return X, y, X.index  # groups=customer_ID

def train_lgbm(X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X))
    feat_imp = pd.Series(0.0, index=X.columns)

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=5000,
        random_state=SEED
    )

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va])
        model = lgb.train(params, dtr, valid_sets=[dtr, dva],
                          valid_names=["train","valid"],
                          num_boost_round=5000,
                          callbacks=[lgb.early_stopping(200, verbose=False)])
        pred = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        oof[va] = pred
        feat_imp += pd.Series(model.feature_importance(importance_type="gain"), index=X.columns)
        model.save_model(str(MODELS/f"lgbm_v1_fold{fold}.txt"))
        print(f"[LGBM] fold {fold} done.")

    metrics = evaluate(y, oof)
    print("[LGBM] OOF:", metrics)
    pd.Series(oof, index=X.index, name="oof").to_frame().to_parquet(OOF/"lgbm_oof_v1.parquet")
    feat_imp.to_frame("gain").to_csv(REPORTS/"lgbm_featimp_v1.csv")
    return metrics

def train_xgb(X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X))
    feat_imp = pd.Series(0.0, index=X.columns)

    params = dict(
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=5000,
        tree_method="hist",  # 有GPU可改 "gpu_hist"
        random_state=SEED,
        eval_metric="auc"
    )

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[va], y.iloc[va])],
                  verbose=False,
                  early_stopping_rounds=200)
        pred = model.predict_proba(X.iloc[va])[:,1]
        oof[va] = pred

        imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
        imp.index = [X.columns[int(k[1:])] if k.startswith("f") else k for k in imp.index]
        feat_imp = feat_imp.add(imp, fill_value=0)

        model.save_model(str(MODELS/f"xgb_v1_fold{fold}.json"))
        print(f"[XGB] fold {fold} done.")

    metrics = evaluate(y, oof)
    print("[XGB] OOF:", metrics)
    pd.Series(oof, index=X.index, name="oof").to_frame().to_parquet(OOF/"xgb_oof_v1.parquet")
    feat_imp.to_frame("gain").to_csv(REPORTS/"xgb_featimp_v1.csv")
    return metrics

if __name__ == "__main__":
    X, y, groups = load_train()
    m1 = train_lgbm(X, y, groups)
    m2 = train_xgb(X, y, groups)
