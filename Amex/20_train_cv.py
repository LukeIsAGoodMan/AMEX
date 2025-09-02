import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from utils.metrics import basic_classification_metrics
import lightgbm as lgb
import xgboost as xgb

FEAT = Path("features")
OOF  = Path("oof");     OOF.mkdir(exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
REPORTS= Path("reports");REPORTS.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 5

def get_data():
    df = pd.read_parquet(FEAT/"train_customer_v1.parquet")
    y  = df["target"].astype(int)
    X  = df.drop(columns=["target"])
    groups = X.index  # customer_ID 作为 index
    return X, y, groups

def train_lgbm(X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X), dtype=float)
    feat_imp = pd.Series(0.0, index=X.columns)

    params = dict(objective="binary",
                  learning_rate=0.05,
                  num_leaves=64,
                  feature_fraction=0.8,
                  bagging_fraction=0.8,
                  bagging_freq=1,
                  n_estimators=5000,
                  random_state=SEED)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va])
        model = lgb.train(params, dtr, valid_sets=[dtr, dva],
                          valid_names=["train","valid"],
                          num_boost_round=5000,
                          callbacks=[lgb.early_stopping(200, verbose=False)])
        pred = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        oof[va] = pred
        feat_imp += pd.Series(model.feature_importance(importance_type="gain"),
                              index=X.columns)
        model.save_model(str(MODELS/f"lgbm_v1_fold{fold}.txt"))
        print(f"[LGBM] fold{fold} done.")

    metrics = basic_classification_metrics(y, oof)
    print("[LGBM] OOF:", metrics)
    pd.Series(oof, index=X.index).to_frame("oof").to_parquet(OOF/"lgbm_oof_v1.parquet")
    feat_imp.to_frame("gain").to_csv(REPORTS/"lgbm_featimp_v1.csv")
    return metrics

def train_xgb(X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X), dtype=float)
    feat_imp = pd.Series(0.0, index=X.columns)

    params = dict(
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=5000,
        tree_method="hist",  # 若有GPU: "gpu_hist"
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

        # xgb 的特征重要性
        imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
        imp.index = [k.replace("f","f_") for k in imp.index]  # 防止列名不匹配
        feat_imp = feat_imp.add(imp, fill_value=0.0)

        model.save_model(str(MODELS/f"xgb_v1_fold{fold}.json"))
        print(f"[XGB] fold{fold} done.")

    metrics = basic_classification_metrics(y, oof)
    print("[XGB] OOF:", metrics)
    pd.Series(oof, index=X.index).to_frame("oof").to_parquet(OOF/"xgb_oof_v1.parquet")
    feat_imp.to_frame("gain").to_csv(REPORTS/"xgb_featimp_v1.csv")
    return metrics

if __name__ == "__main__":
    X, y, groups = get_data()
    lgbm_metrics = train_lgbm(X, y, groups)
    xgb_metrics  = train_xgb(X, y, groups)
    # V1 不做融合，只记录成绩，报告里对比即可
