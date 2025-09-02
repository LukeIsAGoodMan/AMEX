import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path

FEAT = Path("features"); MODELS = Path("models")

def infer_with_lgbm():
    test = pd.read_parquet(FEAT/"test_customer_v1.parquet")
    preds = 0.0
    folds = []
    for p in MODELS.glob("lgbm_v1_fold*.txt"):
        model = lgb.Booster(model_file=str(p))
        preds += model.predict(test, num_iteration=model.best_iteration)
        folds.append(p)
    preds /= len(folds)
    return pd.Series(preds, index=test.index, name="prediction")

def infer_with_xgb():
    test = pd.read_parquet(FEAT/"test_customer_v1.parquet")
    preds = 0.0
    folds = []
    for p in MODELS.glob("xgb_v1_fold*.json"):
        model = xgb.XGBClassifier()
        model.load_model(str(p))
        preds += model.predict_proba(test)[:,1]
        folds.append(p)
    preds /= len(folds)
    return pd.Series(preds, index=test.index, name="prediction")

if __name__ == "__main__":
    # 任选其一生成提交；如果要融合，V1.1 再做
    p = infer_with_lgbm()
    sub = p.reset_index().rename(columns={"index":"customer_ID"})
    sub.to_csv("submission_v1_lgbm.csv", index=False)
