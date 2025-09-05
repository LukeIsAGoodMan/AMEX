import pandas as pd
import numpy as np
y = pd.read_parquet("features/train_customer_v1.parquet")["target"]
lgb_oof = pd.read_parquet("oof/lgbm_oof_v1.parquet")["oof"]
xgb_oof = pd.read_parquet("oof/xgb_oof_v1.parquet")["oof"]
def snap(s): 
    return dict(mean=float(s.mean()), p10=float(np.percentile(s,10)), 
                p50=float(np.percentile(s,50)), p90=float(np.percentile(s,90)))
print("pos_rate", float(y.mean()))
print("lgb_oof", snap(lgb_oof))
print("xgb_oof", snap(xgb_oof))
