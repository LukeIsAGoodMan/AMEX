import pandas as pd
import polars as pl
import numpy as np
import sklearn
import xgboost
import lightgbm
import catboost
import pyarrow
import shap
import matplotlib

from sklearn.linear_model import LogisticRegression
print("All packages imported successfully!")
# 做一个简单的 DataFrame
df = pd.DataFrame({
    "x1": np.random.randn(10),
    "x2": np.random.randn(10),
    "y": np.random.randint(0, 2, 10)
})
print(df)

# 简单训练一个模型
X = df[["x1", "x2"]]
y = df["y"]
model = LogisticRegression()
model.fit(X, y)
print("Model coefficients:", model.coef_)