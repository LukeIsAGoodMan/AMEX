import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def amex_metric(y_true, y_pred):
    # 参考实现（简化版），用于线下对照；面试用“我们能复现竞赛指标思路”
    def top_four_pct_captured(y_true, y_pred):
        # 按得分降序取前4%
        n = int(0.04 * len(y_pred))
        idx = np.argsort(-y_pred)[:n]
        return y_true[idx].sum() / y_true.sum() if y_true.sum() > 0 else 0.0

    def weighted_gini(y_true, y_score):
        df = np.c_[y_true, y_score, np.arange(len(y_true))]
        df = df[np.lexsort((df[:,2], -df[:,1]))]  # 按得分降序
        total_pos = y_true.sum()
        total_neg = len(y_true) - total_pos
        cum_pos = np.cumsum(df[:,0])
        cum_neg = np.cumsum(1 - df[:,0])
        lorentz = cum_pos / total_pos if total_pos > 0 else cum_pos
        gini = np.sum(lorentz - (cum_neg / total_neg if total_neg > 0 else cum_neg))
        return gini / len(y_true)

    gini = weighted_gini(y_true, y_pred)
    dr4 = top_four_pct_captured(y_true, y_pred)
    return 0.5 * (gini + dr4)

def basic_classification_metrics(y_true, y_pred_proba):
    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "amex_m": amex_metric(y_true.values if hasattr(y_true, "values") else y_true,
                              y_pred_proba)
    }
