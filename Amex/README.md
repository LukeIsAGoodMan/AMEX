# AMEX Default Prediction — v1.2 Project

## 🏦 Competition Background & Challenges

American Express hosted the **Default Prediction Competition** on Kaggle, asking participants to predict whether a customer would default in the future based on their transaction and profile data. The dataset includes a large amount of sequential account-level information.

### Key Challenges

* **Massive scale**: tens of millions of rows — efficiency and memory management are critical.
* **Sequential nature**: customer behavior evolves over time, requiring temporal feature engineering (lags, trends, rolling statistics).
* **Extreme class imbalance**: defaults are much rarer than non-defaults.
* **Custom evaluation metric**: the AMEX metric combines Top-4% recall and a weighted Gini, different from conventional ROC-AUC or PR-AUC.

---

## 🚀 Project Iteration

This project was designed as a practical interview exercise, evolving from an initial **1.0 prototype** to the optimized **v1.2** version. The latest version focuses on:

1. **Efficient feature engineering**: addressing fragmentation and memory issues.
2. **Robust model training**: with cross-validation, standard and custom metrics, and feature importance.
3. **Automated reporting**: generating interpretable HTML summaries.

### Modules

1. **10\_make\_features.py**

   * Input: raw parquet/csv with `customer_ID`, `S_2` timestamp, and numeric columns.
   * Output: feature table (shift/delta/delta\_pct + rolling statistics).
   * Improvements:

     * Bulk feature creation to avoid DataFrame fragmentation.
     * Automatic dtype downcasting to reduce memory usage.

2. **20\_train\_model.py**

   * Input: feature table + label table.
   * Output: trained model, out-of-fold (OOF) predictions, cross-validation metrics, permutation feature importance.
   * Model: `HistGradientBoostingClassifier` (sklearn-native, efficient and accurate).
   * Includes a custom implementation of the AMEX metric.

3. **30\_report.py**

   * Input: metrics.json, feat\_importance.parquet, oof.parquet.
   * Output: HTML report with:

     * CV fold metrics
     * Overall OOF metrics
     * Top-N feature importance plot
     * OOF prediction histogram

---

## ⚙️ Usage

### 1. Feature Engineering

```bash
python 10_make_features.py \
  --input data/amex_train_proc.parquet \
  --output features/amex_features_v12.parquet \
  --id customer_ID --time S_2 \
  --cols B_1,B_2,D_39,D_44 \
  --lags 1,2 \
  --roll 3,5 \
  --roll_stats mean,std
```

### 2. Model Training

```bash
python 20_train_model.py \
  --features features/amex_features_v12.parquet \
  --labels data/amex_labels.csv \
  --id customer_ID --target target \
  --model_out models/hgb_v12.joblib \
  --oof_out models/oof_preds_v12.parquet \
  --metrics_out models/metrics_v12.json \
  --featimp_out models/feat_importance_v12.parquet
```

### 3. Report Generation

```bash
python 30_report.py \
  --metrics models/metrics_v12.json \
  --featimp models/feat_importance_v12.parquet \
  --oof models/oof_preds_v12.parquet \
  --id customer_ID --target target \
  --out reports/report_v12.html
```

---

## 📌 Summary

* v1.2 resolves feature engineering bottlenecks and provides a structured, explainable pipeline.
* The modular structure makes it easy to present in interviews: from engineering to modeling to reporting, demonstrating both technical depth and business understanding.
