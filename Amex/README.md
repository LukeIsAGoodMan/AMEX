# AMEX Default Prediction — v1.2 Project

## 🏦 Competition Background & Challenges

American Express hosted the **Default Prediction Competition** on Kaggle, asking participants to predict whether a customer would default in the future based on their transaction and profile data. The dataset includes a large amount of sequential account-level information.

### Key Challenges

* **Massive scale**: tens of millions of rows — efficiency and memory management are critical.
* **Sequential nature**: customer behavior evolves over time, requiring temporal feature engineering (lags, trends, rolling statistics).
* **Extreme class imbalance**: defaults are much rarer than non-defaults.
* **Custom evaluation metric**: the AMEX metric combines Top-4% recall and a weighted Gini, different from conventional ROC-AUC or PR-AUC.

---

## 🚀 Project Iteration Journey

This project was designed as a practical interview exercise, evolving in multiple phases. Below is the roadmap from project initiation through the current **v1.2**, and a look ahead to **v1.3**.

### 📍 Project Initiation

* Goal: Build a reproducible pipeline to simulate solving a real Kaggle challenge under interview conditions.
* Focus: Show end-to-end capability — from data processing to modeling and reporting.

### 🔖 v1.0 — Prototype

* First working pipeline.
* Feature engineering done with iterative column insertions, leading to fragmentation and performance warnings.
* Simple model training with baseline metrics.
* Delivered minimal results to prove feasibility.

### 🔖 v1.0.x Performance Fixes

* Addressed runtime bottlenecks in feature creation.
* Began experimenting with dtype optimization to lower memory usage.
* Added preliminary CV training loop.

### 🔖 v1.1 — Stabilization

* Restructured scripts into modular steps (`10_make_features.py`, `20_train_model.py`, `30_report.py`).
* Introduced clearer outputs (parquet, json, HTML).
* Improved reproducibility and reusability for interview demo.

### 🔖 v1.2 — Current Version (Optimization)

* **Feature Engineering**: moved to vectorized, bulk feature creation to avoid DataFrame fragmentation.
* **Model Training**: robust 5-fold CV with multiple metrics (ROC-AUC, PR-AUC, AMEX).
* **Explainability**: added permutation feature importance and OOF predictions.
* **Automated Reporting**: generates HTML report with metrics tables and charts.

### 🔖 v1.3 — Planned

* **Parallelization**: leverage multiprocessing or Dask for feature generation on large datasets.
* **Model Stacking/Ensembling**: combine multiple learners (LightGBM, CatBoost, NN) for higher accuracy.
* **Hyperparameter Search**: integrate Optuna for automated tuning.
* **Deployment Readiness**: package as a lightweight inference pipeline for production-style usage.

---

## 📦 Current v1.2 Modules

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

## 📈 Extended Project Iteration History

Below is a detailed, end‑to‑end history from project initiation through v1.2, plus the roadmap for v1.3.

### 🧭 Project Kickoff (Initiation)

* **Objective:** Build an interview‑ready, reproducible pipeline for the AMEX default prediction task, balancing accuracy, speed, and explainability.
* **Primary risks:** very large sequential data; memory pressure; metric not aligned with standard AUC.
* **Design tenets:** correctness before cleverness; avoid leakage; prefer simple, inspectable steps; produce artifacts for storytelling (metrics, OOF, importance, report).

### v1.0 — MVP Prototype

* **Scope:** fast baseline to validate data joins and evaluation.
* **Features:** last observation, simple deltas; minimal handling of missing values.
* **Models:** Logistic Regression (sanity), sklearn HistGradientBoosting (quick tree‑based baseline).
* **CV & Metrics:** Stratified 5‑fold; ROC‑AUC / PR‑AUC / early AMEX metric.
* **Issues:**

  * Column‑by‑column inserts → **DataFrame fragmentation** & slow wall‑time.
  * Inconsistent `(customer_ID, S_2)` ordering caused occasional mis‑lagging.
  * No CLI or reports; limited reproducibility.

### v1.0 Hotfixes — Performance Patches

* Switched to **Parquet + PyArrow**; reduced chained assignments and unnecessary copies.
* Partially vectorized some loops; precomputed feature column lists.
* Result: better, but **fragmentation warnings** and rolling‑window slowness remained.

### v1.1 — Structure & Reliability

* **Pipeline split:** `10_make_features.py` / `20_train_model.py` / `30_report.py` with **CLI**.
* **Correctness:** enforced sorting by `(customer_ID, S_2)` before lags; consistent seeds.
* **Artifacts:** OOF predictions; finalized AMEX metric; permutation feature importance.
* **Memory:** dtype downcasting; optional caching; clearer separation of raw vs features.
* **Remaining gaps:** iterative feature construction still created many temporaries; rolling stats not optimized; duplicate logic across blocks.

### v1.2 — Optimization (Current)

* **Bulk & vectorized feature engineering:** lags, deltas, and pct‑deltas computed per group in **one pass**, then **single concat** → removes fragmentation warnings.
* **Rolling windows refactor:** one `groupby.rolling` pipeline per window with consolidated concat; order restored via index mapping.
* **Memory guardrails:** dtype downcasting on by default; optional rolling windows; minimized deep copies and intermediate frames.
* **DevX:** stable CLI; deterministic CV; **HTML report** (CV table, top‑N importance, score histogram); saved model & OOF for auditability.
* **Outcome:** lower wall‑time & RAM, cleaner code, and interview‑friendly deliverables.

### v1.3 — Roadmap (Planned)

* **Model backends:** LightGBM/XGBoost/CatBoost options (with early stopping); group‑aware CV by `customer_ID` (GroupKFold); probability calibration (Platt/Isotonic).
* **Feature depth:** expanding stats; last‑k trend signals; change‑point heuristics; leakage‑safe target encoding; interaction screening.
* **Efficiency:** optional GPU for LGBM/XGB; out‑of‑core; Polars/Arrow pipelines; multiprocessing & smarter caching.
* **MLOps:** unit tests; CI (GitHub Actions); schema checks (pandera); YAML configs; experiment tracking (MLflow); Docker image.
* **Explainability:** SHAP (sampled) and feature documentation cards.
* **Acceptance gates:** (to be measured) wall‑time ↓ on multi‑million rows; RAM peak within budget; **no AMEX metric regression** vs v1.2; reproducible runs.

### 🔧 Technology Stack & Design Rationale

* **GroupKFold CV**: chosen because leakage is a high risk in sequential financial data. Splitting by `customer_ID` ensures that all records for a customer appear in only one fold, preserving independence between train/validation and simulating real‑world deployment.
* **LightGBM & XGBoost Fusion**:

  * **LightGBM** excels in speed and memory usage on large tabular datasets with many features.
  * **XGBoost** provides robust regularization and consistent performance across folds.
  * By ensembling/blending the two, we capture the strengths of both frameworks, improve generalization, and reduce variance.
* **Why not only sklearn HGB?**: sklearn’s HistGradientBoosting was used as a clean, portable baseline, but it lacks some advanced boosting features (GPU, monotone constraints, custom losses). LGBM/XGB are industry‑standard for financial risk modeling.
* **Supporting libraries**: pandas/NumPy for preprocessing, joblib for model persistence, matplotlib for lightweight plots, pyarrow/parquet for efficient storage.

### Changelog Snapshot

| Version | Focus        | Key additions                                                           |
| ------- | ------------ | ----------------------------------------------------------------------- |
| v1.0    | MVP          | Baseline features & models; first CV & metrics                          |
| Hotfix  | Performance  | Parquet I/O; fewer copies; partial vectorization                        |
| v1.1    | Structure    | 10/20/30 split; CLI; AMEX metric; OOF & importance                      |
| v1.2    | Optimization | Vectorized bulk features; rolling refactor; dtype downcast; HTML report |
| v1.3    | Roadmap      | LGBM/XGB/CatBoost; GPU; MLOps; SHAP                                     |

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
* The iteration history demonstrates growth from prototype to optimization, with a clear roadmap toward v1.3.
* This journey shows both technical capability and structured problem-solving — highly relevant for interview storytelling.
