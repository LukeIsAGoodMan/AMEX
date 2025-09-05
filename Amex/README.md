# AMEX Default Prediction – Interview Practicum

This repository contains a compact, reproducible pipeline to build a “sophisticated back-of-the-envelope” default prediction model per the interview brief. The focus is on **sound statistical approach**, **clear communication**, **traceable iteration**, and **reasonable performance** (not leaderboard chasing).

> **Branches**  
> - `v1`: minimal baseline (level-only aggregation)  
> - `v1.1`: adds targeted time-series features (trend block), unified logging, robust inference  
> - `v1.2` (WIP): calibration, feature pruning, slope-on-selected, optimized blending

---

## 1) Project Structure

├─ data_raw/ # raw CSVs (train_data, train_labels, test_data)
├─ data_proc/ # parquet after 00_prepare_data.py
├─ features/ # train/test customer-level features (parquet)
├─ models/ # fold models (LightGBM .txt, XGBoost .json)
├─ oof/ # OOF predictions (parquet)
├─ reports/ # metrics, feature importance, manifest.json
├─ logs/ # per-run logs with timings & exceptions
├─ 00_prepare_data.py # CSV → parquet; basic cleaning / typing
├─ 10_make_features.py # V1.1: aggregation + trend block (+ optional slope)
├─ 20_train_cv.py # GroupKFold CV: LightGBM & XGBoost, metrics & OOF
├─ 30_infer.py # Inference; categorical alignment; (optional) ensemble
├─ features_list.txt # column order for training/inference alignment
└─ README.md


## 2) Environment

- Python ≥ 3.9 (tested on 3.13)
- Packages: `pandas>=2.3`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`, `xgboost`, `psutil`

Create environment (conda example):
```bash
conda create -n amex python=3.10 -y
conda activate amex
pip install -r requirements.txt
Minimal requirements.txt:

pandas>=2.3
numpy
pyarrow
scikit-learn
lightgbm
xgboost
psutil
3) End-to-end Run
Prepare parquet

python 00_prepare_data.py --log-file logs/00_prepare.log
Feature engineering (V1.1)

# Quick smoke test (10% customers, fast stats)
python 10_make_features.py --debug-frac 0.1 --fast --log-file logs/10_fast.log

# Full run (trend block on; slope off by default)
python 10_make_features.py --log-file logs/10_full.log

# Optional: enable slope on selected columns (slower)
python 10_make_features.py --trend-slope --log-file logs/10_full_slope.log
Train (GroupKFold, LightGBM + XGBoost)

python 20_train_cv.py --log-file logs/20_cv.log
Inference (test set)

# LGBM only
python 30_infer.py --log-file logs/30_infer_lgb.log

# LGBM + XGB + 0.5/0.5 ensemble
python 30_infer.py --xgb --ensemble --log-file logs/30_infer_ens.log
Artifacts:

features/train_customer_v1.parquet, features/test_customer_v1.parquet

models/lgbm_v1_fold*.txt, models/xgb_v1_fold*.json

oof/lgbm_oof_v1.parquet, oof/xgb_oof_v1.parquet

reports/lgbm_featimp_v1.csv, reports/xgb_featimp_v1.csv, reports/train_manifest_v1.json

submission_v1_lgbm.csv (and optional submission_v1_xgb.csv, submission_v1_ensemble.csv)

4) What’s in V1 vs V1.1
V1: level-only aggregates (mean/std/min/max/last) + categorical (last/top1freq), GroupKFold CV, LGBM+XGB baselines.

V1.1: adds a targeted trend block on selected strong columns:

prev_last, delta, delta_pct, rolling_mean_3, rolling_std_3, ewm3 (+ optional slope)

Unified file logging: every script writes a timestamped log (args, stages, exceptions, total time, memory)

Robust inference: categorical dtypes aligned with training; XGB probabilities clipped to [0,1]; column order fixed by features_list.txt.

Why this matters: trend features capture trajectory (improving vs deteriorating) beyond static levels, often improving ranking metrics like ROC-AUC/PR-AUC and AMEX-M.

5) Metrics
We report:

ROC-AUC (ranking)

PR-AUC (precision-recall; sensitive to class imbalance)

AMEX-M (competition metric; simplified version implemented)

OOF metrics are logged and saved to reports/train_manifest_v1.json. OOF predictions are saved under oof/ for diagnostics and ensembling.

6) Logging
All scripts accept --log-file. If omitted, a default file is created under logs/ with a timestamp. Logs include:

Arguments snapshot

Stage start/end with elapsed minutes

Memory usage snapshots

Full traceback on exceptions

Final summary (shapes, counts, metrics)

7) Reproducibility & Branching
Determinism: fixed seeds, explicit feature list, explicit categorical handling.

Branches document iteration history:

v1 → v1.1 → v1.2 (WIP)

Each branch includes the code diff, logs, and short notes in PR descriptions.

8) Troubleshooting
LightGBM “categorical_feature do not match” → ensure 30_infer.py loads categorical_features from reports/train_manifest_v1.json and casts test columns to category.

Parquet duplicate column error → ensured by avoiding duplicated names in trend block; safety guard drops duplicates post-concat.

XGB probabilities < 0 or > 1 → we clip to [0,1] post-predict.