# AMEX Default Prediction Practicum (V1)

This repository contains my practicum exam solution (V1 baseline) for a default prediction task, 
based on the [AMEX Kaggle competition dataset](https://www.kaggle.com/competitions/amex-default-prediction).

## Project Roadmap
- **V1 (this branch)**: 
  - Customer-level feature aggregation (last/mean/std/min/max/slope, categorical last/nunique/top1freq)
  - Models: LightGBM & XGBoost with GroupKFold (grouped by customer_ID)
  - Evaluation metrics: AUC, PR-AUC, AMEX-M
- **V1.1 (planned)**:
  - Add lag/rolling/recency features
  - Time-aware validation split
- **V2 (planned)**:
  - CatBoost baseline
  - Simple ensemble (LGBM+XGB)

## Repository Structure
- `00_prepare_data.py` : Convert raw CSV → Parquet, merge labels
- `10_make_features.py`: Aggregate features at customer level
- `20_train_cv.py`     : Train LightGBM/XGB with GroupKFold, save OOF + feature importance
- `30_infer.py`        : Run inference on test set
- `utils/metrics.py`   : Custom AMEX metric + evaluation
- `utils/eda.py`       : Simple EDA utilities

## How to Run
1. Place raw data in `data_raw/`:
   - `train_data.csv`, `train_label.csv`, `test_data.csv`
2. Run preprocessing:
   ```bash
   python 00_prepare_data.py
   python 10_make_features.py
   python 20_train_cv.py
