# Predicting Credit Card Fraud

My solution for Kaggle's [Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11):
predict the probability that a card transaction is fraudulent. Submissions are scored on **ROC AUC**.

## Results

5-fold stratified CV on the same fold split for every model.

| Model | CV AUC |
|---|---|
| Logistic regression (baseline) | 0.94182 |
| CatBoost | 0.94548 |
| LightGBM | 0.94633 |
| XGBoost | 0.94642 |
| XGBoost + red-flag count feature | 0.94665 |
| Rank average (LR + LGB + XGB + CAT) | 0.94666 |
| LR stack | 0.94679 |
| **Weighted rank average** | **0.94690** |

Every run is logged in `results.csv`.

## What I found

- **1.67% fraud**, 700k train rows, 300k test rows, no missing values, and no shift between train and test.
- The strongest signals are `merchant_risk_score`, `velocity_score`, `cvv_retry_count` and `auth_method`.
- **CVV retries:** 0.9% fraud with no retries, 8.8% with one, 42% with two or more, and 73% when
  two or more retries happen with no authentication.
- **Network flags compound:** 1.1% fraud with none of foreign / IP mismatch / VPN / billing mismatch,
  6.9% with one, 26% with two, 63% with three.
- **Amount:** risk is flat below $100, peaks at $100-200 (2.4x), and stays elevated above.
- **Not supported by this data:** desktop devices being riskier, and card age (every card is 25-73 months old).
- Logistic regression is only ~0.005 behind the trees: the flags multiply risk, which is additive in
  log-odds, so a linear model already captures most of it. Hand-built domain features add at most +0.0002.

## Project layout

```
src/
  common.py          paths, data loading, shared folds, CV loop
  models.py          logistic regression, LightGBM, XGBoost, CatBoost
  features.py        domain feature groups (network, cvv, amount, riskcount)
  train.py           train one model with CV, save OOF + submission
  blend.py           rank average, weighted average, and LR stack of saved models
  eda.py             per-feature signal and train/test comparison
  explore_domain.py  fraud-domain hypothesis checks
results.csv          CV score of every experiment
```

Generated and git-ignored: `data/`, `oof/`, `submissions/`, `logs/`.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the competition data from Kaggle and unzip it into `data/`. Then:

```bash
python src/eda.py
python src/train.py lr
python src/train.py lgb
python src/train.py xgb --features riskcount
python src/train.py cat
python src/blend.py lr lgb xgb cat
```

Each `train.py` run saves `oof/oof_<name>.npy`, `oof/test_<name>.npy` and `submissions/sub_<name>.csv`.
XGBoost and CatBoost are set to run on an NVIDIA GPU; change `device="cuda"` / `task_type="GPU"` in
`models.py` to run on CPU.
