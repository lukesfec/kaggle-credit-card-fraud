"""Blend saved OOF predictions three ways: rank average, weighted rank average, and an LR stack.

Usage: python src/blend.py lr lgb xgb cat   (defaults to every saved model)
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import OOF, get_folds, load_data, log_result, save_submission

names = sys.argv[1:] or sorted(p.stem.removeprefix("oof_") for p in OOF.glob("oof_*.npy"))
_, y, _, test_ids = load_data()
oof = {n: np.load(OOF / f"oof_{n}.npy") for n in names}
test = {n: np.load(OOF / f"test_{n}.npy") for n in names}

print("Single models:")
for n in names:
    print(f"  {n:12s} {roc_auc_score(y, oof[n]):.5f}")
print("\nSpearman correlation:")
print(pd.DataFrame(oof).corr(method="spearman").round(4).to_string())


def save(name, pred, auc, notes):
    save_submission(name, test_ids, pred)
    log_result(name, auc, notes)


def ranks(preds):
    return np.column_stack([rankdata(preds[n]) / len(preds[n]) for n in names])


def log_odds(preds):
    return logit(np.clip(np.column_stack([preds[n] for n in names]), 1e-6, 1 - 1e-6))


R, R_test = ranks(oof), ranks(test)

auc = roc_auc_score(y, R.mean(axis=1))
print(f"\nRank average:       {auc:.5f}")
save("blend_rankavg", R_test.mean(axis=1), auc, "+".join(names))


def neg_auc(w):
    w = np.abs(w) / np.abs(w).sum()
    return -roc_auc_score(y, R @ w)


result = minimize(neg_auc, x0=np.full(len(names), 1 / len(names)), method="Nelder-Mead",
                  options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-6})
weights = np.abs(result.x) / np.abs(result.x).sum()
weight_str = " ".join(f"{n}={w:.3f}" for n, w in zip(names, weights))
print(f"Weighted rank avg:  {-result.fun:.5f}  ({weight_str})")
save("blend_weighted", R_test @ weights, -result.fun, weight_str)

S, S_test = log_odds(oof), log_odds(test)
meta_oof = np.zeros(len(y))
for tr, va in get_folds(y):
    meta = LogisticRegression(max_iter=1000).fit(S[tr], y.iloc[tr])
    meta_oof[va] = meta.predict_proba(S[va])[:, 1]
auc = roc_auc_score(y, meta_oof)
meta = LogisticRegression(max_iter=1000).fit(S, y)
coef_str = " ".join(f"{n}={c:.3f}" for n, c in zip(names, meta.coef_[0]))
print(f"LR stack:           {auc:.5f}  ({coef_str})")
save("blend_stack", meta.predict_proba(S_test)[:, 1], auc, coef_str)
