"""Per-feature fraud signal, train-vs-test shift, and fraud rate by category."""
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_data

X, y, X_test, _ = load_data()
print(f"train {X.shape}, test {X_test.shape}, fraud rate {y.mean():.4%}\n")


def as_numeric(s):
    return s.cat.codes if s.dtype == "category" else s


rows = []
for col in X.columns:
    if X[col].dtype == "category":
        score = X[col].map(y.groupby(X[col], observed=True).mean()).astype(float)
    else:
        score = X[col]
    auc = roc_auc_score(y, score)
    rows.append({
        "feature": col,
        "nunique": X[col].nunique(),
        "single_auc": round(max(auc, 1 - auc), 4),
        "train_mean": round(float(as_numeric(X[col]).mean()), 3),
        "test_mean": round(float(as_numeric(X_test[col]).mean()), 3),
        "skew": round(float(as_numeric(X[col]).skew()), 2),
    })
print(pd.DataFrame(rows).sort_values("single_auc", ascending=False).to_string(index=False))

print("\nFraud rate by category:")
for col in X.select_dtypes("category").columns:
    rate = y.groupby(X[col], observed=True).agg(["mean", "size"]).sort_values("mean", ascending=False)
    print(f"\n{col}\n{rate.round(4).to_string()}")
