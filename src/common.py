"""Paths, data loading, the shared fold split, and the CV loop every model runs through."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OOF = ROOT / "oof"
SUBS = ROOT / "submissions"
RESULTS = ROOT / "results.csv"

TARGET = "is_fraud"
ID = "id"
SEED = 42
N_FOLDS = 5


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train.pop(TARGET).astype(int)
    train = train.drop(columns=[ID])
    test_ids = test.pop(ID)

    for col in train.columns:
        if pd.api.types.is_bool_dtype(train[col]):
            train[col] = train[col].astype("int8")
            test[col] = test[col].astype("int8")
        elif pd.api.types.is_string_dtype(train[col]):
            categories = sorted(set(train[col]) | set(test[col]))
            train[col] = pd.Categorical(train[col], categories=categories)
            test[col] = pd.Categorical(test[col], categories=categories)

    return train, y, test[train.columns], test_ids


def get_folds(y):
    """Create the fold split once and reuse it, so every model's OOF lines up."""
    path = DATA / "folds.npy"
    if path.exists():
        fold_id = np.load(path)
        assert len(fold_id) == len(y), "folds.npy doesn't match train; delete it to rebuild"
    else:
        fold_id = np.zeros(len(y), dtype=np.int8)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold_id[va] = k
        np.save(path, fold_id)
    return [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(N_FOLDS)]


def log_result(name, auc, notes=""):
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"),
                         "name": name, "cv_auc": round(auc, 6), "notes": notes}])
    row.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def save_submission(name, test_ids, pred):
    SUBS.mkdir(exist_ok=True)
    pd.DataFrame({ID: test_ids, TARGET: pred}).to_csv(SUBS / f"sub_{name}.csv", index=False)


def run_cv(fit_fn, X, y, X_test, test_ids, name, notes=""):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for k, (tr, va) in enumerate(get_folds(y)):
        model = fit_fn(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  fold {k}: AUC {roc_auc_score(y.iloc[va], oof[va]):.5f}")

    auc = roc_auc_score(y, oof)
    print(f"{name} CV AUC: {auc:.5f}")

    OOF.mkdir(exist_ok=True)
    np.save(OOF / f"oof_{name}.npy", oof)
    np.save(OOF / f"test_{name}.npy", test_pred)
    save_submission(name, test_ids, test_pred)
    log_result(name, auc, notes)
    return oof, test_pred
