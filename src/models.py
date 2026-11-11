"""Model definitions. Each fit_* function takes (X_tr, y_tr, X_va, y_va) and returns a fitted model."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from common import SEED

ONE_HOT_NUMERIC = ["amount_band"]


def fit_lr(X_tr, y_tr, X_va, y_va):
    numeric = [c for c in X_tr.select_dtypes("number").columns if c not in ONE_HOT_NUMERIC]
    categorical = [c for c in X_tr.columns if c not in numeric]
    skewed = [c for c in numeric
              if X_tr[c].nunique() > 2 and X_tr[c].min() >= 0 and X_tr[c].skew() > 2]
    other = [c for c in numeric if c not in skewed]

    preprocess = make_column_transformer(
        (make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), skewed),
        (StandardScaler(), other),
        (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), categorical),
    )
    model = make_pipeline(preprocess, LogisticRegression(C=1.0, max_iter=2000))
    return model.fit(X_tr, y_tr)


def fit_lgb(X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(
        n_estimators=5000, learning_rate=0.03,
        num_leaves=63, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_X=(X_va,), eval_y=(y_va,), eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    return model


def fit_xgb(X_tr, y_tr, X_va, y_va):
    model = xgb.XGBClassifier(
        n_estimators=5000, learning_rate=0.03,
        max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        tree_method="hist", device="cuda", enable_categorical=True,
        eval_metric="auc", early_stopping_rounds=200, random_state=SEED,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def fit_cat(X_tr, y_tr, X_va, y_va):
    model = CatBoostClassifier(
        iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        eval_metric="AUC", od_type="Iter", od_wait=200,
        task_type="GPU", random_seed=SEED, verbose=0,
        cat_features=X_tr.select_dtypes("category").columns.tolist(),
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return model


MODELS = {"lr": fit_lr, "lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}
