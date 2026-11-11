"""Train one model with 5-fold CV and save its OOF, test predictions and submission.

Usage:
    python src/train.py lr
    python src/train.py xgb --features riskcount
    python src/train.py lgb --features all
"""
import argparse

from common import load_data, run_cv
from features import GROUPS, add_features
from models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=MODELS)
parser.add_argument("--features", default="", help="comma-separated groups from features.py, or 'all'")
args = parser.parse_args()

groups = list(GROUPS) if args.features == "all" else [g for g in args.features.split(",") if g]
name = args.model if not groups else f"{args.model}_fe_{'+'.join(groups)}"

X, y, X_test, test_ids = load_data()
if groups:
    X, X_test = add_features(X, X_test, groups)
print(f"{name}: {X.shape[1]} features")

run_cv(MODELS[args.model], X, y, X_test, test_ids, name, notes=f"features: {groups or 'raw'}")
