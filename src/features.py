"""Domain feature groups. None use the target, so train and test are built together."""
import numpy as np
import pandas as pd

NET_FLAGS = ["is_foreign_transaction", "ip_country_mismatch", "used_vpn", "billing_shipping_mismatch"]
AMOUNT_BINS = [0, 50, 100, 200, 500, 1000, np.inf]


def _no_auth(df):
    return (df["auth_method"] == "No Authentication").astype("int8")


def network(df):
    df["net_flag_count"] = df[NET_FLAGS].sum(axis=1).astype("int8")


def cvv(df):
    no_auth = _no_auth(df)
    df["no_auth"] = no_auth
    df["cvv_any"] = (df["cvv_retry_count"] > 0).astype("int8")
    df["cvv_x_no_auth"] = (df["cvv_retry_count"] * no_auth).astype("int8")


def amount(df):
    category_median = df.groupby("merchant_category", observed=True)["amount_usd"].transform("median")
    df["amount_band"] = pd.cut(df["amount_usd"], AMOUNT_BINS, labels=False).astype("int8")
    df["amount_over_100"] = (df["amount_usd"] > 100).astype("int8")
    df["amount_vs_cat_median"] = df["amount_usd"] / category_median
    df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"] + 1)


def riskcount(df):
    flags = df[NET_FLAGS].sum(axis=1) + (df["cvv_retry_count"] > 0) + _no_auth(df) + df["is_new_merchant"]
    df["risk_flag_count"] = flags.astype("int8")


GROUPS = {"network": network, "cvv": cvv, "amount": amount, "riskcount": riskcount}


def add_features(X, X_test, groups):
    both = pd.concat([X, X_test], keys=["train", "test"])
    for group in groups:
        GROUPS[group](both)
    return both.loc["train"], both.loc["test"]
