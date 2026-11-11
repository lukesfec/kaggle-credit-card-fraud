"""Check fraud-domain hypotheses: CVV retries, amount, card age, device, IP and network flags."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_data
from features import NET_FLAGS

X, y, _, _ = load_data()
base = y.mean()
pd.set_option("display.width", 200)


def rate(by, title, min_n=200):
    g = y.groupby(by, observed=True).agg(fraud_rate="mean", n="size")
    g["lift"] = g["fraud_rate"] / base
    print(f"\n== {title} ==\n{g[g['n'] >= min_n].round(4).to_string()}")


print(f"Overall fraud rate: {base:.4%}")

rate(X["cvv_retry_count"], "CVV retries")
rate([X["cvv_retry_count"].clip(upper=2), X["auth_method"]], "CVV retries (2 = 2+) x auth method")

rate(pd.cut(X["amount_usd"], [0, 10, 25, 50, 100, 200, 300, 500, 1000, np.inf]), "Amount bands")
rate([pd.cut(X["amount_usd"], [0, 50, 200, 500, np.inf]), X["merchant_category"]],
     "Amount band x merchant category", min_n=500)
amount_to_balance = X["amount_usd"] / (X["account_balance_usd"] + 1)
for name, s in [("amount_usd", X["amount_usd"]), ("account_balance_usd", X["account_balance_usd"]),
                ("amount / balance", amount_to_balance)]:
    auc = roc_auc_score(y, s)
    print(f"AUC {name:20s} {max(auc, 1 - auc):.4f}")

rate(X["card_age_months"] // 6 * 6, "Card age (6-month bins)", min_n=50)

desktop = X["device_type"].isin(["Windows PC", "Mac"]).map({True: "desktop", False: "other"})
rate(desktop, "Desktop vs other devices")
rate([desktop, X["ip_country_mismatch"]], "Desktop x IP country mismatch")

rate(X[NET_FLAGS].sum(axis=1), "Network flag count")
rate([X["used_vpn"], X["ip_country_mismatch"], X["is_foreign_transaction"]],
     "VPN x IP mismatch x foreign (possible hidden proxy)", min_n=50)
for i, a in enumerate(NET_FLAGS):
    for b in NET_FLAGS[i + 1:]:
        both = (X[a] == 1) & (X[b] == 1)
        print(f"  {a} & {b}: n={both.sum():5d}, fraud rate {y[both].mean():.4f}")
