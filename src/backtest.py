import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import os
import numpy as np
import pandas as pd

COST = 0.001          # 0.1% fee each time we change position
DAYS_PER_YEAR = 365   # crypto trades every day

# ---------- PART 1: tomorrow's guess and tomorrow's real result ----------
pred = pd.read_csv(r"F:\ml_model_webtak\ML-projects\btc-sentiment\data\processed\predictions.csv", parse_dates=["date"])
p0 = pred[pred["h"] == 0].set_index("date")
p1 = pred[pred["h"] == 1].set_index("date")


# (Yesterday -> Tomorrow) - (Yesterday -> Today) = (Today -> Tomorrow)$$
bt = pd.DataFrame({
    "pred_next": p1["pred_ret"] - p0["pred_ret"],   # what the model expects tomorrow to do
    "real_next": p1["y_ret"] - p0["y_ret"],         # what tomorrow really did
}).dropna()

# ---------- PART 2: positions, fees, account value ----------
bt["asset_ret"] = np.expm1(bt["real_next"])         # log return -> normal % return
bt["pos"] = np.where(bt["pred_next"] > 0, 1, -1)    # +1 = buy, -1 = short

"""
If yesterday you were Long (1) and today you stay Long (1), 
the difference is 0 ,no fees.

If yesterday you were Long (1) and today you flip to Short (-1), 
the difference is |-1 - 1| = 2. You had to execute two trades: one to sell your long position, 
and one to open the short position.
"""
turnover = bt["pos"].diff().abs().fillna(1)         # how much we changed position
bt["cost"] = COST * turnover

#what profit u make.
bt["strat_ret"] = bt["pos"] * bt["asset_ret"] - bt["cost"]

"""
Building the Account Balance (Equity Curve): .cumprod() (cumulative product) takes a starting balance of 1.00 (e.g., ₹1) and compounds it day by day. 
This simulates your actual bank account growing or shrinking over the backtest. 
We calculate this for both your ML strategy and a "Buy and Hold" baseline to see if your active trading actually beats simply buying and doing nothing.
"""
bt["equity_strat"] = (1 + bt["strat_ret"]).cumprod()   # 1.00 grows into this
bt["equity_hold"] = (1 + bt["asset_ret"]).cumprod()    # same, for just holding

# ---------- PART 3: the three numbers Gilad asked for ----------
def metrics(r, equity):
    mean, std = r.mean(), r.std()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())   # only the bad days
    return {
        "total_return": equity.iloc[-1] - 1,
        "sharpe": mean / std * np.sqrt(DAYS_PER_YEAR) if std > 0 else np.nan,
        "sortino": mean / downside * np.sqrt(DAYS_PER_YEAR) if downside > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }

res = pd.DataFrame({
    "strategy": metrics(bt["strat_ret"], bt["equity_strat"]),
    "buy_and_hold": metrics(bt["asset_ret"], bt["equity_hold"]),
}).T

print("days tested:", len(bt))
print("direction right:", round((np.sign(bt["pred_next"]) == np.sign(bt["real_next"])).mean() * 100, 1), "%")
print("position changes:", int((turnover > 0).sum()))
print(res.round(3).to_string())

os.makedirs("data/processed", exist_ok=True)
bt.to_csv("data/processed/backtest_daily.csv")
res.to_csv("data/processed/backtest_metrics.csv")
