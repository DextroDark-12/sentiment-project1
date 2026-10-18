import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)

# 1. Price vs Sentiment
feat = pd.read_csv("data/processed/features.csv")
feat["date"] = pd.to_datetime(feat["date"])

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(feat["date"], feat["close"], color="blue", label="BTC Price")
ax1.set_ylabel("Price ($)", color="blue")

"""
creates a secondary vertical axis (y-axis) on the right side of your chart so you can 
plot two completely different scales on the same time timeline
"""
ax2 = ax1.twinx()
ax2.plot(feat["date"], feat["mean_sent"], color="orange", label="Sentiment", alpha=0.7)
ax2.set_ylabel("Sentiment Score", color="orange")

plt.title("BTC Price vs Sentiment")
plt.tight_layout()
plt.savefig("reports/figures/price_vs_sentiment.png")
plt.close()
print("Saved: reports/figures/price_vs_sentiment.png")

# 2. Predicted vs Actual Return (h=1)
pred = pd.read_csv("data/processed/predictions.csv")
h1 = pred[pred["h"] == 1].copy()
h1["date"] = pd.to_datetime(h1["date"])

plt.figure(figsize=(10, 4))
plt.plot(h1["date"], h1["y_ret"], label="Actual Return", color="green")
plt.plot(h1["date"], h1["pred_ret"], label="Predicted Return", color="red", linestyle="--")
plt.axhline(0, color="gray", linestyle=":")
plt.title("Predicted vs Actual Returns (1-Day)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/predicted_vs_actual.png")
plt.close()
print("Saved: reports/figures/predicted_vs_actual.png")

# 3. Strategy Returns & 4. Drawdowns
daily = pd.read_csv("data/processed/backtest_daily.csv")
daily["date"] = pd.to_datetime(daily["date"])

# Normalize equity curves to start at 1.0
strat_cum = daily["equity_strat"] / daily["equity_strat"].iloc[0]
bh_cum = daily["equity_hold"] / daily["equity_hold"].iloc[0]

# Chart 3: Cumulative Return
plt.figure(figsize=(10, 4))
plt.plot(daily["date"], strat_cum, label="Strategy (-15.6%)", color="red")
plt.plot(daily["date"], bh_cum, label="Buy & Hold (+32.4%)", color="blue", linestyle="--")
plt.title("Strategy vs Buy & Hold")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/strategy_vs_buy_and_hold.png")
plt.close()
print("Saved: reports/figures/strategy_vs_buy_and_hold.png")

# Chart 4: Drawdowns
strat_dd = (strat_cum - strat_cum.cummax()) / strat_cum.cummax()
bh_dd = (bh_cum - bh_cum.cummax()) / bh_cum.cummax()

plt.figure(figsize=(10, 4))
plt.plot(daily["date"], strat_dd, label="Strategy Drawdown", color="red")
plt.plot(daily["date"], bh_dd, label="Buy & Hold Drawdown", color="blue", linestyle="--")
plt.title("Portfolio Drawdowns")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/drawdown.png")
plt.close()
print("Saved: reports/figures/drawdown.png")
