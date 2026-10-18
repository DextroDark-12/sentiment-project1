import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import sys
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ONE switch: dev data now, real data later
# python src/sentiment.py          -> dev
# python src/sentiment.py real     -> real
USE_REAL = len(sys.argv) > 1 and sys.argv[1] == "real"
IN_PATH = "data/raw/tweets.jsonl" if USE_REAL else "data/dev/tweets_dev.jsonl"
OUT_PATH = "data/processed/daily_sentiment.csv"

import json
with open(IN_PATH, encoding="utf-8") as f:
    df = pd.json_normalize([json.loads(line) for line in f])
df["id"] = df["id"].astype(str)
print("raw rows:", len(df))

# --- timestamps: UTC everywhere ---
df["ts"] = pd.to_datetime(df["createdAt"], utc=True)
df["date"] = df["ts"].dt.floor("D")

# --- tolerate missing columns (your real actor may not have them) ---
for col in ["likeCount", "retweetCount", "viewCount", "author.followers", "isRetweet"]:
    if col not in df.columns:
        df[col] = 0 if col != "isRetweet" else False
df = df.rename(columns={"author.followers": "followers"})
df[["likeCount", "retweetCount", "followers"]] = (
    df[["likeCount", "retweetCount", "followers"]].fillna(0)
)

# --- cleaning ---
df = df.drop_duplicates(subset="id")
df = df[~df["isRetweet"].fillna(False).astype(bool)]
df = df[df["text"].str.contains(r"\bbtc\b|bitcoin", case=False, regex=True)]
df = df[df["text"].str.len() >= 20]
print("after cleaning:", len(df))

# --- sentiment (-1 bearish ... +1 bullish) ---
vader = SentimentIntensityAnalyzer()
df["sent"] = df["text"].apply(lambda t: vader.polarity_scores(t)["compound"])

# --- weights: log keeps huge accounts from dominating ---
df["w_followers"] = np.log1p(df["followers"])
df["w_engage"] = np.log1p(df["likeCount"] + df["retweetCount"])
df["engage"] = df["likeCount"] + df["retweetCount"]

def wavg(g, w):
    return np.average(g["sent"], weights=g[w]) if g[w].sum() > 0 else g["sent"].mean()

daily = df.groupby("date").apply(
    lambda g: pd.Series({
        "mean_sent": g["sent"].mean(),
        "sent_std": g["sent"].std(),
        "wsent_followers": wavg(g, "w_followers"),
        "wsent_engage": wavg(g, "w_engage"),
        "post_count": len(g),
        "total_engage": g["engage"].sum(),
        "pct_bullish": (g["sent"] > 0.05).mean(),
        "pct_bearish": (g["sent"] < -0.05).mean(),
    }),
    include_groups=False,
).reset_index()

import os
os.makedirs("data/processed", exist_ok=True)
daily.to_csv(OUT_PATH, index=False)
print(daily.head(), "\n", len(daily), "days written to", OUT_PATH)
