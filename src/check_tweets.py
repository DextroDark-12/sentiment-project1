import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import json
import pandas as pd

rows = [json.loads(l) for l in open("data/raw/tweets_acc1.jsonl", encoding="utf-8")]
df = pd.json_normalize(rows)
df["ts"] = pd.to_datetime(df["createdAt"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
df["day"] = df["ts"].dt.floor("D")

print("tweets:", len(df), "| unique ids:", df["id"].nunique())
print("range:", df["day"].min().date(), "->", df["day"].max().date())

per_day = df.groupby("day").size()
all_days = pd.date_range(df["day"].min(), df["day"].max(), freq="D", tz="UTC")
print("days with tweets:", len(per_day), "of", len(all_days))
print("missing days:", [d.date().isoformat() for d in all_days.difference(per_day.index)])
print(per_day.describe().round(0))
print("followers present:", round(df["author.followers"].notna().mean(), 3))
print("mentions btc/bitcoin:", round(df["text"].str.contains(r"\bbtc\b|bitcoin", case=False, regex=True).mean(), 3))

