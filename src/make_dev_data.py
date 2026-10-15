import json, random
from datetime import datetime, timedelta, timezone

random.seed(0)
texts = [
    "$BTC breaking out, bullish!", "Bitcoin looks weak, selling",
    "Bitcoin ETF inflows strong today", "$BTC rejected at resistance",
    "Great day for bitcoin", "Bitcoin is crashing, terrible",
]
start = datetime.now(timezone.utc) - timedelta(days=180)
rows = []
for i in range(180 * 40):
    t = start + timedelta(seconds=random.randint(0, 180 * 86400))
    rows.append({
        "id": str(10**18 + i),
        "text": random.choice(texts),
        "createdAt": t.strftime("%a %b %d %H:%M:%S +0000 %Y"),
        "likeCount": random.randint(0, 200),
        "retweetCount": random.randint(0, 30),
        "author": {"followers": random.randint(10, 100000)},
    })

import os
os.makedirs("data/dev", exist_ok=True)
with open("data/dev/tweets_dev.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "fake tweets written")