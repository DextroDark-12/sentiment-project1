import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# 1. Fetch live news headlines via Tavily
tavily_key = os.getenv("TAVILY_API_KEY")
headlines = []
if tavily_key:
    try:
        client = TavilyClient(api_key=tavily_key)
        res = client.search(query="Bitcoin market price movement", search_depth="basic", max_results=3)
        headlines = res.get("results", [])
    except Exception as e:
        print(f"Warning: Tavily query failed: {e}")

# 2. Extract latest row from features
feat_path = "data/processed/features.csv"
latest_feat = {}
if os.path.exists(feat_path):
    df_feat = pd.read_csv(feat_path)
    latest_feat = df_feat.iloc[-1].to_dict()

# 3. Read live forecast
live_path = "data/processed/live_forecast.csv"
forecast_rows = []
if os.path.exists(live_path):
    df_live = pd.read_csv(live_path)
    forecast_rows = df_live.to_dict(orient="records")

# 4. Write Markdown report
os.makedirs("reports", exist_ok=True)
summary_path = "reports/daily_sentiment_summary.md"
date_str = str(latest_feat.get("date", datetime.utcnow().strftime("%Y-%m-%d"))).split()[0]

md_content = f"""# Daily Bitcoin Sentiment & Market Summary
**Date:** {date_str}

## 1. Sentiment & Price Snapshot
* **BTC Close Price:** ${latest_feat.get('close', 0):,.2f}
* **Mean Daily Sentiment:** {latest_feat.get('mean_sent', 0):.4f}
* **Engagement-Weighted Sentiment:** {latest_feat.get('wsent_engage', 0):.4f}
* **Total Tracked Posts:** {int(latest_feat.get('post_count', 0))}
* **Bullish vs Bearish Ratio:** {latest_feat.get('pct_bullish', 0)*100:.1f}% Bullish / {latest_feat.get('pct_bearish', 0)*100:.1f}% Bearish

---

## 2. Multi-Horizon Forecast
| Horizon | Predicted Return | Probability Up | Target Price |
| :--- | :--- | :--- | :--- |
"""

for row in forecast_rows:
    h = int(row.get("h", 0))
    ret = row.get("pred_ret", 0) * 100
    prob = row.get("prob_up", 0) * 100
    price = row.get("pred_price", 0)
    md_content += f"| +{h}d | {ret:+.2f}% | {prob:.1f}% | ${price:,.2f} |\n"

md_content += """
---

## 3. Real-Time News (Tavily)
"""

if headlines:
    for h in headlines:
        snippet = h.get("content", "").strip().replace("\n", " ")
        md_content += f"* **[{h.get('title')}]({h.get('url')})**\n  > {snippet[:180]}...\n\n"
else:
    md_content += "*No headlines retrieved.*\n"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Summary generated at: {summary_path}")
