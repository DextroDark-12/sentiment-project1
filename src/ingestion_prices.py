import os
import pandas as pd
import yfinance as yf

os.makedirs("data/raw", exist_ok=True)

df = yf.download("BTC-USD", period="200d", interval="1d", auto_adjust=True, progress=False)

# newer yfinance versions return two-level column names; flatten them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.index = pd.to_datetime(df.index, utc=True)
df.index.name = "date"
df.to_csv("data/raw/btc_prices.csv")

print(df.tail())
print(len(df), "rows | first:", df.index.min().date(), "| last:", df.index.max().date())