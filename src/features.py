import numpy as np
import pandas as pd

prices = pd.read_csv("data/raw/btc_prices.csv")
prices["date"] = pd.to_datetime(prices['date'] , utc=True).dt.floor("D")
prices = prices[["date" , "Close"]].rename(columns={"Close":"close"}).sort_values("date")

today = pd.Timestamp.now(tz="UTC").floor("D")
prices = prices[prices["date"] < today]

#sent = sentiments
sent = pd.read_csv(r"F:\ml_model_webtak\ML-projects\data\processed\daily_sentiment.csv")
sent["date"] = pd.to_datetime(sent["date"] , utc=True).dt.floor("D")

#making the final dataframe with sentiments and the prices.
df = prices.merge(sent , on="date" , how="left").reset_index(drop=True)


#just if the dataframe doesnt have any of this value then tirn them to 0.
sent_cols = ["mean_sent" , "sent_std" , "wsent_followers" , "wsent_engage",
             "post_count" , "total_engage" , "pct_bullish" , "pct_bearish"]

df[sent_cols] = df[sent_cols].fillna(0)     # no posts that day -> neutral/zero


"""
now our model needs that what the price are doing recently.So, we have to use returns,
not prices directly because if our model learns that $1000 id good then it will always assume that in any era
but actually it is not beacuse the increase of thousand when BTC was $20k and $100k id different entirely
so we need returns(percentage change in prices) because it is same for both(or any other for that matter).

"""

#ret = returns
df["ret"] = np.log(df["close"]).diff()

# this is just the return k days ago to find some patterns.
for k in [1 , 2,3 ,5]:
    df[f"ret_lag{k}"] = df["ret"].shift(k)

#here from 9th day onwards we will able to see the avg. behaviour of the returns
df["vol_7d"] = df["ret"].shift(1).rolling(7).std()
df["ret_mean_7d"] = df["ret"].shift(1).rolling(7).mean()

# print(df.columns)

"""
gentrating features that add more values(clues) for the model.
"""

df["log_posts"] = np.log1p(df["post_count"]) #we squash the number of tweets, so that a large tweeted day doesnt dominate.


df["sent_chg"] = df["mean_sent"].diff() #did the mood got better or worse?
df["wsent_chg"] = df["wsent_followers"].diff() #same, but weighted by big accounts.
df["posts_chg"] = df["log_posts"].diff() #is more or less being said than yesterday?


#ma = moving avg. of last n days.
"""we did this because the single day's mood is jumpy(maybe some viral news) 
so this moving avg. smooth the noise"""

df["sent_ma3"] = df["mean_sent"].rolling(3).mean() 
df["sent_ma7"] = df["mean_sent"].rolling(7).mean()
df["wsent_ma7"] = df["wsent_followers"].rolling(7).mean()


"""Positive means more bulls than bears, and negative the other way around. 
It’s a more “vote count” view of the mood than the average score, 
because it ignores how strongly each tweet was worded."""

df["bull_bear_spread"] = df["pct_bullish"] - df["pct_bearish"]


#### targets ####

"""
TARGETS (the answer key), explained with an example

We build this table from HISTORY, so for every past day we already know
what happened next. Each row is a "pretend day": we imagine sitting on
that morning and only knowing the past.

Example: pretend we are sitting on 20 July.
  - close_prev  = the close of 19 July (the last price we really know).
                  It is the SAME starting price for all six horizons.
  - h = 0  ->  y_ret_0 = ln( close of 20 July / close of 19 July )
               (the move of 20 July itself)
  - h = 1  ->  y_ret_1 = ln( close of 21 July / close of 19 July )
  - h = 2  ->  y_ret_2 = ln( close of 22 July / close of 19 July )
  ...
  - h = 5  ->  y_ret_5 = ln( close of 25 July / close of 19 July )

So y_ret_h is the TOTAL move from 19 July's close to the close h days
after 20 July. It is NOT the move of one single day.

df["close"].shift(-h) pulls the close from h days in the FUTURE onto the
current row (a negative shift looks forward). That is only possible
because the future of a past day is already in our history file.

y_up_h is 1 if y_ret_h > 0 (price ended above 19 July's close), else 0.

Edges of the table:
  - First row: close_prev is blank (no day before it), so all targets
    are blank and the row is dropped later.
  - Last rows: their future has not happened yet, so the later horizons
    are blank. Last close is 13 Oct, so 13 Oct only has h=0, 12 Oct has
    h=0..1, and so on. The last row is used for the live forecast.

IMPORTANT: y_ret_* and y_up_* are never given to the model as clues.
They are only used (a) for training on past days and (b) to grade the
model's guesses afterwards.
"""

df["close_prev"] = df["close"].shift(1)

for h in range(6):
    df[f"y_ret_{h}"] = np.log(df["close"].shift(-h) / df["close_prev"])
    df[f"y_up_{h}"] = (df[f"y_ret_{h}"] > 0).astype(float).where(df[f"y_ret_{h}"].notna())


####  final cleaning , safety-check and saving the dataframe. ####

assert np.allclose(df["ret_lag1"].iloc[10], df["ret"].iloc[9])

feature_cols = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "vol_7d", "ret_mean_7d",
                "mean_sent", "wsent_followers", "wsent_engage", "sent_std", "log_posts",
                "total_engage", "bull_bear_spread", "sent_chg", "wsent_chg", "posts_chg",
                "sent_ma3", "sent_ma7", "wsent_ma7"]

#checks whether this columns are correct or not if not then script stops with assertionError else move-on.
df = df.dropna(subset=feature_cols).reset_index(drop=True)

df.to_csv("data/processed/features.csv", index=False)
with open("data/processed/feature_cols.txt", "w") as f:
    f.write("\n".join(feature_cols))

print(df.shape)
print(df[["date", "close", "mean_sent", "post_count", "y_ret_0", "y_ret_5"]].tail(8))
print("rows with all 6 targets:", df[[f"y_ret_{h}" for h in range(6)]].dropna().shape[0])


