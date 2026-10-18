# Evaluation Report: X Sentiment and Price Forecasting for BTC

**Asset:** Bitcoin (BTC-USD), daily bars, all times UTC
**Sample:** 21 Apr 2026 to 17 Oct 2026 (180 days)
**Horizons:** n+0 to n+5
**Short version:** with this data and method, **sentiment and price features did not forecast BTC better than a "no change" guess, and a strategy built on the forecasts lost to buy-and-hold.** The sections below explain how this was measured and why the result should be read cautiously.

---

## 1. Data

**Prices.** Daily BTC-USD closes from Yahoo Finance (`yfinance`), UTC.

**X posts.** Collected with Apify, actor `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`.

- Query: `($BTC OR #bitcoin OR bitcoin) lang:en -filter:replies`, sorted by latest.
- **Sampling:** three one-hour windows per day (00:00, 08:00 and 16:00 UTC), not every post. Posts within each window are the newest ones the actor returned.
- Fields used: text, timestamp, likes, reposts, author follower count. Views and reply counts were collected but not used.

| Stage | Count |
|---|---|
| Rows returned by the actor | 20,636 |
| Unique tweets (the actor returned overlapping pages) | 10,576 |
| After cleaning | 10,075 (about 55 per day) |
| Days covered | 180 of 180, no gaps |

**Cleaning.** Duplicate IDs removed, retweets removed, text must contain "btc" or "bitcoin", minimum 20 characters. The sample is therefore a small, consistent slice of X, not all conversation about BTC. Spam that mentions Bitcoin was not filtered further.

**News.** Tavily headlines are used only in the one-page sentiment summary (`reports/daily_sentiment_summary.md`), never as a model input, since the model is restricted to sentiment and price.

## 2. Method

**Sentiment.** Each tweet is scored with VADER (compound score, -1 to +1). Daily columns: mean sentiment, its spread, follower-weighted and engagement-weighted means (weights are log(1 + followers) and log(1 + likes + reposts)), post count, total engagement, share bullish and share bearish.

**Features (19), sentiment and price only.**
- Price: returns lagged 1, 2, 3 and 5 days, 7-day volatility, 7-day mean return.
- Sentiment: the daily columns above, plus day-to-day changes in sentiment and post volume, and 3- and 7-day moving averages.

**Targets.** For day n and horizon h = 0..5: `y_ret_h = ln(close[n+h] / close[n-1])`, and `y_up_h = 1` if that return is positive. Predicted price = `close[n-1] * exp(predicted return)`.

**Timing rule.** On day n the model sees prices up to the close of day n-1 and the posts of day n.

**Models.** Per horizon, Ridge regression (alpha 10) for the return and logistic regression (C 0.1) for the direction, each preceded by a standard scaler fitted on the training window only. Simple models were chosen deliberately because only about 170 daily rows exist.

**Walk-forward validation.** First prediction after 60 training days. For every test day, a fresh model is trained on past rows only. For horizon h, training rows stop h days before the test day, because the answer of a more recent row contains prices from the test period. One prediction per day per horizon is recorded, then the window moves forward one day.

**No tuning.** Hyperparameters were set before looking at test results and were not changed afterwards.

## 3. Forecast accuracy

Graded days per horizon: 102 to 112.

| h | n | MAE model | MAE naive | RMSE model | RMSE naive | Direction (Ridge) | Direction (Logistic) | Direction (majority guess) |
|---|---|---|---|---|---|---|---|---|
| 0 | 112 | 0.0153 | 0.0131 | 0.0221 | 0.0194 | 0.518 | 0.455 | 0.482 |
| 1 | 110 | 0.0237 | 0.0192 | 0.0338 | 0.0281 | 0.491 | 0.509 | 0.500 |
| 2 | 108 | 0.0297 | 0.0242 | 0.0427 | 0.0361 | 0.444 | 0.481 | 0.352 |
| 3 | 106 | 0.0345 | 0.0276 | 0.0520 | 0.0434 | 0.443 | 0.576 | 0.491 |
| 4 | 104 | 0.0395 | 0.0305 | 0.0592 | 0.0497 | 0.442 | 0.462 | 0.510 |
| 5 | 102 | 0.0434 | 0.0345 | 0.0657 | 0.0557 | 0.422 | 0.422 | 0.441 |

"Naive" predicts zero return, which means the price stays at the previous close. Errors are in log-return units, so 0.0153 is about 1.5%.

**Reading.**
- At every horizon the model's error is **17% to 30% larger** than the naive error. The model has no forecasting skill here.
- Direction accuracy sits around 50% with no consistent pattern. The single best figure (logistic, h = 3, 57.6%) is not evidence of skill: with about 106 test days it is within chance range, and it is the best of 12 numbers examined.
- Errors grow with the horizon for both the model and the naive guess, as expected.

## 4. Strategy backtest

**Rule.** After each day's close, take the model's forecast for the next day only (the h = 1 forecast minus the h = 0 forecast). Go long if positive, short if negative, hold until the next close. A 0.1% cost is charged on every position change. Metrics are annualised with 365 days.

| | Strategy | Buy and hold |
|---|---|---|
| Days tested | 110 | 110 |
| Total return | **-22.3%** | **+35.3%** |
| Sharpe ratio | -2.02 | 2.85 |
| Sortino ratio | -2.35 | 5.72 |
| Max drawdown | -27.2% | -7.0% |
| Position changes | 36 | 0 |

- The strategy's calls were right on 55.5% of days. On 110 days that is within the range expected from chance, and it did not offset the losses.
- BTC rose strongly in this window, so time spent short was costly. Buy-and-hold's high Sharpe reflects one strong uptrend and should not be treated as typical.
- **Return before fees: [FILL IN from the command below]. Fees paid: [FILL IN].**

## 5. Charts

Saved in `reports/figures/`:
- `price_vs_sentiment.png`: BTC price and daily sentiment over time
- `predicted_vs_actual.png`: forecasts against realised values
- `strategy_vs_buy_and_hold.png`: account value of both
- `drawdown.png`: drawdowns of both

## 6. Current outlook (latest model output)

Forecast from the latest row (17 Oct 2026, base = close of 16 Oct, 81,500.80):

| h | Predicted return | Probability up | Predicted price |
|---|---|---|---|
| 0 | -0.24% | 50.6% | 81,304 |
| 1 | -0.17% | 42.6% | 81,361 |
| 2 | -0.44% | 42.5% | 81,142 |
| 3 | -0.91% | 33.4% | 80,766 |
| 4 | -1.15% | 35.4% | 80,568 |
| 5 | -1.10% | 41.3% | 80,606 |

**This is model output, not a trading signal.** Given Section 3, it carries no demonstrated predictive value. The one-page sentiment summary is in `reports/daily_sentiment_summary.md`.

## 7. Limitations

- **Small sample.** About 170 daily rows and about 105 graded days per horizon. Metrics are noisy, and conclusions can easily change with a different period.
- **Thin sentiment signal.** About 55 cleaned posts a day from three one-hour windows, so each daily value is noisy and not a census of X.
- **VADER** is a general-English lexicon and can misread crypto slang and sarcasm. A finance-specific model might do better.
- **Same-day echo.** Posts of day n partly react to day n's price move, which flatters the n+0 horizon and makes it the weakest test of forecasting.
- **One market regime.** BTC rose strongly in this window, and the majority-direction baseline differs a lot between horizons because of it.
- **Backtest simplifications.** Trades at the daily close, a flat 0.1% cost, no slippage or funding costs, and a long/short rule that was fixed in advance.
- **Many comparisons.** Twelve accuracy figures were examined, so one or two looking decent by chance is expected.
- **Not done.** No multi-asset model, no heavier models, no ablation comparing sentiment-and-price against price-only features. Without the ablation, we cannot say from this run whether sentiment adds or subtracts anything beyond the price features.

## 8. Reproducing the results

```
pip install -r requirements.txt
python src/ingest_prices.py
python src/ingest_x.py --token-env APIFY_TOKEN_1 --start 2026-04-21 --end 2026-10-17 --out data/raw/tweets_acc1.jsonl
copy data\raw\tweets_acc1.jsonl data\raw\tweets.jsonl
python src/sentiment.py real
python src/features.py
python src/model.py
python src/backtest.py
python src/plots.py
python src/summary_generator.py
```

API keys go in `.env` (see `.env.example`). The actor ID is given above, and tokens are provided separately.
