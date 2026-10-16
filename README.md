# BTC X-Sentiment Price Prediction

A Python pipeline that forecasts Bitcoin's price at **n+0, n+1, n+2, n+3, n+4 and n+5** (daily bars) using **only X (Twitter) sentiment and price history**, validated out-of-sample with walk-forward validation.

Built as an internship assessment (Quantitative Data Scientist). **Work in progress.**

---

## Status

| Stage | Status |
|---|---|
| Price ingestion (yfinance) | Done |
| Tweet ingestion via Apify | Blocked by free-tier limits (see [Data](#data-situation-read-this-first)) |
| Synthetic tweet generator (for debugging only) | Done |
| Sentiment scoring and daily aggregation | Done |
| Feature engineering and targets | Done |
| Single-split model demo (`train_one.py`) | Done |
| Walk-forward model, all 6 horizons (`model.py`) | Written, being reviewed |
| Strategy backtest (Sharpe, Sortino, Max Drawdown) | To do |
| News via Tavily and 1-page sentiment summary | To do |
| Charts (sentiment vs price, equity curve, drawdown) | To do |
| Multi-asset extension | To do |
| Evaluation report | To do |

## Data situation (read this first)

- **Real tweet data is not in the pipeline yet.** The Apify free plan capped what could be scraped (monthly run limit, small per-run item cap, and a limit on concurrent runs), and no public dataset tested so far has the needed fields and time range.
- Everything downstream was therefore built and debugged on **synthetic placeholder tweets** (`src/make_dev_data.py`). **No result computed from synthetic data is a finding.** None will appear in the final report.
- A public Bitcoin tweet dataset on Hugging Face (mirror of a Kaggle set) was inspected and rejected: it covers 2014-2019, is dominated by spam and ads, and has only date and text.
- Plan: validate on a suitable public historical tweet dataset (date and text are the only mandatory fields) and add a small recent Apify scrape for the current sentiment outlook. Missing fields (likes, reposts, views, followers) will be documented as limitations.

## What the model predicts

For each day **n** (a "pretend day" in history) and each horizon **h = 0..5**:

- **Regression target** `y_ret_h = ln( close[n+h] / close[n-1] )`: the log return from the previous day's close to the close h days after day n.
- **Direction target** `y_up_h = 1` if `y_ret_h > 0`, else `0` (blank when the future is not yet known).
- Predicted price = `close[n-1] * exp(predicted return)`.

All six targets share the same base price (`close[n-1]`).

**Timing rule (no leakage):** on day n the model may use prices up to the close of day n-1 and the tweets of day n. Nothing later.

## Project structure

```
.
├── .env                      # secrets (never committed)
├── .env.example              # template with blank keys
├── requirements.txt
├── src/
│   ├── ingest_prices.py      # BTC daily prices (yfinance)
│   ├── ingest_x.py           # Apify scraper loop (work in progress)
│   ├── make_dev_data.py      # synthetic tweets for debugging only
│   ├── sentiment.py          # clean tweets, VADER score, daily aggregation
│   ├── features.py           # join with prices, build clues and targets
│   ├── train_one.py          # single train/test split for one horizon (teaching demo)
│   └── model.py              # walk-forward validation, all horizons
├── data/
│   ├── raw/                  # downloaded data, untouched
│   ├── dev/                  # synthetic data (debugging only)
│   └── processed/            # computed tables
└── reports/                  # report and charts (to do)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install pandas numpy yfinance apify-client tavily-python vaderSentiment scikit-learn matplotlib python-dotenv
```

Create a `.env` file (never commit it):

```
APIFY_TOKEN=your_token_here
TAVILY_API_KEY=your_key_here
```

## How to run

```bash
python src/ingest_prices.py     # -> data/raw/btc_prices.csv
python src/make_dev_data.py     # -> data/dev/tweets_dev.jsonl   (synthetic)
python src/sentiment.py         # -> data/processed/daily_sentiment.csv   (add "real" to use data/raw/tweets.jsonl)
python src/features.py          # -> data/processed/features.csv, feature_cols.txt
python src/train_one.py         # one split, one horizon (demo)
python src/model.py             # walk-forward -> predictions.csv, forecast_metrics.csv, live_forecast.csv
```

Each script reads the previous script's output file.

## Pipeline details

### 1. Prices (`ingest_prices.py`)
Downloads about 200 days of BTC-USD daily bars from Yahoo Finance (via `yfinance`), flattens the column names, converts the index to UTC and saves `data/raw/btc_prices.csv`. Only the close is used later. The extra days beyond 180 absorb the warm-up needed by lags and rolling windows.

### 2. Tweets (`ingest_x.py` and Apify)
Search query: `($BTC OR #bitcoin OR bitcoin) lang:en -filter:replies since:... until:...`.

Actors tested:

| Actor | Outcome |
|---|---|
| `xquik/x-tweet-scraper` | Returned text and author followers; no likes/views; timestamp can be decoded from the tweet ID |
| `apidojo/twitter-scraper-lite` | Returned text, timestamp, likes, retweets; no views/followers; free-plan item cap and monthly run limit |
| `feedminer/x-tweet-scraper` | Run succeeded but returned 0 tweets for the test query |

Findings: search results come back newest-first and the item cap applies to the whole run, so one query per day only samples the last minutes of the day. Sampling needs small time windows. Known issue: with newer `apify-client` versions, use `run.default_dataset_id` instead of `run["defaultDatasetId"]`.

### 3. Synthetic tweets (`make_dev_data.py`)
Generates 7,200 fake tweets (about 40 per day over 180 days) in the same schema as real data: `id`, `text`, `createdAt`, `likeCount`, `retweetCount`, `author.followers`. For debugging only.

### 4. Sentiment (`sentiment.py`)
1. Load JSONL line by line with `json.loads` and keep tweet IDs as **strings** (reading IDs as floats silently merged distinct tweets and the duplicate filter deleted about 99% of the data).
2. Convert `createdAt` to a UTC timestamp and floor it to a UTC date.
3. Tolerate missing columns (likes, retweets, views, followers, `isRetweet`) by creating them with defaults.
4. Clean: drop duplicate IDs, drop retweets, keep only text containing `btc` or `bitcoin`, drop tweets shorter than 20 characters.
5. Score every tweet with **VADER** (compound score from -1 to +1).
6. Weights: `log(1 + followers)` and `log(1 + likes + retweets)`.
7. Aggregate to one row per UTC day:

| Column | Meaning |
|---|---|
| `mean_sent`, `sent_std` | average sentiment and its spread |
| `wsent_followers`, `wsent_engage` | follower-weighted and engagement-weighted sentiment |
| `post_count`, `total_engage` | volume and total likes + retweets |
| `pct_bullish`, `pct_bearish` | share of tweets above +0.05 / below -0.05 |

### 5. Features and targets (`features.py`)
Joins daily prices with daily sentiment on the UTC date (left join on prices; days without tweets get zeros). The incomplete current day is dropped.

**19 features (clues):**

| Group | Features |
|---|---|
| Price | `ret_lag1`, `ret_lag2`, `ret_lag3`, `ret_lag5`, `vol_7d`, `ret_mean_7d` |
| Sentiment level | `mean_sent`, `wsent_followers`, `wsent_engage`, `sent_std`, `log_posts`, `total_engage`, `bull_bear_spread` |
| Sentiment dynamics | `sent_chg`, `wsent_chg`, `posts_chg`, `sent_ma3`, `sent_ma7`, `wsent_ma7` |

Price features use `.shift(1)` or more, so day n's own close never enters its clues. Rows with missing features (warm-up) are dropped, while rows at the end keep blank targets (their future has not happened) and are used for the live forecast. The saved table has 38 columns: 19 features, 12 targets (`y_ret_0..5`, `y_up_0..5`) and 7 helper columns that are never used as inputs.

### 6. Models and validation (`model.py`)
- **Regression:** `StandardScaler` then `Ridge(alpha=10)`, one per horizon (predicts the return).
- **Classification:** `StandardScaler` then `LogisticRegression(C=0.1)`, one per horizon (predicts the probability of "up").
- The scaler sits **inside** the pipeline, so it is fitted on the training window only.
- **Walk-forward:** minimum 60 training days. For each test day `i` and horizon `h`, train on rows `0 .. i-h-1` only (rows whose answers were already known), predict day `i`, record, advance one day. A fresh model is built every round.
- Simple models are used on purpose: only about 190 daily rows exist, so flexible models would overfit.

**Baselines:** predict zero return (naive), always predict the more common direction so far (majority class).

**Forecast metrics:** MAE, RMSE (returns and prices), directional accuracy, per horizon.

**Outputs:** `predictions.csv`, `forecast_metrics.csv`, `live_forecast.csv` (latest n+0..n+5 forecast).

### Sanity run on synthetic sentiment (`train_one.py`)
A single time-ordered split for horizon 1 (113 training rows, 77 test rows) gave a model error larger than the naive "no change" error and about 47% directional accuracy. This is the expected outcome with random sentiment (no signal to learn), and it shows the pipeline runs without leaking. It is **not** a result.

## Planned evaluation

- Forecast error and directional accuracy per horizon versus the baselines.
- Strategy built on the up/down predictions (long/short, with transaction costs): **Sharpe ratio, Sortino ratio, Max Drawdown**, compared with buy-and-hold. Annualised with 365 days.
- Charts: sentiment vs price over time, predicted vs actual, equity curve, drawdown.

## Leakage controls

- All features are lagged or computed from data available on day n.
- Targets are built from future prices but are never in the feature list.
- Training windows stop `h` days before each test day, because answers for the most recent rows contain prices from the test period.
- Scaling is fitted inside each training window.
- An `assert` in `features.py` checks that the lag column really is a lag.
- Row counts are printed after each step (this caught the ID bug above).

## Known limitations

- Real tweet data not yet integrated; no real results yet.
- Small sample (about 190 daily rows, about 125 graded days per horizon) makes metrics noisy.
- VADER is a general-English lexicon and can misread crypto slang; a finance-specific model would be more accurate.
- Same-day tweets partly react to the same day's price move, so the n+0 horizon is the easiest and least informative one.
- Fields such as views, likes and followers may be unavailable depending on the data source.

## Reproducibility and secrets

Tokens and API keys live only in `.env` (ignored by git). Actor IDs used are listed above. Keep `.env`, `data/raw/` and `data/dev/` in `.gitignore`.

## Roadmap

1. Integrate real tweet data (public historical dataset and/or small Apify scrape).
2. Run walk-forward for all horizons on real data.
3. `backtest.py`: long/short strategy, Sharpe, Sortino, Max Drawdown vs buy-and-hold.
4. Tavily news and the 1-page current sentiment outlook.
5. Charts and the evaluation report.
6. Multi-asset version (pooled BTC, ETH, ...), split by date, never by asset.
