import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

"""
Scrape BTC tweets from X through Apify, one run per day.

Each run holds several 1-hour time windows (one search term per window), so the
sample is spread across the day instead of piling up at the end of it.

Usage:
    python src/ingest_x.py --token-env APIFY_TOKEN_1 --start 2026-04-21 --end 2026-10-17 --out data/raw/tweets_acc1.jsonl

Quick one-day test:
    python src/ingest_x.py --token-env APIFY_TOKEN_1 --start 2026-10-10 --end 2026-10-10 --out data/raw/test_day.jsonl

Safe to stop and re-run: finished days are remembered in <out>.done
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from apify_client import ApifyClient
from dotenv import load_dotenv

ACTOR = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"

TERMS = "($BTC OR #bitcoin OR bitcoin)"
FILTERS = "lang:en -filter:replies"

WINDOW_START_HOURS = [0, 8, 16]   # 3 windows a day (was 6)   # UTC hours; each window is 1 hour long
WINDOW_LENGTH_H = 1
TWEETS_PER_WINDOW = 10
CAP_IS_PER_TERM = True      # set False if the test day shows only window 0 filled
MAX_COST_PER_RUN = 0.10     # USD, hard cap for one run (one day)
MAX_FAILS_IN_A_ROW = 3      # stop instead of burning credit on repeated errors


def make_query(start, end):
    # IMPORTANT: use the same time syntax that worked in your manual test
    return f"{TERMS} {FILTERS} since_time:{int(start.timestamp())} until_time:{int(end.timestamp())}"


def dataset_id(run):
    # works with old (dict) and new (object) apify-client versions
    return run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id


def slim(item, window_idx):
    """Keep only the fields the pipeline uses."""
    author = item.get("author") or {}
    followers = author.get("followers", item.get("author.followers"))
    return {
        "id": str(item.get("id")),
        "text": item.get("text", ""),
        "createdAt": item.get("createdAt"),
        "likeCount": item.get("likeCount", 0),
        "retweetCount": item.get("retweetCount", 0),
        "replyCount": item.get("replyCount", 0),
        "quoteCount": item.get("quoteCount", 0),
        "viewCount": item.get("viewCount", 0),
        "lang": item.get("lang"),
        "isReply": bool(item.get("isReply", False)),
        "isQuote": bool(item.get("isQuote", False)),
        "isRetweet": bool(item.get("retweeted_tweet")),
        "author": {"followers": followers},
        "_window": window_idx,
    }


def tweet_day(created_at):
    try:
        return datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").astimezone(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token-env", required=True, help="name of the .env variable holding the token, e.g. APIFY_TOKEN_1")
    p.add_argument("--start", required=True, help="first day, YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="last day, YYYY-MM-DD (UTC), inclusive")
    p.add_argument("--out", required=True, help="output .jsonl file")
    p.add_argument("--max-tweets", type=int, default=20000, help="stop after this many tweets in total")
    a = p.parse_args()

    load_dotenv()
    token = os.environ.get(a.token_env)
    if not token:
        sys.exit(f"{a.token_env} not found in .env")
    client = ApifyClient(token)

    start = datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done_path = a.out + ".done"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()

    total, fails = 0, 0
    day = start
    while day <= end:
        key = day.strftime("%Y-%m-%d")
        if key in done:
            day += timedelta(days=1)
            continue
        if total >= a.max_tweets:
            print("Reached --max-tweets, stopping.")
            break

        windows = [(day + timedelta(hours=h), day + timedelta(hours=h + WINDOW_LENGTH_H))
                   for h in WINDOW_START_HOURS]
        terms = [make_query(s, e) for s, e in windows]
        max_items = TWEETS_PER_WINDOW if CAP_IS_PER_TERM else TWEETS_PER_WINDOW * len(terms)

        try:
            run = client.actor(ACTOR).call(
                run_input={"searchTerms": terms, "queryType": "Latest", "maxItems": max_items},
                max_total_charge_usd=MAX_COST_PER_RUN,
            )
            items = list(client.dataset(dataset_id(run)).iterate_items())
            if not items:
                raise RuntimeError("0 tweets returned")
        except Exception as e:
            fails += 1
            print(f"{key}: FAILED ({e}) [{fails}/{MAX_FAILS_IN_A_ROW}]")
            if fails >= MAX_FAILS_IN_A_ROW:
                sys.exit("Too many failures in a row. Stopping to protect your credit.")
            day += timedelta(days=1)
            continue
        fails = 0

        per_window = Counter(int(i.get("searchTermIndex", 0)) for i in items)
        off_day = sum(1 for i in items if tweet_day(i.get("createdAt")) != key)

        with open(a.out, "a", encoding="utf-8") as f:
            for i in items:
                f.write(json.dumps(slim(i, int(i.get("searchTermIndex", 0))), ensure_ascii=False) + "\n")
        with open(done_path, "a") as f:
            f.write(key + "\n")

        total += len(items)
        counts = [per_window.get(k, 0) for k in range(len(windows))]
        print(f"{key}: {len(items)} tweets | per window {counts} | off-day: {off_day} | total {total}")
        if min(counts) == 0:
            print("   WARNING: at least one window returned nothing. Check CAP_IS_PER_TERM and the query format.")

        day += timedelta(days=1)

    print("Finished. Total tweets this session:", total)


if __name__ == "__main__":
    main()
