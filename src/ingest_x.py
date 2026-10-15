import os, sys, json
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()
client = ApifyClient(os.environ["APIFY_TOKEN"])

ACTOR = "apidojo/twitter-scraper-lite"
QUERY = "($BTC OR #bitcoin OR bitcoin) lang:en -filter:replies"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 180   # test: python src/ingest_x.py 2
START_HOURS = [0, 6, 12, 18]    # UTC hours; each is a 1-hour window
TWEETS_PER_WINDOW = 12
MAX_COST_PER_RUN = 0.02         # USD cap per run
WORKERS = 8

OUT = "data/raw/tweets.jsonl"
DONE = "data/raw/done_windows.txt"
os.makedirs("data/raw", exist_ok=True)
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()

def fmt(dt):
    return dt.strftime("%Y-%m-%d_%H:%M:%S_UTC")

today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
jobs = []
for d in range(DAYS, 0, -1):
    for h in START_HOURS:
        start = today - timedelta(days=d) + timedelta(hours=h)
        if fmt(start) not in done:
            jobs.append((start, start + timedelta(hours=1)))

def scrape(start, end):
    run = client.actor(ACTOR).call(
        run_input={
            "searchTerms": [f"{QUERY} since:{fmt(start)} until:{fmt(end)}"],
            "maxItems": TWEETS_PER_WINDOW,
        },
        max_total_charge_usd=MAX_COST_PER_RUN,
    )
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    return fmt(start), items

total = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = [ex.submit(scrape, s, e) for s, e in jobs]
    for fut in as_completed(futures):
        try:
            key, items = fut.result()
        except Exception as e:
            print("FAILED:", e)
            continue
        with open(OUT, "a", encoding="utf-8") as f:
            for it in items:
                it["_window_start"] = key
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        open(DONE, "a").write(key + "\n")
        total += len(items)
        print(key, len(items), "tweets | total", total)