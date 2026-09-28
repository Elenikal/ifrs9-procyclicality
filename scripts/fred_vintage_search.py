"""Find FRED series matching a search whose first ALFRED vintage predates a cutoff.

Answers: is there any real-time (vintage) house price series covering the pre-2008 training window?
"""
import os, sys, time, json
import requests, pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
KEY = os.environ["FRED_API_KEY"]
BASE = "https://api.stlouisfed.org/fred"


def get(path, **params):
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE}/{path}", params={**params, "api_key": KEY, "file_type": "json"}, timeout=60)
        except requests.ConnectionError:
            time.sleep(10); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(20); continue
        r.raise_for_status()
        time.sleep(0.55)  # FRED limit is 120 req/min
        return r.json()
    raise RuntimeError(f"failed after retries: {path}")


def search(text, max_obs_start="2000-12-31"):
    rows, offset = [], 0
    while True:
        d = get("series/search", search_text=text, limit=1000, offset=offset)
        rows += d["seriess"]
        offset += 1000
        if offset >= min(d["count"], 5000):  # API rejects offset >= 5000
            break
    df = pd.DataFrame(rows)
    df = df[df.observation_start <= max_obs_start]
    # sub-state geographies (MSA, county, ZIP) number in the thousands and cannot have earlier vintages than their states
    return df[~df.title.str.contains(r"\(MSA\)|\(MSAD\)|County|ZIP|Metropolitan Division|Census Tract", regex=True)]


def first_vintage(sid):
    d = get("series/vintagedates", series_id=sid, limit=1, sort_order="asc")
    return d["vintage_dates"][0] if d["vintage_dates"] else None


if __name__ == "__main__":
    texts = sys.argv[1].split("|") if len(sys.argv) > 1 else ["house price index"]
    cutoff = sys.argv[2] if len(sys.argv) > 2 else "2005-01-01"
    text = texts[0]
    cands = pd.concat([search(t) for t in texts]).drop_duplicates("id")
    print(f"{len(cands)} candidate series with observations before 2001", file=sys.stderr)
    cands = cands.assign(first_vintage=[first_vintage(s) for s in cands.id])
    out = cands[["id", "title", "frequency_short", "observation_start", "first_vintage", "popularity"]]
    out.to_csv(f"data/raw/fred_search_{text.replace(' ', '_')}.csv", index=False)
    early = out[out.first_vintage < cutoff].sort_values("first_vintage")
    print(early.to_string(index=False) if len(early) else f"none with first vintage before {cutoff}")
