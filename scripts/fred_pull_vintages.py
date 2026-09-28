"""Pull full ALFRED vintage histories for the series in docs/macro_series.csv.

Stores FRED's "observations by real-time period" form: one row per (date, value) for the span of vintage
dates [realtime_start, realtime_end] over which that value was the published one.
Output: data/raw/fred/<series_id>.parquet (resumable: existing files are skipped).
"""
import os, sys, time
from pathlib import Path
import requests, pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
KEY = os.environ["FRED_API_KEY"]
OUT = Path("data/raw/fred")
REALTIME_START = "2014-01-01"   # vintages before this are not needed for 2016+ forecast dates
OBS_START = "2008-01-01"        # enough history for 12-month changes and pre-period levels
STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
          "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()


def series_list():
    cfg = pd.read_csv("docs/macro_series.csv")
    rows = []
    for r in cfg.itertuples():
        for st in (STATES if "{ST}" in r.series_id else [None]):
            rows.append(dict(series_id=r.series_id.replace("{ST}", st) if st else r.series_id,
                             template=r.series_id, geo=st or "US", block=r.block, transform=r.transform))
    return pd.DataFrame(rows)


def get(path, **params):
    for attempt in range(8):
        try:
            r = requests.get(f"https://api.stlouisfed.org/fred/{path}",
                             params={**params, "api_key": KEY, "file_type": "json"}, timeout=120)
        except requests.RequestException:
            time.sleep(10); continue
        time.sleep(0.55)  # 120 requests/minute
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(20 * (attempt + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"failed: {path} {params.get('series_id')}")


def pull_window(sid, start, end):
    rows, offset = [], 0
    while True:
        d = get("series/observations", series_id=sid, realtime_start=start, realtime_end=end,
                observation_start=OBS_START, output_type=1, limit=100000, offset=offset)
        rows += d["observations"]
        offset += len(d["observations"])
        if offset >= d["count"] or not d["observations"]:
            return rows


def pull(sid):
    n_vintages = get("series/vintagedates", series_id=sid, realtime_start=REALTIME_START, limit=1)["count"]
    if n_vintages <= 2000:
        rows = pull_window(sid, REALTIME_START, "9999-12-31")
    else:
        # API caps a request at 2000 vintage dates (daily series). Rows are clipped at window edges, which leaves
        # the value in force on any given day unchanged.
        years = list(range(int(REALTIME_START[:4]), pd.Timestamp.today().year + 1, 4))
        rows = []
        for y in years:
            end = f"{y + 3}-12-31" if y != years[-1] else "9999-12-31"
            rows += pull_window(sid, f"{y}-01-01", end)
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df.value, errors="coerce")  # "." marks missing
    df["date"] = pd.to_datetime(df.date)
    df["realtime_start"] = pd.to_datetime(df.realtime_start)
    df["realtime_end"] = pd.to_datetime(df.realtime_end.replace("9999-12-31", "2262-04-11"))
    return df.dropna(subset=["value"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = series_list()
    s.to_csv("data/raw/fred/_series_expanded.csv", index=False)
    for k, sid in enumerate(s.series_id, 1):
        dest = OUT / f"{sid}.parquet"
        if dest.exists():
            continue
        try:
            df = pull(sid)
            df.to_parquet(dest, index=False)
            print(f"{k}/{len(s)} {sid}: {len(df)} rows, {df.realtime_start.nunique()} vintages", flush=True)
        except Exception as e:
            print(f"{k}/{len(s)} {sid}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
