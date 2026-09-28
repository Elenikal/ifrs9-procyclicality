"""Download every CMBS ABS-EE filing's EX-102, parse it in memory, and keep only parquet.

Output: data/interim/cmbs_filings/cik=<cik>/<accession>.parquet (all fields as strings; cast later
in DuckDB with union_by_name), plus data/interim/cmbs_manifest.csv, which makes the run resumable.
XML is never written to disk; any filing can be re-fetched from EDGAR.
"""
import csv, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests, pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from edgar_parse_ex102 import parse_bytes

load_dotenv(".env")
UA = os.environ["SEC_USER_AGENT"]
OUT = Path("data/interim/cmbs_filings")
MANIFEST = Path("data/interim/cmbs_manifest.csv")
CACHE = Path("data/raw/edgar/ex102")  # XML left over from discovery runs; consumed then deleted
MAX_RPS = 8  # SEC fair access limit is 10 requests/second across all threads


class RateLimiter:
    def __init__(self, rps):
        self.gap, self.next, self.lock = 1 / rps, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now)
            delay, self.next = self.next - now, self.next + self.gap
        time.sleep(delay)


LIMIT = RateLimiter(MAX_RPS)
local = threading.local()


def get(url):
    if not hasattr(local, "s"):
        local.s = requests.Session(); local.s.headers["User-Agent"] = UA
    for attempt in range(8):
        LIMIT.wait()
        try:
            r = local.s.get(url, timeout=180)
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.Timeout) as e:
            print(f"retry {attempt}: {type(e).__name__} {url}", flush=True)
            time.sleep(5 * (attempt + 1)); continue
        if r.status_code == 429 or r.status_code >= 500:
            print(f"retry {attempt}: HTTP {r.status_code} {url}", flush=True)
            time.sleep(15 * (attempt + 1)); continue
        if r.status_code == 403:
            raise PermissionError("403 from SEC: User-Agent likely rejected; stop and set a real contact")
        r.raise_for_status()
        return r
    raise RuntimeError(f"gave up: {url}")


def fetch(cik, accession, form):
    cached = CACHE / str(cik) / f"{accession}.xml"
    if cached.exists():
        xml = cached.read_bytes()
    else:
        folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
        items = get(f"{folder}/index.json").json()["directory"]["item"]
        names = [i["name"] for i in items if i["name"].lower().endswith(".xml") and "102" in i["name"]]
        if not names:
            return dict(cik=cik, accession=accession, form=form, status="no_ex102", n_rows=0, xml_bytes=0)
        xml = get(f"{folder}/{names[0]}").content
    df = parse_bytes(xml, cik, accession)
    if df is None:
        status, n = "not_cmbs", 0
    else:
        df.insert(2, "form", form)
        dest = OUT / f"cik={cik}" / f"{accession}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.astype("string").to_parquet(dest, index=False)
        status, n = "ok", len(df)
    if cached.exists():
        cached.unlink()
    return dict(cik=cik, accession=accession, form=form, status=status, n_rows=n, xml_bytes=len(xml))


def main(workers=12, limit=None):  # SEC responses stall randomly for seconds; workers ride the stalls, the limiter caps the rate
    filings = pd.read_parquet("data/interim/absee_filings.parquet")
    trusts = pd.read_parquet("data/interim/trust_latest.parquet")
    cmbs = trusts[trusts.namespace.str.contains("cmbs", na=False)].cik
    todo = filings[filings.cik.isin(cmbs)].sort_values("date_filed")
    fields = ["cik", "accession", "form", "status", "n_rows", "xml_bytes"]
    done = set(pd.read_csv(MANIFEST).accession) if MANIFEST.exists() else set()
    todo = todo[~todo.accession.isin(done)].head(limit)
    print(f"{len(todo)} filings to fetch ({len(done)} already in manifest)", flush=True)

    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as fh, ThreadPoolExecutor(workers) as pool:
        w = csv.DictWriter(fh, fields)
        if new:
            w.writeheader()
        futures = {pool.submit(fetch, r.cik, r.accession, r.form): r for r in todo.itertuples()}
        t0, total_mb = time.time(), 0.0
        for k, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                row = fut.result()
            except PermissionError as e:
                print(e, flush=True); pool.shutdown(cancel_futures=True); raise
            except Exception as e:
                print(f"ERROR {r.cik} {r.accession}: {e}", flush=True)
                continue  # not written to manifest, so a rerun retries it
            w.writerow(row); fh.flush()
            total_mb += row["xml_bytes"] / 1e6
            if k % 100 == 0:
                rate = k / (time.time() - t0)
                print(f"{k}/{len(todo)}  {rate:.1f} filings/s  {total_mb:,.0f} MB xml processed  "
                      f"eta {(len(todo) - k) / rate / 60:,.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
