"""For each ABS-EE trust, download the EX-102 asset data file from its latest filing and record
the asset class (from the XML namespace) plus a loan-level summary.

Output: data/interim/trust_latest.parquet; XML cached under data/raw/edgar/ex102/<cik>/<accession>.xml
"""
import os, re, sys, time
from pathlib import Path
from collections import Counter
import requests, pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
S = requests.Session()
S.headers["User-Agent"] = os.environ["SEC_USER_AGENT"]
XML = Path("data/raw/edgar/ex102")
OUT = Path("data/interim/trust_latest.parquet")
AUTO = r"(?i)auto|automobile|vehicle|lease|leasing|motor|carmax|drive|equipment|fleet|harley|honda|toyota|nissan|hyundai|ford credit|world omni|volkswagen|mercedes|bmw|porsche|kubota|deere|cnh|verizon|dell|carvana|exeter|santander|ally|gm financial|americredit|capital one prime|fifth third auto|usaa auto|california republic|efcar"


def get(url):
    for attempt in range(6):
        try:
            r = S.get(url, timeout=180)
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            time.sleep(10 * (attempt + 1)); continue
        time.sleep(0.15)  # SEC fair access: <= 10 req/s
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(30 * (attempt + 1)); continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"throttled: {url}")


def ex102_path(cik, accession):
    folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
    items = get(f"{folder}/index.json").json()["directory"]["item"]
    names = [i["name"] for i in items if i["name"].lower().endswith(".xml") and "102" in i["name"]]
    return f"{folder}/{names[0]}" if names else None


def summarise(xml):
    ns = re.search(r'xmlns(?::\w+)?="(http://www\.sec\.gov/edgar/document/absee/[^"]+)"', xml[:3000])
    tags = Counter(re.findall(r"<(?:\w+:)?(\w+)>", xml))
    status = Counter(re.findall(r"<(?:\w+:)?paymentStatusLoanCode>([^<]+)<", xml))
    return {
        "namespace": ns.group(1) if ns else None,
        "n_assets": tags.get("assets", 0),
        "payment_status": dict(status),
        "distress_tags": sorted(t for t in tags if re.search(r"(?i)loss|liquidat|apprais|specialservic|workout|modification|reo|forbear|defer", t)),
    }


def main(limit=None):
    f = pd.read_parquet("data/interim/absee_filings.parquet")
    trusts = f[f.company.str.contains(r"\b(?:19|20)\d{2}-", regex=True)]  # trusts carry a series id; depositors do not
    # auto/equipment files run to tens of thousands of assets; the namespace check is only needed for the rest
    trusts = trusts[~trusts.company.str.contains(AUTO, regex=True)]
    latest = trusts.sort_values("date_filed").groupby("cik").tail(1)
    done = set(pd.read_parquet(OUT).cik) if OUT.exists() else set()
    todo = latest[~latest.cik.isin(done)].head(limit)
    rows = pd.read_parquet(OUT).to_dict("records") if OUT.exists() else []
    for k, r in enumerate(todo.itertuples(), 1):
        try:
            url = ex102_path(r.cik, r.accession)
            if url is None:
                continue
            dest = XML / str(r.cik) / f"{r.accession}.xml"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(get(url).content)
            s = summarise(dest.read_text(errors="replace"))
            s["payment_status"] = str(s["payment_status"]); s["distress_tags"] = ",".join(s["distress_tags"])
            rows.append({"cik": r.cik, "company": r.company, "accession": r.accession, "date_filed": r.date_filed, "url": url, **s})
        except Exception as e:
            print(f"{r.cik} {r.company}: {e}", file=sys.stderr)
        if k % 50 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False); print(f"{k}/{len(todo)}", flush=True)
    pd.DataFrame(rows).to_parquet(OUT, index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
