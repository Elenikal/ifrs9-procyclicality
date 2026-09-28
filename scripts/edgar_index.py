"""Build the list of ABS-EE (Reg AB II asset-level) filings from EDGAR quarterly full indexes.

Output: data/interim/absee_filings.parquet, one row per ABS-EE filing (all asset classes;
CMBS is separated later by the EX-102 XML namespace, not by issuer name).
"""
import os, time, datetime as dt
from pathlib import Path
import requests, pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
UA = os.environ["SEC_USER_AGENT"]
RAW = Path("data/raw/edgar/full-index")
OUT = Path("data/interim/absee_filings.parquet")
FIRST = (2016, 4)  # Reg AB II asset-level reporting compliance date: Nov 2016


def quarters(today=None):
    today = today or dt.date.today()
    y, q = FIRST
    while (y, q) <= (today.year, (today.month - 1) // 3 + 1):
        yield y, q
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)


def fetch_index(y, q, session):
    path = RAW / f"{y}Q{q}_form.idx"
    current = (y, q) == list(quarters())[-1]
    if path.exists() and not current:
        return path
    r = session.get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx", timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    time.sleep(0.2)
    return path


def parse_index(path):
    lines = path.read_text(encoding="latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("-----")) + 1
    rows = []
    for l in lines[start:]:
        if not l.startswith("ABS-EE"):
            continue
        # fixed-width, but company names can overflow: split from the right on runs of spaces
        parts = l.rsplit(None, 3)
        head, cik, date, fname = parts
        form, _, company = head.partition(" ")
        rows.append((form.strip(), company.strip(), int(cik), date, fname))
    return pd.DataFrame(rows, columns=["form", "company", "cik", "date_filed", "filename"])


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    frames = [parse_index(fetch_index(y, q, s)) for y, q in quarters()]
    df = pd.concat(frames, ignore_index=True)
    df["date_filed"] = pd.to_datetime(df.date_filed)
    df["accession"] = df.filename.str.extract(r"(\d{10}-\d{2}-\d{6})")
    df.to_parquet(OUT, index=False)
    print(df.groupby("form").size())
    print(f"{df.cik.nunique()} distinct filer CIKs, {len(df)} filings, {df.date_filed.min():%Y-%m} to {df.date_filed.max():%Y-%m}")


if __name__ == "__main__":
    main()
