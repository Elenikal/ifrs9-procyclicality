"""Compact per-filing CMBS parquet (data/interim/cmbs_filings) into one zstd parquet per trust.

Columns are unioned by name across filings (schemas differ by filer and over time); values stay strings.
"""
from pathlib import Path
import duckdb

SRC = Path("data/interim/cmbs_filings")
DST = Path("data/interim/cmbs_by_deal")

DST.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
for d in sorted(SRC.glob("cik=*")):
    out = DST / f"{d.name}.parquet"
    if out.exists():
        continue
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{d}/*.parquet', union_by_name=true))
        TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """)
n_in = con.execute(f"SELECT count(*) FROM read_parquet('{SRC}/*/*.parquet', union_by_name=true)").fetchone()[0]
n_out = con.execute(f"SELECT count(*) FROM read_parquet('{DST}/*.parquet', union_by_name=true)").fetchone()[0]
print(f"rows in {n_in:,}  rows out {n_out:,}  deals {len(list(DST.glob('*.parquet')))}")
assert n_in == n_out
