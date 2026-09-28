"""Parse CMBS EX-102 XML files into a flat loan-level table (one row per asset per filing).

Loan-level leaf fields keep their schema names; the first property's fields are prefixed `prop_`
and the property count is kept, so multi-property loans are flagged rather than silently exploded.
Field names are taken from the XML, not hardcoded, so schema differences across filers survive.
"""
import sys
from pathlib import Path
import pandas as pd
from lxml import etree

CMBS_NS = "http://www.sec.gov/edgar/document/absee/cmbs/assetdata"


def parse_file(path):
    path = Path(path)
    return parse_bytes(path.read_bytes(), int(path.parent.name), path.stem)


def parse_bytes(xml, cik, accession):
    root = etree.fromstring(xml, parser=etree.XMLParser(huge_tree=True))
    if etree.QName(root).namespace != CMBS_NS:  # some filers use a prefix (ns1:assetData), so not nsmap[None]
        return None
    rows = []
    for a in root.iterfind(f"{{{CMBS_NS}}}assets"):
        row, n_prop = {}, 0
        for e in a:
            name = etree.QName(e).localname
            if len(e):  # nested block: property
                n_prop += 1
                if n_prop == 1:
                    row.update({f"prop_{etree.QName(c).localname}": c.text for c in e if not len(c)})
            else:
                row[name] = e.text
        row["n_property_blocks"] = n_prop
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "cik", cik)
    df.insert(1, "accession", accession)
    return df


def main(paths, out):
    frames = [f for f in (parse_file(Path(p)) for p in paths) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if c.endswith(("Amount", "Percentage", "Number")) and c not in ("assetNumber", "assetTypeNumber"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif c.endswith("Date"):
            df[c] = pd.to_datetime(df[c], format="%m-%d-%Y", errors="coerce")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} asset rows, {df.accession.nunique()} filings, {df.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
