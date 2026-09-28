"""Generate the CMBS field dictionary and code lists from the SEC ABS XML schema (spec v3.1).

Outputs (small, public, committed):
  docs/cmbs_fields.csv    element, parent block, xsd type, storage kind, Reg AB item text
  docs/cmbs_codes.csv     element, code, description
"""
import re
from pathlib import Path
import pandas as pd
from lxml import etree

XSD = Path("data/raw/sec_spec/ABS XML Technical Specification/EDGAR Form ABS XML Technical Specification/"
           "ABS XML Schema Files/eis_ABS_CMBSAssetData.xsd")
XS = "{http://www.w3.org/2001/XMLSchema}"


def kind(xsd_type, enum_types):
    t = xsd_type.split(":")[-1]
    if t in enum_types:
        return "code"
    if t.startswith("DECIMAL"):
        return "decimal"
    if t.startswith(("INTEGER", "INT", "NON_NEGATIVE_INT", "YEAR")):
        return "integer"
    if t == "DATE_TYPE":
        return "date"
    if t == "TRUE_FALSE_TYPE":
        return "bool"
    return "string"


def parse_codes(doc):
    body = doc.split("descriptions:", 1)[-1].strip()
    parts = re.split(r",\s*(?=[A-Za-z0-9]{1,3}:\s)", body)
    return [tuple(s.strip().rstrip(".,") for s in p.split(":", 1)) for p in parts if ":" in p]


def main():
    tree = etree.parse(str(XSD))
    enum_docs = {}
    for st in tree.iter(f"{XS}simpleType"):
        doc = st.findtext(f"{XS}annotation/{XS}documentation") or ""
        if st.find(f"{XS}restriction/{XS}enumeration") is not None:
            enum_docs[st.get("name")] = doc

    fields, codes = [], []
    for ct in tree.iter(f"{XS}complexType"):
        parent = ct.get("name")
        last_comment = ""
        for node in ct.iter():
            if isinstance(node, etree._Comment):
                last_comment = node.text.strip()
            elif node.tag == f"{XS}element" and node.get("type"):
                name, typ = node.get("name"), node.get("type")
                fields.append(dict(element=name, parent=parent, xsd_type=typ,
                                   kind=kind(typ, enum_docs), item=last_comment))
                if typ in enum_docs:
                    codes += [dict(element=name, code=c, description=d) for c, d in parse_codes(enum_docs[typ])]
    Path("docs").mkdir(exist_ok=True)
    pd.DataFrame(fields).to_csv("docs/cmbs_fields.csv", index=False)
    pd.DataFrame(codes).drop_duplicates().to_csv("docs/cmbs_codes.csv", index=False)
    print(f"{len(fields)} fields, {len(codes)} code rows")


if __name__ == "__main__":
    main()
