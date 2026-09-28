"""Build the CMBS loan-month panel from the per-deal EX-102 parquet.

Rules (each motivated by a diagnostic in the data, see comments):
 1. Types come from docs/cmbs_fields.csv (generated from the SEC schema), not hardcoded.
 2. A filing's period is the month of its modal reportingPeriodEndDate (liquidated loans carry blank dates).
 3. Rows without loan terms are property-detail rows from filers that report properties as assets; dropped.
 4. assetNumber is not stable (two filing agents alternate ID schemes on the same deal), so the loan key is
    origination date | original rate. Maturity date is excluded because some filers overwrite it on extension
    (450 key breaks vs 84); state/type are excluded because the "first property" differs across ID schemes.
    Notes sharing the key are pari passu pieces of one mortgage and are summed. Compare keys with
    `build_cmbs_panel.py <key>`; counts are in the log.
 5. Some filings sit under the wrong trust CIK. Each filing is matched to the deal whose modal loan set it
    overlaps; reassigned if >= 80% overlap with another deal, dropped if it matches none.
 6. One filing per deal-month: best deal match, then amendment over original, then latest filed.

Outputs: data/interim/cmbs_loan_month.parquet, data/interim/cmbs_loans.parquet, data/interim/cmbs_panel_log.json
"""
import json, sys
from pathlib import Path
import duckdb, pandas as pd

SRC = "data/interim/cmbs_by_deal/*.parquet"
OUT = Path("data/interim")
REASSIGN_MIN, OWN_MIN, MODAL_MIN = 0.8, 0.5, 0.5
TERMINAL = ("2", "3", "4", "5", "6", "8", "9")  # payoff, disposition, repurchase, maturity payoff, DPO, penalty/YM payoff
CREDIT_EXIT = ("3", "6")                        # disposition/liquidation, discounted payoff
PREPAY_EXIT = ("2", "8", "9")
log = {}
_RATE = "CAST(round(CASE WHEN originalInterestRatePercentage > 1 THEN originalInterestRatePercentage / 100 ELSE originalInterestRatePercentage END, 5) AS VARCHAR)"
_OD = "strftime(originationDate, '%Y%m%d')"
KEYS = {  # candidate loan keys; see diagnostics in the build log
    "od_md_rate": f"{_OD} || '|' || strftime(maturityDate, '%Y%m%d') || '|' || {_RATE}",
    "od_rate_state_type": f"{_OD} || '|' || {_RATE} || '|' || coalesce(trim(prop_propertyState), '') || '|' || coalesce(trim(prop_propertyTypeCode), '')",
    "od_rate_amt": f"{_OD} || '|' || {_RATE} || '|' || CAST(round(originalLoanAmount) AS VARCHAR)",
    "od_rate": f"{_OD} || '|' || {_RATE}",
}


def typed_select(con):
    f = pd.read_csv("docs/cmbs_fields.csv")
    present = {c[0] for c in con.execute(f"SELECT * FROM read_parquet('{SRC}', union_by_name=true) LIMIT 0").description}
    exprs, sum_cols = [], []
    for r in f.itertuples():
        col = r.element if r.parent == "ASSET_TYPE" else f"prop_{r.element}"
        if col not in present or col in ("reportingPeriodEndDate",):
            continue
        q = f'"{col}"'
        e = {"decimal": f"TRY_CAST({q} AS DOUBLE)",
             "integer": f"CAST(TRY_CAST({q} AS DOUBLE) AS BIGINT)",
             "date": f"TRY_STRPTIME({q}, '%m-%d-%Y')::DATE",
             "bool": f"lower({q}) = 'true'"}.get(r.kind, f"trim({q})")
        exprs.append(f"{e} AS {q}")
        if r.parent == "ASSET_TYPE" and r.kind == "decimal" and col.endswith("Amount"):
            sum_cols.append(col)
    return ",\n  ".join(exprs), sum_cols


def main(key="od_rate"):
    log["loan_key"] = key
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    filings = pd.read_parquet("data/interim/absee_filings.parquet")[["accession", "date_filed"]].drop_duplicates("accession")
    con.register("filing_dates", filings)

    con.execute(f"""
        CREATE TEMP TABLE filing_period AS
        SELECT accession, date_trunc('month', STRPTIME(mode(reportingPeriodEndDate), '%m-%d-%Y'))::DATE AS period
        FROM read_parquet('{SRC}', union_by_name=true)
        WHERE reportingPeriodEndDate IS NOT NULL GROUP BY 1""")

    select, sum_cols = typed_select(con)
    con.execute(f"""
        CREATE TEMP TABLE rows AS
        WITH t AS (SELECT cik AS filed_cik, accession, form, assetNumber, n_property_blocks, {select}
                   FROM read_parquet('{SRC}', union_by_name=true))
        SELECT t.*, fp.period,
               round(CASE WHEN originalInterestRatePercentage > 1 THEN originalInterestRatePercentage / 100
                          ELSE originalInterestRatePercentage END, 5) AS orig_rate,
               {KEYS[key]} AS loan_key
        FROM t JOIN filing_period fp USING (accession)""")
    total = con.execute("SELECT count(*) FROM rows").fetchone()[0]
    con.execute("DELETE FROM rows WHERE originationDate IS NULL OR maturityDate IS NULL OR originalInterestRatePercentage IS NULL")
    kept = con.execute("SELECT count(*) FROM rows").fetchone()[0]
    log["rows_total"], log["rows_without_loan_terms_dropped"] = total, total - kept

    # --- rule 5: deal attribution by loan-set overlap
    k = con.execute("SELECT DISTINCT filed_cik, accession, loan_key FROM rows").df()
    nfil = k.groupby("filed_cik").accession.nunique()
    freq = k.groupby(["filed_cik", "loan_key"]).accession.nunique().div(nfil, level="filed_cik")
    ref = freq[freq >= MODAL_MIN].reset_index()[["filed_cik", "loan_key"]].rename(columns={"filed_cik": "deal_cik"})
    ov = k.merge(ref, on="loan_key").groupby(["accession", "filed_cik", "deal_cik"]).size().rename("hits").reset_index()
    ov = ov.merge(k.groupby("accession").size().rename("n_keys").reset_index(), on="accession")
    ov["share"] = ov.hits / ov.n_keys
    own = ov[ov.filed_cik == ov.deal_cik][["accession", "share"]].rename(columns={"share": "own_share"})
    best = ov.sort_values("share").groupby("accession").tail(1)
    a = k[["accession", "filed_cik"]].drop_duplicates().merge(own, on="accession", how="left").merge(
        best[["accession", "deal_cik", "share"]], on="accession", how="left")
    a["own_share"] = a.own_share.fillna(0)
    a["deal"] = a.filed_cik.where(a.own_share >= OWN_MIN, a.deal_cik.where(a.share >= REASSIGN_MIN))
    a["match"] = a.own_share.where(a.own_share >= OWN_MIN, a.share)
    log["filings"] = len(a)
    log["filings_reassigned_to_other_deal"] = int(((a.own_share < OWN_MIN) & a.deal.notna()).sum())
    log["filings_dropped_unmatched"] = int(a.deal.isna().sum())
    a = a.dropna(subset=["deal"]).astype({"deal": "int64"})

    # --- rule 6: one filing per deal-month
    per = con.execute("SELECT DISTINCT accession, form, period FROM rows").df()
    a = a.merge(per, on="accession").merge(filings, on="accession", how="left")
    a["amend"] = a.form.eq("ABS-EE/A")
    a = a.sort_values(["deal", "period", "match", "amend", "date_filed"], ascending=[True, True, False, False, False])
    log["deal_months_with_competing_filings"] = int(a.duplicated(["deal", "period"]).sum())
    chosen = a.drop_duplicates(["deal", "period"])[["accession", "deal", "period"]]
    con.register("chosen", chosen)

    # --- rule 4: collapse notes into loans
    first_cols = [c[0] for c in con.execute("SELECT * FROM rows LIMIT 0").description
                  if c[0] not in sum_cols + ["filed_cik", "accession", "form", "assetNumber", "period", "loan_key",
                                             "n_property_blocks", "paymentStatusLoanCode"]]
    aggs = [f'sum("{c}") AS "{c}"' for c in sum_cols] + [f'any_value("{c}") AS "{c}"' for c in first_cols]
    con.execute(f"""
        CREATE TEMP TABLE lm AS
        SELECT c.deal AS cik, r.period, r.loan_key, c.deal || ':' || r.loan_key AS loan_id,
               any_value(r.accession) AS accession, count(*) AS n_notes, max(r.n_property_blocks) AS n_property_blocks,
               arg_max(r.paymentStatusLoanCode, CASE r.paymentStatusLoanCode WHEN '5' THEN 5 WHEN '3' THEN 4
                   WHEN '2' THEN 3 WHEN '1' THEN 2 WHEN 'B' THEN 1 ELSE 0 END) AS paymentStatusLoanCode,
               {", ".join(aggs)}
        FROM rows r JOIN chosen c USING (accession)
        GROUP BY ALL""")

    # --- events. Special-servicer transfer date is overwritten each time, so collect every distinct value seen.
    con.execute(f"""
        CREATE TEMP TABLE ss AS
        SELECT loan_id, date_trunc('month', mostRecentSpecialServicerTransferDate)::DATE AS ss_month
        FROM lm WHERE mostRecentSpecialServicerTransferDate IS NOT NULL GROUP BY ALL""")
    con.execute(f"""
        CREATE TEMP TABLE loans AS
        SELECT loan_id, any_value(cik) AS cik, any_value(loan_key) AS loan_key,
               min(period) AS first_period, max(period) AS last_period, count(*) AS n_months,
               max(originalLoanAmount) AS originalLoanAmount, any_value(originationDate) AS originationDate,
               any_value(maturityDate) AS maturityDate, any_value(prop_propertyTypeCode) AS property_type,
               any_value(prop_propertyState) AS state, max(n_notes) AS n_notes,
               min(period) FILTER (WHERE paymentStatusLoanCode IN ('3', '5')) AS first_dq90_period,
               min(period) FILTER (WHERE paymentStatusLoanCode IN ('1', '2', '3', '5')) AS first_dq30_period,
               arg_min(liquidationPrepaymentCode, period) FILTER (WHERE liquidationPrepaymentCode IN {TERMINAL}) AS exit_code,
               min(liquidationPrepaymentDate) FILTER (WHERE liquidationPrepaymentCode IN {TERMINAL}) AS exit_date,
               max(realizedLossToTrustAmount) AS realized_loss
        FROM lm GROUP BY loan_id""")
    con.execute(f"""
        CREATE TEMP TABLE loans2 AS
        SELECT l.*, s.first_ss_month, s.n_ss_transfers,
               exit_code IN {CREDIT_EXIT} OR coalesce(realized_loss, 0) > 0 AS credit_exit,
               exit_code IN {PREPAY_EXIT} AS prepay_exit
        FROM loans l LEFT JOIN (SELECT loan_id, min(ss_month) AS first_ss_month, count(*) AS n_ss_transfers
                                FROM ss GROUP BY 1) s USING (loan_id)""")
    con.execute(f"COPY (SELECT * FROM lm ORDER BY cik, loan_id, period) TO '{OUT}/cmbs_loan_month.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY loans2 TO '{OUT}/cmbs_loans.parquet' (FORMAT parquet, COMPRESSION zstd)")

    s = con.execute("""SELECT count(*) loans, count(DISTINCT cik) deals, sum(n_months) loan_months,
        count(first_dq90_period) dq90, count(first_ss_month) ss_transfer, sum(credit_exit::INT) credit_exits,
        sum(prepay_exit::INT) prepay_exits, sum(CASE WHEN exit_code = '5' THEN 1 ELSE 0 END) maturity_payoffs,
        round(sum(CASE WHEN realized_loss > 0 THEN realized_loss END) / 1e9, 2) loss_bn FROM loans2""").df()
    log.update({k: (v.item() if hasattr(v, "item") else v) for k, v in s.iloc[0].items()})
    # gaps: loans that vanish before the deal's last month without a terminal code
    g = con.execute("""
        WITH d AS (SELECT cik, max(period) AS deal_last FROM lm GROUP BY 1)
        SELECT count(*) FROM loans2 JOIN d USING (cik) WHERE last_period < deal_last AND exit_code IS NULL""").fetchone()[0]
    log["loans_disappearing_without_exit_code"] = g
    # key breaks: a loan vanishes and a loan with the same origination date and amount starts the next month or so
    log["key_breaks_same_od_amount_reappears"] = con.execute("""
        WITH d AS (SELECT cik, max(period) AS deal_last FROM lm GROUP BY 1),
        v AS (SELECT l.* FROM loans2 l JOIN d USING (cik) WHERE last_period < deal_last AND exit_code IS NULL)
        SELECT count(DISTINCT v.loan_id) FROM v JOIN loans2 n ON n.cik = v.cik AND n.originationDate = v.originationDate
          AND abs(n.originalLoanAmount - v.originalLoanAmount) < 1 AND n.loan_id <> v.loan_id
          AND n.first_period > v.last_period AND n.first_period <= v.last_period + INTERVAL 2 MONTH""").fetchone()[0]
    (OUT / "cmbs_panel_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
