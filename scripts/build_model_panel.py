"""Loan-month modelling panel for discrete-time competing-risks hazards.

Row = loan i at risk in month t. Features use only information available before month t:
  loan state from the month t-1 servicer report; macro from the ALFRED vintage in force at the end of t-1.
Outcome at t: 0 stays, 1 default (headline: first 90+ DPD or credit exit), 2 payoff (prepayment or maturity payoff).
Rows after the first event are dropped; loans that stop reporting without an exit code are censored.

Output: data/interim/model_panel.parquet
"""
import os
import duckdb, numpy as np, pandas as pd

LM = "data/interim/cmbs_loan_month.parquet"
SECTOR_EMP = {"LO": "CES7072100001_yoy", "RT": "USTRADE_yoy", "OF": "CES6054000001_yoy", "IN": "INDPRO_yoy"}
SECTOR_PRICE = {"LO": "PCU721110721110_yoy"}
MAIN_TYPES = ["RT", "OF", "MF", "LO", "IN", "SS", "MU", "MH"]

# VARIANT="_ss" swaps in the special-servicer-transfer default definition end to end
VARIANT = os.environ.get("VARIANT", "")


def loan_months(con):
    trigger = ("ss_new OR status IN ('3', '5') OR liq IN ('3', '6') OR loss > 0" if VARIANT == "_ss"
               else "status IN ('3', '5') OR liq IN ('3', '6') OR loss > 0")
    return con.execute(f"""
    WITH m AS (
      SELECT loan_id, cik, period, paymentStatusLoanCode AS status, liquidationPrepaymentCode AS liq,
             realizedLossToTrustAmount AS loss, reportPeriodEndActualBalanceAmount AS bal, originalLoanAmount AS orig_amt,
             originationDate, maturityDate, firstLoanPaymentDueDate, originalInterestRatePercentage AS rate_raw,
             originalTermLoanNumber AS orig_term, originalInterestOnlyTermNumber AS io_term, interestOnlyIndicator AS io,
             NumberProperties AS n_props, n_notes, prop_propertyTypeCode AS ptype, prop_propertyState AS state,
             prop_yearBuiltNumber AS year_built, prop_valuationSecuritizationAmount AS val_sec,
             prop_mostRecentValuationAmount AS val_recent, prop_mostRecentValuationDate AS val_recent_date,
             prop_valuationSecuritizationDate AS val_sec_date,
             prop_debtServiceCoverageNetOperatingIncomeSecuritizationPercentage AS dscr_sec,
             prop_mostRecentDebtServiceCoverageNetOperatingIncomePercentage AS dscr_recent,
             prop_physicalOccupancySecuritizationPercentage AS occ_sec, prop_mostRecentPhysicalOccupancyPercentage AS occ_recent,
             mostRecentSpecialServicerTransferDate AS ss_in, mostRecentMasterServicerReturnDate AS ss_out,
             modifiedIndicator AS modified, prop_DefeasedStatusCode AS defeased, prepaymentLockOutEndDate AS lockout_end,
             mostRecentSpecialServicerTransferDate IS NOT NULL
               AND (mostRecentMasterServicerReturnDate IS NULL OR mostRecentMasterServicerReturnDate < mostRecentSpecialServicerTransferDate)
               AND NOT lag(mostRecentSpecialServicerTransferDate IS NOT NULL
                 AND (mostRecentMasterServicerReturnDate IS NULL OR mostRecentMasterServicerReturnDate < mostRecentSpecialServicerTransferDate),
                 1, FALSE) OVER (PARTITION BY loan_id ORDER BY period) AS ss_new
      FROM '{LM}'),
    ev AS (  -- first event period per loan
      SELECT loan_id,
             min(period) FILTER (WHERE {trigger}) AS default_period,
             min(period) FILTER (WHERE liq IN ('2', '5', '8', '9')) AS payoff_period,
             min(period) AS first_period
      FROM m GROUP BY 1)
    SELECT cur.loan_id, cur.cik, cur.period,
           CASE WHEN cur.period = ev.default_period THEN 1
                WHEN cur.period = ev.payoff_period AND (ev.default_period IS NULL OR ev.payoff_period < ev.default_period) THEN 2
                ELSE 0 END AS outcome,
           cur.status AS cur_status,
           prev.* EXCLUDE (loan_id, cik, period, liq, loss)
    FROM m cur
    JOIN ev USING (loan_id)
    JOIN m prev ON prev.loan_id = cur.loan_id AND prev.period = cur.period - INTERVAL 1 MONTH
    WHERE cur.period <= least(coalesce(ev.default_period, DATE '9999-01-01'), coalesce(ev.payoff_period, DATE '9999-01-01'))
    """).df()


def flag(s):
    return pd.Series(s).fillna(False).astype(bool).astype(int)


def add_features(d):
    d = d.convert_dtypes(dtype_backend="numpy_nullable")
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].astype("float64")
    t = pd.to_datetime(d.period)
    months = lambda a, b: (a.dt.year - b.dt.year) * 12 + (a.dt.month - b.dt.month)
    d["rate"] = np.where(d.rate_raw > 1, d.rate_raw / 100, d.rate_raw)
    d["log_orig_amt"] = np.log(d.orig_amt.clip(lower=1e5))
    d["loan_age"] = months(t, pd.to_datetime(d.firstLoanPaymentDueDate).fillna(pd.to_datetime(d.originationDate)))
    d["months_to_maturity"] = months(pd.to_datetime(d.maturityDate), t)
    d["balloon_12m"] = flag(d.months_to_maturity <= 12)
    d["full_io"] = flag((d.io_term >= d.orig_term).fillna(False) | (d.io.fillna(False).astype(bool) & d.io_term.isna()))
    d["multi_property"] = flag(d.n_props.fillna(1) > 1)
    d["pari_passu_notes"] = flag(d.n_notes > 1)
    d["building_age"] = (t.dt.year - d.year_built).clip(0, 150)
    occ = lambda s: np.where(s > 1.5, s / 100, s)
    d["occ_sec"], d["occ_recent"] = occ(d.occ_sec), occ(d.occ_recent)
    d["dscr_sec"] = d.dscr_sec.clip(0, 6)
    d["dscr_recent"] = d.dscr_recent.fillna(d.dscr_sec).clip(0, 6)
    d["occ_recent"] = pd.Series(d.occ_recent).fillna(pd.Series(d.occ_sec)).clip(0, 1)
    d["occ_sec"] = pd.Series(d.occ_sec).clip(0, 1)
    d["ltv_sec"] = (d.orig_amt / d.val_sec).clip(0.05, 1.5)
    reappraised = (pd.to_datetime(d.val_recent_date) > pd.to_datetime(d.val_sec_date)).fillna(False).astype(bool)
    d["reappraised"] = flag(reappraised)
    d["ltv_current"] = (d.bal.fillna(d.orig_amt) / d.val_recent.where(reappraised, d.val_sec)).clip(0.05, 3)
    d["dscr_change"] = d.dscr_recent - d.dscr_sec
    d["occ_change"] = d.occ_recent - d.occ_sec
    d["dq30"] = flag(d.status == "1")
    d["dq60"] = flag(d.status == "2")
    d["matured_performing"] = flag(d.status == "4")
    ss_in, ss_out = pd.to_datetime(d.ss_in), pd.to_datetime(d.ss_out)
    d["in_special_servicing"] = flag(ss_in.notna() & (ss_out.isna() | (ss_out < ss_in)))
    d["modified"] = flag(d.modified)
    d["defeased"] = flag(d.defeased.isin(["F"]))
    d["lockout"] = flag(pd.to_datetime(d.lockout_end) > t)
    d["ptype"] = d.ptype.where(d.ptype.isin(MAIN_TYPES), "OT")
    # at-maturity defaults (balloon not repaid) are kept separate from in-term defaults for LGD
    d["default_at_maturity"] = flag((d.outcome == 1) & ((d.cur_status == "5") | (d.months_to_maturity <= 1)))
    return d


def add_macro(d):
    us = pd.read_parquet("data/interim/macro_asof_us.parquet").rename(columns={"asof": "period"})
    st = pd.read_parquet("data/interim/macro_asof_state.parquet").rename(columns={"asof": "period"})
    d["period"] = pd.to_datetime(d.period)
    d = d.merge(us, on="period", how="left").merge(st, on=["period", "state"], how="left")
    d["cre_price_yoy"] = d.BOGZ1FL075035503Q_yoy.fillna(d.COMREPUSQ159N_level)
    d["sector_emp_yoy"] = d.PAYEMS_yoy
    d["sector_price_yoy"] = d.PCU531120531120_yoy
    for pt, col in SECTOR_EMP.items():
        d.loc[d.ptype == pt, "sector_emp_yoy"] = d.loc[d.ptype == pt, col]
    for pt, col in SECTOR_PRICE.items():
        d.loc[d.ptype == pt, "sector_price_yoy"] = d.loc[d.ptype == pt, col]
    # loans with no state series (missing/non-US codes) take the national counterpart
    d["state_matched"] = flag(d.st_UR_level.notna())
    d["st_UR_level"] = d.st_UR_level.fillna(d.UNRATE_level)
    d["st_UR_yoy"] = d.st_UR_yoy.fillna(d.UNRATE_yoy)
    d["st_NA_yoy"] = d.st_NA_yoy.fillna(d.PAYEMS_yoy)
    d["st_STHPI_yoy"] = d.st_STHPI_yoy.fillna(d.groupby("period").st_STHPI_yoy.transform("median"))
    return d


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    d = add_macro(add_features(loan_months(con)))
    keep = ["loan_id", "cik", "period", "outcome", "ptype", "state", "bal", "orig_amt", "rate", "log_orig_amt",
            "loan_age", "months_to_maturity", "balloon_12m", "full_io", "multi_property", "pari_passu_notes",
            "building_age", "occ_sec", "occ_recent", "occ_change", "dscr_sec", "dscr_recent", "dscr_change", "ltv_sec",
            "ltv_current", "reappraised", "dq30", "dq60", "matured_performing", "in_special_servicing", "modified",
            "defeased", "lockout", "state_matched", "default_at_maturity",
            "UNRATE_level", "UNRATE_yoy", "PAYEMS_yoy", "BAA10Y_level", "NFCI_level", "GS10_level", "GS10_yoy",
            "FEDFUNDS_level", "SUBLPDRCSN_level", "DRCRELEXFACBS_level", "cre_price_yoy", "sector_emp_yoy",
            "sector_price_yoy", "st_UR_level", "st_UR_yoy", "st_NA_yoy", "st_STHPI_yoy"]
    d = d[keep]
    d.to_parquet(f"data/interim/model_panel{VARIANT}.parquet", index=False)
    print(f"{len(d):,} loan-months at risk, {d.loan_id.nunique():,} loans")
    print(d.outcome.value_counts().rename({0: "stay", 1: "default", 2: "payoff"}).to_string())
    print("defaults by year:", d[d.outcome == 1].period.dt.year.value_counts().sort_index().to_dict())
    miss = d.isna().mean()
    print("features with missing values:\n", miss[miss > 0].round(3).to_string())


if __name__ == "__main__":
    main()
