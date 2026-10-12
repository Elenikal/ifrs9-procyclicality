"""IFRS 9 and CECL provisions from each PD arm, with LGD and exposure held identical across arms.

Per loan-month and arm:
  Stage 3 (credit-impaired): 90+ DPD, non-performing matured balloon, or in special servicing -> LGD x EAD.
  Stage 2: residual lifetime PD has risen by SICR_MULT against the same model's score of the loan's
    origination-month characteristics for that same remaining life, or 30+ DPD backstop -> lifetime ECL.
    (Benchmarking against the loan's first-observed PD instead puts 45-87% of exposure in Stage 2: that compares a
    2019-vintage model with a 2025-vintage one, so staging tracks model updating rather than credit deterioration.)
  Stage 1: everything else -> 12-month ECL.
  CECL: lifetime ECL for every loan, no staging.
Lifetime ECL = sum_k S_(k-1) (q_term_k LGD_term + q_mat_k LGD_mat) EAD / (1+r)^(k-0.5), discounted at the loan rate.
EAD is the reported balance; LGD comes from docs/lgd_table.csv (default type only).

Outputs: data/interim/ecl_loan_month.parquet, data/processed/ecl_aggregate.parquet (date x arm x regime x segment)
"""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd

ARMS = ["ttc_logit", "pit_logit", "pit_gbm"]
SICR_MULT, K_PRED = 2.0, 10

# VARIANT="_ss" swaps in the special-servicer-transfer default definition end to end
VARIANT = os.environ.get("VARIANT", "")


def lgd_table(which="headline_resolved"):
    t = pd.read_csv(f"docs/lgd_table{VARIANT}.csv")
    t = t[(t.table == which) & (t.group == "all")].set_index("type").lgd
    return float(t.in_term), float(t.at_maturity)


def loan_state():
    lm = pd.read_parquet("data/interim/cmbs_loan_month.parquet", columns=[
        "loan_id", "cik", "period", "paymentStatusLoanCode", "reportPeriodEndActualBalanceAmount",
        "reportPeriodEndScheduledLoanBalanceAmount", "originalLoanAmount", "originalInterestRatePercentage",
        "realizedLossToTrustAmount", "mostRecentSpecialServicerTransferDate", "mostRecentMasterServicerReturnDate",
        "prop_propertyTypeCode"])
    lm["period"] = pd.to_datetime(lm.period)
    lm["ead"] = lm.reportPeriodEndActualBalanceAmount.fillna(lm.reportPeriodEndScheduledLoanBalanceAmount).fillna(lm.originalLoanAmount)
    r = lm.originalInterestRatePercentage
    lm["rate"] = np.where(r > 1, r / 100, r).clip(0.01, 0.15)
    ss_in, ss_out = pd.to_datetime(lm.mostRecentSpecialServicerTransferDate), pd.to_datetime(lm.mostRecentMasterServicerReturnDate)
    lm["in_ss"] = ss_in.notna() & (ss_out.isna() | (ss_out < ss_in))
    lm["stage3"] = lm.paymentStatusLoanCode.isin(["3", "5"]) | lm.in_ss
    lm["dq30"] = lm.paymentStatusLoanCode.isin(["1", "2"])
    lm["charge_off"] = lm.realizedLossToTrustAmount.fillna(0).clip(lower=0)
    lm["ptype"] = lm.prop_propertyTypeCode.where(lm.prop_propertyTypeCode.isin(["RT", "OF", "MF", "LO", "IN", "SS", "MU", "MH"]), "OT")
    return lm[["loan_id", "cik", "period", "ead", "rate", "stage3", "dq30", "charge_off", "ptype"]]


def residual_benchmark(d, arm, source="horizon"):
    """SICR benchmark: lifetime PD over the remaining life implied by the loan's origination-month characteristics
    scored with the same (current) model.

    pd_origin_<arm> holds, per refit, the conditional year-k curve for each loan's first observed month. After j-1
    whole years the benchmark is sum_{k>=j} (S_{k-1}/S_{j-1}) q_k from that curve, so remaining maturity is held
    fixed and both sides of the ratio come from one model: only new information moves it.
    """
    o = pd.read_parquet(f"data/interim/pd_origin_{arm}{VARIANT}.parquet")  # origin curves are raw in both PD sources
    qd = (o[[f"q_term_{k}" for k in range(1, K_PRED + 1)]].to_numpy(dtype=float)
          + o[[f"q_mat_{k}" for k in range(1, K_PRED + 1)]].to_numpy(dtype=float))
    qp = o[[f"q_pay_{k}" for k in range(1, K_PRED + 1)]].to_numpy(dtype=float)
    qd, qp = np.nan_to_num(qd), np.nan_to_num(qp)
    surv = np.cumprod(np.clip(1 - qd - qp, 0, 1), axis=1)          # S_k
    surv_prev = np.hstack([np.ones((len(qd), 1)), surv[:, :-1]])    # S_{k-1}
    tail = np.cumsum((surv_prev * qd)[:, ::-1], axis=1)[:, ::-1]    # sum_{k>=j} S_{k-1} q_k
    o = o[["loan_id", "refit", "origin_period"]].reset_index(drop=True)
    o["row"] = np.arange(len(o))
    m = d[["loan_id", "period", "refit"]].merge(o, on=["loan_id", "refit"], how="left")
    elapsed = (m.period.dt.year - m.origin_period.dt.year) * 12 + (m.period.dt.month - m.origin_period.dt.month)
    j = np.clip((elapsed // 12).fillna(0).astype(int), 0, K_PRED - 1)
    ok = m.row.notna().to_numpy()
    r = m.row.fillna(0).astype(int).to_numpy()
    sp = surv_prev[r, j]
    bench = np.where(ok, tail[r, j] / np.where(sp > 0, sp, np.nan), np.nan)
    return pd.Series(bench, index=d.index)


def ecl_for_arm(arm, lm, lgd_term, lgd_mat, at_maturity_type, sicr_mult=SICR_MULT, source="horizon"):
    p = pd.read_parquet(f"data/interim/pd_{source}_{arm}{VARIANT}.parquet")
    if "pd_life" not in p.columns:   # scenario files hold only the conditional curves
        surv = np.ones(len(p))
        life = np.zeros(len(p))
        for k in range(1, K_PRED + 1):
            qt, qm, qp = (p[f"q_{n}_{k}"].fillna(0).to_numpy() for n in ("term", "mat", "pay"))
            life += surv * (qt + qm)
            surv *= np.clip(1 - qt - qm - qp, 0, 1)
        p["pd12"], p["pd_life"] = p.q_term_1 + p.q_mat_1, life.astype(np.float32)
    d = lm.merge(p, on=["loan_id", "period"], how="left")
    surv = np.ones(len(d))
    life = np.zeros(len(d))
    twelve = np.zeros(len(d))
    for k in range(1, K_PRED + 1):
        qt = d[f"q_term_{k}"].fillna(0).to_numpy()
        qm = d[f"q_mat_{k}"].fillna(0).to_numpy()
        qp = d[f"q_pay_{k}"].fillna(0).to_numpy()
        loss_k = surv * (qt * lgd_term + qm * lgd_mat) / (1 + d.rate.to_numpy()) ** (k - 0.5)
        life += loss_k
        if k == 1:
            twelve = loss_k
        surv *= np.clip(1 - qt - qm - qp, 0, 1)
    d["ecl_rate_12m"], d["ecl_rate_life"] = twelve, life
    d["sicr_benchmark"] = residual_benchmark(d, arm)
    ratio = d.pd_life / d.sicr_benchmark.replace(0, np.nan)
    d["stage"] = np.where(d.stage3, 3, np.where(d.dq30 | (ratio >= sicr_mult), 2, 1))
    d.loc[d.pd_life.isna() & ~d.stage3, "stage"] = 1
    lgd_s3 = np.where(d.loan_id.map(at_maturity_type).fillna(0) == 1, lgd_mat, lgd_term)
    d["ecl_ifrs9"] = np.where(d.stage == 3, lgd_s3 * d.ead,
                              np.where(d.stage == 2, d.ecl_rate_life * d.ead, d.ecl_rate_12m * d.ead))
    d["ecl_cecl"] = np.where(d.stage3, lgd_s3 * d.ead, d.ecl_rate_life * d.ead)
    d["arm"] = arm
    return d[["loan_id", "cik", "period", "arm", "ptype", "ead", "stage", "charge_off", "pd12", "pd_life",
              "ecl_rate_12m", "ecl_rate_life", "ecl_ifrs9", "ecl_cecl"]]


def main(arms=ARMS, lgd_which="headline_resolved", sicr_mult=SICR_MULT, source="horizon", suffix="", weighted=False):
    """weighted=True replaces the single macro path with the probability-weighted scenarios that IFRS 9 intends:
    ECL is computed under each scenario and averaged with SCENARIO_WEIGHTS, staging included, so both the
    provision and the stage allocation are probability-weighted rather than conditional on one path."""
    lgd_term, lgd_mat = lgd_table(lgd_which)
    lm = loan_state()
    panel = pd.read_parquet(f"data/interim/model_panel{VARIANT}.parquet", columns=["loan_id", "period", "outcome", "default_at_maturity"])
    at_maturity_type = panel[panel.outcome == 1].groupby("loan_id").default_at_maturity.max()
    if weighted:
        import fit_pd_horizon as F
        parts = []
        for tag, w in F.SCENARIO_WEIGHTS.items():
            src = source if tag == "base" else f"scn{tag}"
            d = pd.concat([ecl_for_arm(a, lm, lgd_term, lgd_mat, at_maturity_type, float(sicr_mult), src) for a in arms],
                          ignore_index=True).assign(scenario=tag, weight=w)
            parts.append(d)
        allc = pd.concat(parts, ignore_index=True)
        num = ["ead", "charge_off", "pd12", "pd_life", "ecl_rate_12m", "ecl_rate_life", "ecl_ifrs9", "ecl_cecl"]
        allc[num] = allc[num].mul(allc.weight, axis=0)
        g = allc.groupby(["loan_id", "cik", "period", "arm", "ptype"], as_index=False)
        out = g[num].sum()
        out["ead"] = out.ead / sum(F.SCENARIO_WEIGHTS.values())     # exposure is common across scenarios
        out["charge_off"] = out.charge_off / sum(F.SCENARIO_WEIGHTS.values())
        # weighted stage: the probability-weighted share of exposure in each stage, summarised by the modal stage
        st = allc.assign(w=allc.weight).groupby(["loan_id", "period", "arm", "stage"]).w.sum().reset_index()
        st = st.sort_values("w").groupby(["loan_id", "period", "arm"]).tail(1)[["loan_id", "period", "arm", "stage"]]
        out = out.merge(st, on=["loan_id", "period", "arm"], how="left")
    else:
        out = pd.concat([ecl_for_arm(a, lm, lgd_term, lgd_mat, at_maturity_type, float(sicr_mult), source) for a in arms], ignore_index=True)
    out = out[out.period >= "2019-01-01"]
    Path("data/processed").mkdir(exist_ok=True)
    out.to_parquet(f"data/interim/ecl_loan_month{suffix}.parquet", index=False)

    agg = []
    for seg_name, keys in [("all", []), ("ptype", ["ptype"])]:
        g = out.groupby(["period", "arm"] + keys)
        a = g.agg(ead=("ead", "sum"), loans=("loan_id", "size"), charge_off=("charge_off", "sum"),
                  ecl_ifrs9=("ecl_ifrs9", "sum"), ecl_cecl=("ecl_cecl", "sum")).reset_index()
        for s in (1, 2, 3):
            a[f"ead_stage{s}"] = g.apply(lambda x, s=s: x.ead[x.stage == s].sum(), include_groups=False).values
            a[f"ecl_stage{s}"] = g.apply(lambda x, s=s: x.ecl_ifrs9[x.stage == s].sum(), include_groups=False).values
        a["segment"] = seg_name if not keys else a[keys[0]]
        agg.append(a.drop(columns=keys))
    ag = pd.concat(agg, ignore_index=True)
    ag["lgd_table"] = lgd_which
    ag.to_parquet(f"data/processed/ecl_aggregate{suffix}.parquet", index=False)
    pd.set_option("display.width", 220)
    t = ag[ag.segment == "all"].assign(ecl_pct=lambda x: x.ecl_ifrs9 / x.ead * 100,
                                       cecl_pct=lambda x: x.ecl_cecl / x.ead * 100,
                                       stage2_pct=lambda x: x.ead_stage2 / x.ead * 100)
    print(f"LGD in-term {lgd_term:.3f}, at-maturity {lgd_mat:.3f}")
    print(t.pivot(index="period", columns="arm", values=["ecl_pct", "cecl_pct", "stage2_pct"]).round(2).iloc[::6].to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ARMS)
