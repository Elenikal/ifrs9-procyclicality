"""Loss given default from realised resolutions, by default type and property group.

A default is resolved when the loan either leaves the pool with a terminal code, or returns to performing
(current, out of special servicing) for 6 consecutive months. LGD = realised loss to trust / exposure at default,
with zero for resolutions that cost the trust nothing.

Headline table: resolved defaults only (a lower bound: 70% of defaults are still open, and open cases skew to slow
workouts). Bound table: open defaults imputed from the latest appraisal, balance less appraisal plus selling costs.
Cells are shrunk towards the default-type mean by credibility weight n / (n + 20). Property group is reported but
not used in the headline: once no-loss cures are included, LGD is the same across groups (11.6% vs 11.7% in-term).
A macro-linked variant (LGD on CRE price growth at default) is estimated for the robustness row.
The same LGD applies to every PD arm.

Outputs: docs/lgd_table.csv, data/interim/lgd_defaults.parquet, docs/lgd_macro_fit.json
"""
import os
import json
import numpy as np, pandas as pd

CRED_K, SELLING_COSTS, CURE_MONTHS = 20.0, 0.10, 6
HIGH_LOSS_TYPES = ["LO", "RT", "OF", "MU"]  # lodging, retail, office, mixed use
TERMINAL = ("2", "3", "4", "5", "6", "8", "9")

# VARIANT="_ss" swaps in the special-servicer-transfer default definition end to end
VARIANT = os.environ.get("VARIANT", "")


def group(ptype):
    return np.where(pd.Series(ptype).isin(HIGH_LOSS_TYPES), "LO_RT_OF_MU", "other")


def main():
    lm = pd.read_parquet("data/interim/cmbs_loan_month.parquet", columns=[
        "loan_id", "period", "paymentStatusLoanCode", "liquidationPrepaymentCode", "realizedLossToTrustAmount",
        "reportPeriodBeginningScheduleLoanBalanceAmount", "reportPeriodEndActualBalanceAmount",
        "mostRecentSpecialServicerTransferDate", "mostRecentMasterServicerReturnDate",
        "prop_mostRecentValuationAmount", "prop_valuationSecuritizationAmount", "prop_propertyTypeCode"])
    lm["period"] = pd.to_datetime(lm.period)
    panel = pd.read_parquet(f"data/interim/model_panel{VARIANT}.parquet", columns=["loan_id", "period", "outcome", "default_at_maturity"])
    d = panel[panel.outcome == 1].groupby("loan_id").agg(default_period=("period", "min"), at_maturity=("default_at_maturity", "max"))

    a = lm.merge(d.reset_index(), on="loan_id")
    at_default = a[a.period == a.default_period].groupby("loan_id").agg(
        ead=("reportPeriodBeginningScheduleLoanBalanceAmount", "max"), ptype=("prop_propertyTypeCode", "first"))
    post = a[a.period >= a.default_period].sort_values(["loan_id", "period"])
    ss_in, ss_out = pd.to_datetime(post.mostRecentSpecialServicerTransferDate), pd.to_datetime(post.mostRecentMasterServicerReturnDate)
    post["performing"] = post.paymentStatusLoanCode.isin(["0", "A", "B"]) & ~(ss_in.notna() & (ss_out.isna() | (ss_out < ss_in)))
    cure = post.groupby("loan_id").performing.apply(
        lambda s: s.rolling(CURE_MONTHS).sum().max() == CURE_MONTHS if len(s) >= CURE_MONTHS else False)
    res = post.groupby("loan_id").agg(
        loss=("realizedLossToTrustAmount", "max"), months_open=("period", "size"),
        exited=("liquidationPrepaymentCode", lambda x: x.dropna().isin(TERMINAL).any()),
        last_balance=("reportPeriodEndActualBalanceAmount", "last"),
        appraisal=("prop_mostRecentValuationAmount", "last"), appraisal_sec=("prop_valuationSecuritizationAmount", "last"))
    x = d.join(at_default).join(res).join(cure.rename("cured"))
    x["resolved"] = x.exited.fillna(False) | x.cured.fillna(False)
    x["lgd"] = (x.loss.fillna(0) / x.ead).clip(0, 2)
    val = x.appraisal.fillna(x.appraisal_sec)
    x["lgd_open_proxy"] = ((x.last_balance - val * (1 - SELLING_COSTS)) / x.ead).clip(0, 1.5)
    x["type"] = np.where(x.at_maturity == 1, "at_maturity", "in_term")
    x["group"] = group(x.ptype)
    x.to_parquet(f"data/interim/lgd_defaults{VARIANT}.parquet")

    rows = []
    for label, sub, col in [("headline_resolved", x[x.resolved], "lgd"),
                            ("bound_with_open", x.assign(lgd_all=np.where(x.resolved, x.lgd, x.lgd_open_proxy)), "lgd_all")]:
        overall = sub.groupby("type")[col].mean()
        g = sub.groupby(["type", "group"])[col].agg(["size", "mean"])
        g["shrunk"] = (g["mean"] * g["size"] + overall.reindex(g.index.get_level_values(0)).to_numpy() * CRED_K) / (g["size"] + CRED_K)
        for (t, grp), r in g.iterrows():
            rows.append(dict(table=label, type=t, group=grp, n=int(r["size"]), raw_lgd=r["mean"], lgd=r["shrunk"]))
        for t, m in overall.items():  # headline: default type only
            rows.append(dict(table=label, type=t, group="all", n=int((sub.type == t).sum()), raw_lgd=m, lgd=m))
    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(f"docs/lgd_table{VARIANT}.csv", index=False)

    # robustness: LGD on CRE price growth at the default date (same for every arm)
    us = pd.read_parquet("data/interim/macro_asof_us.parquet").rename(columns={"asof": "default_period"})
    us["cre_price_yoy"] = us.BOGZ1FL075035503Q_yoy.fillna(us.COMREPUSQ159N_level)
    r = x[x.resolved].reset_index().merge(us[["default_period", "cre_price_yoy"]], on="default_period", how="left").dropna(subset=["cre_price_yoy", "lgd"])
    fit = {}
    for t, g in r.groupby("type"):
        if len(g) > 30:
            b = np.polyfit(g.cre_price_yoy, g.lgd, 1)
            fit[t] = dict(n=len(g), slope=round(float(b[0]), 5), intercept=round(float(b[1]), 4),
                          corr=round(float(np.corrcoef(g.cre_price_yoy, g.lgd)[0, 1]), 3))
    json.dump(fit, open("docs/lgd_macro_fit.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(x.groupby(["type", "resolved"]).size().unstack(fill_value=0).to_string())
    print(tab.round(3).to_string(index=False))
    print("macro fit:", json.dumps(fit))


if __name__ == "__main__":
    main()
