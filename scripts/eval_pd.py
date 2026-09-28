"""Out-of-sample evaluation of the PD arms.

12-month horizon, loan-months whose full 12-month window has been reported. Reported for all loans and for
performing loans (current, not in special servicing), by default type.
Outputs: data/interim/pd_eval_yearly.csv, data/interim/pd_paths_quarterly.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

PD_COLS = ["pd12", "pd12_term", "pd12_mat", "pd_life", "pd_life_term", "pd_life_mat"]


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def main(arms, source="cal"):
    panel = pd.read_parquet("data/interim/model_panel.parquet", columns=[
        "loan_id", "period", "outcome", "default_at_maturity", "dq30", "dq60", "in_special_servicing", "bal"])
    P = pd.concat([pd.read_parquet(f"data/interim/pd_{source}_{a}.parquet", columns=["loan_id", "period", "arm"] + PD_COLS)
                   for a in arms if Path(f"data/interim/pd_{source}_{a}.parquet").exists()])
    d = P.merge(panel, on=["loan_id", "period"])
    d["performing"] = (d.dq30 + d.dq60 + d.in_special_servicing) == 0

    paths = d[d.performing].groupby([d.period.dt.to_period("Q").astype(str), "arm"])[PD_COLS].mean().mul(100)
    paths.round(3).to_csv("data/interim/pd_paths_quarterly.csv")

    dd = panel[panel.outcome == 1].groupby("loan_id").agg(dper=("period", "min"), mat=("default_at_maturity", "max"))
    e = d.join(dd, on="loan_id").reset_index(drop=True)
    e["we"] = e.period + pd.DateOffset(months=11)
    e = e[e.we <= panel.period.max()].copy()
    hit = (e.dper >= e.period) & (e.dper <= e.we)
    e["y"], e["y_term"], e["y_mat"] = hit.astype(int), (hit & (e.mat == 0)).astype(int), (hit & (e.mat == 1)).astype(int)
    e["year"] = e.period.dt.year
    rows = []
    for (arm, yr), g_all in e.groupby(["arm", "year"]):
        for sample, g in (("all", g_all), ("performing", g_all[g_all.performing])):
            rows.append(dict(arm=arm, year=yr, sample=sample, n=len(g),
                             realised=g.y.mean() * 100, predicted=g.pd12.mean() * 100, auc=auc(g.y, g.pd12),
                             realised_term=g.y_term.mean() * 100, predicted_term=g.pd12_term.mean() * 100,
                             auc_term=auc(g.y_term, g.pd12_term),
                             realised_mat=g.y_mat.mean() * 100, predicted_mat=g.pd12_mat.mean() * 100))
    ev = pd.DataFrame(rows)
    ev.round(4).to_csv("data/interim/pd_eval_yearly.csv", index=False)
    pd.set_option("display.width", 250)
    for sample in ("all", "performing"):
        x = ev[ev["sample"] == sample].pivot(index="year", columns="arm",
                                            values=["realised", "predicted", "auc", "predicted_term", "auc_term"])
        print(f"\n12-month default, {sample} loans (%):\n{x.round(2).to_string()}")
    print("\nPerforming loans, mean PD by quarter (%):")
    print(paths[["pd12", "pd_life_term", "pd_life_mat"]].unstack("arm").round(2).iloc[::2].to_string())


if __name__ == "__main__":
    # usage: eval_pd.py [cal|horizon] [arms...]   (cal = recalibrated, horizon = raw)
    args = sys.argv[1:]
    source = args.pop(0) if args and args[0] in ("cal", "horizon") else "cal"
    main(args or ["ttc_logit", "pit_logit", "pit_gbm"], source)
