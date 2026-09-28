"""Expanding-window competing-risks logits: TTC (loan features only) and PIT (+ real-time macro).

Monthly multinomial logit over {stay, default, payoff}. At each quarterly refit date R the model is trained on
loan-months with period <= R - 2 months (outcomes whose servicer reports were filed before R) and predicts
loan-months in [R, R + 3 months). Stays are subsampled in training and reweighted, so hazards stay calibrated.

Output: data/interim/pd_logit_oos.parquet  (loan_id, period, arm, refit, h_default, h_payoff)
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PANEL = "data/interim/model_panel.parquet"
OUT = "data/interim/pd_logit_oos.parquet"
FIRST_REFIT, STAY_SAMPLE, C, SEED = "2019-01-01", 0.2, 0.1, 7
CLIP = True

LOAN = ["rate", "log_orig_amt", "loan_age", "months_to_maturity", "balloon_12m", "full_io", "multi_property",
        "pari_passu_notes", "building_age", "occ_sec", "dscr_sec", "ltv_sec", "occ_recent", "dscr_recent", "ltv_current",
        "reappraised", "dq30", "dq60", "matured_performing", "in_special_servicing", "modified", "defeased", "lockout"]
MACRO = ["UNRATE_level", "UNRATE_yoy", "BAA10Y_level", "NFCI_level", "GS10_level", "GS10_yoy", "SUBLPDRCSN_level",
         "DRCRELEXFACBS_level", "cre_price_yoy", "sector_emp_yoy", "sector_price_yoy", "st_UR_level", "st_UR_yoy",
         "st_NA_yoy", "st_STHPI_yoy"]
ARMS = {"ttc_logit": LOAN, "pit_logit": LOAN + MACRO}


def design(d, cols, stats=None):
    X = pd.get_dummies(d[cols + ["ptype"]], columns=["ptype"], dtype=float).drop(columns="ptype_OT", errors="ignore")
    if stats is None:
        stats = (X.median(), X.fillna(X.median()).mean(), X.fillna(X.median()).std().replace(0, 1), X.columns,
                 X.min(), X.max())
    med, mu, sd, columns, lo, hi = stats
    # inputs are clipped to the training range: a linear logit shown 2020 unemployment after training on 2017-19
    # extrapolates to ~25x the realised default count
    X = X.reindex(columns=columns, fill_value=0.0).fillna(med).clip(lo, hi, axis=1) if CLIP else \
        X.reindex(columns=columns, fill_value=0.0).fillna(med)
    return ((X - mu) / sd).to_numpy(), stats


def main(arms=tuple(ARMS)):
    d = pd.read_parquet(PANEL)
    rng = np.random.default_rng(SEED)
    d["train_draw"] = rng.random(len(d))
    refits = pd.date_range(FIRST_REFIT, d.period.max(), freq="QS")
    preds, fits = [], []
    for arm in arms:
        for R in refits:
            tr = d[(d.period <= R - pd.DateOffset(months=2)) & ((d.outcome > 0) | (d.train_draw < STAY_SAMPLE))]
            te = d[(d.period >= R) & (d.period < R + pd.DateOffset(months=3))]
            if te.empty:
                continue
            w = np.where(tr.outcome == 0, 1 / STAY_SAMPLE, 1.0)
            X, stats = design(tr, ARMS[arm])
            m = LogisticRegression(C=C, max_iter=2000).fit(X, tr.outcome, sample_weight=w)
            p = m.predict_proba(design(te, ARMS[arm], stats)[0])
            cls = list(m.classes_)
            preds.append(pd.DataFrame({"loan_id": te.loan_id.values, "period": te.period.values, "arm": arm, "refit": R,
                                       "h_default": p[:, cls.index(1)] if 1 in cls else 0.0,
                                       "h_payoff": p[:, cls.index(2)] if 2 in cls else 0.0,
                                       "outcome": te.outcome.values}))
            fits.append(dict(arm=arm, refit=R.date(), train_defaults=int((tr.outcome == 1).sum()),
                             train_payoffs=int((tr.outcome == 2).sum()), test_rows=len(te)))
            print(f"{arm} {R.date()} train defaults {fits[-1]['train_defaults']:4d}  test rows {len(te):6d}", flush=True)
    out = pd.concat(preds, ignore_index=True)
    out.to_parquet(OUT, index=False)
    pd.DataFrame(fits).to_csv("data/interim/pd_logit_fits.csv", index=False)

    out["year"] = out.period.dt.year
    rows = []
    for (arm, yr), g in out.groupby(["arm", "year"]):
        y = (g.outcome == 1).astype(int)
        rows.append(dict(arm=arm, year=yr, defaults=int(y.sum()), predicted=round(g.h_default.sum(), 1),
                         auc=round(roc_auc_score(y, g.h_default), 3) if 0 < y.sum() < len(y) else np.nan))
    ev = pd.DataFrame(rows)
    ev["pred_to_actual"] = (ev.predicted / ev.defaults.replace(0, np.nan)).round(2)
    print(ev.pivot(index="year", columns="arm", values=["defaults", "predicted", "pred_to_actual", "auc"]).to_string())


if __name__ == "__main__":
    main(*(sys.argv[1:] and [tuple(sys.argv[1:])]))
