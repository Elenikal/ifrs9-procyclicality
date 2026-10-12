"""Direct PD term structure: pooled horizon models refit on an expanding window.

For origin (loan i, month t) and horizon year k the model gives conditional probabilities of in-term default,
at-maturity default and payoff in year k given survival to its start (LGD is set per default type). One multinomial model per arm pools k = 1..3 with horizon dummies and window
features (time to maturity at window start, maturity inside window). Years 4..10 reuse the k = 3 dummy with the
loan aged forward. Lifetime runs to the maturity year.

Refit quarterly: train on rows whose window was fully reported by R - 2 months; predict every loan-month in
[R, R + 3). Inputs clipped to the training range (see design_decisions.md).

Each refit also scores every loan's first observed month (its origination proxy) with the same model, giving the
SICR benchmark curve: both sides of the Stage 2 ratio then come from one model, so staging responds to new
information rather than to model vintage.

Outputs: data/interim/pd_horizon_<arm>.parquet with q_term_k, q_mat_k, q_pay_k (k = 1..10) and pd12 / pd_life in
         total and by type (_term, _mat); data/interim/pd_origin_<arm>.parquet with the origin-month curve per refit
"""
import os
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from calibrate_pd import solve_shift

FIRST_REFIT, NEG_SAMPLE, SEED, K_PRED = "2019-01-01", 0.3, 7, 10
# IFRS 9 reasonable-and-supportable period: macro at the forecast date drives years 1-2; from year 3 on, macro inputs
# revert to their training-sample mean (standardised value 0). Without this, lifetime PD is maturity-year default
# under today's rates projected up to a decade ahead.
RS_YEARS = 2

# IFRS 9 requires a probability-weighted view rather than a single path. Scenarios are applied as shifts to the
# macro inputs at the forecast date, in each series' own units, and are then clipped to the training range like
# any other input. Magnitudes are stylised: a moderate recession and a mild expansion.
SCENARIOS = {
    "dn": {"UNRATE_level": 1.5, "UNRATE_yoy": 1.5, "st_UR_level": 1.5, "st_UR_yoy": 1.5,
           "cre_price_yoy": -10.0, "sector_emp_yoy": -3.0, "sector_price_yoy": -3.0,
           "BAA10Y_level": 1.0, "NFCI_level": 0.5, "st_STHPI_yoy": -5.0, "st_NA_yoy": -2.0},
    "up": {"UNRATE_level": -0.5, "UNRATE_yoy": -0.5, "st_UR_level": -0.5, "st_UR_yoy": -0.5,
           "cre_price_yoy": 4.0, "sector_emp_yoy": 1.0, "sector_price_yoy": 1.0,
           "BAA10Y_level": -0.3, "NFCI_level": -0.2, "st_STHPI_yoy": 2.0, "st_NA_yoy": 1.0},
}
SCENARIO_WEIGHTS = {"base": 0.5, "dn": 0.3, "up": 0.2}
LOAN = ["rate", "log_orig_amt", "full_io", "multi_property", "pari_passu_notes", "building_age", "occ_sec", "dscr_sec",
        "ltv_sec", "occ_recent", "dscr_recent", "ltv_current", "reappraised", "dq30", "dq60", "matured_performing",
        "in_special_servicing", "modified", "defeased", "lockout"]
MACRO = ["UNRATE_level", "UNRATE_yoy", "BAA10Y_level", "NFCI_level", "GS10_level", "GS10_yoy", "SUBLPDRCSN_level",
         "DRCRELEXFACBS_level", "cre_price_yoy", "sector_emp_yoy", "sector_price_yoy", "st_UR_level", "st_UR_yoy",
         "st_NA_yoy", "st_STHPI_yoy"]
WINDOW = ["age_ws", "mtm_ws", "maturity_in_window", "past_maturity", "k2", "k3"]

# VARIANT="_ss" swaps in the special-servicer-transfer default definition end to end
VARIANT = os.environ.get("VARIANT", "")


def logit():
    return LogisticRegression(C=0.1, max_iter=3000)


def gbm(seed=None):
    # hyperparameters fixed once by scripts/tune_gbm.py on data reported before the 2022Q3 refit
    return HistGradientBoostingClassifier(**json.load(open("docs/gbm_params.json"))["params"], early_stopping=False,
                                          random_state=SEED if seed is None else seed)


ARMS = {"ttc_logit": (LOAN, logit), "pit_logit": (LOAN + MACRO, logit), "pit_gbm": (LOAN + MACRO, gbm)}


def window_features(d, k):
    k = np.asarray(k)
    mtm = d.months_to_maturity.to_numpy() - 12 * (k - 1)
    return d.assign(age_ws=d.loan_age.to_numpy() + 12 * (k - 1), mtm_ws=np.clip(mtm, -24, 120),
                    maturity_in_window=((mtm >= 0) & (mtm <= 11)).astype(float), past_maturity=(mtm < 0).astype(float),
                    k2=(np.minimum(k, 3) == 2).astype(float), k3=(np.minimum(k, 3) == 3).astype(float))


def design(d, cols, stats=None, revert_macro=False, scenario=None):
    if scenario:
        d = d.copy()
        for c, shift in SCENARIOS[scenario].items():
            if c in d.columns:
                d[c] = d[c] + shift
    X = pd.get_dummies(d[cols + WINDOW + ["ptype"]], columns=["ptype"], dtype=float).drop(columns="ptype_OT", errors="ignore")
    if stats is None:
        med = X.median()
        Xf = X.fillna(med)
        stats = (med, Xf.mean(), Xf.std().replace(0, 1), X.columns, Xf.min(), Xf.max())
    med, mu, sd, columns, lo, hi = stats
    X = X.reindex(columns=columns, fill_value=0.0).fillna(med).clip(lo, hi, axis=1)
    Z = (X - mu) / sd
    if revert_macro:
        Z[[c for c in MACRO if c in Z.columns]] = 0.0
    return Z.to_numpy(dtype=np.float32), stats


def holdout_calibration(h, panel, cols, make, R, realised):
    """Fit the same specification with data cut 12 months earlier; score the most recent 12 months of origins whose
    12-month outcome is fully reported by R - 2 (unseen by that model). Returns mean predicted and realised 12-month
    PD by default type and the log-odds shift that matches them (0 when there is no class or no event)."""
    cut = R - pd.DateOffset(months=14)
    tr = h[(h.label_known <= cut) & ((h.label > 0) | (h.draw < NEG_SAMPLE))]
    last_origin = R - pd.DateOffset(months=13)  # window end + 1 month <= R - 2 months
    ev = panel[(panel.period > last_origin - pd.DateOffset(months=12)) & (panel.period <= last_origin)]
    rec = dict(refit=R, holdout_rows=len(ev), holdout_train_rows=len(tr), shift_term=0.0, shift_mat=0.0)
    if len(ev) == 0 or not tr.label.isin([1, 3]).any():
        return rec
    X, stats = design(tr, cols)
    m = make().fit(X, tr.label, sample_weight=np.where(tr.label == 0, 1 / NEG_SAMPLE, 1.0))
    p = m.predict_proba(design(window_features(ev, np.ones(len(ev), dtype=int)), cols, stats)[0])
    cls = list(m.classes_)
    r = ev[["loan_id", "period"]].join(realised, on="loan_id")
    hit = (r.dper >= r.period) & (r.dper <= r.period + pd.DateOffset(months=11))
    for name, c, y in (("term", 1, hit & (r.mat == 0)), ("mat", 3, hit & (r.mat == 1))):
        rec[f"realised_{name}"] = y.mean() * 100
        if c in cls:
            rec[f"pred_{name}"] = p[:, cls.index(c)].mean() * 100
            if y.sum() > 0:
                rec[f"shift_{name}"] = solve_shift(p[:, cls.index(c)], y.mean())
    return rec


def main(arm):
    # "<arm>_s<seed>" refits with a different random seed: it drives both the estimator and, more importantly,
    # which stay-rows are sampled into training (HistGradientBoosting itself is deterministic with early stopping off)
    base, _, seed = arm.partition("_s")
    cols, make = ARMS[base if seed else arm]
    if seed and base == "pit_gbm":
        make = (lambda s=int(seed): gbm(s))
    draw_seed = int(seed) if seed else SEED
    panel = pd.read_parquet(f"data/interim/model_panel{VARIANT}.parquet")
    h = pd.read_parquet(f"data/interim/horizon_train{VARIANT}.parquet")
    h = h.merge(panel.drop(columns=["outcome"]), on=["loan_id", "period"])
    h = window_features(h, h.k)
    h["draw"] = np.random.default_rng(draw_seed).random(len(h))
    out, shifts, origin_out = [], [], []
    scn = {k: [] for k in SCENARIOS}
    origin = panel.sort_values("period").groupby("loan_id").first().assign(
        period=panel.sort_values("period").groupby("loan_id").period.first())
    realised = panel[panel.outcome == 1].groupby("loan_id").agg(dper=("period", "min"), mat=("default_at_maturity", "max"))
    for R in pd.date_range(FIRST_REFIT, panel.period.max(), freq="QS"):
        tr = h[(h.label_known <= R - pd.DateOffset(months=2)) & ((h.label > 0) | (h.draw < NEG_SAMPLE))]
        te = panel[(panel.period >= R) & (panel.period < R + pd.DateOffset(months=3))]
        if te.empty or not tr.label.isin([1, 3]).any():
            continue
        X, stats = design(tr, cols)
        m = make().fit(X, tr.label, sample_weight=np.where(tr.label == 0, 1 / NEG_SAMPLE, 1.0))
        cls = list(m.classes_)
        shifts.append(holdout_calibration(h, panel, cols, make, R, realised))
        n_years = np.clip(np.ceil((te.months_to_maturity.to_numpy() + 1) / 12), 1, K_PRED)
        for scenario in (None,) + tuple(SCENARIOS):   # base path, then the weighted-scenario variants
            res = {"loan_id": te.loan_id.values, "period": te.period.values, "refit": R}
            for k in range(1, K_PRED + 1):
                p = m.predict_proba(design(window_features(te, np.full(len(te), k)), cols, stats,
                                           revert_macro=k > RS_YEARS, scenario=scenario)[0])
                live = k <= n_years
                col = lambda c: p[:, cls.index(c)] if c in cls else np.zeros(len(te))
                for name, c in (("term", 1), ("pay", 2), ("mat", 3)):
                    res[f"q_{name}_{k}"] = np.where(live, col(c), np.nan).astype(np.float32)
            (out if scenario is None else scn[scenario]).append(pd.DataFrame(res))
        og = origin.loc[origin.index.intersection(te.loan_id.unique())]
        ores = {"loan_id": og.index.values, "refit": R, "origin_period": og.period.values}
        for k in range(1, K_PRED + 1):
            p = m.predict_proba(design(window_features(og, np.full(len(og), k)), cols, stats, revert_macro=k > RS_YEARS)[0])
            col = lambda c: p[:, cls.index(c)] if c in cls else np.zeros(len(og))
            for name, c in (("term", 1), ("pay", 2), ("mat", 3)):
                ores[f"q_{name}_{k}"] = col(c).astype(np.float32)
        origin_out.append(pd.DataFrame(ores))
        print(f"{arm} {R.date()} train rows {len(tr):7d} term defaults {int((tr.label == 1).sum()):5d} "
              f"maturity defaults {int((tr.label == 3).sum()):4d}", flush=True)
    o = pd.concat(out, ignore_index=True)
    surv = np.ones(len(o))
    life = {"term": np.zeros(len(o)), "mat": np.zeros(len(o))}
    for k in range(1, K_PRED + 1):
        q = {n: o[f"q_{n}_{k}"].fillna(0).to_numpy() for n in ("term", "mat", "pay")}
        for n in life:
            life[n] += surv * q[n]
        surv *= 1 - q["term"] - q["mat"] - q["pay"]
    o["pd12_term"], o["pd12_mat"] = o.q_term_1, o.q_mat_1
    o["pd12"] = o.pd12_term + o.pd12_mat
    o["pd_life_term"], o["pd_life_mat"] = life["term"].astype(np.float32), life["mat"].astype(np.float32)
    o["pd_life"] = o.pd_life_term + o.pd_life_mat
    o.insert(2, "arm", arm)
    o.to_parquet(f"data/interim/pd_horizon_{arm}{VARIANT}.parquet", index=False)
    pd.DataFrame(shifts).assign(arm=arm).to_csv(f"data/interim/pd_holdout_{arm}{VARIANT}.csv", index=False)
    pd.concat(origin_out, ignore_index=True).to_parquet(f"data/interim/pd_origin_{arm}{VARIANT}.parquet", index=False)
    for name, frames in scn.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(f"data/interim/pd_scn{name}_{arm}{VARIANT}.parquet", index=False)


if __name__ == "__main__":
    for a in (sys.argv[1:] or ARMS):
        main(a)
