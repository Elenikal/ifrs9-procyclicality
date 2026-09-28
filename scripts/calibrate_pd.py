"""Common PD recalibration rule, applied identically to every arm.

At each refit R, scripts/fit_pd_horizon.py also fits the same specification with data cut 12 months earlier and
scores the most recent 12 months of origins whose 12-month outcome is reported by R - 2 months (unseen by that
model). The constant log-odds shift that equates mean predicted and realised 12-month default, separately for
in-term and at-maturity default, is applied here to q_term_k / q_mat_k at every horizon for loan-months predicted
in [R, R + 3).

Raw shifts are noisy: the window holds 12 overlapping monthly origins, so independent events are roughly
(events / 12), and a fresh shift each quarter swings with a handful of loans. Three dampeners, in order:
  1. credibility shrinkage towards no adjustment, weight n_eff / (n_eff + CRED_K);
  2. exponential smoothing across refits (EWMA_ALPHA on the shrunk shift);
  3. a cap on the result.

Output: data/interim/pd_cal_<arm>.parquet (recalibrated) and data/interim/pd_calibration_shifts.csv
"""
import sys
import numpy as np, pandas as pd
from scipy.special import expit, logit

CAP, K_PRED, CRED_K, EWMA_ALPHA, ORIGINS_PER_LOAN = 1.0, 10, 50.0, 0.35, 12


def solve_shift(p, target):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    lo, hi = -CAP, CAP
    f = lambda a: expit(logit(p) + a).mean() - target
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def damp(sh, cred_k=CRED_K, alpha=EWMA_ALPHA, cap=CAP):
    """Shrink each raw shift by the credibility of its window, then smooth across refits, then cap."""
    sh = sh.sort_values("refit").copy()
    for name in ("term", "mat"):
        events = sh[f"realised_{name}"].fillna(0) / 100 * sh.holdout_rows.fillna(0)
        n_eff = events / ORIGINS_PER_LOAN
        sh[f"n_eff_{name}"] = n_eff
        shrunk = sh[f"shift_{name}"].fillna(0) * (n_eff / (n_eff + cred_k))
        sh[f"shift_{name}"] = shrunk.ewm(alpha=alpha, adjust=False).mean().clip(-cap, cap)
    return sh


def main(arm, cred_k=CRED_K, alpha=EWMA_ALPHA, cap=CAP):
    raw = pd.read_parquet(f"data/interim/pd_horizon_{arm}.parquet")
    sh = damp(pd.read_csv(f"data/interim/pd_holdout_{arm}.csv", parse_dates=["refit"]), cred_k, alpha, cap).set_index("refit")
    out = []
    for R, g in raw.groupby("refit"):
        g = g.copy()
        for name in ("term", "mat"):
            a = float(sh.loc[R, f"shift_{name}"]) if R in sh.index else 0.0
            if a == 0:
                continue
            for k in range(1, K_PRED + 1):
                c = f"q_{name}_{k}"
                v = g[c].to_numpy(dtype=float)
                g[c] = np.where(np.isnan(v), np.nan, expit(logit(np.clip(v, 1e-7, 1 - 1e-7)) + a)).astype(np.float32)
        out.append(g)
    o = pd.concat(out)
    surv = np.ones(len(o))
    life = {"term": np.zeros(len(o)), "mat": np.zeros(len(o))}
    for k in range(1, K_PRED + 1):
        q = {n: o[f"q_{n}_{k}"].fillna(0).to_numpy() for n in ("term", "mat", "pay")}
        for n in life:
            life[n] += surv * q[n]
        surv *= np.clip(1 - q["term"] - q["mat"] - q["pay"], 0, 1)
    o["pd12_term"], o["pd12_mat"] = o.q_term_1, o.q_mat_1
    o["pd12"] = o.pd12_term + o.pd12_mat
    o["pd_life_term"], o["pd_life_mat"] = life["term"].astype(np.float32), life["mat"].astype(np.float32)
    o["pd_life"] = o.pd_life_term + o.pd_life_mat
    o.to_parquet(f"data/interim/pd_cal_{arm}.parquet", index=False)
    return sh.reset_index().assign(arm=arm)


if __name__ == "__main__":
    arms = sys.argv[1:] or ["ttc_logit", "pit_logit", "pit_gbm"]
    s = pd.concat([main(a) for a in arms])
    s.to_csv("data/interim/pd_calibration_shifts.csv", index=False)
    pd.set_option("display.width", 220)
    s["refit"] = pd.to_datetime(s.refit).dt.date
    print(s.pivot(index="refit", columns="arm", values=["shift_term", "shift_mat"]).round(2).iloc[::2].to_string())
