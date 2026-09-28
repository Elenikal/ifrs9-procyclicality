"""H1-H3 statistics on the provision paths.

H1 amplitude: relative trough-to-peak change in ECL / exposure over the evaluation window, per arm and regime.
   Paired differences between arms with a deal-clustered bootstrap (deals resampled with replacement, the whole
   path recomputed each draw), so the comparison is not contaminated by loan-sampling noise.
H2 timing: lead/lag at which the correlation between the change in ECL / exposure and each realised stress series
   is highest (negative lead = provisions move first). Realised charge-offs are lumpy (a handful of liquidations),
   so the denser flow series are the primary ones: newly 90+ delinquent balances and new special-servicer
   transfers, both as a share of exposure.
H3 stage cliff: the change in IFRS 9 ECL between trough and peak split into stage migration and within-stage
   movement, computed in both orders because the decomposition is not unique.

The evaluation window starts at the first refit whose training data holds at least EPV events of every outcome
class for the arm with the most parameters (events-per-parameter rule, applied identically to all arms and decided
from training data alone, never from the provision paths). EPV 5 -> 2023-10; the rule is swept in robustness.
Output: docs/procyclicality_results.csv
"""
import sys
import numpy as np, pandas as pd

START, END, B, SEED, EPV = "2022-07-01", "2026-08-01", 400, 11, 5  # the current month is partially filed


def window_start(epv=EPV, variant=""):
    """First refit whose training data holds >= epv events of every outcome class, for the widest arm."""
    import fit_pd_horizon as F
    panel = pd.read_parquet(f"data/interim/model_panel{variant}.parquet")
    h = pd.read_parquet(f"data/interim/horizon_train{variant}.parquet").merge(panel.drop(columns=["outcome"]), on=["loan_id", "period"])
    h = F.window_features(h, h.k)
    n_par = max(F.design(h.head(1000), cols)[0].shape[1] for cols, _ in F.ARMS.values())
    need = epv * n_par
    for R in pd.date_range("2019-01-01", panel.period.max(), freq="QS"):
        tr = h[h.label_known <= R - pd.DateOffset(months=2)].label
        if all((tr == c).sum() >= need for c in (1, 2, 3)):
            return R, need, n_par
    return None, need, n_par


def stress_series(d, lm_path="data/interim/cmbs_loan_month.parquet"):
    """Realised stress flows as a share of exposure: new 90+ DPD balances, new special-servicer transfers,
    and charge-offs (the last is lumpy and kept only for comparison)."""
    lm = pd.read_parquet(lm_path, columns=["loan_id", "period", "paymentStatusLoanCode",
                                           "reportPeriodEndActualBalanceAmount", "originalLoanAmount",
                                           "mostRecentSpecialServicerTransferDate", "mostRecentMasterServicerReturnDate",
                                           "realizedLossToTrustAmount"])
    lm["period"] = pd.to_datetime(lm.period)
    lm = lm.sort_values(["loan_id", "period"])
    bal = lm.reportPeriodEndActualBalanceAmount.fillna(lm.originalLoanAmount)
    dq90 = lm.paymentStatusLoanCode.isin(["3", "5"])
    prev_dq90 = dq90.groupby(lm.loan_id).shift(1).fillna(False)
    ss_in, ss_out = pd.to_datetime(lm.mostRecentSpecialServicerTransferDate), pd.to_datetime(lm.mostRecentMasterServicerReturnDate)
    in_ss = ss_in.notna() & (ss_out.isna() | (ss_out < ss_in))
    prev_ss = in_ss.groupby(lm.loan_id).shift(1).fillna(False)
    f = pd.DataFrame({"period": lm.period, "bal": bal,
                      "new_dq90": bal.where(dq90 & ~prev_dq90, 0),
                      "new_ss": bal.where(in_ss & ~prev_ss, 0),
                      "charge_off": lm.realizedLossToTrustAmount.fillna(0).clip(lower=0)})
    g = f.groupby("period").sum()
    return pd.DataFrame({c: g[c] / g.bal * 1e4 for c in ("new_dq90", "new_ss", "charge_off")})


def paths(d, col="ecl_ifrs9"):
    g = d.groupby(["arm", "period"]).agg(ecl=(col, "sum"), ead=("ead", "sum"))
    return (g.ecl / g.ead).unstack("arm") * 100


def amplitude(p):
    """Relative trough-to-peak rise, and the same in logs; level-free so a constant PD scale does not matter."""
    return pd.Series({"trough": p.min(), "peak": p.max(), "rel_rise": p.max() / p.min() - 1,
                      "log_range": np.log(p.max() / p.min()),
                      "trough_date": p.idxmin().date(), "peak_date": p.idxmax().date()})


def lead_lag(ecl, charge_off, max_lag=12):
    de, dc = ecl.diff().dropna(), charge_off.reindex(ecl.index).diff().dropna()
    idx = de.index.intersection(dc.index)
    de, dc = de[idx], dc[idx]
    out = {k: de.corr(dc.shift(-k)) for k in range(-max_lag, max_lag + 1)}
    s = pd.Series(out).dropna()
    return (np.nan, np.nan) if s.empty else (s.idxmax(), s.max())


def bootstrap_amplitude(d, col, arms, rng):
    deals = d.cik.unique()
    per = d.groupby(["cik", "arm", "period"]).agg(ecl=(col, "sum"), ead=("ead", "sum"))
    draws = []
    for _ in range(B):
        pick = rng.choice(deals, len(deals), replace=True)
        s = per.loc[pick].groupby(["arm", "period"]).sum()
        p = (s.ecl / s.ead).unstack("arm") * 100
        draws.append({a: p[a].max() / p[a].min() - 1 for a in arms if a in p})
    return pd.DataFrame(draws)


def main(col="ecl_ifrs9", start=START, end=END, suffix="", quiet=False):
    d = pd.read_parquet(f"data/interim/ecl_loan_month{suffix}.parquet")
    d = d[(d.period >= start) & (d.period <= end)]
    arms = sorted(d.arm.unique())
    rows = []
    stress = stress_series(d)
    for regime, c in (("ifrs9", "ecl_ifrs9"), ("cecl", "ecl_cecl")):
        p = paths(d, c)
        for a in arms:
            r = amplitude(p[a]).to_dict()
            for name in ("new_dq90", "new_ss", "charge_off"):
                lag, corr = lead_lag(p[a], stress[name].reindex(p.index))
                r[f"lead_{name}"], r[f"corr_{name}"] = lag, round(corr, 3) if corr == corr else np.nan
            rows.append(dict(regime=regime, arm=a, **r))
    res = pd.DataFrame(rows)

    rng = np.random.default_rng(SEED)
    bs = bootstrap_amplitude(d, col, arms, rng)
    pairs = [("pit_gbm", "pit_logit"), ("pit_gbm", "ttc_logit"), ("pit_logit", "ttc_logit")]
    diffs = []
    for a, b in pairs:
        if a in bs and b in bs:
            x = bs[a] - bs[b]
            diffs.append(dict(comparison=f"{a} - {b}", mean_diff=x.mean(), lo=x.quantile(.025), hi=x.quantile(.975),
                              share_positive=(x > 0).mean()))
    dif = pd.DataFrame(diffs)

    # H3: trough-to-peak change split into stage migration vs within-stage movement, both orders
    h3 = []
    for a in arms:
        s = d[d.arm == a]
        p = paths(d, "ecl_ifrs9")[a]
        t0, t1 = p.idxmin(), p.idxmax()
        w = s[s.period.isin([t0, t1])]
        mix = w.groupby(["period", "stage"]).ead.sum().unstack(fill_value=0)
        mix = mix.div(mix.sum(axis=1), axis=0)                      # stage shares of exposure
        rate = w.groupby(["period", "stage"]).apply(lambda x: (x.ecl_ifrs9.sum() / x.ead.sum()), include_groups=False).unstack()
        tot = lambda m, r: float((m * r).sum())
        base, final = tot(mix.loc[t0], rate.loc[t0]), tot(mix.loc[t1], rate.loc[t1])
        mig_first = tot(mix.loc[t1], rate.loc[t0]) - base           # move stage shares, hold rates
        rate_first = tot(mix.loc[t0], rate.loc[t1]) - base          # move rates, hold stage shares
        h3.append(dict(arm=a, trough=t0.date(), peak=t1.date(), total_change=final - base,
                       migration_first=mig_first, within_stage_first=rate_first,
                       migration_share_lo=min(mig_first, final - base - rate_first) / (final - base),
                       migration_share_hi=max(mig_first, final - base - rate_first) / (final - base)))
    h3 = pd.DataFrame(h3)

    if quiet:
        return res, dif, h3
    pd.set_option("display.width", 240)
    print(f"window {start} to {end}")
    print("H1 amplitude (ECL / exposure, %) and H2 lead/lag (months; negative = provisions move first):\n",
          res.round(3).to_string(index=False))
    print("\nPaired amplitude differences, deal-clustered bootstrap (relative rise):\n", dif.round(3).to_string(index=False))
    print("\nH3 stage decomposition (percentage points of ECL/EAD):\n", h3.round(4).to_string(index=False))
    res.assign(kind="amplitude").to_csv("docs/procyclicality_results.csv", index=False)
    dif.assign(kind="paired_diff").to_csv("docs/procyclicality_results.csv", mode="a", index=False)
    h3.assign(kind="stage_decomposition").to_csv("docs/procyclicality_results.csv", mode="a", index=False)
    return res, dif, h3


if __name__ == "__main__":
    main(*sys.argv[1:])
