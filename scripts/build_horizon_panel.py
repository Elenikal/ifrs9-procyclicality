"""Stacked horizon panel for direct PD term-structure models.

Origin = loan i at month t in model_panel (features from report t-1, macro vintage end of t-1).
For horizon year k = 1..3 the window is months [t + 12(k-1), t + 12k - 1]. A row exists if the loan is still at
risk and observed at the window start; its label is 1 in-term default / 3 at-maturity default / 2 payoff /
0 neither within the window. Default type follows `default_at_maturity` in model_panel (LGD differs by type).
Windows cut short by the loan vanishing without an event are dropped (censored).
`label_known` is the first month in which the whole window has been reported, so a model refit at R may only
use rows with label_known <= R - 2 months (same reporting lag as the monthly panel).

Training origins are quarterly (Jan/Apr/Jul/Oct): adjacent monthly origins share 11 of 12 outcome months.
Output: data/interim/horizon_train.parquet
"""
import os
import pandas as pd

K_MAX = 3

# VARIANT="_ss" swaps in the special-servicer-transfer default definition end to end
VARIANT = os.environ.get("VARIANT", "")


def main():
    p = pd.read_parquet(f"data/interim/model_panel{VARIANT}.parquet", columns=["loan_id", "period", "outcome", "default_at_maturity"])
    ev = p.groupby("loan_id").agg(last_period=("period", "max"))
    ev["default_period"] = p[p.outcome == 1].groupby("loan_id").period.min()
    ev["maturity_default"] = p[p.outcome == 1].groupby("loan_id").default_at_maturity.max()
    ev["payoff_period"] = p[p.outcome == 2].groupby("loan_id").period.min()
    o = p[p.period.dt.month.isin([1, 4, 7, 10])][["loan_id", "period"]].join(ev, on="loan_id")
    end = p.period.max()
    rows = []
    for k in range(1, K_MAX + 1):
        ws = o.period + pd.DateOffset(months=12 * (k - 1))
        we = o.period + pd.DateOffset(months=12 * k - 1)
        event = o[["default_period", "payoff_period"]].min(axis=1)
        at_risk = (event.isna() | (event >= ws)) & (o.last_period >= ws)
        in_win = event.notna() & (event >= ws) & (event <= we)
        observed = in_win | (o.last_period >= we)
        label = pd.Series(0, index=o.index)
        label[in_win & (o.default_period == event)] = 1
        label[in_win & (o.default_period == event) & (o.maturity_default == 1)] = 3
        label[in_win & (o.payoff_period == event) & (o.default_period != event)] = 2
        r = o.loc[at_risk & observed, ["loan_id", "period"]].assign(k=k, label=label[at_risk & observed],
                                                                    label_known=(we + pd.DateOffset(months=1))[at_risk & observed])
        r = r[r.label_known <= end + pd.DateOffset(months=1)]
        rows.append(r)
    h = pd.concat(rows, ignore_index=True)
    h.to_parquet(f"data/interim/horizon_train{VARIANT}.parquet", index=False)
    print(f"{len(h):,} origin-horizon rows from {h[['loan_id', 'period']].drop_duplicates().shape[0]:,} origins")
    print(h.groupby("k").label.value_counts().unstack().rename(columns={0: "none", 1: "term default", 2: "payoff", 3: "maturity default"}).to_string())


if __name__ == "__main__":
    main()
