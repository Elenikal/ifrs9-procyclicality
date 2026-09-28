"""Real-time macro features: what was published before each forecast month.

For forecast month A (first day of month), the information set is the ALFRED vintage in force on A - 1 day.
Within that vintage: daily/weekly series are averaged to months, then for the latest available observation
  level, yoy (index: % change vs 12 months earlier; rate: difference), chg3 (same over 3 months), lag_m.
The same quantities from the latest vintage for the same observation date are stored as final_* so revisions
can be measured.

Outputs: data/interim/macro_asof_long.parquet, macro_asof_us.parquet (wide), macro_asof_state.parquet (wide),
         data/interim/macro_revisions.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

RAW = Path("data/raw/fred")
OUT = Path("data/interim")
ASOF = pd.date_range("2016-01-01", pd.Timestamp.today().normalize().replace(day=1) + pd.offsets.MonthBegin(1), freq="MS")


def to_monthly(obs):
    """obs: Series indexed by date. Returns Series indexed by month start (quarterly stays on quarter starts)."""
    gaps = np.diff(obs.index.values).astype("timedelta64[D]").astype(int)
    if len(gaps) and np.median(gaps) < 25:  # daily or weekly
        return obs.groupby(obs.index.to_period("M")).mean().to_timestamp()
    return obs


def features(m, transform, at=None):
    """Latest observation (or the observation dated `at`) with its 12- and 3-month changes, within one vintage."""
    if m.empty:
        return None
    t = m.index.max() if at is None else at
    if t not in m.index:
        return None
    v = m[t]
    def past(k):
        return m.get(t - pd.DateOffset(months=k), np.nan)
    if transform == "index":
        yoy, chg3 = v / past(12) * 100 - 100, v / past(3) * 100 - 100
    else:
        yoy, chg3 = v - past(12), v - past(3)
    return t, v, yoy, chg3


def vintage(df, day):
    live = df[(df.realtime_start <= day) & (df.realtime_end >= day)]
    return to_monthly(live.set_index("date").value.sort_index())


def main():
    spec = pd.read_csv(RAW / "_series_expanded.csv")
    rows = []
    for s in spec.itertuples():
        path = RAW / f"{s.series_id}.parquet"
        if not path.exists():
            print("missing", s.series_id); continue
        df = pd.read_parquet(path)
        final = vintage(df, df.realtime_start.max())
        for a in ASOF:
            f = features(vintage(df, a - pd.Timedelta(days=1)), s.transform)
            if f is None:
                continue
            t, v, yoy, chg3 = f
            ff = features(final, s.transform, at=t) or (t, np.nan, np.nan, np.nan)
            rows.append(dict(asof=a, series_id=s.series_id, template=s.template, geo=s.geo, block=s.block,
                             transform=s.transform, obs_date=t, lag_m=(a.year - t.year) * 12 + a.month - t.month,
                             level=v, yoy=yoy, chg3=chg3, final_level=ff[1], final_yoy=ff[2]))
    long = pd.DataFrame(rows)
    long.to_parquet(OUT / "macro_asof_long.parquet", index=False)

    us = long[long.geo == "US"].pivot(index="asof", columns="series_id", values=["level", "yoy", "chg3", "lag_m"])
    us.columns = [f"{sid}_{v}" for v, sid in us.columns]
    us.reset_index().to_parquet(OUT / "macro_asof_us.parquet", index=False)
    st = long[long.geo != "US"].assign(t=lambda d: d.template.str.replace("{ST}", "st_", regex=False))
    st = st.pivot_table(index=["asof", "geo"], columns="t", values=["level", "yoy", "chg3", "lag_m"])
    st.columns = [f"{t}_{v}" for v, t in st.columns]
    st.reset_index().rename(columns={"geo": "state"}).to_parquet(OUT / "macro_asof_state.parquet", index=False)

    rev = long.dropna(subset=["yoy", "final_yoy"]).assign(abs_rev=lambda d: (d.yoy - d.final_yoy).abs())
    summary = rev.groupby("template").agg(
        n=("abs_rev", "size"), mean_abs_rev_yoy=("abs_rev", "mean"), p90_abs_rev_yoy=("abs_rev", lambda x: x.quantile(.9)),
        sd_yoy=("final_yoy", "std"), corr_rt_final=("yoy", lambda x: np.corrcoef(x, rev.loc[x.index, "final_yoy"])[0, 1]),
        median_lag_m=("lag_m", "median"), first_asof=("asof", "min"))
    summary["rev_to_sd"] = summary.mean_abs_rev_yoy / summary.sd_yoy
    num = summary.select_dtypes("number").columns
    summary[num] = summary[num].round(3)
    summary.to_csv(OUT / "macro_revisions.csv")
    print(f"{len(long):,} as-of rows; US wide {us.shape}; state wide {st.shape}")
    print(summary.sort_values("rev_to_sd", ascending=False).to_string())


if __name__ == "__main__":
    main()
