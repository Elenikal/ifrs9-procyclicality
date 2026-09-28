"""External benchmark for the CRE recovery calibration: Freddie Mac Multifamily Loan Performance Database.

The CMBS sample cannot observe a severe downturn (191 of 196 credit events fall in 2021-2026). The MLPD covers
loans funded from 1994 with quarterly status, liquidation balances and realised credit losses, so it spans the
2008-2012 crisis for multifamily collateral.

Default here is any quarter at 60+ days delinquent, modification with a loss, foreclosure or REO. Severity is
credit loss / unpaid balance at payoff. Only defaults that have reached a liquidation are used.

Output: docs/mlpd_benchmark.json, calibration/tables/mlpd_benchmark.tex
"""
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from scipy import stats

SRC = "data/raw/mlpd/mlpd_y*.csv"
DEFAULT_STATUS = ("200", "250", "300", "450")


def load(con):
    con.execute(f"create view m as select * from read_csv_auto('{SRC}', union_by_name=true)")
    con.execute(f"""create table d as
    with ever as (select lnno,
        min(case when mrtg_status in {DEFAULT_STATUS} then quarter end) as first_default_q,
        max(case when mrtg_status in ('300','450') then 1 else 0 end) as ever_fcl_reo,
        max(liq_upb_amt) as upb, max(credit_loss) as loss,
        max(strptime(liq_dte,'%d%b%Y')) as liq_date, max(rate_ltv) as ltv
      from m group by 1)
    select *, year(liq_date) as liq_year, case when upb>0 then coalesce(loss,0)/upb end as severity
    from ever where first_default_q is not null""")


def stats_for(con, where):
    r = con.execute(f"""select count(*) as n,
        avg(case when loss>0 then 1.0 else 0.0 end) as p_loss,
        avg(severity) filter (where loss>0) as sev_given_loss,
        avg(severity) as expected_lgd from d where liq_date is not null and {where}""").df().iloc[0]
    return dict(n=int(r.n), p_loss=round(float(r.p_loss), 4), sev_given_loss=round(float(r.sev_given_loss), 4),
                expected_lgd=round(float(r.expected_lgd), 4), expected_recovery=round(1 - float(r.expected_lgd), 4))


def main():
    con = duckdb.connect()
    load(con)
    eras = {"pre_2008": "liq_year<=2007", "gfc_2008_2012": "liq_year between 2008 and 2012",
            "y2013_2019": "liq_year between 2013 and 2019", "y2020_2025": "liq_year>=2020", "all": "true"}
    out = {k: stats_for(con, w) for k, w in eras.items()}
    sev = con.execute("select severity from d where loss>0 and severity>0 and severity<1").df().severity
    a, b, _, _ = stats.beta.fit(sev.clip(1e-3, 1 - 1e-3), floc=0, fscale=1)
    out["severity_beta"] = dict(alpha=round(float(a), 4), beta=round(float(b), 4), n=int(len(sev)),
                                mean=round(float(a / (a + b)), 4),
                                ks_p=round(float(stats.kstest(sev.clip(1e-3, 1 - 1e-3), "beta", args=(a, b)).pvalue), 3))
    out["source"] = "Freddie Mac Multifamily Loan Performance Database, funded 1994 onwards, quarterly to 2025Q4"
    json.dump(out, open("docs/mlpd_benchmark.json", "w"), indent=2)

    rows = [("Liquidations to 2007", out["pre_2008"]), ("2008--2012 (financial crisis)", out["gfc_2008_2012"]),
            ("2013--2019", out["y2013_2019"]), ("2020--2025", out["y2020_2025"])]
    t = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Liquidation period & $n$ & $P(\text{loss})$ & Severity $|$ loss & E[LGD] \\", r"\midrule"]
    for lab, v in rows:
        t.append(f"{lab} & {v['n']} & {v['p_loss']:.3f} & {v['sev_given_loss']:.3f} & {v['expected_lgd']:.3f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    Path("calibration/tables").mkdir(parents=True, exist_ok=True)
    (Path("calibration/tables") / "mlpd_benchmark.tex").write_text("\n".join(t) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
