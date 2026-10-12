"""Macro sensitivity of each PD arm, and the probability-weighted ECL that IFRS 9 intends.

Each arm is scored at the same forecast dates under a stylised downside and upside macro state (see SCENARIOS in
fit_pd_horizon). The ratio of mean 12-month PD to the base path measures how much each estimator actually
responds to macroeconomic information, holding the information set identical.

Outputs: docs/scenario_sensitivity.json, paper/tables/scenario_sensitivity.tex
"""
import json
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, "scripts")
import analyse_procyclicality as A

ARMS = {"pit_gbm": "Gradient boosting (PIT)", "pit_logit": "Logit (PIT)", "ttc_logit": "Logit (TTC)"}


def main():
    start = A.window_start(A.EPV)[0]
    rows = []
    for arm, label in ARMS.items():
        base = pd.read_parquet(f"data/interim/pd_horizon_{arm}.parquet", columns=["loan_id", "period", "q_term_1", "q_mat_1"])
        base = base[base.period >= start]
        b = (base.q_term_1 + base.q_mat_1).mean()
        r = dict(arm=arm, label=label, base_pd12=round(float(b * 100), 3))
        for tag in ("dn", "up"):
            s = pd.read_parquet(f"data/interim/pd_scn{tag}_{arm}.parquet", columns=["loan_id", "period", "q_term_1", "q_mat_1"])
            m = base.merge(s, on=["loan_id", "period"], suffixes=("_b", "_s"))
            x = (m.q_term_1_s + m.q_mat_1_s).mean()
            r[f"pd12_{tag}"] = round(float(x * 100), 3)
            r[f"ratio_{tag}"] = round(float(x / (m.q_term_1_b + m.q_mat_1_b).mean()), 3)
        rows.append(r)
    d = pd.DataFrame(rows)
    json.dump(d.to_dict("records"), open("docs/scenario_sensitivity.json", "w"), indent=2)

    t = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"& Base & \multicolumn{2}{c}{Downside} & \multicolumn{2}{c}{Upside} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Model & PD (\%) & PD (\%) & ratio & PD (\%) & ratio \\", r"\midrule"]
    for r in rows:
        t.append(f"{r['label']} & {r['base_pd12']:.2f} & {r['pd12_dn']:.2f} & {r['ratio_dn']:.3f} & "
                 f"{r['pd12_up']:.2f} & {r['ratio_up']:.3f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    (Path("paper/tables") / "scenario_sensitivity.tex").write_text("\n".join(t) + "\n")
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
