"""Figures and LaTeX tables for the paper.

Colours pass the categorical palette checks; line style carries identity as well as colour, so the figures read
in greyscale print and under deuteranopia. Series are identified by legend rather than by end labels, because the
provision paths converge at the right edge and end labels collide there.

Outputs: paper/figures/*.pdf, paper/tables/*.tex
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

sys.path.insert(0, "scripts")
import analyse_procyclicality as A

FIG, TAB = Path("paper/figures"), Path("paper/tables")
STYLE = {"pit_gbm": ("#2f6fb2", "-", "Gradient boosting (PIT)"),
         "pit_logit": ("#d1701a", "--", "Logit (PIT)"),
         "ttc_logit": ("#4f9e5c", ":", "Logit (TTC)")}
mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 150,
                     "axes.labelsize": 9, "legend.frameon": False})


def fig_paths(d, start, out):
    p = A.paths(d, "ecl_ifrs9")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for arm in STYLE:
        c, ls, lab = STYLE[arm]
        ax.plot(p.index, p[arm], color=c, ls=ls, lw=1.6, label=lab)
    ax.axvline(pd.Timestamp(start), color="#888", lw=1, ls="-", alpha=0.6)
    ax.annotate("evaluation window starts", xy=(pd.Timestamp(start), ax.get_ylim()[0]), xytext=(5, 6),
                textcoords="offset points", fontsize=7.5, color="#555")
    ax.set_ylabel("IFRS 9 ECL / exposure (%)")
    ax.set_xlabel("")
    ax.margins(x=0.02)
    ax.set_ylim(top=p.max().max() * 1.22)
    ax.legend(loc="upper left", fontsize=8, ncol=3, columnspacing=1.2, handlelength=2.4,
              frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_leadlag(d, out, ref="new_dq90", max_lag=12):
    p = A.paths(d, "ecl_ifrs9")
    stress = A.stress_series(d)[ref].reindex(p.index)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for arm in STYLE:
        c, ls, lab = STYLE[arm]
        de, dc = p[arm].diff().dropna(), stress.diff().dropna()
        idx = de.index.intersection(dc.index)
        corr = [de[idx].corr(dc[idx].shift(-k)) for k in range(-max_lag, max_lag + 1)]
        ax.plot(range(-max_lag, max_lag + 1), corr, color=c, ls=ls, lw=1.6, label=lab)
    ax.axvline(0, color="#888", lw=1, alpha=0.6)
    ax.set_xlabel("lead of provisions over newly 90+ delinquent balances (months)")
    ax.set_ylabel("correlation of monthly changes")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_stages(d, out):
    g = d.groupby(["arm", "period"]).apply(lambda x: x.ead[x.stage == 2].sum() / x.ead.sum() * 100, include_groups=False).unstack("arm")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for arm in STYLE:
        c, ls, lab = STYLE[arm]
        ax.plot(g.index, g[arm], color=c, ls=ls, lw=1.6, label=lab)
    ax.set_ylabel("Stage 2 share of exposure (%)")
    ax.margins(x=0.02)
    ax.set_ylim(top=g.max().max() * 1.25)
    ax.legend(loc="upper left", fontsize=8, ncol=3, columnspacing=1.2, handlelength=2.4,
              frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def signed(v, dp=2):
    """LaTeX-safe signed number: proper minus sign, no '+0'."""
    if abs(v) < 10 ** -dp / 2:
        return "$0$"
    return f"${v:+.{dp}f}$".replace("+", "+").replace("-", "-")


def ci(text):
    """[-0.44,-0.09] -> $[-0.44, -0.09]$ with maths minus signs."""
    return "$" + text.replace(",", ",\,") + "$"


def tables(res, dif, start):
    name = {a: STYLE[a][2] for a in STYLE}
    r = res[res.regime == "ifrs9"].set_index("arm")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"& Trough & Peak & Relative & \multicolumn{2}{c}{Lead (months) vs} \\",
             r"\cmidrule(lr){5-6}",
             r"Model & (\% exp.) & (\% exp.) & rise & new 90+ & new SS \\", r"\midrule"]
    for a in STYLE:
        x = r.loc[a]
        lead = lambda v: "$0$" if int(v) == 0 else f"${int(v):+d}$"
        lines.append(f"{name[a]} & {x.trough:.2f} & {x.peak:.2f} & {signed(x.rel_rise)} & "
                     f"{lead(x.lead_new_dq90)} & {lead(x.lead_new_ss)} \\\\")
    lines += [r"\addlinespace", r"\midrule",
              r"\multicolumn{6}{l}{\emph{Paired differences in relative rise, deal-clustered bootstrap}} \\",
              r"\addlinespace[2pt]"]
    for row in dif.itertuples():
        a, b = row.comparison.split(" - ")
        lines.append(f"\\multicolumn{{4}}{{l}}{{{name[a]} $-$ {name[b]}}} & "
                     f"\\multicolumn{{2}}{{r}}{{{signed(row.mean_diff)}\\;$[{row.lo:+.2f},\\,{row.hi:+.2f}]$}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "amplitude.tex").write_text("\n".join(lines) + "\n")

    rb = pd.read_csv("docs/robustness_results.csv")
    lab = {"sicr": "SICR multiple", "source": "PD source", "lgd": "LGD table", "start": "Window start"}
    pretty = {"horizon": "raw", "cal": "recalibrated", "ss_default_def": "special-servicer default definition",
              "headline_resolved": "resolved defaults",
              "bound_with_open": "bound incl.\\ open", "2.0": "2", "1.5": "1.5", "3.0": "3"}
    show = lambda v: pretty.get(str(v), str(v).replace("_", " "))
    base = rb.iloc[0]
    out = [r"\begin{tabular}{lrr}", r"\toprule",
           r"Specification & GB $-$ Logit(PIT) & Logit(PIT) $-$ Logit(TTC) \\", r"\midrule"]
    for _, row in rb.iterrows():  # column names contain "-", so index by label rather than itertuples
        changed = [f"{lab[k]} {show(row[k])}" for k in lab if row[k] != base[k]] or ["Baseline"]
        if row["source"] == "ss_default_def":   # this variant also brings its own rule-based window
            changed = ["Special-servicer default definition"]
        out.append(f"{changed[0]} & {signed(row['diff_pit_gbm-pit_logit'])}\\;{ci(row['ci_pit_gbm-pit_logit'])} & "
                   f"{signed(row['diff_pit_logit-ttc_logit'])}\\;{ci(row['ci_pit_logit-ttc_logit'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "robustness.tex").write_text("\n".join(out) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    start = A.window_start(A.EPV)[0]
    d_full = pd.read_parquet("data/interim/ecl_loan_month.parquet")
    d_full = d_full[(d_full.period >= "2022-01-01") & (d_full.period <= A.END)]
    d_win = d_full[d_full.period >= start]
    fig_paths(d_full, start, FIG / "ecl_paths.pdf")
    fig_leadlag(d_win, FIG / "lead_lag.pdf")
    fig_stages(d_full, FIG / "stage2_share.pdf")
    res, dif, h3 = A.main("ecl_ifrs9", start.strftime("%Y-%m-%d"), A.END, "", quiet=True)
    tables(res, dif, start)
    h3.round(3).to_csv("paper/tables/stage_decomposition.csv", index=False)
    print("window start", start.date())
    print("wrote", [str(p) for p in sorted(FIG.glob('*.pdf')) + sorted(TAB.glob('*'))])


if __name__ == "__main__":
    main()
