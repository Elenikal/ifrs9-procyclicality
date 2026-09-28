"""Robustness of the H1 result to the design choices that could overturn it.

The special-servicer default definition is a separate pipeline (VARIANT=_ss rebuilds panel, LGD and PD arms), so
its row is read from the files that variant produced rather than recomputed here.

Varies, one at a time from the baseline (SICR 2x, raw PDs, resolved-LGD, window 2022-07 to 2026-08):
  SICR multiple 1.5 / 2 / 3, recalibrated PDs, the LGD bound with open defaults imputed, and the window start.
Each configuration reports the relative trough-to-peak rise per arm and the paired differences with the
deal-clustered bootstrap. Model refits (GBM seeds, alternative default definition) are run separately.

Output: docs/robustness_results.csv
"""
import itertools
import pandas as pd

import analyse_procyclicality as A
import build_ecl as E

START_RULE = A.window_start(A.EPV)[0].strftime("%Y-%m-%d")   # pre-committed events-per-parameter rule
BASE = dict(sicr=2.0, source="horizon", lgd="headline_resolved", start=START_RULE)
GRID = [dict(sicr=s) for s in (1.5, 3.0)] + \
       [dict(source="cal"), dict(lgd="bound_with_open")] + \
       [dict(start=s) for s in ("2022-01-01", "2022-07-01", A.window_start(10)[0].strftime("%Y-%m-%d"))]


def run(cfg):
    tag = "_".join(f"{k}{v}" for k, v in cfg.items() if k != "start").replace(".", "") or "base"
    suffix = f"_{tag}"
    E.main(E.ARMS, cfg["lgd"], cfg["sicr"], cfg["source"], suffix)
    res, dif, _ = A.main("ecl_ifrs9", cfg["start"], A.END, suffix, quiet=True)
    res = res[res.regime == "ifrs9"][["arm", "rel_rise"]].set_index("arm").rel_rise
    out = {f"rise_{a}": v for a, v in res.items()}
    for r in dif.itertuples():
        out[f"diff_{r.comparison.replace(' ', '')}"] = r.mean_diff
        out[f"ci_{r.comparison.replace(' ', '')}"] = f"[{r.lo:.2f},{r.hi:.2f}]"
    return {**cfg, **out}


def ss_variant():
    """Special-servicer default definition: separate end-to-end build (see VARIANT in build_ecl/fit_pd_horizon)."""
    start = A.window_start(A.EPV, "_ss")[0].strftime("%Y-%m-%d")
    res, dif, _ = A.main("ecl_ifrs9", start, A.END, "_ss", quiet=True)
    res = res[res.regime == "ifrs9"][["arm", "rel_rise"]].set_index("arm").rel_rise
    out = dict(sicr=2.0, source="ss_default_def", lgd="headline_resolved", start=start,
               **{f"rise_{a}": v for a, v in res.items()})
    for r in dif.itertuples():
        out[f"diff_{r.comparison.replace(' ', '')}"] = r.mean_diff
        out[f"ci_{r.comparison.replace(' ', '')}"] = f"[{r.lo:.2f},{r.hi:.2f}]"
    return out


if __name__ == "__main__":
    rows = [run(BASE)]
    for g in GRID:
        rows.append(run({**BASE, **g}))
    rows.append(ss_variant())
    df = pd.DataFrame(rows)
    df.to_csv("docs/robustness_results.csv", index=False)
    pd.set_option("display.width", 260)
    print(df.round(3).to_string(index=False))
