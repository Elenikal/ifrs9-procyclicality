"""Learning curve and dispersion diagnostics: is the ML arm limited by sample size?

At a fixed refit date, the gradient boosting and logit arms are trained on random subsamples of loans and scored
on the same out-of-sample year. If the ML arm's accuracy is still rising at the full sample, the portfolio is
binding on it; if its predictions were merely shrunk towards a constant, its cross-sectional dispersion would also
be compressed, which is tested separately.

Outputs: docs/learning_curve.json, paper/figures/learning_curve.pdf, paper/tables/learning_curve.tex
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fit_pd_horizon as F

REFIT, FRACS, SEED = pd.Timestamp("2025-01-01"), (0.15, 0.3, 0.5, 0.75, 1.0), 7
STYLE = {"pit_gbm": ("#2f6fb2", "-", "Gradient boosting (PIT)"), "pit_logit": ("#d1701a", "--", "Logit (PIT)")}


def learning_curve():
    panel = pd.read_parquet("data/interim/model_panel.parquet")
    h = pd.read_parquet("data/interim/horizon_train.parquet").merge(panel.drop(columns=["outcome"]), on=["loan_id", "period"])
    h = F.window_features(h, h.k)
    rng = np.random.default_rng(SEED)
    h["draw"] = rng.random(len(h))
    tr_all = h[(h.label_known <= REFIT - pd.DateOffset(months=2)) & ((h.label > 0) | (h.draw < F.NEG_SAMPLE))]
    te = h[(h.label_known > REFIT - pd.DateOffset(months=2)) & (h.label_known <= REFIT + pd.DateOffset(months=10))]
    y = te.label.isin([1, 3]).astype(int)
    rows = []
    for frac in FRACS:
        loans = tr_all.loan_id.unique()
        pick = set(rng.choice(loans, max(2, int(len(loans) * frac)), replace=False))
        tr = tr_all[tr_all.loan_id.isin(pick)]
        for arm, (cols, make) in ((a, F.ARMS[a]) for a in STYLE):
            X, stats = F.design(tr, cols)
            m = make().fit(X, tr.label, sample_weight=np.where(tr.label == 0, 1 / F.NEG_SAMPLE, 1.0))
            p = m.predict_proba(F.design(te, cols, stats)[0])
            cls = list(m.classes_)
            pdef = sum(p[:, cls.index(c)] for c in (1, 3) if c in cls)
            rows.append(dict(frac=frac, n_train=len(tr), train_defaults=int(tr.label.isin([1, 3]).sum()),
                             arm=arm, auc=round(float(roc_auc_score(y, pdef)), 4)))
    return pd.DataFrame(rows), dict(test_rows=int(len(te)), test_defaults=int(y.sum()), refit=str(REFIT.date()))


def dispersion():
    P = pd.concat([pd.read_parquet(f"data/interim/pd_horizon_{a}.parquet", columns=["loan_id", "period", "arm", "pd12"])
                   for a in ("ttc_logit", "pit_logit", "pit_gbm")])
    P = P[(P.period >= "2023-10-01") & (P.period <= "2026-08-01")]
    cs = P.groupby(["arm", "period"]).pd12.std().groupby("arm").mean()
    ts = P.groupby(["arm", "period"]).pd12.mean().unstack("arm")
    return {a: dict(cross_sectional_sd=round(float(cs[a]), 4),
                    time_series_sd_pp=round(float(ts[a].std() * 100), 3),
                    sd_month_on_month_pct_change=round(float(ts[a].pct_change().std()), 3)) for a in cs.index}


def main():
    lc, meta = learning_curve()
    out = dict(meta=meta, learning_curve=lc.to_dict("records"), dispersion=dispersion())
    json.dump(out, open("docs/learning_curve.json", "w"), indent=2)

    w = lc.pivot(index=["frac", "n_train", "train_defaults"], columns="arm", values="auc").reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    for arm in STYLE:
        c, ls, lab = STYLE[arm]
        ax.plot(w.train_defaults, w[arm], color=c, ls=ls, lw=1.8, marker="o", ms=4, label=lab)
    ax.set_xlabel("defaults in the training sample")
    ax.set_ylabel("out-of-sample AUC")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("paper/figures/learning_curve.pdf", bbox_inches="tight"); plt.close(fig)

    t = [r"\begin{tabular}{rrrrr}", r"\toprule",
         r"Training loans & Training rows & Defaults & GB AUC & Logit AUC \\", r"\midrule"]
    for _, r in w.iterrows():
        t.append(f"{r.frac:.0%} & {int(r.n_train):,} & {int(r.train_defaults):,} & "
                 f"{r.pit_gbm:.3f} & {r.pit_logit:.3f} \\\\".replace("%", "\\%"))
    t += [r"\bottomrule", r"\end{tabular}"]
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    (Path("paper/tables") / "learning_curve.tex").write_text("\n".join(t) + "\n")
    print(json.dumps(out["dispersion"], indent=2)); print(w.to_string(index=False))


if __name__ == "__main__":
    main()
