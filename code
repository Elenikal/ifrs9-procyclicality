"""Tune the gradient-boosting PD arm once, on data available before the evaluation window.

Tuning refit date 2022-07-01 (first stable refit). Train: horizon rows with label_known <= 2021-11-01.
Validation: label_known in (2021-11-01, 2022-05-01], i.e. still before the tuning refit's reporting cut-off.
Selection by weighted multiclass log loss; the chosen parameters and iteration count are then held fixed for
every refit. Output: docs/gbm_params.json
"""
import itertools, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import fit_pd_horizon as F

TRAIN_END, VALID_END = pd.Timestamp("2021-11-01"), pd.Timestamp("2022-05-01")
GRID = dict(max_leaf_nodes=[7, 15, 31], min_samples_leaf=[50, 200, 800], l2_regularization=[0.0, 1.0])


def main():
    panel = pd.read_parquet("data/interim/model_panel.parquet")
    h = pd.read_parquet("data/interim/horizon_train.parquet").merge(panel.drop(columns=["outcome"]), on=["loan_id", "period"])
    h = F.window_features(h, h.k)
    h["draw"] = np.random.default_rng(F.SEED).random(len(h))
    tr = h[(h.label_known <= TRAIN_END) & ((h.label > 0) | (h.draw < F.NEG_SAMPLE))]
    va = h[(h.label_known > TRAIN_END) & (h.label_known <= VALID_END)]
    cols = F.LOAN + F.MACRO
    Xtr, stats = F.design(tr, cols)
    Xva = F.design(va, cols, stats)[0]
    wtr = np.where(tr.label == 0, 1 / F.NEG_SAMPLE, 1.0)
    labels = sorted(tr.label.unique())
    yva = va.label.where(va.label.isin(labels), 0)
    print(f"train rows {len(tr):,} (labels {tr.label.value_counts().to_dict()}), validation rows {len(va):,} "
          f"(labels {va.label.value_counts().to_dict()})")

    def score(m):
        p = m.predict_proba(Xva)
        cls = list(m.classes_)
        pdef = sum(p[:, cls.index(c)] for c in (1, 3) if c in cls)
        return log_loss(yva, p, labels=cls), roc_auc_score(va.label.isin([1, 3]), pdef)

    ll, auc = score(F.logit().fit(Xtr, tr.label, sample_weight=wtr))
    print(f"logit reference: logloss {ll:.5f}  default AUC {auc:.3f}")
    rows = []
    for vals in itertools.product(*GRID.values()):
        params = dict(zip(GRID, vals))
        best = None
        for n_iter in (50, 100, 200, 400):
            m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=n_iter, early_stopping=False,
                                               random_state=F.SEED, **params).fit(Xtr, tr.label, sample_weight=wtr)
            ll, auc = score(m)
            rows.append(dict(**params, max_iter=n_iter, logloss=ll, auc=auc))
    res = pd.DataFrame(rows).sort_values("logloss")
    print(res.head(10).round(5).to_string(index=False))
    best = res.iloc[0]
    params = dict(learning_rate=0.05, max_leaf_nodes=int(best.max_leaf_nodes), min_samples_leaf=int(best.min_samples_leaf),
                  l2_regularization=float(best.l2_regularization), max_iter=int(best.max_iter))
    json.dump(dict(params=params, tuned_on=dict(train_label_known_max=str(TRAIN_END.date()),
                                                 valid_label_known_max=str(VALID_END.date())),
                   validation=dict(logloss=round(best.logloss, 5), auc=round(best.auc, 3))),
              open("docs/gbm_params.json", "w"), indent=2)
    print("chosen:", params)


if __name__ == "__main__":
    main()
