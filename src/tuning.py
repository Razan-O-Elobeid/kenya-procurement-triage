"""Isolation Forest tuning by rank stability, with synthetic anomaly injection as a sensitivity check.

Injected copies are built from the clean features of their originals, so they never enter
any buyer or supplier history, any fitted parameter, or the clean validation ranking.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit

from src.features import IndicatorBuilder
from src.pipeline import build_preprocessor

DATE = "tender_tenderPeriod_endDate"


def inject(F, ib, kind, frac=0.02, seed=0):
    """Perturbed copies of a random sample of clean feature rows.

    value: value x 50 (non-zero values only); value flags recomputed with fitted thresholds.
    date: signed 200 days before tender close (rows with a signing date only).
    f4 and f5 use strictly earlier awards, so they are unchanged by a contract's own value or date.
    """
    rng = np.random.default_rng(seed)
    if kind == "value":
        pool = F.index[F["value_amount"] > 0]
    else:
        pool = F.index[F["days_to_signing"].notna()]
    pick = rng.choice(pool, size=max(1, int(frac * len(F))), replace=False)
    G = F.loc[pick].copy()
    if kind == "value":
        v = G["value_amount"] * 50
        G["value_amount"] = v
        G["zero_value_award"] = (v == 0).astype(int)
        G["above_p99"] = (v > ib.p99_).astype(int)
        G["below_p1"] = ((v > 0) & (v < ib.p1_)).astype(int)
    else:
        G["days_to_signing"] = -200.0
        G["is_signed_pre_closure"] = 1
    return G


def run_grid(X, trees=(100, 200, 400), samples=(256, 1024, 4096), seeds=(0, 1),
             n_splits=5, frac=0.02, top=0.95, log=print):
    """Returns one row per (setting, fold). X must be the training period only."""
    X = X.sort_values(DATE, kind="stable")
    rows = []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ib = IndicatorBuilder().fit(Xtr)
        Ftr, Fva = ib.transform(Xtr), ib.transform(Xva)
        prep = clone(build_preprocessor()).fit(Ftr)
        Ztr, Zva = prep.transform(Ftr), prep.transform(Fva)
        Zinj = {k: prep.transform(inject(Fva, ib, k, frac, seed=fold)) for k in ("value", "date")}
        log(f"fold {fold + 1}: train {len(Xtr):,} | validation {len(Xva):,} | "
            f"injected {len(Zinj['value']):,} value + {len(Zinj['date']):,} date")
        for n, m in itertools.product(trees, samples):
            clean, rec = [], {"value": [], "date": []}
            for s in seeds:
                model = IsolationForest(n_estimators=n, max_samples=min(m, len(Ztr)),
                                        random_state=s, n_jobs=-1).fit(Ztr)
                sc = -model.score_samples(Zva)
                thr = np.quantile(sc, top)  # cut-off from clean scores only
                clean.append(sc)
                for k, Z in Zinj.items():
                    rec[k].append(float((-model.score_samples(Z) > thr).mean()))
            rows.append(dict(trees=n, max_samples=m, fold=fold + 1,
                             stability=spearmanr(clean[0], clean[1])[0],
                             recall_value=np.mean(rec["value"]), recall_date=np.mean(rec["date"])))
    return pd.DataFrame(rows)


def summarise(res, tie=0.005):
    """Mean over folds per setting, and the pre-specified selection."""
    g = (res.groupby(["trees", "max_samples"])
            .agg(stability=("stability", "mean"), stability_sd=("stability", "std"),
                 recall_value=("recall_value", "mean"), recall_date=("recall_date", "mean"))
            .reset_index())
    g["recall_mean"] = (g["recall_value"] + g["recall_date"]) / 2
    best = g["stability"].max()
    near = g[g["stability"] >= best - tie]
    pick = near.sort_values(["recall_mean", "stability"], ascending=False).iloc[0]
    return g.sort_values("stability", ascending=False).reset_index(drop=True), pick
