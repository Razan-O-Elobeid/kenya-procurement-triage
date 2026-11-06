"""Shared evaluation harness for model comparison (see MODEL_EVALUATION_PROTOCOL.md, sections 5 and 6).

Training period only. Folds, fitting and injection follow src/tuning.py::run_grid exactly,
so the Phase 2 grid numbers can be reproduced (checkpoint CP1) before any new model is run.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit

from src.features import IndicatorBuilder
from src.pipeline import build_preprocessor
from src.tuning import DATE, inject

EXPECTED = dict(n_train=72_585, n_test=36_399)


def load_training_period(root):
    """Training period only, built exactly as in notebook cell 0 and src/baseline.py."""
    from src.data import load_merged_table, parse_dates, temporal_split
    data = root / "data" / "raw" / "full"
    if not data.exists():
        data = root / "data" / "raw" / "kenya_ppra_full.csv.tar.gz"
    aw, train = temporal_split(parse_dates(load_merged_table(data)))
    if (int(train.sum()), int((~train).sum())) != (EXPECTED["n_train"], EXPECTED["n_test"]):
        raise ValueError(f"unexpected split sizes {int(train.sum())} / {int((~train).sum())}")
    return aw[train]  # the test period is never returned


def inject_value(F, ib, factor, frac=0.02, seed=0):
    """Same as tuning.inject(kind='value') but with any factor. Same seed picks the same rows."""
    rng = np.random.default_rng(seed)
    pool = F.index[F["value_amount"] > 0]
    pick = rng.choice(pool, size=max(1, int(frac * len(F))), replace=False)
    G = F.loc[pick].copy()
    v = G["value_amount"] * factor
    G["value_amount"] = v
    G["zero_value_award"] = (v == 0).astype(int)
    G["above_p99"] = (v > ib.p99_).astype(int)
    G["below_p1"] = ((v > 0) & (v < ib.p1_)).astype(int)
    return G


def build_folds(X, n_splits=5, frac=0.02, log=print):
    """Fit indicators and preprocessing per fold on the fold's training part only.

    Returns one dict per fold with clean and injected model inputs. Built once, reused by every model.
    """
    X = X.sort_values(DATE, kind="stable")
    folds = []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ib = IndicatorBuilder().fit(Xtr)
        Ftr, Fva = ib.transform(Xtr), ib.transform(Xva)
        prep = clone(build_preprocessor()).fit(Ftr)
        inj = {"value": inject(Fva, ib, "value", frac, seed=fold),
               "date": inject(Fva, ib, "date", frac, seed=fold),
               "value_x5": inject_value(Fva, ib, 5, frac, seed=fold)}
        folds.append(dict(fold=fold + 1, ib=ib, Ftr=Ftr, Fva=Fva,
                          Ztr=prep.transform(Ftr), Zva=prep.transform(Fva),
                          Zinj={k: prep.transform(G) for k, G in inj.items()}))
        log(f"fold {fold + 1}: train {len(Xtr):,} | validation {len(Xva):,} | injected {len(inj['value']):,} per type")
    return folds


class SklearnIF:
    """Phase 2 model. score() is higher = more anomalous."""

    def __init__(self, n_estimators=400, max_samples=256, seed=0):
        self.kw = dict(n_estimators=n_estimators, max_samples=max_samples)
        self.seed = seed

    def fit(self, Z):
        m = min(self.kw["max_samples"], len(Z))
        self.model = IsolationForest(n_estimators=self.kw["n_estimators"], max_samples=m,
                                     random_state=self.seed, n_jobs=-1).fit(Z)
        return self

    def score(self, Z):
        return -self.model.score_samples(Z)


def top_set(scores, top=0.95):
    return set(np.flatnonzero(scores > np.quantile(scores, top)))


def evaluate(make_model, folds, seeds=(0, 1), top=0.95, baseline=None):
    """One row per fold. make_model(seed) returns an unfitted model with fit(Z) and score(Z).

    baseline: optional dict fold -> clean validation scores of the frozen baseline (seed 0),
    used for the top-5% Jaccard overlap.
    """
    rows = []
    for f in folds:
        clean, rec, fit_s = [], {k: [] for k in f["Zinj"]}, []
        for s in seeds:
            t0 = time.perf_counter()
            model = make_model(s).fit(f["Ztr"])
            fit_s.append(time.perf_counter() - t0)
            sc = model.score(f["Zva"])
            thr = np.quantile(sc, top)  # cut-off from clean scores only
            clean.append(sc)
            for k, Z in f["Zinj"].items():
                rec[k].append(float((model.score(Z) > thr).mean()))
        row = dict(fold=f["fold"], stability=spearmanr(clean[0], clean[1])[0],
                   recall_value=np.mean(rec["value"]), recall_date=np.mean(rec["date"]),
                   recall_value_x5=np.mean(rec["value_x5"]),
                   spearman_R=spearmanr(clean[0], f["Fva"]["R"], nan_policy="omit")[0],
                   fit_seconds=float(np.mean(fit_s)))
        row["recall_primary"] = (row["recall_value"] + row["recall_date"]) / 2
        if baseline is not None:
            a, b = top_set(clean[0], top), top_set(baseline[f["fold"]], top)
            row["jaccard_top5_baseline"] = len(a & b) / len(a | b)
        rows.append(row)
    return pd.DataFrame(rows)
