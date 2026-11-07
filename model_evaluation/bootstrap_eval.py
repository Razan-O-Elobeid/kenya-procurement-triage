"""Bootstrap and jackknife stability (protocol 7.1), on the 2018 to 2023 validation folds only.

Run from the repo root:  python -m model_evaluation.bootstrap_eval
(a) model-stage bootstrap, every fold: indicators and preprocessing are fitted once in date
    order; only the model's training rows are resampled (20 resamples, seed = resample number).
(b) quarter jackknife, fold 5 only: each calendar quarter of the training part is dropped once
    and the full pipeline is refitted on the rest, in date order.
Model: Isolation Forest 400 / 256, random_state 0, as in the frozen baseline.
Writes only results/model_evaluation/bootstrap_by_fold.csv and jackknife_fold5_by_quarter.csv,
and refuses to overwrite them.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

from model_evaluation.harness import SklearnIF, build_folds, load_training_period, top_set
from src.pipeline import build_pipeline
from src.tuning import DATE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation"
FILES = {"boot": OUT / "bootstrap_by_fold.csv", "jack": OUT / "jackknife_fold5_by_quarter.csv"}
N_BOOT, TOP = 20, 0.95


def jaccard(a, b):
    return len(a & b) / len(a | b)


def bootstrap(folds):
    rows = []
    for f in folds:
        Ztr = np.asarray(f["Ztr"])
        ref = SklearnIF(400, 256, seed=0).fit(Ztr).score(f["Zva"])
        ref_top = top_set(ref, TOP)
        S = []
        for b in range(N_BOOT):
            idx = np.random.default_rng(b).integers(0, len(Ztr), len(Ztr))  # rows with replacement
            S.append(SklearnIF(400, 256, seed=0).fit(Ztr[idx]).score(f["Zva"]))
        S = np.column_stack(S)
        rho = spearmanr(S)[0]
        pairs = rho[np.triu_indices(N_BOOT, k=1)]
        rows.append(dict(fold=f["fold"],
                         spearman_pairwise_mean=float(pairs.mean()), spearman_pairwise_min=float(pairs.min()),
                         spearman_vs_full_mean=float(np.mean([spearmanr(S[:, b], ref)[0] for b in range(N_BOOT)])),
                         jaccard_top5_vs_full_mean=float(np.mean([jaccard(top_set(S[:, b], TOP), ref_top)
                                                                  for b in range(N_BOOT)])),
                         jaccard_top5_vs_full_min=float(min(jaccard(top_set(S[:, b], TOP), ref_top)
                                                            for b in range(N_BOOT)))))
        print(f"bootstrap fold {f['fold']} done")
    return pd.DataFrame(rows)


def jackknife(X):
    X = X.sort_values(DATE, kind="stable")
    tr, va = list(TimeSeriesSplit(n_splits=5).split(X))[-1]
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    full = build_pipeline(n_estimators=400, max_samples=256, random_state=0).fit(Xtr)
    ref = -full.score_samples(Xva)
    ref_top = top_set(ref, TOP)
    q = Xtr[DATE].dt.to_period("Q")
    rows = []
    for quarter in sorted(q.unique()):
        keep = (q != quarter).to_numpy()
        t0 = time.perf_counter()
        sc = -build_pipeline(n_estimators=400, max_samples=256, random_state=0).fit(Xtr[keep]).score_samples(Xva)
        rows.append(dict(quarter=str(quarter), n_dropped=int((~keep).sum()),
                         spearman_vs_full=spearmanr(sc, ref)[0],
                         jaccard_top5_vs_full=jaccard(top_set(sc, TOP), ref_top),
                         fit_seconds=time.perf_counter() - t0))
        print(f"jackknife {quarter}: dropped {int((~keep).sum()):,}")
    return pd.DataFrame(rows)


def main():
    for p in FILES.values():
        if p.exists():
            print("STOPPED: output exists:", p.relative_to(ROOT), file=sys.stderr)
            return 1
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    boot = bootstrap(build_folds(X))
    jack = jackknife(X)
    OUT.mkdir(parents=True, exist_ok=True)
    boot.to_csv(FILES["boot"], index=False)
    jack.to_csv(FILES["jack"], index=False)

    print("\n(a) model-stage bootstrap, 20 resamples per fold:")
    print(boot.round(3).to_string(index=False))
    print("all folds:", boot.drop(columns="fold").mean().round(3).to_dict())
    print("\n(b) fold 5 quarter jackknife:")
    print(jack.round(3).to_string(index=False))
    print("summary:", {"spearman_median": round(float(jack["spearman_vs_full"].median()), 3),
                       "spearman_min": round(float(jack["spearman_vs_full"].min()), 3),
                       "jaccard_median": round(float(jack["jaccard_top5_vs_full"].median()), 3),
                       "jaccard_min": round(float(jack["jaccard_top5_vs_full"].min()), 3)})
    print("\nsaved", *(p.relative_to(ROOT) for p in FILES.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
