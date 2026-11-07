"""R(c) reweighting stability (protocol 7.1), on the 2018 to 2023 validation folds only.

Run from the repo root:  python -m model_evaluation.r_reweighting_eval
200 weight vectors from a flat Dirichlet over f1, f3, f4, f5 (seed 0). Each reweighted R(c)
is the weighted mean of the indicators available for a contract, so equal weights give the
Phase 2 R(c) exactly (checked). Compared with equal weights on each clean validation fold.
Writes only results/model_evaluation/r_reweighting_by_vector.csv and refuses to overwrite it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model_evaluation.harness import build_folds, load_training_period, top_set
from src.features import INDICATORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation" / "r_reweighting_by_vector.csv"
N_VECTORS, SEED, TOP = 200, 0, 0.95


def weighted_R(F, w):
    """Weighted mean of the available indicators."""
    A = F[INDICATORS].to_numpy(dtype=float)
    have = ~np.isnan(A)
    return (np.nan_to_num(A) @ w) / (have @ w)


def main():
    if OUT.exists():
        print("STOPPED: output exists:", OUT.relative_to(ROOT), file=sys.stderr)
        return 1
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)
    W = np.random.default_rng(SEED).dirichlet(np.ones(len(INDICATORS)), size=N_VECTORS)

    rows = []
    for f in folds:
        F = f["Fva"]
        base = weighted_R(F, np.full(len(INDICATORS), 0.25))
        if not np.allclose(base, F["R"].to_numpy(), rtol=0, atol=1e-12):
            print(f"STOPPED: equal weights do not reproduce R(c) in fold {f['fold']}", file=sys.stderr)
            return 1
        tie = 1e-9 * np.random.default_rng([SEED, f["fold"]]).random(len(F))  # same tie-break for every vector
        base_top = top_set(base + tie, TOP)
        for i, w in enumerate(W):
            r = weighted_R(F, w)
            top = top_set(r + tie, TOP)
            rows.append(dict(fold=f["fold"], vector=i, **{f"w_{k.split('_')[0]}": w[j] for j, k in enumerate(INDICATORS)},
                             spearman=spearmanr(r, base)[0],
                             jaccard_top5=len(top & base_top) / len(top | base_top)))
        print(f"fold {f['fold']}: equal weights reproduce R(c) exactly")

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    q = lambda p: (lambda s: s.quantile(p))
    summ = res.groupby("fold").agg(spearman_median=("spearman", "median"), spearman_p5=("spearman", q(0.05)),
                                   spearman_min=("spearman", "min"), jaccard_median=("jaccard_top5", "median"),
                                   jaccard_p5=("jaccard_top5", q(0.05)), jaccard_min=("jaccard_top5", "min"))
    print("\nagreement with equal-weight R(c), over 200 weight vectors:")
    print(summ.round(3).to_string())
    print("\nall folds:", summ.mean().round(3).to_dict())
    # which weight moves the top set most (correlation across vectors, pooled over folds)
    wcols = [c for c in res.columns if c.startswith("w_")]
    print("\ncorrelation of each weight with top-5% Jaccard:",
          {c: round(float(np.corrcoef(res[c], res["jaccard_top5"])[0, 1]), 3) for c in wcols})
    print("\nsaved", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
