"""Pre-run check for the naive red-flag count (protocol 7.1). No scores, rankings or recall.

Run from the repo root:  python -m model_evaluation.naive_count_check
Prints, per fold and on the training part only: the f3, f4 and f5 thresholds, how often
each count component fires, and whether a threshold is degenerate (equal to the training
minimum). Also checks the f3 and R(c) recomputation for synthetic date copies.
Writes nothing.
"""
import sys
from pathlib import Path

import numpy as np

from model_evaluation.harness import build_folds, load_training_period
from model_evaluation.indicator_scores import (COUNT_CONTINUOUS, COUNT_FLAGS, fit_count_thresholds,
                                               injected_frames)
from src.features import ecdf_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)
    ok = True
    for f in folds:
        Ftr = f["Ftr"]
        thr = fit_count_thresholds(Ftr)
        print(f"\nfold {f['fold']} (training part {len(Ftr):,})")
        for c in COUNT_FLAGS:
            print(f"  {c:28s} fires {Ftr[c].mean():6.1%}")
        for c in COUNT_CONTINUOUS:
            x = Ftr[c]
            degenerate = np.isclose(thr[c], x.min())
            print(f"  {c:28s} P95 {thr[c]:.4f} | min {x.min():.4f} | covered {x.notna().mean():6.1%} | "
                  f"fires {(x >= thr[c]).mean():6.1%}" + ("  << threshold equals minimum" if degenerate else ""))

        # date copies: f3 must equal 1 - ECDF(-200) and R(c) must be the mean of available indicators
        G = injected_frames(f)["date"]
        f3_expected = float(1 - ecdf_score(f["ib"].d_ref_, [-200.0]).iloc[0])
        f3_ok = np.allclose(G["f3_decision_period"], f3_expected)
        r_ok = np.allclose(G["R"], G[["f1_direct_award", "f3_decision_period", "f4_supplier_concentration",
                                     "f5_benford_deviation"]].mean(axis=1))
        stale = f["Fva"].loc[G.index, "f3_decision_period"].mean()
        print(f"  date copies: {len(G)} | f3 recomputed to {f3_expected:.4f} (was mean {stale:.4f}) "
              f"| f3 check {f3_ok} | R check {r_ok}")
        ok &= f3_ok and r_ok
    print("\nrecomputation checks passed" if ok else "\nrecomputation checks FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
