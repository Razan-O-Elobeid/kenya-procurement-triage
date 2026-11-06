"""Checkpoint CP1: the shared harness must reproduce the Phase 2 grid row for 400 trees / 256 samples.

Run from the repo root:  python -m model_evaluation.cp1_reproduce
Reads results/if_tuning_by_fold.csv (Phase 2, never written). Writes only
results/model_evaluation/cp1_if_400_256_by_fold.csv, and refuses to overwrite it.
Any failed check exits with a nonzero status.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model_evaluation.harness import SklearnIF, build_folds, evaluate, load_training_period

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "if_tuning_by_fold.csv"
OUT = ROOT / "results" / "model_evaluation" / "cp1_if_400_256_by_fold.csv"
COLS = ["stability", "recall_value", "recall_date"]
TOL = 1e-12


class CheckFailed(Exception):
    pass


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise CheckFailed(msg)


def run():
    check(not OUT.exists(), f"output does not exist yet: {OUT.relative_to(ROOT)}")
    check(REF.exists(), f"Phase 2 reference exists: {REF.relative_to(ROOT)}")
    ref = pd.read_csv(REF)
    ref = ref[(ref["trees"] == 400) & (ref["max_samples"] == 256)].sort_values("fold").reset_index(drop=True)
    check(list(ref["fold"]) == [1, 2, 3, 4, 5], "reference has folds 1 to 5 for 400 / 256")

    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)
    res = evaluate(lambda s: SklearnIF(400, 256, seed=s), folds)

    for c in COLS:
        diff = float(np.max(np.abs(res[c].to_numpy() - ref[c].to_numpy())))
        check(diff <= TOL, f"{c} per fold matches Phase 2 (max difference {diff:.1e})")
    m = res[COLS].mean()
    print(f"fold means: stability {m['stability']:.4f} | value {m['recall_value']:.3f} | date {m['recall_date']:.3f}")
    check(round(m["stability"], 4) == 0.9916 and round(m["recall_value"], 3) == 0.117
          and round(m["recall_date"], 3) == 0.842, "fold means match the Phase 2 report (0.9916, 0.117, 0.842)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print("saved", OUT.relative_to(ROOT))
    print(res.round(4).to_string(index=False))
    print("CP1 passed")


def main():
    try:
        run()
    except CheckFailed as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
