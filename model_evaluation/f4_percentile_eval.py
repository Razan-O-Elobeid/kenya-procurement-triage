"""f4 percentile sensitivity of R(c) (protocol 7.4, amendment 6c0cd50). Training period only.

Run from the repo root:  python -m model_evaluation.f4_percentile_eval
For each validation fold: f4 percentiles referenced to the fold's training part, R(c) recomputed,
mean R(c) by number of available indicators (2, 3, 4) for the original and modified R(c).
Decision (committed): the range across the 2-, 3- and 4-indicator groups is computed on each
clean validation fold and averaged over folds; adopt only if the averaged range falls by at least half.
Also reported: Spearman and top-5% Jaccard between the original and modified R(c).
No model is fitted. Writes only results/model_evaluation/f4_percentile_by_fold.csv and
f4_percentile_summary.json, and refuses to overwrite them.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model_evaluation.f4_percentile import (F4, adopt, fit_reference, group_means, midrank_percentile,
                                            n_available, r_with_f4_percentile, require_finite,
                                            require_groups, require_reference)
from model_evaluation.harness import build_folds, load_training_period, top_set
from src.features import INDICATORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation"
FILES = {"folds": OUT / "f4_percentile_by_fold.csv", "summary": OUT / "f4_percentile_summary.json"}
TOP = 0.95


class CheckFailed(Exception):
    pass


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise CheckFailed(msg)


def prerequisite(fn, *args):
    """Run a prerequisite check; stop clearly if it fails."""
    try:
        fn(*args)
        return True, ""
    except ValueError as e:
        return False, str(e)


def run():
    for p in FILES.values():
        check(not p.exists(), f"output does not exist yet: {p.relative_to(ROOT)}")
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)

    rows = []
    for f in folds:
        Ftr, Fva = f["Ftr"], f["Fva"]
        ref = fit_reference(Ftr)
        good, why = prerequisite(require_reference, ref)
        check(good, f"fold {f['fold']}: training f4 reference has {len(ref):,} finite values {why}".rstrip())
        check(np.allclose(Fva[INDICATORS].mean(axis=1), Fva["R"], rtol=0, atol=1e-12),
              f"fold {f['fold']}: original R(c) equals the mean of available indicators")
        tr_pct = midrank_percentile(Ftr[F4].dropna(), ref)
        check(abs(tr_pct.mean() - 0.5) < 1e-12, f"fold {f['fold']}: training mean percentile is 0.5")

        r0 = Fva["R"].to_numpy(dtype=float)
        r1 = r_with_f4_percentile(Fva, ref).to_numpy()
        n = n_available(Fva).to_numpy()
        good, why = prerequisite(require_groups, n)
        check(good, f"fold {f['fold']}: validation contracts in the 2-, 3- and 4-indicator groups {why}".rstrip())
        check(np.array_equal(np.isnan(r0), np.isnan(r1)), f"fold {f['fold']}: same contracts have R(c)")
        m0, range0 = group_means(r0, n)
        m1, range1 = group_means(r1, n)
        for name, value in (("original range", range0), ("percentile range", range1)):
            good, why = prerequisite(require_finite, value, name)
            check(good, f"fold {f['fold']}: {name} is finite {why}".rstrip())

        tie = 1e-9 * np.random.default_rng([0, f["fold"]]).random(len(r0))  # same tie-break for both
        a, b = top_set(r0 + tie, TOP), top_set(r1 + tie, TOP)
        zero_share = float((ref == 0).mean())
        rows.append({"fold": f["fold"], "range_original": range0, "range_percentile": range1,
                     **{f"mean_R_{k}_original": float(m0.get(k, np.nan)) for k in (1, 2, 3, 4)},
                     **{f"mean_R_{k}_percentile": float(m1.get(k, np.nan)) for k in (1, 2, 3, 4)},
                     **{f"n_{k}": int((n == k).sum()) for k in (1, 2, 3, 4)},
                     "spearman": float(spearmanr(r0, r1, nan_policy="omit")[0]),
                     "jaccard_top5": len(a & b) / len(a | b),
                     "train_zero_share_f4": zero_share, "pct_of_zero": 0.5 * zero_share})

    res = pd.DataFrame(rows)
    mean0, mean1 = float(res["range_original"].mean()), float(res["range_percentile"].mean())
    adopted = adopt(mean0, mean1)
    summary = {"protocol": "7.4, amendment 6c0cd50",
               "rule": "adopt only if the fold-averaged range across the 2-, 3- and 4-indicator groups falls by at least half",
               "mean_range_original": mean0, "mean_range_percentile": mean1,
               "range_ratio": mean1 / mean0 if mean0 else None, "adopted": adopted,
               "mean_spearman": float(res["spearman"].mean()), "mean_jaccard_top5": float(res["jaccard_top5"].mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(FILES["folds"], index=False)
    FILES["summary"].write_text(json.dumps(summary, indent=2))

    show = ["fold", "n_2", "n_3", "n_4", "mean_R_2_original", "mean_R_3_original", "mean_R_4_original",
            "mean_R_2_percentile", "mean_R_3_percentile", "mean_R_4_percentile",
            "range_original", "range_percentile", "spearman", "jaccard_top5", "pct_of_zero"]
    print("\nper fold:")
    print(res[show].round(4).to_string(index=False))
    print(f"\nmean range: original {mean0:.4f} | percentile {mean1:.4f} | ratio {summary['range_ratio']:.3f}")
    print(f"agreement with original R(c): Spearman {summary['mean_spearman']:.3f} | top-5% Jaccard {summary['mean_jaccard_top5']:.3f}")
    print("adopted:", adopted)
    print("\nsaved", *(p.relative_to(ROOT) for p in FILES.values()))


def main():
    try:
        run()
    except CheckFailed as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
