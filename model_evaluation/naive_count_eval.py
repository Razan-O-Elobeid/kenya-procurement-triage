"""Naive red-flag count baseline (protocol 7.1), on the 2018 to 2023 validation folds only.

Run from the repo root:  python -m model_evaluation.naive_count_eval
Scores each clean validation fold and its injected copies with:
  count: nine red flags (protocol 7.1), ties broken at random
  R:     the Phase 2 composite R(c), for comparison (ties broken at random)
and compares them with the frozen baseline configuration (Isolation Forest 400 / 256, seed 0).
Writes only results/model_evaluation/naive_count_by_fold.csv and
naive_count_top5_composition.csv, and refuses to overwrite them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model_evaluation.harness import SklearnIF, build_folds, load_training_period, top_set
from model_evaluation.indicator_scores import (COUNT_CONTINUOUS, COUNT_FLAGS, fit_count_thresholds,
                                               injected_frames)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation"
FILES = {"folds": OUT / "naive_count_by_fold.csv", "comp": OUT / "naive_count_top5_composition.csv"}
TYPES = ["value", "date", "value_x5"]
TOP = 0.95


def raw_count(F, thr):
    flags = F[COUNT_FLAGS].fillna(0).sum(axis=1)
    cont = sum((F[c] >= thr[c]).astype(int) for c in COUNT_CONTINUOUS)  # missing counts as 0
    return (flags + cont).to_numpy(dtype=float)


def jitter(n, seed, fold, stream, scale):
    # random tie-break; scale is below the gap between distinct values
    return scale * np.random.default_rng([seed, fold, stream]).random(n)


def scorers(thr):
    return {"count": (lambda F: raw_count(F, thr), 0.5),
            "R": (lambda F: F["R"].to_numpy(dtype=float), 1e-9)}


def evaluate_fold(f, if_clean):
    thr = fit_count_thresholds(f["Ftr"])
    inj = injected_frames(f)
    rows, tops = [], {"IF": top_set(if_clean, TOP)}
    raws = {}
    for name, (fn, scale) in scorers(thr).items():
        raw = fn(f["Fva"])
        raws[name] = raw
        clean, rec = [], {k: [] for k in TYPES}
        for seed in (0, 1):
            sc = raw + jitter(len(raw), seed, f["fold"], 0, scale)
            cut = np.quantile(sc, TOP)  # cut-off from clean scores only
            clean.append(sc)
            for i, k in enumerate(TYPES, start=1):
                g = fn(inj[k])
                rec[k].append(float((g + jitter(len(g), seed, f["fold"], i, scale) > cut).mean()))
        tops[name] = top_set(clean[0], TOP)
        row = dict(fold=f["fold"], scorer=name, tiebreak_stability=spearmanr(clean[0], clean[1])[0],
                   **{f"recall_{k}": np.mean(v) for k, v in rec.items()},
                   spearman_IF=spearmanr(raw, if_clean)[0])
        row["recall_primary"] = (row["recall_value"] + row["recall_date"]) / 2
        if name == "count":
            row["count_at_cutoff"] = float(np.floor(np.quantile(clean[0], TOP)))
            row["share_count_0"] = float((raw == 0).mean())
        rows.append(row)
    for row in rows:
        other = "R" if row["scorer"] == "count" else "count"
        row[f"spearman_{other}"] = spearmanr(raws[row["scorer"]], raws[other])[0]
        a = tops[row["scorer"]]
        row["jaccard_top5_IF"] = len(a & tops["IF"]) / len(a | tops["IF"])
        row[f"jaccard_top5_{other}"] = len(a & tops[other]) / len(a | tops[other])

    # which red flags fire inside each top-5% set
    comp = []
    Fva = f["Fva"].reset_index(drop=True)
    for name, idx in tops.items():
        sub = Fva.iloc[sorted(idx)]
        for c in COUNT_FLAGS + COUNT_CONTINUOUS:
            fires = sub[c] if c in COUNT_FLAGS else (sub[c] >= thr[c]).astype(int)
            comp.append(dict(fold=f["fold"], top5_of=name, component=c, share_firing=float(fires.mean())))

    # f3 ties at the cut-off: how many training contracts share the smallest f3 value at or above P95
    f3 = f["Ftr"]["f3_decision_period"]
    at = f3[f3 >= thr["f3_decision_period"]].min()
    print(f"fold {f['fold']}: f3 P95 {thr['f3_decision_period']:.4f} | smallest firing f3 {at:.4f} "
          f"held by {(f3 == at).mean():.1%} of the training part | f3 fires {(f3 >= thr['f3_decision_period']).mean():.1%}")
    return rows, comp


def main():
    for p in FILES.values():
        if p.exists():
            print("STOPPED: output exists:", p.relative_to(ROOT), file=sys.stderr)
            return 1
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)
    rows, comp = [], []
    for f in folds:
        if_clean = SklearnIF(400, 256, seed=0).fit(f["Ztr"]).score(f["Zva"])
        r, c = evaluate_fold(f, if_clean)
        rows += r
        comp += c
    res = pd.DataFrame(rows)
    comp = pd.DataFrame(comp)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(FILES["folds"], index=False)
    comp.to_csv(FILES["comp"], index=False)

    cols = ["recall_value", "recall_date", "recall_value_x5", "recall_primary", "tiebreak_stability",
            "spearman_IF", "jaccard_top5_IF"]
    print("\nper fold:")
    print(res.round(3).to_string(index=False))
    print("\nfold means:")
    print(res.groupby("scorer")[cols].mean().round(3).to_string())
    cp1 = OUT / "cp1_if_400_256_by_fold.csv"
    if cp1.exists():
        m = pd.read_csv(cp1)[["recall_value", "recall_date", "recall_value_x5", "recall_primary"]].mean()
        print("\nfor comparison, Isolation Forest 400 / 256 (CP1):", m.round(3).to_dict())
    print("\nshare of each top-5% set where a flag fires (fold means):")
    print(comp.pivot_table(index="component", columns="top5_of", values="share_firing").round(3).to_string())
    print("\nsaved", *(p.relative_to(ROOT) for p in FILES.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
