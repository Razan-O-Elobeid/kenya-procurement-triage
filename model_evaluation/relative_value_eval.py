"""Relative-value experiment (protocol 7.3, amendment 6c0cd50). Training period only.

Run from the repo root:  python -m model_evaluation.relative_value_eval
Steps:
  1. rebuild the 14-input fold inputs and check they equal the harness exactly, and that the
     Isolation Forest 400 / 256 reproduces CP1 (stops otherwise)
  2. add each variant (buyer, category) as a 15th input, fitted per fold on the training part;
     injected copies pass through the same fitted preprocessing
  3. evaluate Isolation Forest 400 / 256, seeds 0 and 1, twice (identical tables required)
  4. apply the committed adoption rule and variant tie-breaker
  5. if a variant is adopted, re-evaluate EIF 400 / 256 / ndim 2 with it (grid not rerun)
Writes only results/model_evaluation/relative_value_by_fold.csv and relative_value_summary.json,
and refuses to overwrite them.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from model_evaluation.harness import SklearnIF, build_folds, evaluate, load_training_period
from model_evaluation.indicator_scores import injected_frames
from model_evaluation.relative_value import COLUMNS, build_preprocessor_relative
from src.pipeline import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation"
FILES = {"folds": OUT / "relative_value_by_fold.csv", "summary": OUT / "relative_value_summary.json"}
CP1 = OUT / "cp1_if_400_256_by_fold.csv"
VARIANTS = ("buyer", "category")
SEEDS = (0, 1)
GAIN, STAB_MIN, PREFER_MARGIN = 0.02, 0.95, 0.02  # protocol 7.3
METRICS = ["stability", "recall_value", "recall_date", "recall_value_x5", "recall_primary",
           "spearman_R", "jaccard_top5_baseline"]


class CheckFailed(Exception):
    pass


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise CheckFailed(msg)


def variant_folds(folds, make_prep):
    """Same folds and injected copies, with a different preprocessor fitted per fold."""
    out, levels = [], []
    for f in folds:
        prep = make_prep().fit(f["Ftr"])
        inj = injected_frames(f)
        g = {"fold": f["fold"], "Fva": f["Fva"], "Ztr": prep.transform(f["Ftr"]),
             "Zva": prep.transform(f["Fva"]), "Zinj": {k: prep.transform(G) for k, G in inj.items()}}
        out.append(g)
        if "relvalue" in prep.named_transformers_:
            rel = prep.named_transformers_["relvalue"].steps[0][1]
            share = rel.reference_level(f["Fva"][COLUMNS]).value_counts(normalize=True)
            levels.append({"fold": f["fold"], **{k: round(float(v), 4) for k, v in share.items()}})
    return out, levels


def adopt(row, base):
    return {"value_gain": float(row["recall_value"] - base["recall_value"]),
            "value_x5_gain": float(row["recall_value_x5"] - base["recall_value_x5"]),
            "primary_change": float(row["recall_primary"] - base["recall_primary"]),
            "passes": bool(row["recall_value"] - base["recall_value"] >= GAIN
                           and row["recall_value_x5"] - base["recall_value_x5"] >= GAIN
                           and row["recall_primary"] >= base["recall_primary"]
                           and row["stability"] >= STAB_MIN)}


def choose(results):
    """Committed tie-breaker: one passes -> it; both -> higher primary, within 0.02 -> buyer."""
    passing = [v for v in VARIANTS if results[v]["passes"]]
    if not passing:
        return None
    if len(passing) == 1:
        return passing[0]
    pb, pc = results["buyer"]["primary"], results["category"]["primary"]
    return "category" if pc - pb > PREFER_MARGIN else "buyer"


def run():
    for p in FILES.values():
        check(not p.exists(), f"output does not exist yet: {p.relative_to(ROOT)}")
    check(CP1.exists(), f"CP1 reference exists: {CP1.relative_to(ROOT)}")
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)

    # 1. the 14-input path must equal the harness and reproduce CP1
    base_folds, _ = variant_folds(folds, lambda: clone(build_preprocessor()))
    same = all(np.array_equal(a["Ztr"], f["Ztr"]) and np.array_equal(a["Zva"], f["Zva"])
               and all(np.array_equal(a["Zinj"][k], f["Zinj"][k]) for k in f["Zinj"])
               for a, f in zip(base_folds, folds))
    check(same, "rebuilt 14-input fold inputs and injected copies equal the harness exactly")
    base_scores = {f["fold"]: SklearnIF(400, 256, seed=0).fit(f["Ztr"]).score(f["Zva"]) for f in folds}
    base = evaluate(lambda s: SklearnIF(400, 256, seed=s), base_folds, seeds=SEEDS, baseline=base_scores)
    cp1 = pd.read_csv(CP1)
    diff = float(np.max(np.abs(base[["stability", "recall_value", "recall_date"]].to_numpy()
                               - cp1[["stability", "recall_value", "recall_date"]].to_numpy())))
    check(diff <= 1e-12, f"14-input Isolation Forest reproduces CP1 (max difference {diff:.1e})")
    base_mean = base[METRICS].mean()

    # 2-3. each variant as a 15th input, evaluated twice
    tables, levels, decisions = [base.assign(variant="none", model="IF 400/256")], {}, {}
    for v in VARIANTS:
        vf, levels[v] = variant_folds(folds, lambda v=v: build_preprocessor_relative(v))
        check(all(g["Ztr"].shape[1] == 15 for g in vf), f"{v}: 15 model inputs")
        r1 = evaluate(lambda s: SklearnIF(400, 256, seed=s), vf, seeds=SEEDS, baseline=base_scores)
        r2 = evaluate(lambda s: SklearnIF(400, 256, seed=s), vf, seeds=SEEDS, baseline=base_scores)
        check(r1[METRICS].equals(r2[METRICS]), f"{v}: two runs give identical metric tables")
        tables.append(r1.assign(variant=v, model="IF 400/256"))
        m = r1[METRICS].mean()
        decisions[v] = {**adopt(m, base_mean), "primary": float(m["recall_primary"]),
                        "means": {k: float(m[k]) for k in METRICS}}

    # 4. adoption
    chosen = choose(decisions)

    # 5. EIF re-evaluation only if a variant is adopted
    eif = None
    if chosen:
        from model_evaluation.eif_eval import IsoTreeEIF
        vf, _ = variant_folds(folds, lambda: build_preprocessor_relative(chosen))
        e14 = evaluate(lambda s: IsoTreeEIF(400, 256, 2, s), base_folds, seeds=SEEDS, baseline=base_scores)
        e15 = evaluate(lambda s: IsoTreeEIF(400, 256, 2, s), vf, seeds=SEEDS, baseline=base_scores)
        tables += [e14.assign(variant="none", model="EIF 400/256/2"), e15.assign(variant=chosen, model="EIF 400/256/2")]
        eif = {"without": {k: float(e14[METRICS].mean()[k]) for k in METRICS},
               "with": {k: float(e15[METRICS].mean()[k]) for k in METRICS}}

    by_fold = pd.concat(tables, ignore_index=True)
    summary = {"protocol": "7.3, amendment 6c0cd50", "baseline_14_inputs": {k: float(base_mean[k]) for k in METRICS},
               "variants": decisions, "adopted": chosen, "reference_levels_validation": levels,
               "eif_reevaluation": eif,
               "note": "synthetic recall measures sensitivity to injected distortions, not corruption detection"}
    OUT.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(FILES["folds"], index=False)
    FILES["summary"].write_text(json.dumps(summary, indent=2))

    show = ["stability", "recall_value", "recall_date", "recall_value_x5", "recall_primary", "jaccard_top5_baseline"]
    print("\nfold means:")
    print(by_fold.groupby(["model", "variant"], sort=False)[show].mean().round(4).to_string())
    for v in VARIANTS:
        d = decisions[v]
        print(f"{v}: value {d['value_gain']:+.3f} | value x5 {d['value_x5_gain']:+.3f} | "
              f"primary {d['primary_change']:+.3f} | passes rule {d['passes']}")
    print("reference levels used on validation folds:", levels)
    print("adopted:", chosen or "none")
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
