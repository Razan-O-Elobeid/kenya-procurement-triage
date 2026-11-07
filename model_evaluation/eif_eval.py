"""Extended Isolation Forest comparison (protocol 7.2) and checkpoint CP2.

Run from the repo root:  python -m model_evaluation.eif_eval
Training period only, on the shared harness folds and injections.
Grid: ntrees {100, 200, 400} x sample_size {256, 1024, 4096} x ndim {2, 4}, seeds {0, 1}.
Control: ndim=1, 400 trees, sample_size 256 (reported only, never selectable).
Every other isotree parameter stays at the installed default.
Protocol seed s is passed to isotree as random_seed = 100000 * s (seeds 0 and 100000), see the
deviations log. The first run with seeds 0 and 1 is kept in superseded_eif_seed_overlap/.
CP2: the whole evaluation runs twice with fresh models; the metric tables must be identical,
otherwise nothing is saved and the exit status is 1.
Writes only results/model_evaluation/eif_grid_by_fold.csv, eif_grid_summary.csv and
eif_selection.json, and refuses to overwrite them.
"""
import itertools
import json
import platform
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import isotree
from model_evaluation.harness import SklearnIF, build_folds, evaluate, load_training_period

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation"
FILES = {"folds": OUT / "eif_grid_by_fold.csv", "summary": OUT / "eif_grid_summary.csv",
         "selection": OUT / "eif_selection.json"}
CP1 = OUT / "cp1_if_400_256_by_fold.csv"

GRID = list(itertools.product((100, 200, 400), (256, 1024, 4096), (2, 4)))
CONTROL = (400, 256, 1)
SEEDS = (0, 1)
METRICS = ["stability", "recall_value", "recall_date", "recall_value_x5", "recall_primary",
           "spearman_R", "jaccard_top5_baseline"]
STAB_MIN, WITHIN_TIE, ACROSS_TIE, GUARD = 0.95, 0.005, 0.02, 0.05  # protocol section 6

# isotree seeds tree i with random_seed + i (src/fit_model.hpp:1756), so seeds 0 and 1 would
# share all but one tree. Protocol seed s is passed as 100000 * s (deviations log, 8 Nov 2026).
SEED_STRIDE = 100_000

# the build has no OpenMP, so isotree runs on one thread and warns about the default nthreads=-1
warnings.filterwarnings("ignore", message="Attempting to use more than 1 thread")


class IsoTreeEIF:
    """isotree with only ntrees, sample_size, ndim and the seed set. score() is higher = more anomalous."""

    def __init__(self, ntrees, sample_size, ndim, seed):
        self.kw = dict(ntrees=ntrees, sample_size=sample_size, ndim=ndim, random_seed=SEED_STRIDE * seed)

    def fit(self, Z):
        self.model = isotree.IsolationForest(**self.kw).fit(np.asarray(Z))
        return self

    def score(self, Z):
        return self.model.predict(np.asarray(Z))  # standardized outlier score


def run_all(folds, baseline):
    """One full pass over the grid and the control. Returns the per-fold table."""
    tables = []
    for nt, ss, nd in GRID + [CONTROL]:
        t0 = time.perf_counter()
        res = evaluate(lambda s: IsoTreeEIF(nt, ss, nd, s), folds, seeds=SEEDS, baseline=baseline)
        res.insert(0, "role", "control" if (nt, ss, nd) == CONTROL else "grid")
        res.insert(1, "ntrees", nt)
        res.insert(2, "sample_size", ss)
        res.insert(3, "ndim", nd)
        tables.append(res)
        print(f"  ntrees {nt:3d} | sample_size {ss:4d} | ndim {nd} | {time.perf_counter() - t0:5.1f} s")
    return pd.concat(tables, ignore_index=True)


def summarise(by_fold):
    keys = ["role", "ntrees", "sample_size", "ndim"]
    g = by_fold.groupby(keys)[METRICS + ["fit_seconds"]].mean().reset_index()
    return g.sort_values("stability", ascending=False).reset_index(drop=True)


def select_within(summary):
    """Phase 2 rule: highest stability; within 0.005 tied; ties broken by primary recall."""
    grid = summary[summary["role"] == "grid"]
    near = grid[grid["stability"] >= grid["stability"].max() - WITHIN_TIE]
    return near.sort_values(["recall_primary", "stability"], ascending=False).iloc[0]


def compare_with_baseline(eif, base):
    """Cross-family rule, section 6.2 and 6.3. Validation folds only."""
    d = float(eif["recall_primary"] - base["recall_primary"])
    if d > ACROSS_TIE:
        primary = "EIF better by more than 0.02"
    elif d < -ACROSS_TIE:
        primary = "EIF worse by more than 0.02"
    else:
        primary = "within plus or minus 0.02 (tie)"
    out = {"eif_stability_ok": bool(eif["stability"] >= STAB_MIN),
           "guard_value_ok": bool(eif["recall_value"] >= base["recall_value"] - GUARD),
           "guard_date_ok": bool(eif["recall_date"] >= base["recall_date"] - GUARD),
           "primary_difference": d, "primary_comparison": primary}
    # same rules and order as before; only the wording separates the three cases
    if not out["eif_stability_ok"]:
        out["scientific_result"] = "EIF below the stability requirement; baseline kept"
    elif not (out["guard_value_ok"] and out["guard_date_ok"]):
        out["scientific_result"] = "per-type guard fails; reported as a trade-off; baseline kept"
    elif d > ACROSS_TIE:
        out["scientific_result"] = "EIF ranks above the baseline"
    elif d < -ACROSS_TIE:
        out["scientific_result"] = "EIF worse than the baseline by more than 0.02; baseline kept"
    else:
        out["scientific_result"] = "within the 0.02 tie margin; simpler model (baseline) kept"
    return out


def full_period_cost(X, nt, ss, nd):
    """Timing on the full training period, stage by stage.

    features_and_preprocessing_fit_seconds: IndicatorBuilder fit + transform and preprocessor
    fit + transform, shared by both models. model_fit_seconds: model only, on preprocessed inputs.
    end_to_end_fit_seconds is the sum of the two separately timed stages, not one pipeline call.
    Scoring time is the model stage only, on preprocessed inputs.
    """
    from sklearn.base import clone
    from src.features import IndicatorBuilder
    from src.pipeline import build_preprocessor
    t0 = time.perf_counter()
    F = IndicatorBuilder().fit(X).transform(X)
    Z = np.asarray(clone(build_preprocessor()).fit(F).transform(F))
    prep_s = time.perf_counter() - t0
    cost = {"features_and_preprocessing_fit_seconds": round(prep_s, 2),
            "threads": "baseline IF uses all cores (n_jobs=-1); EIF runs on one thread"}
    for name, model in (("eif", IsoTreeEIF(nt, ss, nd, 0)), ("baseline_if", SklearnIF(400, 256, 0))):
        t0 = time.perf_counter()
        model.fit(Z)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.score(Z[:10_000])
        cost[name] = {"model_fit_seconds": round(fit_s, 2),
                      "end_to_end_fit_seconds": round(prep_s + fit_s, 2),
                      "model_score_seconds_per_10000": round(time.perf_counter() - t0, 2)}
    return cost


def main():
    for p in FILES.values():
        if p.exists():
            print("STOPPED: output exists:", p.relative_to(ROOT), file=sys.stderr)
            return 1
    if not CP1.exists():
        print("STOPPED: CP1 reference missing:", CP1.relative_to(ROOT), file=sys.stderr)
        return 1
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)

    # baseline on the same folds; must match CP1 exactly
    base_by_fold = evaluate(lambda s: SklearnIF(400, 256, seed=s), folds, seeds=SEEDS)
    cp1 = pd.read_csv(CP1)
    diff = float(np.max(np.abs(base_by_fold[["stability", "recall_value", "recall_date"]].to_numpy()
                               - cp1[["stability", "recall_value", "recall_date"]].to_numpy())))
    if diff > 1e-12:
        print(f"STOPPED: baseline differs from CP1 (max difference {diff:.1e})", file=sys.stderr)
        return 1
    print(f"baseline matches CP1 (max difference {diff:.1e})")
    base_scores = {f["fold"]: SklearnIF(400, 256, seed=0).fit(f["Ztr"]).score(f["Zva"]) for f in folds}

    print("\nrun 1 of 2")
    run1 = run_all(folds, base_scores)
    print("\nrun 2 of 2 (CP2 reproducibility)")
    run2 = run_all(folds, base_scores)
    cols = ["ntrees", "sample_size", "ndim", "fold"] + METRICS
    same = run1[cols].equals(run2[cols])
    print(f"\nCP2: metric tables identical across the two runs: {same}")
    if not same:
        print("STOPPED: CP2 failed; nothing saved", file=sys.stderr)
        return 1

    summary = summarise(run1)
    base = base_by_fold[[m for m in METRICS if m in base_by_fold] + ["fit_seconds"]].mean()
    base["jaccard_top5_baseline"] = 1.0
    pick = select_within(summary)
    decision = compare_with_baseline(pick, base)
    cost = full_period_cost(X, int(pick["ntrees"]), int(pick["sample_size"]), int(pick["ndim"]))

    effective = isotree.IsolationForest(ntrees=int(pick["ntrees"]), sample_size=int(pick["sample_size"]),
                                        ndim=int(pick["ndim"]), random_seed=SEED_STRIDE * SEEDS[0]).get_params()
    record = {
        "protocol_section": "7.2", "seeds": list(SEEDS), "grid": GRID, "control": CONTROL,
        "seed_mapping": f"isotree random_seed = {SEED_STRIDE} x protocol seed",
        "isotree_random_seeds": [SEED_STRIDE * s for s in SEEDS],
        "superseded_run": "results/model_evaluation/superseded_eif_seed_overlap/ (seeds 0 and 1 shared 399 of 400 trees)",
        "selected_within_family": {k: int(pick[k]) for k in ("ntrees", "sample_size", "ndim")},
        "selected_metrics": {m: float(pick[m]) for m in METRICS},
        "baseline_metrics": {m: float(base[m]) for m in METRICS},
        "cross_family": decision,
        "deployment_eligibility": {
            "end_to_end_fit_under_10_minutes": cost["eif"]["end_to_end_fit_seconds"] < 600,
            "fit_measure": "features and preprocessing plus model fit, timed as separate stages",
            "validated_explanation": "pending, protocol section 8 (scheduled 17 Nov)"},
        "cost": cost,
        "cp2_reproducible": True,
        "isotree_version": version("isotree"),
        "isotree_has_openmp": bool(getattr(isotree, "_get_has_openmp", lambda: None)()),
        "threading": "nthreads left at default (-1); build has no OpenMP, so isotree runs single-threaded",
        "effective_params_selected": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                      for k, v in effective.items()},
        "python": platform.python_version(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    run1.to_csv(FILES["folds"], index=False)
    summary.to_csv(FILES["summary"], index=False)
    FILES["selection"].write_text(json.dumps(record, indent=2))

    show = ["role", "ntrees", "sample_size", "ndim", "stability", "recall_value", "recall_date",
            "recall_value_x5", "recall_primary", "jaccard_top5_baseline", "fit_seconds"]
    print("\nsummary (fold means), sorted by stability:")
    print(summary[show].round(4).to_string(index=False))
    print("\nbaseline IF 400 / 256:", {m: round(float(base[m]), 4) for m in
                                       ["stability", "recall_value", "recall_date", "recall_value_x5", "recall_primary"]})
    print("selected EIF:", record["selected_within_family"])
    print("cross-family:", decision)
    print("cost:", cost)
    print("\nsaved", *(p.relative_to(ROOT) for p in FILES.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
