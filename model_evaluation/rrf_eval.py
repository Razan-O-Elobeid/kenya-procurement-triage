"""Reciprocal rank fusion on the 2018 to 2023 validation folds (protocol 7.5 and its 13 Nov clarification).

Run from the repo root:  python -m model_evaluation.rrf_eval
Components: R(c), the frozen IF (400 / 256) and the selected EIF (400 / 256 / ndim 2).
Equal weights, k = 60, ranks against each fold's training-part scores.
Recall cut-off: 95th percentile of clean validation RRF scores, strictly greater, as in the harness.
Seeds {0, 1} apply to the IF and the EIF; R(c) is deterministic.

Before scoring, the script stops (exit 1, nothing saved) unless:
  1. the saved EIF and explanation results confirm EIF is the eligible third component;
  2. the IF and EIF components reproduce CP1 and the stored EIF grid row;
  3. injected model inputs line up with the recomputed indicator frames;
  4. R(c) has no missing values.
The whole RRF evaluation runs twice and the tables must be identical.
Writes only results/model_evaluation/rrf_by_fold.csv and rrf_result.json; refuses to overwrite.
"""
import json
import platform
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

from model_evaluation.rrf import K, fuse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation"
FILES = {"folds": OUT / "rrf_by_fold.csv", "result": OUT / "rrf_result.json"}
CP1 = OUT / "cp1_if_400_256_by_fold.csv"
EIF_GRID = OUT / "eif_grid_by_fold.csv"
EIF_SEL = OUT / "eif_selection.json"
EXPLAIN = OUT / "explanations" / "explain_validation.json"

EIF_CONFIG = dict(ntrees=400, sample_size=256, ndim=2)
SEEDS = (0, 1)
TYPES = ["value", "date", "value_x5"]
TOP = 0.95
STAB_MIN, ACROSS_TIE, GUARD = 0.95, 0.02, 0.05  # protocol section 6
REPRO_TOL = 1e-12
CHECK_COLS = ["stability", "recall_value", "recall_date", "recall_value_x5", "recall_primary"]

warnings.filterwarnings("ignore", message="Attempting to use more than 1 thread")


class Stop(Exception):
    pass


def require(cond, msg):
    if not cond:
        raise Stop(msg)


def default_models():
    """Model components by name. Each value maps a seed to an unfitted model."""
    from model_evaluation.eif_eval import IsoTreeEIF
    from model_evaluation.harness import SklearnIF
    return {"IF": lambda s: SklearnIF(400, 256, seed=s),
            "EIF": lambda s: IsoTreeEIF(EIF_CONFIG["ntrees"], EIF_CONFIG["sample_size"], EIF_CONFIG["ndim"], s)}


# ---------- eligibility and integrity checks ----------

def check_eligibility(eif_sel, explain):
    """EIF is the third component only if the saved results say it holds under 6.3."""
    sel = eif_sel.get("selected_within_family", {})
    cf = eif_sel.get("cross_family", {})
    checks = {
        "selected_config_matches": {k: sel.get(k) for k in EIF_CONFIG} == EIF_CONFIG,
        "ranks_above_baseline": cf.get("scientific_result") == "EIF ranks above the baseline",
        "stability_ok": cf.get("eif_stability_ok") is True,
        "guard_ok": cf.get("guard_value_ok") is True and cf.get("guard_date_ok") is True,
        "fit_under_10_minutes": eif_sel.get("deployment_eligibility", {}).get("end_to_end_fit_under_10_minutes") is True,
        "explanation_validated": explain.get("eif_perm", {}).get("validated") is True,
    }
    return all(checks.values()), checks


def check_alignment(f, inj):
    """Refitted preprocessing on the injected indicator frames must give the harness inputs exactly."""
    from src.pipeline import build_preprocessor
    prep = clone(build_preprocessor()).fit(f["Ftr"])
    for k in TYPES:
        a = np.asarray(prep.transform(inj[k]), dtype=float)
        if a.shape != np.asarray(f["Zinj"][k]).shape or not np.array_equal(a, np.asarray(f["Zinj"][k], dtype=float)):
            return False
    return True


def max_diff(a, b, cols):
    return float(np.max(np.abs(a[cols].to_numpy(dtype=float) - b[cols].to_numpy(dtype=float))))


# ---------- metrics ----------

def jaccard(a, b):
    return len(a & b) / len(a | b)


def top_set(scores, top=TOP):
    return set(np.flatnonzero(scores > np.quantile(scores, top)))


def fold_metrics(fold, clean, inj, R_va, baseline_clean=None, others=None):
    """Same definitions as harness.evaluate. clean: list of clean scores per seed.
    inj: dict type -> list of injected scores per seed. others: dict name -> seed-0 clean scores."""
    rec = {k: [] for k in inj}
    above = []
    for i, sc in enumerate(clean):
        thr = np.quantile(sc, TOP)  # cut-off from clean scores only
        above.append(float((sc > thr).mean()))
        for k in inj:
            rec[k].append(float((inj[k][i] > thr).mean()))
    row = dict(fold=fold, stability=spearmanr(clean[0], clean[1])[0],
               recall_value=np.mean(rec["value"]), recall_date=np.mean(rec["date"]),
               recall_value_x5=np.mean(rec["value_x5"]),
               spearman_R=spearmanr(clean[0], R_va, nan_policy="omit")[0],
               share_above_cutoff=float(np.mean(above)))
    row["recall_primary"] = (row["recall_value"] + row["recall_date"]) / 2
    if baseline_clean is not None:
        row["jaccard_top5_baseline"] = jaccard(top_set(clean[0]), top_set(baseline_clean))
    for name, sc in (others or {}).items():
        row[f"jaccard_top5_{name}"] = jaccard(top_set(clean[0]), top_set(sc))
        row[f"spearman_{name}"] = spearmanr(clean[0], sc)[0]
    return row


def rrf_fold(f, inj, models, seeds=SEEDS):
    """RRF scores for one fold. Returns clean scores per seed, injected scores per seed,
    seed-0 component scores on the clean fold and mean fit seconds."""
    clean, injs, comp0, fit_s = [], {k: [] for k in TYPES}, None, []
    for s in seeds:
        t0 = time.perf_counter()
        fitted = {name: make(s).fit(f["Ztr"]) for name, make in models.items()}
        refs = {"R": f["Ftr"]["R"].to_numpy(dtype=float)}
        refs.update({name: m.score(f["Ztr"]) for name, m in fitted.items()})
        fit_s.append(time.perf_counter() - t0)
        for v in refs.values():
            if not np.isfinite(v).all():
                raise Stop(f"fold {f['fold']}: non-finite reference scores")

        def score(Z, F):
            out = {"R": F["R"].to_numpy(dtype=float)}
            out.update({name: m.score(Z) for name, m in fitted.items()})
            return out

        comp = score(f["Zva"], f["Fva"])
        clean.append(fuse(refs, comp))
        for k in TYPES:
            injs[k].append(fuse(refs, score(f["Zinj"][k], inj[k])))
        if comp0 is None:
            comp0 = comp
    return clean, injs, comp0, float(np.mean(fit_s))


def evaluate_rrf(folds, injected, models, baseline=None, seeds=SEEDS):
    """One row per fold. injected: dict fold -> {type: indicator frame of the injected copies}."""
    rows = []
    for f in folds:
        inj = injected[f["fold"]]
        clean, injs, comp0, fit_s = rrf_fold(f, inj, models, seeds)
        others = {n: v for n, v in comp0.items() if n != "R"}
        row = fold_metrics(f["fold"], clean, injs, f["Fva"]["R"].to_numpy(dtype=float),
                           None if baseline is None else baseline[f["fold"]], others)
        row["fit_seconds"] = fit_s
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- ranking rule (7.5 clarification of 13 Nov) ----------

def rank_rrf(rrf, base, best, best_name="EIF"):
    """rrf, base, best: fold-mean metrics. best is the best-ranked single family under 6.3."""
    d_best = float(rrf["recall_primary"] - best["recall_primary"])
    out = {"stability_ok": bool(rrf["stability"] >= STAB_MIN),
           "guard_value_ok": bool(rrf["recall_value"] >= base["recall_value"] - GUARD),
           "guard_date_ok": bool(rrf["recall_date"] >= base["recall_date"] - GUARD),
           "primary_difference_vs_baseline": float(rrf["recall_primary"] - base["recall_primary"]),
           f"primary_difference_vs_{best_name}": d_best,
           f"value_difference_vs_{best_name}": float(rrf["recall_value"] - best["recall_value"]),
           f"date_difference_vs_{best_name}": float(rrf["recall_date"] - best["recall_date"])}
    if not out["stability_ok"]:
        out["scientific_result"] = f"RRF below the stability requirement; {best_name} keeps the top rank"
    elif not (out["guard_value_ok"] and out["guard_date_ok"]):
        out["scientific_result"] = f"per-type guard fails; reported as a trade-off; {best_name} keeps the top rank"
    elif d_best > ACROSS_TIE:
        out["scientific_result"] = f"RRF ranks above {best_name}"
    elif d_best < -ACROSS_TIE:
        out["scientific_result"] = f"RRF worse than {best_name} by more than 0.02; {best_name} keeps the top rank"
    else:
        out["scientific_result"] = f"within the 0.02 tie margin; simpler family ({best_name}) keeps the top rank"
    out["ranks_best"] = out["scientific_result"] == f"RRF ranks above {best_name}"
    out["deployment"] = ("not deployment-eligible: no validated explanation method for RRF (section 8); "
                         "the deployed model is chosen by 6.3 among eligible families")
    return out


# ---------- main ----------

def main():
    try:
        for p in FILES.values():
            require(not p.exists(), f"output exists: {p.relative_to(ROOT)}")
        for p in (CP1, EIF_GRID, EIF_SEL, EXPLAIN):
            require(p.exists(), f"required earlier result missing: {p.relative_to(ROOT)}")

        ok, elig = check_eligibility(json.loads(EIF_SEL.read_text()), json.loads(EXPLAIN.read_text()))
        print("EIF eligibility as third component:", elig)
        require(ok, "EIF is not confirmed as the eligible third component")

        from model_evaluation.harness import build_folds, evaluate, load_training_period
        from model_evaluation.indicator_scores import injected_frames
        X = load_training_period(ROOT)
        print(f"training period: {len(X):,} contracts (test period not loaded)")
        folds = build_folds(X)

        injected = {}
        for f in folds:
            inj = injected_frames(f)
            require(check_alignment(f, inj), f"fold {f['fold']}: injected inputs do not match indicator frames")
            for name, F in [("train", f["Ftr"]), ("validation", f["Fva"])] + list(inj.items()):
                require(F["R"].notna().all(), f"fold {f['fold']}: R(c) missing in {name}")
            injected[f["fold"]] = inj
        print("injected inputs aligned; R(c) complete in every fold")

        models = default_models()
        base_by_fold = evaluate(models["IF"], folds, seeds=SEEDS)
        cp1 = pd.read_csv(CP1)
        d_if = max_diff(base_by_fold, cp1, ["stability", "recall_value", "recall_date"])
        require(d_if <= REPRO_TOL, f"IF component differs from CP1 ({d_if:.1e})")
        eif_by_fold = evaluate(models["EIF"], folds, seeds=SEEDS)
        g = pd.read_csv(EIF_GRID)
        g = g[(g["role"] == "grid") & (g["ntrees"] == 400) & (g["sample_size"] == 256)
              & (g["ndim"] == 2)].sort_values("fold").reset_index(drop=True)
        require(len(g) == len(folds), "stored EIF grid row not found")
        d_eif = max_diff(eif_by_fold, g, CHECK_COLS)
        require(d_eif <= REPRO_TOL, f"EIF component differs from the stored grid ({d_eif:.1e})")
        print(f"components reproduce stored results (IF {d_if:.1e}, EIF {d_eif:.1e})")

        base_scores = {f["fold"]: models["IF"](0).fit(f["Ztr"]).score(f["Zva"]) for f in folds}

        print("\nRRF run 1 of 2")
        run1 = evaluate_rrf(folds, injected, models, base_scores)
        print("RRF run 2 of 2 (repeat)")
        run2 = evaluate_rrf(folds, injected, models, base_scores)
        cols = [c for c in run1.columns if c != "fit_seconds"]
        require(run1[cols].equals(run2[cols]), "RRF results differ between two runs")
        print("repeat identical: True")
    except Stop as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1

    means = lambda df: df.drop(columns="fold").mean()
    rrf_m, base_m, eif_m = means(run1), means(base_by_fold), means(eif_by_fold)
    decision = rank_rrf(rrf_m, base_m, eif_m, "EIF")
    record = {
        "protocol_sections": ["7.5", "7.5 clarification of 13 Nov 2026", "6.2", "6.3"],
        "components": ["R(c)", "IF 400 / 256 (frozen)", "EIF 400 / 256 / ndim 2"],
        "k": K, "weights": "equal", "seeds": list(SEEDS),
        "rank": "1 + #{fold training-part scores > s}",
        "eif_eligibility_checks": elig,
        "component_reproduction_max_diff": {"IF_vs_CP1": d_if, "EIF_vs_grid": d_eif},
        "repeat_identical": True,
        "rrf_means": {c: float(v) for c, v in rrf_m.items()},
        "baseline_means": {c: float(base_m[c]) for c in CHECK_COLS},
        "eif_means": {c: float(eif_m[c]) for c in CHECK_COLS},
        "ranking": decision,
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "isotree")},
        "python": platform.python_version(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    run1.to_csv(FILES["folds"], index=False)
    FILES["result"].write_text(json.dumps(record, indent=2))

    show = ["fold"] + CHECK_COLS + ["share_above_cutoff", "jaccard_top5_baseline", "jaccard_top5_EIF"]
    print("\nRRF per fold:")
    print(run1[show].round(4).to_string(index=False))
    print("\nfold means:")
    tab = pd.DataFrame({"IF": base_m[CHECK_COLS], "EIF": eif_m[CHECK_COLS], "RRF": rrf_m[CHECK_COLS]})
    print(tab.round(4).to_string())
    print("\nranking:", json.dumps(decision, indent=2))
    print("\nsaved", *(p.relative_to(ROOT) for p in FILES.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
