"""Explanation validation for the frozen Isolation Forest and the selected EIF (protocol 8 and 8.1).

Run from the repo root:  python -m model_evaluation.explain_eval
Training period only, fold 5. Models: IF 400 / 256 seed 0 and EIF 400 / 256 / ndim 2 seed 0,
both fitted on fold 5's training part.

Per explanation, in this order:
  1. check 5: time 20 contracts (s = 0) and extrapolate; unless under 30 minutes, record it and stop
  2. full run s = 0, a second full run s = 0 (exact repeat), full run s = 1 (check 4)
  3. checks 1 and 2 on the first s = 0 run
IF: interventional TreeExplainer on a(x) = -E[h(x)]; check 3 is not applicable to it.
EIF: permutation explainer; check 3 uses a permutation run on the IF, same a(x), same s = 0 background.
That run also gets the 20-contract timing check first; if it fails, check 3 is not completed.

Failed integrity checks exit with status 1 and save nothing. Failed validation checks are results:
they are saved and the exit status is 0. Writes only results/model_evaluation/explanations/,
and refuses to overwrite it.
"""
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble._iforest import _average_path_length

from model_evaluation.eif_eval import IsoTreeEIF
from model_evaluation.explain_checks import (
    N_TIMING, anomaly_depth, avg_path_length, background, background_index, check_additivity, check_agreement,
    check_feasibility, check_stability, check_target, exact_repeat, explained_set, max_evals_for,
    permutation_explain, timed, tree_explain, varying_counts, verdict)
from model_evaluation.harness import SklearnIF, build_folds, load_training_period
from src.pipeline import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation" / "explanations"
FOLD5 = dict(n_train=60_488, n_val=12_097)
SEEDS = (0, 1)
NEED_TREE = ["feasibility", "additivity", "target", "stability", "exact_repeat"]
NEED_PERM = ["feasibility", "additivity", "target", "method_agreement", "stability", "exact_repeat"]


class Stop(Exception):
    pass


def require(cond, msg):
    if not cond:
        raise Stop(msg)


def finite(*arrays):
    return all(np.all(np.isfinite(a)) for a in arrays)


def per_contract(names, ids, pos, score, output, values, base, extra=None):
    df = pd.DataFrame(values, columns=[f"attr__{n}" for n in names])
    df.insert(0, "base", base)
    df.insert(0, "explained_output", output)
    df.insert(0, "ranking_score", score)
    df.insert(0, "position_in_fold5_validation", pos)
    df.insert(0, "contract_id", ids)
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def validate(label, run, X, output, rank_score):
    """Timing first, then the full runs. Returns (checks, runs or None, seconds)."""
    checks, secs = {}, {}
    _, secs["timing_20"] = timed(run, 0, X[:N_TIMING])
    checks["feasibility"] = check_feasibility(secs["timing_20"], len(X))
    print(f"  {label}: 20 contracts in {secs['timing_20']:.1f} s, "
          f"extrapolated {checks['feasibility']['extrapolated_seconds'] / 60:.1f} min")
    if not checks["feasibility"]["passed"]:
        print(f"  {label}: not under 30 minutes; stopped before the full run (protocol 8.1)")
        return checks, None, secs
    runs = {}
    runs["s0"], secs["full_s0"] = timed(run, 0, X)
    runs["s0_repeat"], secs["full_s0_repeat"] = timed(run, 0, X)
    runs["s1"], secs["full_s1"] = timed(run, 1, X)
    for k, (v, b) in runs.items():
        require(v.shape == X.shape and finite(v, b), f"{label}: non-finite or misshaped attributions ({k})")
    v0, b0 = runs["s0"]
    checks["additivity"] = check_additivity(v0, b0, output)
    checks["target"] = check_target(output, rank_score)
    checks["stability"] = check_stability(v0, runs["s1"][0])
    checks["exact_repeat"] = {"passed": exact_repeat(v0, b0, *runs["s0_repeat"])}
    print(f"  {label}: full s=0 run {secs['full_s0'] / 60:.1f} min | "
          + " | ".join(f"{k} {'pass' if c['passed'] else 'FAIL'}" for k, c in checks.items()))
    return checks, runs, secs


def main():
    if OUT.exists() and any(OUT.iterdir()):
        print("STOPPED: output folder exists and is not empty:", OUT.relative_to(ROOT), file=sys.stderr)
        return 1
    try:
        X = load_training_period(ROOT)
        print(f"training period: {len(X):,} contracts (test period not loaded)")
        f = build_folds(X)[-1]
        Ztr, Zva = np.asarray(f["Ztr"], dtype=float), np.asarray(f["Zva"], dtype=float)
        require(f["fold"] == 5 and (len(Ztr), len(Zva)) == (FOLD5["n_train"], FOLD5["n_val"]),
                f"unexpected fold 5 sizes {len(Ztr)} / {len(Zva)}")
        require(Ztr.ndim == 2 and Zva.shape[1] == Ztr.shape[1] and finite(Ztr, Zva), "inputs not finite or misshaped")
        prep = clone(build_preprocessor()).fit(f["Ftr"])
        require(np.array_equal(np.asarray(prep.transform(f["Fva"]), dtype=float), Zva),
                "refitted preprocessor does not reproduce the harness inputs")
        names = list(prep.get_feature_names_out())
        d = Ztr.shape[1]
        require(len(names) == d, "feature names do not match the input columns")
        max_evals = max_evals_for(d)
        ids = (f["Fva"]["ocid"] if "ocid" in f["Fva"] else pd.Series(f["Fva"].index)).to_numpy()
        bgs = {s: background(Ztr, s) for s in SEEDS}
        print(f"fold 5: train {len(Ztr):,} | validation {len(Zva):,} | d = {d} | max_evals = {max_evals}")

        # frozen IF configuration, explained output a(x) = -E[h(x)]
        m_if = SklearnIF(400, 256, seed=0).fit(Ztr).model
        require(np.isclose(avg_path_length(m_if.max_samples_), _average_path_length([m_if.max_samples_])[0]),
                "c(n) differs from scikit-learn")
        rank_if = -m_if.score_samples(Zva)
        a_if = anomaly_depth(m_if, Zva)
        e_if = explained_set(rank_if)
        X_if = Zva[e_if]

        # selected EIF configuration, explained output = ranking score
        m_eif = IsoTreeEIF(400, 256, 2, 0).fit(Ztr)
        rank_eif = m_eif.score(Zva)
        e_eif = explained_set(rank_eif)
        X_eif = Zva[e_eif]
        require(finite(rank_if, a_if, rank_eif) and len(e_if) > N_TIMING and len(e_eif) > N_TIMING,
                "scores not finite or explained set too small")
        print(f"explained sets: IF {len(e_if)} | EIF {len(e_eif)} contracts")

        result = {"protocol_sections": ["8", "8.1"], "fold": 5, "d": d, "feature_names": names,
                  "max_evals": max_evals, "background_rows": 100, "seeds": list(SEEDS),
                  "background_indices": {s: background_index(len(Ztr), s).tolist() for s in SEEDS}}
        tables = {}

        # 1 IF, TreeExplainer
        print("\nIF 400 / 256, interventional TreeExplainer")
        run_tree = lambda s, Z: tree_explain(m_if, bgs[s], Z)
        c_tree, r_tree, t_tree = validate("IF tree", run_tree, X_if, a_if[e_if], rank_if[e_if])
        c_tree["method_agreement"] = {"status": "not applicable by definition (section 8 check 3)"}
        result["if_tree"] = {"checks": c_tree, "seconds": t_tree, **verdict(c_tree, NEED_TREE)}
        if r_tree:
            for s in SEEDS:
                key = "s0" if s == 0 else "s1"
                tables[f"if_tree_s{s}"] = per_contract(names, ids[e_if], e_if, rank_if[e_if], a_if[e_if], *r_tree[key])

        # 2 IF, permutation explainer on the same a(x), for check 3 only
        print("\nIF 400 / 256, permutation explainer (check 3 only)")
        if r_tree is None:
            agree = {"passed": False, "status": "not run: TreeExplainer stopped at the feasibility check"}
        else:
            f_if = lambda Z: anomaly_depth(m_if, Z)
            _, t20_pi = timed(permutation_explain, f_if, bgs[0], X_if[:N_TIMING], 0, max_evals)
            feas_pi = check_feasibility(t20_pi, len(X_if))
            print(f"  IF perm: 20 contracts in {t20_pi:.1f} s, "
                  f"extrapolated {feas_pi['extrapolated_seconds'] / 60:.1f} min")
            result["if_perm_check3"] = {"feasibility": feas_pi}
        if r_tree is not None and not feas_pi["passed"]:
            print("  IF perm: not under 30 minutes; stopped before the full run (protocol 8.1)")
            agree = {"passed": False, "status": "not completed: IF permutation run failed the feasibility check"}
        elif r_tree is not None:
            (v_pi, b_pi), t_pi = timed(permutation_explain, f_if, bgs[0], X_if, 0, max_evals)
            require(finite(v_pi, b_pi), "IF permutation attributions not finite")
            add_pi = check_additivity(v_pi, b_pi, a_if[e_if])
            m_pi = varying_counts(X_if, bgs[0])
            tables["if_perm_s0"] = per_contract(names, ids[e_if], e_if, rank_if[e_if], a_if[e_if], v_pi, b_pi,
                                                {"m_varying": m_pi, "n_permutations": max_evals // (2 * m_pi + 1)})
            if not (add_pi["passed"] and c_tree["additivity"]["passed"]):
                agree = {"passed": False, "status": "not comparable: one method is not additive to a(x)"}
            else:
                agree = {**check_agreement(v_pi, r_tree["s0"][0]), "status": "compared on a(x), s = 0 background"}
            result["if_perm_check3"].update({"additivity": add_pi, "seconds": t_pi})
        print(f"  check 3: {agree}")

        # 3 EIF, permutation explainer
        print("\nEIF 400 / 256 / ndim 2, permutation explainer")
        run_perm = lambda s, Z: permutation_explain(m_eif.score, bgs[s], Z, s, max_evals)
        c_eif, r_eif, t_eif = validate("EIF perm", run_perm, X_eif, rank_eif[e_eif], rank_eif[e_eif])
        c_eif["method_agreement"] = agree
        result["eif_perm"] = {"checks": c_eif, "seconds": t_eif, **verdict(c_eif, NEED_PERM)}
        if r_eif:
            for s in SEEDS:
                m_e = varying_counts(X_eif, bgs[s])
                key = "s0" if s == 0 else "s1"
                tables[f"eif_perm_s{s}"] = per_contract(
                    names, ids[e_eif], e_eif, rank_eif[e_eif], rank_eif[e_eif], *r_eif[key],
                    {"m_varying": m_e, "n_permutations": max_evals // (2 * m_e + 1)})
    except Stop as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1

    result["environment"] = {
        "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "versions": {p: version(p) for p in ("shap", "scikit-learn", "numpy", "isotree")},
        "threads": "IF n_jobs=-1 as in the harness; isotree single-threaded (no OpenMP)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explain_validation.json").write_text(json.dumps(result, indent=2, default=str))
    for k, df in tables.items():
        df.to_csv(OUT / f"{k}.csv", index=False)
    print("\nIF TreeExplainer validated:", result["if_tree"]["validated"], "-", result["if_tree"]["reason"])
    print("EIF permutation validated:", result["eif_perm"]["validated"], "-", result["eif_perm"]["reason"])
    print("saved", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
