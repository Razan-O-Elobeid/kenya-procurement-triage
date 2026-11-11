"""Helpers for explanation validation (protocol sections 8 and 8.1).

Pure functions only. The run script is model_evaluation/explain_eval.py.
Explained outputs (8.1): for the scikit-learn Isolation Forest the anomaly depth
a(x) = -E[h(x)]; for EIF the isotree standardized score. Higher always means more anomalous.
"""
import time
import warnings

import numpy as np
import shap
from scipy.stats import rankdata, spearmanr

N_BACKGROUND = 100
REL_TOL = 1e-6          # check 1
STAB_MIN = 0.9          # check 4
OVERLAP_MIN = 2.0       # check 3, out of 3
LIMIT_SECONDS = 1800    # check 5, 30 minutes
N_TIMING = 20


def max_evals_for(d):
    """8.1: at least 10 forward-and-backward permutations per contract."""
    return 10 * (2 * d + 1)


def avg_path_length(n):
    """c(n), the average path length of an unsuccessful search (same formula as scikit-learn)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n


def anomaly_depth(model, Z):
    """a(x) = -E[h(x)] for a fitted scikit-learn IsolationForest.

    score_samples = -2 ** (-E[h] / c), so -E[h] = c * log2(-score_samples).
    """
    c = avg_path_length(model.max_samples_)
    return c * np.log2(-model.score_samples(np.asarray(Z)))


def explained_set(scores, top=0.95):
    """Top 5% by score, same cut as harness.top_set, ordered from the highest score down."""
    scores = np.asarray(scores)
    idx = np.flatnonzero(scores > np.quantile(scores, top))
    return idx[np.argsort(-scores[idx], kind="stable")]


def background_index(n_rows, s, n=N_BACKGROUND):
    return np.random.default_rng(s).choice(n_rows, size=n, replace=False)


def background(Ztr, s, n=N_BACKGROUND):
    return np.asarray(Ztr)[background_index(len(Ztr), s, n)]


def varying_counts(X, bg):
    """m per contract: inputs that differ from at least one background row (shap uses np.isclose)."""
    X, bg = np.asarray(X), np.asarray(bg)
    return np.array([int(np.any(~np.isclose(x, bg), axis=0).sum()) for x in X])


def tree_explain(model, bg, X):
    """Interventional TreeSHAP, negated so that it explains a(x) = -E[h(x)]."""
    te = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    vals = te.shap_values(np.asarray(X), check_additivity=False)
    base = float(np.ravel(te.expected_value)[0])
    return -np.asarray(vals, dtype=float), np.full(len(X), -base)


def permutation_explain(f, bg, X, s, max_evals):
    """Explainer built right before its single call, so the seeded global RNG is not disturbed."""
    masker = shap.maskers.Independent(bg, max_samples=N_BACKGROUND)
    ex = shap.explainers.Permutation(f, masker, seed=s)
    r = ex(np.asarray(X), max_evals=max_evals, main_effects=False, error_bounds=False,
           batch_size="auto", silent=True)
    return np.asarray(r.values, dtype=float), np.asarray(r.base_values, dtype=float).ravel()


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


# checks

def check_additivity(values, base, output):
    """Check 1: |base + sum - output| / |output| <= 1e-6 for every contract."""
    diff = np.abs(base + values.sum(axis=1) - output)
    rel = diff / np.maximum(np.abs(output), 1e-12)
    return {"max_relative_error": float(rel.max()), "passed": bool(rel.max() <= REL_TOL)}


def check_target(output, ranking_score):
    """Check 2: rank correlation 1.0, tested as identical ranks (ties included).

    The computed Spearman can print as 0.9999999999999999 from float rounding even when
    the ranks match exactly, so the ranks are compared directly. Spearman is reported too.
    """
    same = np.array_equal(rankdata(output), rankdata(ranking_score))
    rho = float(spearmanr(output, ranking_score)[0])
    return {"identical_ranks": bool(same), "spearman": rho, "passed": bool(same)}


def top3(values):
    """Indices of the 3 largest |attributions|; ties go to the lower column index."""
    return np.argsort(-np.abs(values), axis=1, kind="stable")[:, :3]


def check_agreement(perm_values, tree_values):
    """Check 3: mean top-3 overlap (out of 3) between the two methods, same contracts and output."""
    a, b = top3(perm_values), top3(tree_values)
    overlap = np.array([len(set(x) & set(y)) for x, y in zip(a, b)])
    m = float(overlap.mean())
    return {"mean_top3_overlap": m, "passed": bool(m >= OVERLAP_MIN)}


def check_stability(v0, v1):
    """Check 4: mean per-contract Spearman between runs s=0 and s=1.

    An undefined correlation (a constant vector) counts as 0, the conservative choice.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant vectors are handled below
        rho = np.array([spearmanr(x, y)[0] for x, y in zip(v0, v1)], dtype=float)
    n_undefined = int(np.isnan(rho).sum())
    m = float(np.nan_to_num(rho, nan=0.0).mean())
    return {"mean_spearman": m, "n_undefined": n_undefined, "passed": bool(m >= STAB_MIN)}


def check_feasibility(seconds_20, n_explained, n_timed=N_TIMING):
    """Check 5: linear extrapolation from the 20-contract timing. Must be under 30 minutes."""
    est = seconds_20 * n_explained / n_timed
    return {"seconds_timed": float(seconds_20), "n_timed": n_timed, "n_explained": int(n_explained),
            "extrapolated_seconds": float(est), "passed": bool(est < LIMIT_SECONDS)}


def exact_repeat(v_a, b_a, v_b, b_b):
    return bool(np.array_equal(v_a, v_b) and np.array_equal(b_a, b_b))


def verdict(checks, need):
    """Validated only if every needed check was run and passed. Anything else says why."""
    missing = [k for k in need if k not in checks]
    failed = [k for k in need if k in checks and not checks[k].get("passed", False)]
    if missing:
        return {"validated": False, "reason": "not run: " + ", ".join(missing)}
    if failed:
        return {"validated": False, "reason": "failed: " + ", ".join(failed)}
    return {"validated": True, "reason": "all required checks passed"}
