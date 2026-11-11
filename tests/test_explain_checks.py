"""Checks for explanation validation (protocol 8 and 8.1). Run: python tests/test_explain_checks.py

Synthetic data only. No project data is loaded.
"""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from shap.utils._masked_model import MaskedModel
warnings.simplefilter("ignore", UserWarning)
from model_evaluation.eif_eval import IsoTreeEIF
from model_evaluation.explain_checks import (
    anomaly_depth, avg_path_length, background, background_index, check_additivity, check_agreement,
    check_feasibility, check_stability, check_target, exact_repeat, explained_set, max_evals_for,
    permutation_explain, top3, tree_explain, varying_counts, verdict)
from model_evaluation.harness import top_set

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []

# synthetic data: 6 inputs, one rare binary flag, a few planted outliers
rng = np.random.default_rng(0)
Ztr = rng.normal(size=(3000, 6))
Ztr[:, 5] = (rng.random(3000) < 0.05) * 4.0
Zva = rng.normal(size=(600, 6))
Zva[:, 5] = (rng.random(600) < 0.05) * 4.0
Zva[:10, 0] += 6.0
d = Ztr.shape[1]
m_if = IsolationForest(n_estimators=100, max_samples=256, random_state=0, n_jobs=-1).fit(Ztr)

# 1 settings
results.append(ok(max_evals_for(14) == 290 and max_evals_for(d) == 130, "max_evals = 10 x (2d + 1)"))
results.append(ok(all(np.isclose(avg_path_length(n), _average_path_length([n])[0]) for n in (2, 3, 10, 256, 4096)),
                  "c(n) matches scikit-learn"))

# 2 explained output for the IF: a(x) = -E[h(x)], same direction as the ranking score
a = anomaly_depth(m_if, Zva)
rank = -m_if.score_samples(Zva)
pd_te = shap.TreeExplainer(m_if)  # path-dependent, only to read E[h] for this test
Eh = pd_te.shap_values(Zva, check_additivity=False).sum(1) + np.ravel(pd_te.expected_value)[0]
results.append(ok(np.allclose(a, -Eh, rtol=1e-10, atol=1e-10), "TreeExplainer explains E[h]; a(x) = -E[h(x)]"))
results.append(ok(check_target(a, rank)["passed"], "a(x) has identical ranks to the ranking score (rank correlation 1)"))
results.append(ok(np.corrcoef(Eh, rank)[0, 1] < 0, "raw E[h] runs against the ranking score (so it is negated)"))

# 3 explained set: same cut as the harness, highest first
e = explained_set(rank)
results.append(ok(set(e) == top_set(rank) and np.all(np.diff(rank[e]) <= 0), "explained set = harness top 5%, sorted"))
X = Zva[e]

# 4 background: 100 distinct rows, fixed by seed, different seeds differ
i0, i0b, i1 = background_index(3000, 0), background_index(3000, 0), background_index(3000, 1)
results.append(ok(len(set(i0)) == 100 and np.array_equal(i0, i0b) and not np.array_equal(i0, i1),
                  "background: 100 distinct rows, reproducible, seed-dependent"))
bg0, bg1 = background(Ztr, 0), background(Ztr, 1)

# 5 interventional TreeExplainer on a(x): additive, positive = towards anomalous
tv, tb = tree_explain(m_if, bg0, X)
c1 = check_additivity(tv, tb, a[e])
results.append(ok(c1["passed"], f"TreeExplainer additive to a(x) (max rel error {c1['max_relative_error']:.1e})"))
planted = np.flatnonzero(e < 10)
results.append(ok(len(planted) > 0 and tv[planted, 0].mean() > 0,
                  "planted outliers: input 0 pushes towards anomalous (positive sign)"))
tv2, tb2 = tree_explain(m_if, bg0, X)
results.append(ok(exact_repeat(tv, tb, tv2, tb2), "TreeExplainer exact repeat is bit-identical"))
tv1, _ = tree_explain(m_if, bg1, X)
results.append(ok(not np.array_equal(tv, tv1), "TreeExplainer depends on the background (check 4 is not trivial)"))

# 6 permutation explainer on the same a(x)
f_if = lambda Z: anomaly_depth(m_if, Z)
me = max_evals_for(d)
pv, pb = permutation_explain(f_if, bg0, X, 0, me)
results.append(ok(check_additivity(pv, pb, a[e])["passed"], "permutation additive to a(x)"))
np.random.seed(123); _ = np.random.random(50)  # disturb the global RNG before construction
pv2, pb2 = permutation_explain(f_if, bg0, X, 0, me)
results.append(ok(exact_repeat(pv, pb, pv2, pb2), "permutation exact repeat, even after other global RNG use"))
pv1, _ = permutation_explain(f_if, bg0, X, 1, me)
results.append(ok(not np.array_equal(pv, pv1), "permutation seed changes the permutations"))
agree = check_agreement(pv, tv)
results.append(ok(agree["mean_top3_overlap"] >= 2, f"permutation agrees with TreeExplainer (top-3 {agree['mean_top3_overlap']:.2f})"))

# 7 m and the number of permutations follow shap's own count
fm_m = []
for x in X[:15]:
    fm = MaskedModel(f_if, shap.maskers.Independent(bg0, max_samples=100), shap.links.identity, True, x)
    fm_m.append(len(fm.varying_inputs()))
results.append(ok(np.array_equal(varying_counts(X[:15], bg0), fm_m), "m (varying inputs) matches shap"))
results.append(ok(np.all(me // (2 * np.array(fm_m) + 1) >= 10), "at least 10 permutations per contract"))

# 8 EIF: output is the ranking score, higher for outliers, explanation additive and repeatable
m_eif = IsoTreeEIF(100, 256, 2, 0).fit(Ztr)
s_eif = m_eif.score(Zva)
results.append(ok(s_eif[:10].mean() > np.median(s_eif), "EIF score is higher for planted outliers"))
e2 = explained_set(s_eif)
ev, eb = permutation_explain(m_eif.score, bg0, Zva[e2], 0, me)
ev2, eb2 = permutation_explain(m_eif.score, bg0, Zva[e2], 0, me)
results.append(ok(check_additivity(ev, eb, s_eif[e2])["passed"], "EIF permutation additive to its score"))
results.append(ok(exact_repeat(ev, eb, ev2, eb2), "EIF permutation exact repeat"))
results.append(ok(check_target(s_eif[e2], s_eif[e2])["passed"], "EIF check 2: output is the ranking score"))

# 9 the check functions themselves
bad = tv.copy(); bad[0, 0] += 1.0
results.append(ok(not check_additivity(bad, tb, a[e])["passed"], "check 1 catches a broken sum"))
results.append(ok(not check_target(a[e], -rank[e])["passed"], "check 2 catches a reversed direction"))
t3 = top3(np.array([[1.0, -3.0, 3.0, 0.5], [0.0, 0.0, 0.0, 0.0]]))
results.append(ok(list(t3[0]) == [1, 2, 0] and list(t3[1]) == [0, 1, 2], "top 3 by |value|, ties to lower index"))
ag = check_agreement(np.array([[3.0, 2, 1, 0]]), np.array([[0.0, 1, 2, 3]]))
results.append(ok(ag["mean_top3_overlap"] == 2 and ag["passed"], "check 3 counts shared inputs, not order"))
st = check_stability(np.array([[1, 2, 3.0], [1, 1, 1.0]]), np.array([[1, 2, 3.0], [3, 2, 1.0]]))
results.append(ok(st["n_undefined"] == 1 and np.isclose(st["mean_spearman"], 0.5) and not st["passed"],
                  "check 4: an undefined correlation counts as 0"))
f5a, f5b = check_feasibility(59.0, 605), check_feasibility(60.0, 605)
results.append(ok(f5a["passed"] and not f5b["passed"] and np.isclose(f5b["extrapolated_seconds"], 1815),
                  "check 5: 59 s for 20 passes (1784.75 s), 60 s fails (1815 s)"))
f5c = check_feasibility(60.0, 600)
results.append(ok(f5c["extrapolated_seconds"] == 1800.0 and not f5c["passed"],
                  "check 5: exactly 1800 s (30 minutes) fails; it must be under 30 minutes"))
v = verdict({"feasibility": {"passed": True}, "additivity": {"passed": True}}, ["feasibility", "additivity", "stability"])
results.append(ok(not v["validated"] and "not run" in v["reason"], "a check not run is never counted as passed"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
