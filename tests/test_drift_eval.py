"""Checks for the training-period drift check (protocol 7.9(a)). Run: python tests/test_drift_eval.py

Synthetic data only. No project data is loaded.
"""
import sys, pathlib, tempfile, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
import model_evaluation.drift_eval as de

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(0)

# 1 SMD: known shift, scale invariance, zero variance
a = rng.normal(0, 1, (20000, 2)); b = rng.normal(0, 1, (20000, 2)); b[:, 1] += 0.5
d = de.smd(a, b)
results.append(ok(abs(d[0]) < 0.03 and abs(d[1] - 0.5) < 0.03, "SMD near 0 without shift and near 0.5 for a 0.5 SD shift"))
results.append(ok(np.allclose(de.smd(a * 3 + 7, b * 3 + 7), d), "SMD unchanged by rescaling an input"))
results.append(ok(np.isnan(de.smd(np.ones((5, 1)), np.ones((5, 1))))[0], "zero variance in both parts: no SMD"))

# 2 flag share: threshold from training only; about 5% on training by construction
tr = rng.normal(size=10000)
thr, s_tr, s_va = de.flag_share(tr, tr - 0.5)
results.append(ok(abs(s_tr - 0.05) < 0.002 and s_va < 0.03 and thr == np.quantile(tr, 0.95),
                  "share: 5% on training, lower when validation scores fall"))

# 3 feature summary
F = pd.DataFrame({"f4_supplier_concentration": [0.1, np.nan, 0.3, np.nan], "f5_benford_deviation": [np.nan] * 3 + [0.2],
                  "is_signed_pre_closure": [1, 0, 0, 0], "f3_decision_period": [0.1, 0.2, 0.3, np.nan]})
s = de.feature_summary(F)
results.append(ok(s["f4_coverage"] == 0.5 and s["f5_coverage"] == 0.25 and s["pre_closure_share"] == 0.25
                  and s["f3_coverage"] == 0.75 and np.isclose(s["f3_q50"], 0.2), "coverage, pre-closure share and f3 quantiles"))

# 4 run on synthetic folds with the real model classes; shift in fold 2 lowers nothing by force but SMD shows it
def fold(k, shift):
    Ztr = rng.normal(size=(2000, 3)); Zva = rng.normal(size=(500, 3)); Zva[:, 0] += shift
    Fr = lambda n: pd.DataFrame({"f4_supplier_concentration": rng.random(n), "f5_benford_deviation": rng.random(n),
                                 "is_signed_pre_closure": rng.integers(0, 2, n), "f3_decision_period": rng.random(n)})
    return dict(fold=k, Ztr=Ztr, Zva=Zva, Ftr=Fr(2000), Fva=Fr(500))
folds = [fold(1, 0.0), fold(2, 1.0)]
shares, smds, feats = de.run(folds, ["x0", "x1", "x2"], de.frozen_models)
results.append(ok(len(shares) == 4 and set(shares["model"]) == {"if", "eif"} and shares["share_train"].between(0.04, 0.06).all(),
                  "two models per fold; training share about 5%"))
g = smds.set_index(["fold", "input"])["smd"]
results.append(ok(abs(g[(1, "x0")]) < 0.15 and g[(2, "x0")] > 0.7, "SMD picks up the planted shift in fold 2 only"))
results.append(ok(len(feats) == 4 and set(feats["part"]) == {"train", "validation"}, "feature summary per fold and part"))
bad = [dict(folds[0], Zva=np.full((5, 3), np.nan))]
try:
    de.run(bad, ["x0", "x1", "x2"], de.frozen_models); stopped = False
except ValueError:
    stopped = True
results.append(ok(stopped, "non-finite inputs stop the run"))

# 5 refuses to overwrite
with tempfile.TemporaryDirectory() as tmp:
    de.OUT = pathlib.Path(tmp); de.ROOT = pathlib.Path(tmp).parent
    results.append(ok(de.main() == 1, "refuses to overwrite existing output"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
