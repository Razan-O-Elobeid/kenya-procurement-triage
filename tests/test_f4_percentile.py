"""Checks for the f4 percentile in R(c) (protocol 7.4). Run: python tests/test_f4_percentile.py"""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore", UserWarning)
from src.features import INDICATORS
from model_evaluation.f4_percentile import (F4, adopt, fit_reference, group_means, midrank_percentile,
                                            r_with_f4_percentile, require_finite, require_groups,
                                            require_reference)

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []

# 1 mid-rank formula, tied zeros, values outside the reference
train = pd.DataFrame({F4: [0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, np.nan, np.nan]})
ref = fit_reference(train)
results.append(ok(len(ref) == 10, "reference keeps only non-missing training values"))
p = midrank_percentile([0.0, 0.1, 0.25, 0.4, 1.0, -1.0], ref)
results.append(ok(np.isclose(p[0], 0.30), "tied zeros get half the zero share (6 of 10 zeros: 0.30)"))
results.append(ok(np.isclose(p[1], 0.65), "a single tied value: (6 below + 0.5) / 10 = 0.65"))
results.append(ok(np.isclose(p[2], 0.80), "value between references: 8 below / 10 = 0.80"))
results.append(ok(np.isclose(p[3], 0.95) and np.isclose(p[4], 1.0) and np.isclose(p[5], 0.0),
                  "top value 0.95, above all 1.0, below all 0.0"))
results.append(ok(np.isclose(midrank_percentile(ref, ref).mean(), 0.5), "training mean percentile is 0.5"))

# 2 missing stays missing; zero is not missing
q = midrank_percentile([np.nan, 0.0], ref)
results.append(ok(np.isnan(q[0]) and q[1] > 0, "missing stays missing, zero gets a percentile"))

# 3 R(c) with the percentile: only f4 changes, missing f4 is left out of the average
F = pd.DataFrame({"f1_direct_award": [1, 0, 0], "f3_decision_period": [0.5, 0.2, 0.9],
                  F4: [0.0, np.nan, 0.4], "f5_benford_deviation": [0.7, 0.6, np.nan]})
r = r_with_f4_percentile(F, ref)
results.append(ok(np.isclose(r[0], np.mean([1, 0.5, 0.30, 0.7])), "four indicators: f4 replaced by its percentile"))
results.append(ok(np.isclose(r[1], np.mean([0, 0.2, 0.6])), "missing f4 excluded from the average"))
results.append(ok(np.isclose(r[2], np.mean([0, 0.9, 0.95])), "missing f5 still excluded"))
results.append(ok(F[F4].isna().sum() == 1 and F.loc[0, F4] == 0, "input table not changed"))

# 4 training reference only: validation values never enter the reference
val = pd.DataFrame({F4: [5.0, 6.0]})
before = ref.copy()
midrank_percentile(val[F4], ref)
results.append(ok(np.array_equal(before, ref), "scoring other data leaves the reference unchanged"))

# 5 group means and range over the 2-, 3- and 4-indicator groups
rr = np.array([0.1, 0.3, 0.2, 0.4, 0.9])
n = np.array([2, 2, 3, 4, 1])
means, rng_ = group_means(rr, n)
results.append(ok(np.isclose(means[2], 0.2) and np.isclose(rng_, 0.4 - 0.2), "range uses groups 2, 3, 4 only"))

# 6 committed decision rule
results.append(ok(adopt(0.056, 0.028), "rule: range halved exactly, adopted"))
results.append(ok(adopt(0.056, 0.010), "rule: range falls by more than half, adopted"))
results.append(ok(not adopt(0.056, 0.030), "rule: range falls by less than half, not adopted"))
results.append(ok(not adopt(0.056, 0.080), "rule: range grows, not adopted"))

# 7 prerequisites: stop instead of deciding from incomplete data


def raises(fn, *args):
    try:
        fn(*args)
        return False
    except ValueError:
        return True


results.append(ok(not raises(require_reference, ref), "prerequisite: normal reference accepted"))
results.append(ok(raises(require_reference, fit_reference(pd.DataFrame({F4: [np.nan, np.nan]}))),
                  "prerequisite: all-missing reference rejected"))
results.append(ok(raises(require_reference, np.array([])), "prerequisite: empty reference rejected"))
results.append(ok(raises(require_reference, np.array([0.0, np.inf])), "prerequisite: non-finite reference rejected"))
results.append(ok(not raises(require_groups, np.array([2, 3, 4, 1])), "prerequisite: groups 2, 3, 4 present"))
results.append(ok(raises(require_groups, np.array([3, 4, 4, 1])), "prerequisite: missing 2-indicator group rejected"))
results.append(ok(raises(require_groups, np.array([2, 2, 4])), "prerequisite: missing 3-indicator group rejected"))
results.append(ok(raises(require_groups, np.array([2, 3, 3])), "prerequisite: missing 4-indicator group rejected"))
_, partial = group_means(np.array([0.1, 0.2]), np.array([3, 4]))
results.append(ok(raises(require_finite, np.nan, "range") and not raises(require_finite, 0.05, "range"),
                  "prerequisite: non-finite range rejected, finite accepted"))
results.append(ok(np.isfinite(partial) and raises(require_groups, np.array([3, 4])),
                  "a range from incomplete groups is still caught by the group check"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
