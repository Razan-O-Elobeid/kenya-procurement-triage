"""f4 as a training-referenced mid-rank percentile inside R(c) (protocol 7.4, amendment 6c0cd50).

pct(x) = (#{ref < x} + 0.5 * #{ref == x}) / #{ref}, ref = non-missing training-part f4.
Every f4 = 0 gets half the training share of zeros. Missing f4 stays missing and is still
left out of the R(c) average. Only R(c) changes; the model input f4 is untouched.
"""
import numpy as np
import pandas as pd

from src.features import INDICATORS

F4 = "f4_supplier_concentration"


def fit_reference(Ftr):
    """Sorted non-missing f4 values of the training part."""
    ref = Ftr[F4].to_numpy(dtype=float)
    return np.sort(ref[~np.isnan(ref)])


def midrank_percentile(x, ref):
    x = np.asarray(x, dtype=float)
    below = np.searchsorted(ref, x, side="left")
    upto = np.searchsorted(ref, x, side="right")
    pct = (below + 0.5 * (upto - below)) / len(ref)
    return np.where(np.isnan(x), np.nan, pct)


def r_with_f4_percentile(F, ref):
    """R(c) with f4 replaced by its percentile: mean of the available indicators."""
    ind = F[INDICATORS].astype(float).copy()
    ind[F4] = midrank_percentile(ind[F4], ref)
    return ind.mean(axis=1)


def n_available(F):
    return F[INDICATORS].notna().sum(axis=1)


def group_means(r, n, groups=(2, 3, 4)):
    """Mean R(c) per number of available indicators, and the range across the given groups."""
    means = pd.Series(r).groupby(pd.Series(n).to_numpy()).mean()
    present = [g for g in groups if g in means.index]
    rng = float(means[present].max() - means[present].min()) if len(present) > 1 else np.nan
    return means, rng


def require_reference(ref):
    """The training reference must hold at least one value, all finite."""
    if len(ref) == 0 or not np.all(np.isfinite(ref)):
        raise ValueError("training f4 reference is empty or has non-finite values")


def require_groups(n, groups=(2, 3, 4)):
    """Every required coverage group must hold at least one contract."""
    counts = pd.Series(np.asarray(n)).value_counts()
    missing = [g for g in groups if counts.get(g, 0) < 1]
    if missing:
        raise ValueError(f"coverage groups without contracts: {missing}")


def require_finite(value, name):
    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite: {value}")


def adopt(range_original, range_percentile):
    """Committed rule: adopt only if the mean range falls by at least half."""
    return bool(range_percentile <= 0.5 * range_original)
