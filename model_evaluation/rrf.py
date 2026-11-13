"""Reciprocal rank fusion with training-referenced ranks (protocol 7.5).

rank_m(c) = 1 + #{reference scores of component m > s_m(c)}, higher score = more anomalous.
RRF(c) = sum over components of 1 / (K + rank_m(c)), equal weights, K = 60.
A missing score gets the worst rank, reference size + 1.
Each contract is ranked against the fixed reference only, so no scored contract
affects another's RRF score. Used in validation and, later, at the final test.
"""
import numpy as np

K = 60


def _finite_reference(ref):
    ref = np.asarray(ref, dtype=float)
    if ref.ndim != 1 or len(ref) == 0 or not np.isfinite(ref).all():
        raise ValueError("reference scores must be a non-empty finite 1-d array")
    return np.sort(ref)


def reference_ranks(ref, s):
    """1 + number of reference scores strictly greater than s. NaN gets len(ref) + 1."""
    r = _finite_reference(ref)
    s = np.asarray(s, dtype=float)
    if np.isinf(s).any():
        raise ValueError("scores must not be infinite")
    greater = len(r) - np.searchsorted(r, s, side="right")
    ranks = 1 + greater
    return np.where(np.isnan(s), len(r) + 1, ranks).astype(np.int64)


def fuse(refs, scores, k=K):
    """refs and scores: dicts with the same component names. Returns RRF scores."""
    if set(refs) != set(scores):
        raise ValueError("components differ between references and scores")
    n = {len(np.asarray(v)) for v in scores.values()}
    if len(n) != 1:
        raise ValueError("components scored different numbers of contracts")
    total = np.zeros(n.pop())
    for name in sorted(refs):  # fixed summation order
        total += 1.0 / (k + reference_ranks(refs[name], scores[name]))
    return total


def reference_rrf(refs, k=K):
    """RRF of the reference contracts themselves (used for the final-test threshold)."""
    return fuse(refs, refs, k)
