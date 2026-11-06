"""Rule-based scores for model evaluation: R(c) and the naive red-flag count (protocol 7.1).

Training period only. Thresholds come from each fold's training part.
"""
import numpy as np
import pandas as pd

from src.features import INDICATORS, ecdf_score
from src.tuning import inject
from model_evaluation.harness import inject_value

# binary flags counted as they are
COUNT_FLAGS = ["f1_direct_award", "is_signed_pre_closure", "above_p99", "below_p1",
               "zero_value_award", "is_terminated"]
# continuous indicators counted when at or above their training 95th percentile
COUNT_CONTINUOUS = ["f3_decision_period", "f4_supplier_concentration", "f5_benford_deviation"]


def recompute_date_indicators(G, ib):
    """f3 and R(c) for injected copies, using the fold's fitted IndicatorBuilder.

    inject() changes days_to_signing but not f3 or R. f1, f4 and f5 do not depend on a
    contract's own date or value, so only f3 and the R columns change.
    """
    G = G.copy()
    G["f3_decision_period"] = 1 - ecdf_score(ib.d_ref_, G["days_to_signing"]).to_numpy()
    ind = G[INDICATORS]
    G["R"] = ind.mean(axis=1)
    for k in INDICATORS:
        G[f"R_minus_{k}"] = ind.drop(columns=k).mean(axis=1)
    return G


def injected_frames(f, frac=0.02):
    """The same injected copies as harness.build_folds (same seeds), as indicator frames."""
    seed = f["fold"] - 1
    return {"value": inject(f["Fva"], f["ib"], "value", frac, seed=seed),
            "date": recompute_date_indicators(inject(f["Fva"], f["ib"], "date", frac, seed=seed), f["ib"]),
            "value_x5": inject_value(f["Fva"], f["ib"], 5, frac, seed=seed)}


def fit_count_thresholds(Ftr, q=0.95):
    """95th percentile of each continuous indicator on the training part (missing ignored)."""
    return {c: float(Ftr[c].quantile(q)) for c in COUNT_CONTINUOUS}
