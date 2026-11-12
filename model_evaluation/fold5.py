"""Fold 5 of the harness (training period only), built as in harness.build_folds but without
the injected copies. Shared by the face-validity sheets and the development dashboard bundle."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from src.features import IndicatorBuilder
from src.pipeline import build_preprocessor
from src.tuning import DATE

SIZES = (60_488, 12_097)


def fold5(X, sizes=SIZES):
    X = X.sort_values(DATE, kind="stable")
    tr, va = list(TimeSeriesSplit(n_splits=5).split(X))[-1]
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    if (len(Xtr), len(Xva)) != sizes:
        raise ValueError(f"unexpected fold 5 sizes {len(Xtr)} / {len(Xva)}")
    ib = IndicatorBuilder().fit(Xtr)
    Ftr, Fva = ib.transform(Xtr), ib.transform(Xva)
    prep = clone(build_preprocessor()).fit(Ftr)
    return dict(ib=ib, prep=prep, Ftr=Ftr, Fva=Fva,
                Ztr=np.asarray(prep.transform(Ftr), dtype=float),
                Zva=np.asarray(prep.transform(Fva), dtype=float),
                names=list(prep.get_feature_names_out()))


def fit_models(Ztr):
    """The two frozen candidate configurations, seed 0 (sections 4 and 7.2)."""
    from model_evaluation.eif_eval import IsoTreeEIF
    from model_evaluation.harness import SklearnIF
    return {"if": SklearnIF(400, 256, seed=0).fit(Ztr), "eif": IsoTreeEIF(400, 256, 2, 0).fit(Ztr)}


def rank_desc(scores):
    """Rank 1 = highest score. Ties go to the earlier position (date order)."""
    scores = np.asarray(scores, dtype=float)
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks
