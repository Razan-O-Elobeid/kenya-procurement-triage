"""Full pipeline: indicators -> preprocessing -> Isolation Forest. Everything is fit on X_train only."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import IndicatorBuilder
from src.transformers import ClipGroupImputer, GroupModeImputer, signed_log

# model inputs; f1 and f3 are excluded (collinear by construction with method and days)
FLAGS = ["zero_value_award", "above_p99", "below_p1", "is_signed_pre_closure", "is_terminated"]
HISTORY = ["f4_supplier_concentration", "f5_benford_deviation"]
METHOD = "tender_procurementMethod"
CATEGORY = "tender_mainProcurementCategory"


def _one_hot():
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def build_preprocessor():
    value = make_pipeline(ClipGroupImputer(0.01, 0.99, zero_as_missing=True),
                          FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                          StandardScaler())
    days = make_pipeline(ClipGroupImputer(0.01, 0.99),
                         FunctionTransformer(signed_log, feature_names_out="one-to-one"),
                         StandardScaler())
    return ColumnTransformer([
        ("value", value, ["value_amount", CATEGORY]),
        ("days", days, ["days_to_signing", METHOD]),
        ("method", make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler()), ["method_level"]),
        ("category", make_pipeline(GroupModeImputer(), _one_hot()), [CATEGORY, METHOD]),
        ("history", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), HISTORY),
        ("flags", StandardScaler(), FLAGS),
    ], remainder="drop", verbose_feature_names_out=True)


def build_pipeline(n_estimators=200, max_samples="auto", random_state=0):
    return Pipeline([
        ("indicators", IndicatorBuilder()),
        ("prep", build_preprocessor()),
        ("model", IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                  random_state=random_state, n_jobs=-1)),
    ])
