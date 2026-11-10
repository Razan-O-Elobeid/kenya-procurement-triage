"""Relative-value feature (protocol 7.3, amendment 6c0cd50).

One extra model input that compares a contract's value with a training-only reference:
  buyer:    log(value) - ref, ref = buyer median (>= 5 awards), else category-group
            median (>= 5 awards), else overall median
  category: (log(value) - median_c) / (1.4826 * MAD_c), with the committed fallbacks
References are fixed summaries of the training part (positive values only), not
expanding histories. Computed from the unclipped value, clipped at the training
P0.1 / P99.9 of the feature, then standardized. Zero, negative or missing values give 0.
src/ is not changed: the extended preprocessor adds one column after the 14 Phase 2 inputs.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline import CATEGORY, build_preprocessor

COLUMNS = ["value_amount", "buyer_id", CATEGORY]
MISSING = "__missing__"


class RelativeValue(BaseEstimator, TransformerMixin):
    """Takes three columns [value, buyer_id, category]. Returns one column."""

    def __init__(self, variant="buyer", min_awards=5, lower=0.001, upper=0.999):
        self.variant = variant
        self.min_awards = min_awards
        self.lower = lower
        self.upper = upper

    def _split(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        v = pd.to_numeric(X.iloc[:, 0], errors="coerce").astype(float)
        ok = np.isfinite(v) & (v > 0)
        lv = np.log(v.where(ok))
        b = X.iloc[:, 1].astype(object)  # a missing buyer id never has a buyer reference
        c = X.iloc[:, 2].astype(object).where(X.iloc[:, 2].notna(), MISSING)
        return lv, ok, b, c

    def fit(self, X, y=None):
        if self.variant not in ("buyer", "category"):
            raise ValueError("variant must be 'buyer' or 'category'")
        lv, ok, b, c = self._split(X)
        x = lv[ok]
        self.overall_median_ = float(x.median())
        self.overall_mad_ = float((x - self.overall_median_).abs().median())

        cg = x.groupby(c[ok])
        cats = pd.DataFrame({"n": cg.size(), "median": cg.median()})
        cats["mad"] = [float((x[c[ok] == k] - m).abs().median()) for k, m in cats["median"].items()]
        self.categories_ = cats[cats["n"] >= self.min_awards]

        has_b = ok & b.notna()
        bg = lv[has_b].groupby(b[has_b])
        buyers = pd.DataFrame({"n": bg.size(), "median": bg.median()})
        self.buyers_ = buyers[buyers["n"] >= self.min_awards]

        raw = self._raw(lv, ok, b, c)
        self.lo_, self.hi_ = (float(q) for q in raw[ok].quantile([self.lower, self.upper]))
        return self

    def reference_level(self, X):
        """Which reference each row uses: buyer, category, overall or none (invalid value)."""
        lv, ok, b, c = self._split(X)
        level = np.where(c.isin(self.categories_.index), "category", "overall")
        if self.variant == "buyer":
            level = np.where(b.isin(self.buyers_.index), "buyer", level)
        return pd.Series(np.where(ok, level, "none"))

    def _raw(self, lv, ok, b, c):
        cat_med = c.map(self.categories_["median"]).astype(float)
        if self.variant == "buyer":
            ref = b.map(self.buyers_["median"]).astype(float)
            ref = ref.fillna(cat_med).fillna(self.overall_median_)
            out = lv - ref
        else:
            med = cat_med.fillna(self.overall_median_)
            mad = c.map(self.categories_["mad"]).astype(float)
            mad = mad.where(mad > 0, self.overall_mad_)  # small or unseen group, or zero group MAD
            denom = (1.4826 * mad).where(mad > 0, 1.0)  # overall MAD is also 0
            out = (lv - med) / denom
        return out.where(ok, 0.0)

    def raw(self, X):
        """Feature before clipping, for checks."""
        return self._raw(*self._split(X)).to_numpy()

    def transform(self, X):
        lv, ok, b, c = self._split(X)
        out = self._raw(lv, ok, b, c).clip(self.lo_, self.hi_).where(ok, 0.0)
        return out.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"relative_value_{self.variant}"])


def build_preprocessor_relative(variant):
    """Phase 2 preprocessor plus one relative-value column at the end (15 inputs)."""
    base = build_preprocessor()
    extra = ("relvalue", make_pipeline(RelativeValue(variant), StandardScaler()), COLUMNS)
    return ColumnTransformer(base.transformers + [extra], remainder="drop",
                             verbose_feature_names_out=True)
