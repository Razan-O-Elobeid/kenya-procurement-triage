"""Transformers used inside the ColumnTransformer. All parameters are learned in fit (train only)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClipGroupImputer(BaseEstimator, TransformerMixin):
    """Takes two columns [value, group]. Clips the value to train quantiles, then fills
    missing values with the train median of its group (overall median for unseen groups).

    zero_as_missing: treat 0 as missing, so zeros are imputed rather than clipped.
    Quantiles are computed on observed values only, before any imputation.
    """

    def __init__(self, lower=0.01, upper=0.99, zero_as_missing=False):
        self.lower = lower
        self.upper = upper
        self.zero_as_missing = zero_as_missing

    def _split(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        x = pd.to_numeric(X.iloc[:, 0], errors="coerce").astype(float)
        if self.zero_as_missing:
            x = x.where(x != 0)
        g = X.iloc[:, 1].astype(object).where(X.iloc[:, 1].notna(), "__missing__")
        return x, g

    def fit(self, X, y=None):
        x, g = self._split(X)
        self.lo_, self.hi_ = x.quantile(self.lower), x.quantile(self.upper)
        xc = x.clip(self.lo_, self.hi_)
        self.medians_ = xc.groupby(g).median().to_dict()
        self.overall_ = xc.median()
        return self

    def transform(self, X):
        x, g = self._split(X)
        xc = x.clip(self.lo_, self.hi_)
        fill = g.map(self.medians_).astype(float).fillna(self.overall_)
        return xc.fillna(fill).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = input_features[0] if input_features is not None else "x0"
        return np.array([name])


class GroupModeImputer(BaseEstimator, TransformerMixin):
    """Takes two columns [category, group]. Fills a missing category with the train mode
    of its group (overall mode for unseen groups)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        c, g = X.iloc[:, 0].astype(object), X.iloc[:, 1].astype(object)
        ok = c.notna()
        self.modes_ = c[ok].groupby(g[ok]).agg(lambda s: s.mode().iloc[0]).to_dict()
        self.overall_ = c[ok].mode().iloc[0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        c, g = X.iloc[:, 0].astype(object), X.iloc[:, 1].astype(object)
        fill = g.map(self.modes_).fillna(self.overall_)
        return c.where(c.notna(), fill).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = input_features[0] if input_features is not None else "x0"
        return np.array([name])


def signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))
