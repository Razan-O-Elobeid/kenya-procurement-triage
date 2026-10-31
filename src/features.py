"""Red-flag indicators and the composite risk score R(c), as a sklearn transformer.

fit() learns only from the training contracts it is given.
transform() scores any contracts, using only history dated before each one.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import METHOD_LEVEL
from src.entity_resolution import SupplierResolver

# f2 (award criteria missing) was dropped in Phase 2: the field was discontinued around 2021
INDICATORS = [
    "f1_direct_award",
    "f3_decision_period",
    "f4_supplier_concentration",
    "f5_benford_deviation",
]
BENFORD = np.log10(1 + 1 / np.arange(1, 10))
DIGITS = [f"d{i}" for i in range(1, 10)]


def normalise_names(s):
    """Entity resolution stage 1: case, punctuation, spacing, legal suffixes."""
    return (s.astype("string").str.upper()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.replace(r"\bLIMITED\b", "LTD", regex=True)
            .str.replace(r"\bCOMPANY\b", "CO", regex=True))


def buyer_group(names):
    """Coarse buyer type read from the buyer's name, used for the Benford fallback."""
    n = names.astype("string").str.lower().fillna("")
    return pd.Series(np.select(
        [n.str.contains("county"),
         n.str.contains("universit|college|polytechnic|school|institute|tvet"),
         n.str.contains("ministry|state department")],
        ["county", "education", "ministry"], default="agency"), index=names.index)


def first_digit(v):
    """First significant digit of positive values, 0 for zero or missing."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(len(v), dtype=int)
    pos = np.isfinite(v) & (v > 0)
    d = (v[pos] / 10 ** np.floor(np.log10(v[pos]))).astype(int)
    out[pos] = np.where(d >= 10, 1, np.clip(d, 1, 9))  # float edge cases
    return out


def ecdf_score(ref, x):
    """Share of the training reference that is <= x. NaN stays NaN."""
    x = pd.Series(x, dtype=float)
    if len(ref) == 0:
        return pd.Series(np.nan, index=x.index)
    res = np.searchsorted(ref, x.to_numpy(), side="right") / len(ref)
    return pd.Series(np.where(x.isna(), np.nan, res), index=x.index)


class IndicatorBuilder(BaseEstimator, TransformerMixin):
    """Adds indicators f1, f3, f4, f5, the flags, and R(c) to a contracts frame.

    k_prior: weight (in awards) of the buyer-group Benford pattern when a
        buyer has little history. Small buyers lean on their group, large
        buyers on their own awards.
    min_prior: fewest earlier non-zero awards needed for f5.
    resolve_suppliers: group supplier name variants (option A, built on
        training names only) before computing f4.
    """

    def __init__(self, k_prior=100, min_prior=10, resolve_suppliers=True):
        self.k_prior = k_prior
        self.min_prior = min_prior
        self.resolve_suppliers = resolve_suppliers

    # ---------- helpers ----------

    def _events(self, X):
        """One row per contract with what history-based features need."""
        v = X["value_amount"].astype(float)
        s = normalise_names(X["supplier_name"])
        if self.resolver_ is not None:
            s = self.resolver_.transform(s)
        e = pd.DataFrame({
            "b": X["buyer_id"].to_numpy(),
            "s": s.fillna("__missing__").to_numpy(),
            "t": X["tender_tenderPeriod_endDate"].dt.normalize().to_numpy(),
            "v": v.where(v > 0, 0.0).to_numpy(),
            "g": buyer_group(X["buyer_name"]).to_numpy(),
        }, index=X.index)
        fd = first_digit(v)
        for i, col in enumerate(DIGITS, start=1):
            e[col] = (fd == i).astype(float)
        return e

    def _with_history(self, X):
        """Training history plus X, without counting a contract twice."""
        e = self._events(X)
        hist = self.history_[~self.history_.index.isin(e.index)]
        return pd.concat([hist, e]).sort_values("t", kind="stable"), e.index

    @staticmethod
    def _before(frame, keys, cols):
        """Sum of cols over earlier dates within each key group (same day excluded)."""
        day = frame.groupby(keys + ["t"])[cols].sum()
        prior = day.groupby(level=list(range(len(keys)))).cumsum() - day
        return frame[keys + ["t"]].join(prior, on=keys + ["t"])[cols]

    def _benford_mad(self, X):
        """Buyer MAD vs Benford, shrunk towards the buyer group's training pattern."""
        both, idx = self._with_history(X)
        counts = self._before(both, ["b"], DIGITS).loc[idx]
        n = counts.sum(axis=1)
        group = both.loc[idx, "g"]
        prior = np.vstack([self.group_digits_.get(g, self.global_digits_) for g in group])
        p = (counts.to_numpy() + self.k_prior * prior) / (n.to_numpy()[:, None] + self.k_prior)
        mad = pd.Series(np.abs(p - BENFORD).mean(axis=1), index=idx)
        return mad.where(n >= self.min_prior), n

    # ---------- fit / transform ----------

    def fit(self, X, y=None):
        X = X.copy()
        self.p99_ = X["value_amount"].quantile(0.99)
        self.resolver_ = (SupplierResolver().fit(normalise_names(X["supplier_name"]))
                          if self.resolve_suppliers else None)
        self.history_ = self._events(X)

        # Benford digit pattern per buyer group, learned on training awards
        nonzero = self.history_[self.history_["v"] > 0]
        totals = nonzero.groupby("g")[DIGITS].sum()
        self.group_digits_ = {g: (row / row.sum()).to_numpy() for g, row in totals.iterrows()}
        self.global_digits_ = (totals.sum() / totals.sum().sum()).to_numpy()

        d = (X["dateSigned"] - X["tender_tenderPeriod_endDate"]).dt.days
        self.d_ref_ = np.sort(d.dropna().to_numpy())
        mad, _ = self._benford_mad(X)
        self.mad_ref_ = np.sort(mad.dropna().to_numpy())
        return self

    def transform(self, X):
        X = X.copy()
        method = X["tender_procurementMethod"]
        v = X["value_amount"].astype(float)
        f = pd.DataFrame(index=X.index)

        f["method_level"] = method.map(METHOD_LEVEL)
        f["f1_direct_award"] = method.isin(["direct", "SpeciallyPermitted"]).astype(int)

        # f3: shorter decision period scores higher
        d = (X["dateSigned"] - X["tender_tenderPeriod_endDate"]).dt.days.astype(float)
        f["days_to_signing"] = d
        f["f3_decision_period"] = 1 - ecdf_score(self.d_ref_, d)

        # f4: share of the buyer's earlier award value won by this supplier
        both, idx = self._with_history(X)
        tot_b = self._before(both, ["b"], ["v"])["v"].loc[idx]
        tot_bs = self._before(both, ["b", "s"], ["v"])["v"].loc[idx]
        share = tot_bs / tot_b.where(tot_b > 0)
        f["f4_supplier_concentration"] = share.where(X["supplier_name"].notna())

        # f5: Benford deviation with the buyer-group fallback
        mad, n_prior = self._benford_mad(X)
        f["benford_mad"] = mad
        f["n_prior_awards"] = n_prior
        f["f5_benford_deviation"] = ecdf_score(self.mad_ref_, mad)

        # flags
        f["is_signed_pre_closure"] = (d < 0).astype(int)
        f["zero_value_award"] = (v == 0).astype(int)
        f["above_p99"] = (v > self.p99_).astype(int)
        f["is_terminated"] = (X["status"] == "terminated").astype(int)
        f["supplier_missing"] = X["supplier_name"].isna().astype(int)  # display only
        f["signing_date_missing"] = X["dateSigned"].isna().astype(int)  # display only

        # R(c): average of the indicators available for each contract
        ind = f[INDICATORS]
        f["R"] = ind.mean(axis=1)
        for k in INDICATORS:
            f[f"R_minus_{k}"] = ind.drop(columns=k).mean(axis=1)
        return pd.concat([X, f], axis=1)
