"""Scoring core for the serving stage. Used by the export script, the tests and later the API.

Each contract is scored on its own against the reference history: the frozen training fit plus
the recorded awards up to the end of the release (option C). Contracts in one request never
see each other. History features use strictly earlier dates (same day excluded), ordered by
tender closing date, as in the frozen run. That is the event date, not the date the award was
published, so an offline score can use awards a live system would not yet have seen.
"""
import copy
import hashlib

import numpy as np
import pandas as pd

MODELS = ("eif", "if")
ID = "ocid"
DATES = ["tender_tenderPeriod_endDate", "dateSigned"]
REQUIRED = ["buyer_id", "buyer_name", "supplier_name", "value_amount",
            "tender_mainProcurementCategory", "tender_procurementMethod", "status"] + DATES
SCORECARD = ["days_to_signing", "method_level", "f1_direct_award", "f3_decision_period",
             "f4_supplier_concentration", "f5_benford_deviation", "benford_mad", "n_prior_awards",
             "is_signed_pre_closure", "zero_value_award", "above_p99", "below_p1", "is_terminated",
             "supplier_missing", "signing_date_missing", "R"]
NOTE = ("historical reference: model, preprocessing and thresholds fitted on 2018 to 2023; "
        "lookup history up to the end of the release. Ranking for review, not a corruption finding.")


# ---------- hashes ----------

def _feed(h, obj):
    """Content hash that does not depend on how objects are laid out in memory or in a pickle."""
    if isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=repr):
            _feed(h, k)
            _feed(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for v in obj:
            _feed(h, v)
        h.update(b"]")
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            _feed(h, obj.tolist())
        else:
            h.update(f"{obj.dtype}{obj.shape}".encode())
            h.update(np.ascontiguousarray(obj).tobytes())
    else:
        h.update(f"{type(obj).__name__}:{obj!r};".encode())


def _sha(obj):
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


def fitted_state_hash(ib):
    """Hash of everything IndicatorBuilder learned in fit, except the lookup history."""
    r = ib.resolver_
    return _sha({"params": ib.get_params(), "p1": ib.p1_, "p99": ib.p99_,
                 "d_ref": ib.d_ref_, "mad_ref": ib.mad_ref_,
                 "group_digits": ib.group_digits_,
                 "global_digits": ib.global_digits_,
                 "resolver": None if r is None else (r.threshold, r.name_map_, r.key_map_,
                                                     r.ref_sorted_, list(r.ref_ids_))})


def history_hash(ib):
    h = pd.util.hash_pandas_object(ib.history_, index=True).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def ecdf(ref, x):
    """Share of the training reference at or below x (same as dashboard.bundle.ecdf)."""
    ref = np.sort(np.asarray(ref, dtype=float))
    return np.searchsorted(ref, np.asarray(x, dtype=float), side="right") / len(ref)


# ---------- history ----------

def extend_history(ib, X_more):
    """Copy of ib whose lookup history also holds X_more. Fitted references are not touched."""
    if (ib.history_.index < 0).any() or (X_more.index < 0).any():
        raise ValueError("negative indices are kept for incoming rows")
    if X_more.index.isin(ib.history_.index).any() or X_more.index.duplicated().any():
        raise ValueError("history rows must have unique indices")
    ext = copy.deepcopy(ib)
    ext.history_ = pd.concat([ib.history_, ext._events(X_more)])
    if fitted_state_hash(ext) != fitted_state_hash(ib):
        raise RuntimeError("extending the history changed a fitted reference")
    return ext


def buyer_kind(ib):
    b = ib.history_["b"]
    if pd.api.types.is_numeric_dtype(b):
        return str(b.dtype)
    return "str" if b.dropna().map(type).eq(str).all() else "object"


# ---------- rows ----------

def prepare_rows(df, kind="str"):
    """Checks and normalises incoming contracts. Gives them negative indices."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if len(df) == 0:
        raise ValueError("no contracts")
    X = df[[c for c in [ID] + REQUIRED if c in df.columns]].copy()
    for c in DATES:
        raw = X[c]
        parsed = pd.to_datetime(raw, utc=True, errors="coerce", format="ISO8601")
        if (parsed.isna() & raw.notna()).any():
            raise ValueError(f"{c}: unparseable date")
        X[c] = parsed
    if X["tender_tenderPeriod_endDate"].isna().any():
        raise ValueError("tender_tenderPeriod_endDate is required")
    v = pd.to_numeric(X["value_amount"], errors="coerce")
    if v.isna().any() or not np.isfinite(v).all():
        raise ValueError("value_amount must be a finite number")
    X["value_amount"] = v.astype(float)
    if kind == "str":
        X["buyer_id"] = X["buyer_id"].map(lambda b: b if pd.isna(b) else str(b))
    elif kind != "object":
        X["buyer_id"] = pd.to_numeric(X["buyer_id"], errors="raise").astype(kind)
    X.index = -np.arange(1, len(X) + 1)  # history indices are never negative
    return X


def score_rows(bundle, df):
    """Scores each contract on its own. Returns one row per contract, in input order."""
    ib, prep = bundle["ib"], bundle["prep"]
    X = prepare_rows(df, bundle["buyer_kind"])
    if X.index.isin(ib.history_.index).any():
        raise RuntimeError("incoming index collides with the history")
    F = pd.concat([ib.transform(X.iloc[[i]]) for i in range(len(X))])
    Z = np.asarray(prep.transform(F), dtype=float)
    out = pd.DataFrame({ID: F[ID].to_numpy() if ID in F else None}, index=range(len(F)))
    for m in MODELS:
        s = bundle["models"][m].score(Z)
        out[f"score_{m}"] = s
        out[f"pct_{m}"] = ecdf(bundle["train_scores"][m], s)
        out[f"flag_{m}"] = s > bundle["thresholds"][m]
    for c in SCORECARD:
        out[c] = F[c].to_numpy()
    return out


# ---------- bundle ----------

def build_bundle(ib, prep, models, thresholds, train_scores, X_more, background, names, meta=None):
    """Everything the API needs, in one object. ib is copied, never changed."""
    ext = extend_history(ib, X_more)
    return {"ib": ext, "prep": prep, "models": dict(models),
            "thresholds": {m: float(thresholds[m]) for m in MODELS},
            "train_scores": {m: np.asarray(train_scores[m], dtype=float) for m in MODELS},
            "background": np.asarray(background, dtype=float), "input_names": list(names),
            "buyer_kind": buyer_kind(ext), "fitted_state_sha256": fitted_state_hash(ext),
            "history_sha256": history_hash(ext), "note": NOTE, "meta": meta or {}}
