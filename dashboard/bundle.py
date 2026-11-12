"""Dashboard bundle: frozen files the app reads (protocol section 11).

The builder writes the bundle; the app only reads it. A bundle holds:
  manifest.json        what the bundle is, models, thresholds, explanation status, file hashes
  contracts.csv.gz     one row per contract: display fields, indicators, scores, ranks, flags
  explanations.csv.gz  attributions for contracts, only from checks that passed
  buyers.csv.gz, categories.csv.gz   historical reference from the training data only
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_evaluation.fold5 import rank_desc

FILES = ["contracts.csv.gz", "explanations.csv.gz", "buyers.csv.gz", "categories.csv.gz"]
DISPLAY = ["ocid", "buyer_id", "buyer_name", "supplier_name", "tender_title", "tender_mainProcurementCategory",
           "tender_procurementMethod", "value_amount", "value_currency", "tender_tenderPeriod_endDate",
           "dateSigned", "status"]
INDICATORS = ["days_to_signing", "method_level", "f1_direct_award", "f3_decision_period",
              "f4_supplier_concentration", "f5_benford_deviation", "benford_mad", "n_prior_awards",
              "is_signed_pre_closure", "zero_value_award", "above_p99", "below_p1", "is_terminated",
              "supplier_missing", "signing_date_missing", "R"]
SCORE_TOL = 1e-12   # saved scores must match the builder's scores this closely
ADD_TOL = 1e-6      # section 8 additivity tolerance (relative)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ecdf(ref, x):
    """Share of the training reference at or below x."""
    ref = np.sort(np.asarray(ref, dtype=float))
    return np.searchsorted(ref, np.asarray(x, dtype=float), side="right") / len(ref)


def contract_table(F, supplier_resolved, scores, train_scores, thresholds):
    """Display fields, indicators and per-model score, rank, training percentile and flag."""
    t = F[[c for c in DISPLAY if c in F.columns]].copy()
    t.insert(0, "row_id", np.arange(len(F)))
    t["supplier_resolved"] = np.asarray(supplier_resolved, dtype=object)
    for c in INDICATORS:
        if c in F.columns:
            t[c] = F[c].to_numpy()
    t["rank_R"] = rank_desc(F["R"].fillna(-np.inf).to_numpy())
    for m, s in scores.items():
        t[f"score_{m}"] = s
        t[f"rank_{m}"] = rank_desc(s)
        t[f"pct_{m}"] = ecdf(train_scores[m], s)
        t[f"flag_{m}"] = s > thresholds[m]
    return t.reset_index(drop=True)


def check_explanations(saved, contracts, model, names, score, output, validated):
    """Integrity of saved attributions against this bundle. Returns (rows or None, report).

    Rows are matched by position and contract id, for the given model. Scores and explained
    outputs must match the builder's to SCORE_TOL; attribution columns must follow the input order;
    attributions must add up to the output (section 8). Anything else: no attributions shown.
    """
    rep = {"model": model, "section8_validated": bool(validated), "passed": False}
    attr = [f"attr__{n}" for n in names]
    need = ["contract_id", "position_in_fold5_validation", "ranking_score", "explained_output", "base"]
    if not validated:
        rep["reason"] = "explanation method not validated under section 8"
        return None, rep
    if not set(need) <= set(saved.columns) or [c for c in saved.columns if c.startswith("attr__")] != attr:
        rep["reason"] = "columns or input order differ from the bundle's inputs"
        return None, rep
    pos = saved["position_in_fold5_validation"].to_numpy()
    if len(set(pos)) != len(pos) or pos.min() < 0 or pos.max() >= len(contracts):
        rep["reason"] = "positions missing, repeated or out of range"
        return None, rep
    ids = contracts["ocid"].astype(str).to_numpy()[pos]
    if not np.array_equal(ids, saved["contract_id"].astype(str).to_numpy()):
        rep["reason"] = "contract ids do not match at the saved positions"
        return None, rep
    d_score = float(np.max(np.abs(saved["ranking_score"].to_numpy() - score[pos])))
    d_out = float(np.max(np.abs(saved["explained_output"].to_numpy() - output[pos])))
    v = saved[attr].to_numpy(dtype=float)
    rel = np.abs(saved["base"].to_numpy() + v.sum(axis=1) - saved["explained_output"].to_numpy())
    rel = float(np.max(rel / np.maximum(np.abs(saved["explained_output"].to_numpy()), 1e-12)))
    rep.update(max_score_diff=d_score, max_output_diff=d_out, max_additivity_error=rel, n_rows=len(saved))
    if not (d_score <= SCORE_TOL and d_out <= SCORE_TOL and rel <= ADD_TOL and np.all(np.isfinite(v))):
        rep["reason"] = "scores, outputs or additivity do not match"
        return None, rep
    rows = saved[["base", "explained_output"] + attr].copy()
    rows.insert(0, "row_id", pos)
    rows.insert(1, "model", model)
    rep.update(passed=True, reason="all integrity checks passed")
    return rows, rep


def history_tables(Ftr):
    """Training-only reference: per buyer and per category."""
    v = Ftr["value_amount"].astype(float)
    pos = Ftr.assign(_v=v)[v > 0]
    buyers = pos.groupby("buyer_id").agg(buyer_awards=("_v", "size"), buyer_median_value=("_v", "median"),
                                         buyer_total_value=("_v", "sum")).reset_index()
    cats = pos.groupby("tender_mainProcurementCategory").agg(
        category_awards=("_v", "size"), category_median_value=("_v", "median")).reset_index()
    return buyers, cats


def write_bundle(out, tables, manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, df in tables.items():
        df.to_csv(out / name, index=False, compression="gzip")
    manifest["files"] = {name: sha256(out / name) for name in tables}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def load_bundle(path):
    """Read a bundle and verify every file hash. Raises ValueError on any mismatch."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"{name} does not match its hash in manifest.json")
    tables = {name.split(".")[0]: pd.read_csv(path / name, low_memory=False) for name in FILES}
    return manifest, tables
