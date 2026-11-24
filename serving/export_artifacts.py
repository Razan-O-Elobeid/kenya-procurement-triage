"""Serving artefact export (deployment stage, step 1). Run once, from the repo root:

    python -m serving.export_artifacts

Refits the frozen IF and EIF on the training period exactly as final_eval did, using its own
prepare(), fit_train() and make_models(). Then extends the lookup history with the recorded
test-period awards (option C) and checks parity before anything is saved:
  G1 split sizes
  G2 thresholds equal the frozen values exactly
  G3 training scores equal results/final_test/train_scores.csv.gz
  G4 batch test scores, flags, percentiles and history features equal the official bundle
  G5 one-contract-at-a-time scoring (the serving path) equals the bundle on a fixed sample
  G6 the reloaded artefact repeats G5 bit for bit; fitted state and history unchanged
This is a reproduction check of published scores. It computes no metric and decides nothing.
It never calls final_eval.run() and only reads results/final_test/.

Writes models/serving_bundle.joblib (untracked, holds PPRA-derived history) and
models/serving_bundle.json (no contract records). Refuses to overwrite them.
Any failed gate exits with status 1 and saves nothing.
"""
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from serving.core import ID, MODELS, build_bundle, ecdf, fitted_state_hash, history_hash, score_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = {"model": ROOT / "models" / "serving_bundle.joblib", "meta": ROOT / "models" / "serving_bundle.json"}
FINAL = ROOT / "results" / "final_test"
FREEZE = ROOT / "model_evaluation" / "model_freeze.json"

# frozen values from final_report.json and the results log (3f3d1fe)
EXPECTED = {"n": (72_585, 36_399),
            "thr": {"eif": 0.5439721468420403, "if": 0.5637224653931433},
            "flags": {"eif": 1_295, "if": 1_131}}
TOL = 1e-12
FEATURES = ["days_to_signing", "f3_decision_period", "f4_supplier_concentration",
            "f5_benford_deviation", "benford_mad", "n_prior_awards", "R"]
N_RANDOM, N_TOP, SAMPLE_SEED = 200, 50, 0

warnings.filterwarnings("ignore", message="Attempting to use more than 1 thread")


class Gate(Exception):
    pass


def gate(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise Gate(msg)


def max_diff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    ok = ~np.isnan(a)
    return float(np.max(np.abs(a[ok] - b[ok]), initial=0.0))


def compare(res, ref, label):
    """Scores, percentiles, flags and history features against the official contracts table."""
    for m in MODELS:
        d = max_diff(res[f"score_{m}"], ref[f"score_{m}"])
        gate(d <= TOL, f"{label}: {m} scores match (max difference {d:.1e})")
        d = max_diff(res[f"pct_{m}"], ref[f"pct_{m}"])
        gate(d <= TOL, f"{label}: {m} training percentiles match (max difference {d:.1e})")
        same = np.array_equal(np.asarray(res[f"flag_{m}"], bool), np.asarray(ref[f"flag_{m}"], bool))
        gate(same, f"{label}: {m} flags identical")
    worst = max(max_diff(res[c], ref[c]) for c in FEATURES)
    gate(worst <= TOL, f"{label}: history features and R(c) match (max difference {worst:.1e})")


def model_hash(model):
    return hashlib.sha256(pickle.dumps(model, protocol=4)).hexdigest()


def sample_positions(ref, n_random, n_top, seed=SAMPLE_SEED):
    rng = np.random.default_rng(seed)
    rand = rng.choice(len(ref), size=min(n_random, len(ref)), replace=False)
    top = np.argsort(-ref["score_eif"].to_numpy(), kind="stable")[:n_top]
    return np.array(sorted(set(rand.tolist()) | set(top.tolist())))


def run(out, Xtr, Xte, refs, expected=EXPECTED, n_random=N_RANDOM, n_top=N_TOP, meta=None):
    """All gates, then an atomic save. Returns the exit status."""
    from model_evaluation.explain_checks import background
    from model_evaluation.final_eval import fit_train, make_models, prepare

    for p in out.values():
        if p.exists():
            print("STOPPED: output exists:", p, file=sys.stderr)
            return 1
    tmp = {k: p.with_name(p.name + ".tmp") for k, p in out.items()}
    try:
        gate((len(Xtr), len(Xte)) == tuple(expected["n"]),
             f"G1 split sizes {len(Xtr):,} / {len(Xte):,}")

        t0 = time.perf_counter()
        d = prepare(Xtr, Xte)
        fitted, tr, thr, _ = fit_train(d, make_models())
        fit_s = time.perf_counter() - t0
        for m in MODELS:
            gate(thr[m] == expected["thr"][m] == refs["thresholds"][m],
                 f"G2 {m} threshold {thr[m]!r} equals the frozen value exactly")
        for m in MODELS:
            dm = max_diff(tr[m], refs["train_scores"][m])
            gate(dm <= TOL, f"G3 {m} training scores match (max difference {dm:.1e})")

        state0 = fitted_state_hash(d["ib"])
        models0 = {m: model_hash(fitted[m]) for m in MODELS}
        bundle = build_bundle(d["ib"], d["prep"], {m: fitted[m] for m in MODELS}, thr, tr, Xte,
                              background(d["Ztr"], 0), d["names"], meta)
        gate(bundle["fitted_state_sha256"] == state0, "fitted references unchanged by the history extension")
        gate(len(bundle["ib"].history_) == len(Xtr) + len(Xte),
             f"lookup history holds {len(Xtr):,} training + {len(Xte):,} test-period awards")

        # G4: whole test period in one batch, as the frozen run did
        ref = refs["contracts"]
        gate(len(ref) == len(Xte) and np.array_equal(ref[ID].astype(str).to_numpy(),
                                                     Xte[ID].astype(str).to_numpy()),
             "G4 contract ids in the same order as the official bundle")
        Fb = bundle["ib"].transform(Xte)
        Zb = np.asarray(bundle["prep"].transform(Fb), dtype=float)
        batch = pd.DataFrame({c: Fb[c].to_numpy() for c in FEATURES})
        for m in MODELS:
            s = bundle["models"][m].score(Zb)
            batch[f"score_{m}"], batch[f"flag_{m}"] = s, s > thr[m]
            batch[f"pct_{m}"] = ecdf(tr[m], s)
            gate(int((s > thr[m]).sum()) == expected["flags"][m],
                 f"G4 {m} flags {int((s > thr[m]).sum()):,} (expected {expected['flags'][m]:,})")
        compare(batch, ref, "G4 batch")

        # G5: the serving path, one contract per call
        pos = sample_positions(ref, n_random, n_top)
        rows, lat = [], []
        for p in pos:
            t1 = time.perf_counter()
            rows.append(score_rows(bundle, Xte.iloc[[p]]))
            lat.append(time.perf_counter() - t1)
        single = pd.concat(rows, ignore_index=True)
        gate(np.array_equal(single[ID].astype(str).to_numpy(), ref[ID].astype(str).to_numpy()[pos]),
             f"G5 {len(pos)} sampled contracts returned in order")
        compare(single, ref.iloc[pos].reset_index(drop=True), "G5 single-contract")

        # G6: save, reload, repeat
        out["model"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, tmp["model"])
        loaded = joblib.load(tmp["model"])
        again = score_rows(loaded, Xte.iloc[pos])
        same = all(np.array_equal(again[c].to_numpy(), single[c].to_numpy())
                   for c in [f"{k}_{m}" for m in MODELS for k in ("score", "pct", "flag")])
        gate(same, "G6 reloaded artefact repeats G5 bit for bit")
        gate(fitted_state_hash(loaded["ib"]) == state0 and history_hash(loaded["ib"]) == bundle["history_sha256"],
             "G6 fitted state and lookup history unchanged after scoring")
        models1 = {m: model_hash(bundle["models"][m]) for m in MODELS}
        gate(models1 == models0, "G6 models unchanged after scoring")

        record = {
            "name": "serving_bundle", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": "serving artefact; reproduces the frozen configuration, no new evaluation",
            "deployed_model": "eif", "comparison_model": "if",
            "thresholds": bundle["thresholds"], "threshold_rule": "p95_training_scores (frozen)",
            "history": {"training_awards": len(Xtr), "test_period_awards": len(Xte),
                        "rule": "strictly earlier tender closing dates, same day excluded",
                        "date_convention": "event date (tender closing), not publication date"},
            "input_names": bundle["input_names"], "buyer_kind": bundle["buyer_kind"],
            "fitted_state_sha256": state0, "history_sha256": bundle["history_sha256"],
            "parity": {"gates": "G1 to G6 passed", "tolerance": TOL, "sample_size": int(len(pos)),
                       "sample": f"{n_random} random (seed {SAMPLE_SEED}) plus top {n_top} by EIF score",
                       "note": "reproduction of published scores; no metric computed"},
            "local_timing_development_machine": {
                "fit_seconds": round(fit_s, 2),
                "single_contract_seconds_median": round(float(np.median(lat)), 3),
                "single_contract_seconds_p95": round(float(np.quantile(lat, 0.95)), 3),
                "note": "not the latency benchmark"},
            "versions": {"python": platform.python_version(),
                         **{p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy",
                                                    "isotree", "joblib", "rapidfuzz")}},
            **(meta or {}),
        }
        h = hashlib.sha256(tmp["model"].read_bytes()).hexdigest()
        record["artifact"], record["artifact_sha256"] = out["model"].name, h
        tmp["meta"].write_text(json.dumps(record, indent=2, default=str))
        os.replace(tmp["model"], out["model"])
        os.replace(tmp["meta"], out["meta"])
    except Exception as e:  # any failure: nothing is kept
        for p in tmp.values():
            p.unlink(missing_ok=True)
        print(f"STOPPED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print("\nsaved", out["model"], "and", out["meta"])
    print(f"single contract: median {record['local_timing_development_machine']['single_contract_seconds_median']} s, "
          f"p95 {record['local_timing_development_machine']['single_contract_seconds_p95']} s")
    return 0


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main():
    from model_evaluation.final_eval import FROZEN, load_periods

    fz = json.loads(FREEZE.read_text())
    if fz.get("models") != FROZEN or fz.get("deployed_model") != "eif" \
            or fz.get("threshold_rule") != "p95_training_scores":
        print("STOPPED: model_freeze.json differs from the frozen configuration", file=sys.stderr)
        return 1
    if git("diff", "--quiet", "HEAD", "--", str(FREEZE.relative_to(ROOT))) is None:
        print("STOPPED: model_freeze.json differs from the last commit", file=sys.stderr)
        return 1
    log = json.loads((FINAL / "RUN_LOG.json").read_text())
    if "completed" not in [a.get("status") for a in log.get("attempts", [])]:
        print("STOPPED: no completed official run in results/final_test", file=sys.stderr)
        return 1
    report = json.loads((FINAL / "final_report.json").read_text())
    refs = {"thresholds": report["report"]["thresholds"],
            "train_scores": pd.read_csv(FINAL / "train_scores.csv.gz"),
            "contracts": pd.read_csv(FINAL / "contracts.csv.gz", low_memory=False)}
    try:
        Xtr, Xte = load_periods(ROOT)
    except Exception as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 1
    meta = {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "freeze_commit": git("log", "-1", "--format=%h %cI", "--", str(FREEZE.relative_to(ROOT))),
            "official_run": "results/final_test (737d16d, 3f3d1fe); read only"}
    return run(OUT, Xtr, Xte, refs, meta=meta)


if __name__ == "__main__":
    sys.exit(main())
