"""Final held-out evaluation (protocol section 10, with RRF from 7.5). Scores the test period once.

Real run, only after the freeze is committed:
    python -m model_evaluation.final_eval
Rehearsal on the training period only (fold 5 as a stand-in test set), any time before:
    python -m model_evaluation.final_eval --dry-run --freeze model_evaluation/freeze_dryrun.json

Steps:
  1. gate (real run): model_freeze.json is tracked, committed and unchanged, the working tree is clean,
     and no earlier final run exists. A run log is written before the test period is loaded.
  2. fit the frozen configurations on the full training period (seed 0); thresholds = training P95;
     RRF threshold = P95 of training RRF scores (training-referenced ranks).
  3. score the test period with the frozen IF only.
  4. integrity: the IF reproduces the Phase 2 test scores and its 1,131 flags (real run only).
     Only then are EIF, R(c) and RRF test scores computed. A stop before this point may be retried.
  5. report shares, score drift, rank agreement, top-5% overlap,
     flagged share by month and injection recall (sensitivity only).
  6. explanations for each explained model's flagged test contracts (section 8.1 settings).
  7. save results, then build the final dashboard bundle (dashboard.build_final_bundle).
Thresholds, models and rules are never changed here.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.base import clone

from model_evaluation.rrf import K, fuse, reference_rrf

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "model_evaluation" / "model_freeze.json"
VALIDATION = ROOT / "results" / "model_evaluation" / "explanations" / "explain_validation.json"
PHASE2_REF = ROOT / "results" / "test_scores.csv"
OUTS = {"real": ROOT / "results" / "final_test", "dry": ROOT / "results" / "final_test_dryrun"}
RUN_LOG = "RUN_LOG.json"

EXPECTED = dict(n_train=72_585, n_test=36_399, n_flagged_phase2=1_131)
FROZEN = {"if": dict(family="sklearn_iforest", n_estimators=400, max_samples=256, seed=0),
          "eif": dict(family="isotree_eif", ntrees=400, sample_size=256, ndim=2, seed=0)}
RRF_SPEC = dict(components=["R", "if", "eif"], k=K)
EXPLAINER = {"if": "tree_interventional", "eif": "permutation"}
VALIDATION_KEY = {"if": "if_tree", "eif": "eif_perm"}
TOP = 0.95
TYPES = ["value", "date", "value_x5"]
N_REPEAT = 20       # exact-repeat check on the first 20 explained contracts
REPRO_TOL = 1e-12

warnings.filterwarnings("ignore", message="Attempting to use more than 1 thread")


def rel(p):
    """Path relative to the repo, safe with symlinked folders such as /var on macOS."""
    return Path(p).resolve().relative_to(ROOT.resolve())


class Stop(Exception):
    pass


def require(cond, msg):
    if not cond:
        raise Stop(msg)


# ---------- freeze ----------

def check_freeze(fz, validation):
    """The freeze must name exactly the frozen configurations; returns it unchanged."""
    require(fz.get("models") == FROZEN, "freeze models differ from the frozen configurations")
    require(fz.get("rrf") == RRF_SPEC, "freeze RRF specification differs from 7.5")
    require(fz.get("threshold_rule") == "p95_training_scores", "threshold rule must be p95_training_scores")
    dep, lead = fz.get("deployed_model"), fz.get("scientific_leader")
    require(lead in ("if", "eif"), "scientific_leader must be if or eif")
    require(dep in (None, "if", "eif"), "deployed_model must be if, eif or null")
    q = fz.get("queue_models")
    require(isinstance(q, list) and q and len(set(q)) == len(q) and set(q) <= {"if", "eif"},
            "queue_models must be a non-empty list from if, eif (RRF is never queued)")
    require(q[0] == (dep if dep else lead), "the first queue model must be the deployed model, or the "
            "scientific leader when nothing is deployed (8.1)")
    ex = fz.get("explained_models")
    require(isinstance(ex, list) and set(ex) <= {"if", "eif"}, "explained_models must be from if, eif")
    require(dep is None or dep in ex, "the deployed model must be in explained_models")
    for m in set(ex):
        require(validation.get(VALIDATION_KEY[m], {}).get("validated") is True,
                f"{m}: explanation not validated under section 8")
    require(fz.get("explanation_settings") == {"background_rows": 100, "background_seed": 0},
            "explanation settings must be 100 background rows, seed 0 (8.1)")
    return fz


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def check_committed(path):
    rel_path = str(rel(path))
    require(git("ls-files", "--error-unmatch", rel_path).returncode == 0, f"{rel_path} is not tracked by Git")
    require(git("diff", "--quiet", "HEAD", "--", rel_path).returncode == 0, f"{rel_path} differs from the last commit")
    require(not git("status", "--porcelain", "--untracked-files=no").stdout.strip(),
            "tracked files have uncommitted changes")
    return git("rev-parse", "HEAD").stdout.strip(), git("log", "-1", "--format=%h %cI", "--", rel_path).stdout.strip()


# ---------- data and models ----------

def load_periods(root):
    """Training and test periods, exactly as src/baseline.py. Only called after the gate."""
    from src.data import load_merged_table, parse_dates, temporal_split
    data = root / "data" / "raw" / "full"
    if not data.exists():
        data = root / "data" / "raw" / "kenya_ppra_full.csv.tar.gz"
    aw, train = temporal_split(parse_dates(load_merged_table(data)))
    require((int(train.sum()), int((~train).sum())) == (EXPECTED["n_train"], EXPECTED["n_test"]),
            "unexpected split sizes")
    return aw[train], aw[~train]


def prepare(Xtr, Xte):
    """Indicators and preprocessing fitted on the training period only."""
    from src.features import IndicatorBuilder
    from src.pipeline import build_preprocessor
    ib = IndicatorBuilder().fit(Xtr)
    Ftr, Fte = ib.transform(Xtr), ib.transform(Xte)
    prep = clone(build_preprocessor()).fit(Ftr)
    return dict(ib=ib, Ftr=Ftr, Fte=Fte, Ztr=np.asarray(prep.transform(Ftr), dtype=float),
                Zte=np.asarray(prep.transform(Fte), dtype=float), prep=prep,
                names=list(prep.get_feature_names_out()))


def make_models():
    from model_evaluation.eif_eval import IsoTreeEIF
    from model_evaluation.harness import SklearnIF
    return {"if": SklearnIF(400, 256, seed=0), "eif": IsoTreeEIF(400, 256, 2, 0)}


def fit_train(d, models):
    """Fit on training and score training only. Thresholds and RRF references come from training."""
    fitted = {m: mod.fit(d["Ztr"]) for m, mod in models.items()}
    tr = {m: f.score(d["Ztr"]) for m, f in fitted.items()}
    tr["R"] = d["Ftr"]["R"].to_numpy(dtype=float)
    for k in ("R", "if", "eif"):
        require(np.isfinite(tr[k]).all(), f"non-finite training {k} scores or missing R(c)")
    refs = {k: tr[k] for k in RRF_SPEC["components"]}
    tr["rrf"] = reference_rrf(refs)
    thr = {m: float(np.quantile(tr[m], TOP)) for m in ("if", "eif", "rrf")}  # fixed before any test use
    return fitted, tr, thr, refs


def score_if(d, fitted):
    """Test scores of the frozen IF only: the result already known from Phase 2."""
    s = fitted["if"].score(d["Zte"])
    require(np.isfinite(s).all(), "non-finite IF test scores")
    return s


def score_new(d, fitted, refs, te):
    """New test results: EIF, R(c) and RRF. Called only after the Phase 2 check has passed."""
    te["R"] = d["Fte"]["R"].to_numpy(dtype=float)
    require(np.isfinite(te["R"]).all(), "missing R(c) in the test period")
    te["eif"] = fitted["eif"].score(d["Zte"])
    require(np.isfinite(te["eif"]).all(), "non-finite EIF test scores")
    te["rrf"] = fuse(refs, {k: te[k] for k in refs})
    return te


def fit_score(d, models):
    """Training fit plus all test scores, in the gated order (used by tests and the dry run)."""
    fitted, tr, thr, refs = fit_train(d, models)
    te = score_new(d, fitted, refs, {"if": score_if(d, fitted)})
    return fitted, tr, te, thr, refs


def check_phase2(ref, te_if, thr_if, ids):
    return (len(ref) == EXPECTED["n_test"] and int(ref["top5"].sum()) == EXPECTED["n_flagged_phase2"]
            and np.allclose(np.sort(ref["anomaly_score"].to_numpy()), np.sort(te_if), rtol=0, atol=REPRO_TOL)
            and sorted(ref.loc[ref["top5"], "main_ocid"].astype(str)) == sorted(ids[te_if > thr_if]))


# ---------- report ----------

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else float("nan")


def top_set(s):
    return set(np.flatnonzero(s > np.quantile(s, TOP)))


def drift(tr, te):
    q = [0.5, 0.9, 0.95, 0.99]
    return {"train_quantiles": dict(zip(map(str, q), np.quantile(tr, q).tolist())),
            "test_quantiles": dict(zip(map(str, q), np.quantile(te, q).tolist())),
            "train_mean": float(tr.mean()), "test_mean": float(te.mean()),
            "ks_statistic": float(ks_2samp(tr, te).statistic)}


def injection_recall(d, fitted, refs, thr, te):
    """Injected copies of test contracts (seed 0), as in validation; f3 and R(c) recomputed for dates.
    Recall against (a) the training threshold and (b) the P95 of clean test scores."""
    from model_evaluation.indicator_scores import injected_frames
    inj = injected_frames({"fold": 1, "Fva": d["Fte"], "ib": d["ib"]})
    out = {}
    for k, G in inj.items():
        Z = np.asarray(d["prep"].transform(G), dtype=float)
        s = {m: f.score(Z) for m, f in fitted.items()}
        s["R"] = G["R"].to_numpy(dtype=float)
        s["rrf"] = fuse(refs, {c: s[c] for c in refs})
        out[k] = {"n": len(G), **{m: {"vs_training_threshold": float((s[m] > thr[m]).mean()),
                                      "vs_test_p95": float((s[m] > np.quantile(te[m], TOP)).mean())}
                                  for m in ("if", "eif", "rrf")}}
    return out


def build_report(d, tr, te, thr, fitted, refs, date_col):
    ms = ("if", "eif", "rrf")
    flags = {m: te[m] > thr[m] for m in ms}
    rep = {"n_train": len(tr["if"]), "n_test": len(te["if"]),
           "thresholds": thr,
           "share_above_threshold": {m: float(flags[m].mean()) for m in ms},
           "n_flagged": {m: int(flags[m].sum()) for m in ms},
           "train_share_above_threshold": {m: float((tr[m] > thr[m]).mean()) for m in ms},
           "score_drift": {m: drift(tr[m], te[m]) for m in ("if", "eif", "rrf", "R")}}
    names = ["if", "eif", "rrf", "R"]
    rep["spearman"] = {f"{a}_{b}": float(spearmanr(te[a], te[b])[0])
                       for i, a in enumerate(names) for b in names[i + 1:]}
    rep["jaccard_top5_test"] = {f"{a}_{b}": jaccard(top_set(te[a]), top_set(te[b]))
                                for i, a in enumerate(names) for b in names[i + 1:]}
    rep["jaccard_flagged"] = {f"{a}_{b}": jaccard(set(np.flatnonzero(flags[a])), set(np.flatnonzero(flags[b])))
                              for i, a in enumerate(ms) for b in ms[i + 1:]}
    month = pd.to_datetime(d["Fte"][date_col], errors="coerce", utc=True).dt.strftime("%Y-%m").to_numpy()
    by_month = pd.DataFrame({"month": month, **{f"flag_{m}": flags[m] for m in ms}})
    by_month = by_month.groupby("month", dropna=False).agg(
        n=("flag_if", "size"), **{f"share_{m}": (f"flag_{m}", "mean") for m in ms}).reset_index()
    rep["injection_recall"] = injection_recall(d, fitted, refs, thr, te)
    return rep, by_month


# ---------- explanations ----------

def explain(model_key, fitted, d, idx, s=0):
    """Attributions for the test rows idx with the section 8.1 settings. Returns values, base, output."""
    from model_evaluation.explain_checks import (anomaly_depth, background, max_evals_for,
                                                 permutation_explain, tree_explain)
    bg = background(d["Ztr"], s)
    X = d["Zte"][idx]
    if model_key == "if":
        v, b = tree_explain(fitted["if"].model, bg, X)
        out = anomaly_depth(fitted["if"].model, X)
    else:
        v, b = permutation_explain(fitted["eif"].score, bg, X, s, max_evals_for(d["Ztr"].shape[1]))
        out = fitted["eif"].score(X)
    return v, b, out


def explanation_table(model_key, fitted, d, te, thr, ids):
    """Flagged test contracts of one model, highest score first, with an exact repeat on the first 20."""
    from model_evaluation.explain_checks import check_additivity, check_target
    idx = np.flatnonzero(te[model_key] > thr[model_key])
    idx = idx[np.argsort(-te[model_key][idx], kind="stable")]
    t0 = time.perf_counter()
    v, b, out = explain(model_key, fitted, d, idx)
    secs = time.perf_counter() - t0
    rep = {"method": EXPLAINER[model_key], "n_explained": len(idx), "seconds": secs}
    if len(idx):
        v2, b2, _ = explain(model_key, fitted, d, idx[:N_REPEAT])
        rep["exact_repeat_first_20"] = bool(np.array_equal(v[:N_REPEAT], v2) and np.array_equal(b[:N_REPEAT], b2))
        rep["additivity"] = check_additivity(v, b, out)
        rep["target"] = check_target(out, te[model_key][idx]) if len(idx) > 2 else {"passed": True}
    df = pd.DataFrame(v, columns=[f"attr__{n}" for n in d["names"]])
    df.insert(0, "base", b)
    df.insert(0, "explained_output", out)
    df.insert(0, "ranking_score", te[model_key][idx])
    df.insert(0, "position", idx)
    df.insert(0, "contract_id", np.asarray(ids)[idx])
    return df, rep


# ---------- run ----------

def write_log(out, entry):
    p = out / RUN_LOG
    log = json.loads(p.read_text()) if p.exists() else {"attempts": []}
    log["attempts"].append(entry)
    p.write_text(json.dumps(log, indent=2, default=str))


def run(args):
    mode = "dry" if args.dry_run else "real"
    out = OUTS[mode]
    freeze_path = (Path(args.freeze) if args.freeze else FREEZE).resolve()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    freeze_commit = None

    # 1. gate, before any data is loaded
    require(VALIDATION.exists(), "section 8 results missing")
    require(freeze_path.exists(), f"freeze file missing: {freeze_path}")
    fz = check_freeze(json.loads(freeze_path.read_text()), json.loads(VALIDATION.read_text()))
    if mode == "real":
        require(freeze_path == FREEZE.resolve(), "the real run reads only model_evaluation/model_freeze.json")
        head, freeze_commit = check_committed(freeze_path)
        prior = json.loads((out / RUN_LOG).read_text())["attempts"] if (out / RUN_LOG).exists() else []
        status = [a.get("status") for a in prior]
        require("completed" not in status, "a final run already exists; the test period is scored once")
        require(not status or status[-1] == "stopped_integrity",
                "an earlier attempt did not finish; it may have scored the test period, so no automatic rerun")
        require(not status or args.after_integrity_stop,
                "an earlier attempt stopped on an integrity check; rerun only with --after-integrity-stop "
                "after documenting the fix in the deviations log")
        require(PHASE2_REF.exists(), "Phase 2 reference scores missing")
        out.mkdir(parents=True, exist_ok=True)
        write_log(out, {"started_utc": started, "status": "started", "git_commit": head,
                        "freeze_commit": freeze_commit})
    else:
        require(not out.exists(), f"output exists: {rel(out)}")

    stage = "before_new_results"
    try:
        # 2 and 3. data, fit, thresholds
        if mode == "real":
            Xtr, Xte = load_periods(ROOT)
            d = prepare(Xtr, Xte)
            ids = Xte["main_ocid"].astype(str).to_numpy()
        else:
            from model_evaluation.fold5 import fold5
            from model_evaluation.harness import load_training_period
            d = fold5(load_training_period(ROOT))
            d["Fte"], d["Zte"] = d.pop("Fva"), d.pop("Zva")
            ids = d["Fte"]["ocid"].astype(str).to_numpy()
        print(f"{'TEST PERIOD' if mode == 'real' else 'DRY RUN (training period only)'}: "
              f"train {len(d['Ztr']):,} | scored {len(d['Zte']):,}")
        fitted, tr, thr, refs = fit_train(d, make_models())
        te = {"if": score_if(d, fitted)}

        # 4. Phase 2 reproduction (real run only), before any new test result exists
        if mode == "real":
            require(check_phase2(pd.read_csv(PHASE2_REF), te["if"], thr["if"], ids),
                    "the frozen IF does not reproduce the Phase 2 test scores and flags")
            print(f"IF reproduces the Phase 2 test scores and its {EXPECTED['n_flagged_phase2']:,} flags")
        stage = "new_results"
        te = score_new(d, fitted, refs, te)
    except Stop as e:
        if mode == "real":  # a stop after new results blocks any rerun
            write_log(out, {"status": "stopped_integrity" if stage == "before_new_results" else "stopped_after_scoring",
                            "reason": str(e),
                            "stopped_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        raise

    # 5. report
    from src.tuning import DATE
    rep, by_month = build_report(d, tr, te, thr, fitted, refs, DATE)

    # 6. explanations
    expl, expl_rep = [], {}
    for m in fz["explained_models"]:
        print(f"explaining {m}: {int((te[m] > thr[m]).sum()):,} flagged contracts")
        df, r = explanation_table(m, fitted, d, te, thr, d["Fte"]["ocid"].astype(str).to_numpy())
        df.insert(1, "model", m)
        expl.append(df)
        expl_rep[m] = r
        print(f"  {r['seconds'] / 60:.1f} min | additivity {r.get('additivity', {}).get('passed')} | "
              f"repeat {r.get('exact_repeat_first_20')}")

    # 7. save
    from dashboard.bundle import contract_table, history_tables
    from src.features import normalise_names
    resolved = d["ib"].resolver_.transform(normalise_names(d["Fte"]["supplier_name"]))
    contracts = contract_table(d["Fte"], resolved, {m: te[m] for m in ("if", "eif", "rrf")},
                               {m: tr[m] for m in ("if", "eif", "rrf")}, thr)
    buyers, cats = history_tables(d["Ftr"])
    out.mkdir(parents=True, exist_ok=True)
    contracts.to_csv(out / "contracts.csv.gz", index=False, compression="gzip")
    pd.DataFrame({m: tr[m] for m in ("if", "eif", "rrf", "R")}).to_csv(out / "train_scores.csv.gz", index=False,
                                                                        compression="gzip")
    empty = pd.DataFrame(columns=["contract_id", "model", "position", "ranking_score", "explained_output", "base"])
    (pd.concat(expl, ignore_index=True) if expl else empty).to_csv(
        out / "explanations.csv.gz", index=False, compression="gzip")
    buyers.to_csv(out / "buyers.csv.gz", index=False, compression="gzip")
    cats.to_csv(out / "categories.csv.gz", index=False, compression="gzip")
    by_month.to_csv(out / "flagged_by_month.csv", index=False)
    record = {"mode": mode, "started_utc": started, "freeze": fz, "freeze_file": str(rel(freeze_path)),
              "freeze_commit": freeze_commit, "git_commit": git("rev-parse", "HEAD").stdout.strip() or None,
              "inputs": d["names"], "report": rep,
              "explanations": expl_rep, "rrf_k": K, "seed": 0,
              "versions": {"python": platform.python_version(),
                           **{p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "isotree", "shap")}}}
    (out / "final_report.json").write_text(json.dumps(record, indent=2, default=str))
    if mode == "real":
        write_log(out, {"status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    print("\nshare above the training threshold:", {m: round(v, 4) for m, v in rep["share_above_threshold"].items()})
    print("flagged:", rep["n_flagged"])
    print("spearman:", {k: round(v, 3) for k, v in rep["spearman"].items()})
    print("injection recall (training threshold):",
          {k: {m: round(v[m]["vs_training_threshold"], 3) for m in ("if", "eif", "rrf")}
           for k, v in rep["injection_recall"].items()})
    print("saved", rel(out))
    from dashboard.build_final_bundle import build
    return build(mode)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--freeze")
    p.add_argument("--after-integrity-stop", action="store_true")
    args = p.parse_args(argv)
    try:
        return run(args)
    except Stop as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
