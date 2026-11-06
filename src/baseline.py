import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
NAME = "baseline_iforest_400_256"
SETTINGS = dict(n_estimators=400, max_samples=256, random_state=0)  # selected in Phase 2
TOP = 0.95  # top-5% cut-off on training scores

# reproduction targets from the submitted Phase 2 run; checked, never tuned against
EXPECTED = dict(n_train=72_585, n_test=36_399, n_flagged=1_131)
REF_COLS = ["main_ocid", "anomaly_score", "top5"]
ID = "main_ocid"


class CheckFailed(Exception):
    pass


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise CheckFailed(msg)


def output_paths():
    return {"model": ROOT / "models" / f"{NAME}.joblib",
            "meta": ROOT / "models" / f"{NAME}.json",
            "scores": ROOT / "results" / "phase3" / "baseline_test_scores.csv"}


def fit_baseline(X_train):
    """Same fit as notebook step 8c. Returns the pipeline, the frozen threshold and train scores."""
    pipe = build_pipeline(**SETTINGS).fit(X_train)
    s_tr = -pipe.score_samples(X_train)
    return pipe, float(np.quantile(s_tr, TOP)), s_tr


def load_baseline(path):
    obj = joblib.load(path)
    return obj["pipeline"], obj["threshold"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        # tracked files only; new outputs in models/ and results/ do not count
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except Exception:
        return None, None


def versions():
    import rapidfuzz, scipy, sklearn
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "rapidfuzz": rapidfuzz.__version__, "joblib": joblib.__version__}


def guard_outputs(overwrite):
    """Refuse to run if outputs exist, unless --overwrite, which moves them aside first."""
    existing = [p for p in output_paths().values() if p.exists()]
    if not existing:
        return
    if not overwrite:
        for p in existing:
            print("exists:", p.relative_to(ROOT))
        raise CheckFailed("outputs already exist; rerun with --overwrite to move them to models/superseded/")
    backup = ROOT / "models" / "superseded" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup.mkdir(parents=True)
    for p in existing:
        shutil.move(str(p), str(backup / p.name))
        print("moved", p.relative_to(ROOT), "->", (backup / p.name).relative_to(ROOT))


def run(overwrite=False):
    from src.data import load_merged_table, parse_dates, temporal_split

    # 1. stop early if anything would be overwritten or the reference is missing
    guard_outputs(overwrite)
    ref_path = ROOT / "results" / "test_scores.csv"
    check(ref_path.exists(), f"Phase 2 reference exists: {ref_path.relative_to(ROOT)}")
    ref = pd.read_csv(ref_path)
    missing = [c for c in REF_COLS if c not in ref.columns]
    check(not missing, f"reference has columns {REF_COLS} (missing: {missing or 'none'})")
    check(ref["top5"].dtype == bool, "reference top5 column is boolean")
    check(len(ref) == EXPECTED["n_test"], f"reference rows {len(ref):,} == {EXPECTED['n_test']:,}")
    check(int(ref["top5"].sum()) == EXPECTED["n_flagged"],
          f"reference flagged {int(ref['top5'].sum()):,} == {EXPECTED['n_flagged']:,}")

    # 2. data and split, exactly as notebook cell 0
    data = ROOT / "data" / "raw" / "full"
    if not data.exists():
        data = ROOT / "data" / "raw" / "kenya_ppra_full.csv.tar.gz"
    check(data.exists(), f"data found: {data.relative_to(ROOT)}")
    files = sorted(data.glob("*.csv")) if data.is_dir() else [data]
    df = parse_dates(load_merged_table(data))
    aw, train = temporal_split(df)
    X_train, X_test = aw[train], aw[~train]
    check(len(X_train) == EXPECTED["n_train"], f"train rows {len(X_train):,} == {EXPECTED['n_train']:,}")
    check(len(X_test) == EXPECTED["n_test"], f"test rows {len(X_test):,} == {EXPECTED['n_test']:,}")
    check(ID in X_test.columns, f"test data has id column {ID}")

    # 3. fit on train only; threshold from training scores only
    pipe, thr, s_tr = fit_baseline(X_train)
    s_te = -pipe.score_samples(X_test)
    flagged = s_te > thr
    print(f"threshold {thr:.6f} | train above {(s_tr > thr).mean():.1%} | test above {int(flagged.sum()):,}")

    # 4. reproduction checks against Phase 2, before anything is written
    check(int(flagged.sum()) == EXPECTED["n_flagged"], f"test flagged {int(flagged.sum()):,} == {EXPECTED['n_flagged']:,}")
    check(np.allclose(np.sort(ref["anomaly_score"].to_numpy()), np.sort(s_te), rtol=0, atol=1e-12),
          "test scores identical to Phase 2 (sorted, tolerance 1e-12)")
    check(sorted(ref.loc[ref["top5"], ID]) == sorted(X_test.loc[flagged, ID]),
          "flagged OCIDs identical to Phase 2")

    # 5. write to temporary names, reload, then rename into place
    paths = output_paths()
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = {k: p.with_name(p.name + ".tmp") for k, p in paths.items()}
    joblib.dump({"pipeline": pipe, "threshold": thr, "settings": SETTINGS}, tmp["model"])
    pipe2, thr2 = load_baseline(tmp["model"])
    check(np.array_equal(-pipe2.score_samples(X_test), s_te), "reloaded model gives identical test scores")
    check(thr2 == thr, "reloaded threshold identical")

    commit, dirty = git_state()
    meta = {
        "name": NAME, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": SETTINGS, "threshold": thr, "threshold_rule": "95th percentile of training scores",
        "score": "-pipeline.score_samples(X), higher = more anomalous",
        "split": "tender_tenderPeriod_endDate year <= 2023 train, 2024 to 2026 test",
        "n_train": len(X_train), "n_test": len(X_test), "n_test_above_threshold": int(flagged.sum()),
        "train_median_score": float(np.median(s_tr)), "test_median_score": float(np.median(s_te)),
        "inputs": [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()],
        "reproduces_phase2": True, "phase2_reference_sha256": sha256(ref_path),
        "git_commit": commit, "git_dirty": dirty, "versions": versions(),
        "data_files": {f.name: sha256(f) for f in files},
        "artifact": paths["model"].name, "artifact_sha256": sha256(tmp["model"]),
    }
    tmp["meta"].write_text(json.dumps(meta, indent=2))
    # row_id is the index of aw, so later steps can join back to the contract record
    (pd.DataFrame({"row_id": X_test.index, ID: X_test[ID].to_numpy(), "anomaly_score": s_te, "top5": flagged})
       .sort_values("anomaly_score", ascending=False)
       .to_csv(tmp["scores"], index=False))
    for k in paths:
        tmp[k].replace(paths[k])
        print("saved", paths[k].relative_to(ROOT))
    print("baseline frozen and verified")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Freeze and verify the Phase 2 Isolation Forest baseline.")
    ap.add_argument("--overwrite", action="store_true",
                    help="move existing outputs to models/superseded/<timestamp>/ before writing")
    args = ap.parse_args(argv)
    try:
        run(overwrite=args.overwrite)
    except CheckFailed as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
