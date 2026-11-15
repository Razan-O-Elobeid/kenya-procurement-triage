"""Build the final auditor dashboard bundle from saved final-evaluation results.

Called at the end of model_evaluation.final_eval. Can be rerun on its own (it never fits or scores):
    python -m dashboard.build_final_bundle            # from results/final_test/
    python -m dashboard.build_final_bundle --dry-run  # from results/final_test_dryrun/

Checks before writing: the run completed; flags equal score > threshold for every model; ranks match
the scores; attributions pass dashboard.bundle.check_explanations (positions, ids, scores within 1e-12,
input order, additivity), cover every flagged contract exactly once, and passed the final run's checks. Failed attributions are not shown;
the scores still are. Wording follows protocol 8.1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.bundle import check_explanations, write_bundle
from model_evaluation.fold5 import rank_desc

ROOT = Path(__file__).resolve().parents[1]
SRC = {"real": ROOT / "results" / "final_test", "dry": ROOT / "results" / "final_test_dryrun"}
OUT = {"real": ROOT / "results" / "dashboard" / "final_test", "dry": ROOT / "results" / "dashboard" / "final_dryrun"}
VALIDATION = ROOT / "results" / "model_evaluation" / "explanations" / "explain_validation.json"
VALIDATION_KEY = {"if": "if_tree", "eif": "eif_perm"}
NAMES = {"if": "Isolation Forest 400 / 256", "eif": "Extended Isolation Forest 400 / 256 / ndim 2"}


def rel(p):
    """Path relative to the repo, safe with symlinked folders such as /var on macOS."""
    return Path(p).resolve().relative_to(ROOT.resolve())


class Stop(Exception):
    pass


def require(cond, msg):
    if not cond:
        raise Stop(msg)


def status_text(mode, fz):
    """Banner, mode and deployment status (protocol 8.1)."""
    dep = fz.get("deployed_model")
    if mode == "dry":
        return ("development", "Dry run: fold 5 validation contracts from the training period (2018 to 2023), "
                "standing in for the test period. Not test data and not a deployment.",
                "dry run: no deployed model")
    if dep:
        return ("deployment", "Final held-out evaluation: test-period contracts (2024 to Aug 2026), scored once "
                "with the models frozen on 19 Nov 2026. Scores rank contracts for review; a high score is not "
                "evidence of wrongdoing.", f"deployed model: {NAMES[dep]} (frozen, validated explanation)")
    lead = fz["scientific_leader"]
    return ("prototype", "Local auditor prototype: no validated model deployment was achieved. The queue shows "
            f"the frozen scores of the best scientifically ranked family ({NAMES[lead]}). Test-period contracts "
            "(2024 to Aug 2026), scored once.", "local auditor prototype: no validated model deployment")


def check_scores(C, thr):
    for m, t in thr.items():
        s = C[f"score_{m}"].to_numpy(dtype=float)
        require(np.array_equal(C[f"flag_{m}"].astype(bool).to_numpy(), s > t), f"{m}: flags differ from score > threshold")
        require(np.array_equal(C[f"rank_{m}"].to_numpy(), rank_desc(s)), f"{m}: ranks differ from the scores")


def build(mode="real"):
    src, out = SRC[mode], OUT[mode]
    try:
        require(not out.exists(), f"bundle exists: {rel(out)}")
        rec = json.loads((src / "final_report.json").read_text())
        require(rec["mode"] == mode, "report mode differs")
        if mode == "real":
            log = json.loads((src / "RUN_LOG.json").read_text())["attempts"]
            require(log and log[-1]["status"] == "completed", "the final run did not complete")
        fz, rep = rec["freeze"], rec["report"]
        thr = rep["thresholds"]
        C = pd.read_csv(src / "contracts.csv.gz", low_memory=False)
        require(len(C) == rep["n_test"], "contract count differs from the report")
        check_scores(C, thr)
        validation = json.loads(VALIDATION.read_text())
        E = pd.read_csv(src / "explanations.csv.gz")
        names = rec["inputs"]

        rows, reports = [], {}
        for m in ("if", "eif"):
            saved = E[E["model"] == m].drop(columns="model").reset_index(drop=True)
            fr = rec["explanations"].get(m)
            if fr is None or saved.empty:
                reports[m] = {"model": m, "passed": False, "reason": "not explained in the final run",
                              "section8_validated": validation.get(VALIDATION_KEY[m], {}).get("validated", False)}
            else:
                score = C[f"score_{m}"].to_numpy(dtype=float)
                if m == "eif":  # EIF explains its own score
                    output = score
                else:  # IF explains -E[h(x)], checked against the score's ranks in the final run
                    output = np.full(len(C), np.nan)
                    output[saved["position"].to_numpy()] = saved["explained_output"].to_numpy()
                r, rp = check_explanations(saved, C, m, names, score, output,
                                           validation[VALIDATION_KEY[m]]["validated"], pos_col="position")
                flagged_pos = np.flatnonzero(C[f"flag_{m}"].astype(bool).to_numpy())
                if r is not None and not np.array_equal(np.sort(saved["position"].to_numpy()), flagged_pos):
                    r, rp = None, {**rp, "passed": False,
                                   "reason": "coverage: flagged contracts and explained contracts differ"}
                final_ok = (fr.get("exact_repeat_first_20") is True and fr.get("additivity", {}).get("passed") is True
                            and fr.get("target", {}).get("passed") is True)
                if r is not None and not final_ok:
                    r, rp = None, {**rp, "passed": False,
                                   "reason": "exact repeat, additivity or target check failed in the final run"}
                if r is not None:
                    rows.append(r)
                flagged = set(C.loc[C[f"flag_{m}"], "row_id"])
                rp.update(method=fr["method"], n_flagged=len(flagged),
                          n_flagged_with_attribution=len(flagged & set(r["row_id"])) if r is not None else 0)
                reports[m] = rp
            print(f"{m}: attributions {'shown' if reports[m]['passed'] else 'NOT shown'} ({reports[m]['reason']})")
        reports["rrf"] = {"model": "rrf", "passed": False, "section8_validated": False,
                          "reason": "no explanation method for RRF (section 8); reporting only"}
        explanations = (pd.concat(rows, ignore_index=True) if rows
                        else pd.DataFrame(columns=["row_id", "model", "base", "explained_output"]))

        bundle_mode, banner, status = status_text(mode, fz)
        rule = "95th percentile of training-period scores" if mode == "real" else "95th percentile of fold 5 training scores"
        manifest = {
            "bundle": out.name, "mode": bundle_mode, "banner": banner, "deployment_status": status,
            "created_from": str(rel(src)), "freeze_commit": rec.get("freeze_commit"),
            "git_commit": rec.get("git_commit"),
            "contracts": (f"test period, 2024 to Aug 2026 ({len(C):,} contracts)" if mode == "real"
                          else f"fold 5 validation, training period ({len(C):,} contracts)"),
            "history_reference": ("full training period, 2018 to 2023" if mode == "real"
                                  else "fold 5 training part only"),
            "queue_models": fz["queue_models"],
            "models": {m: {"threshold": thr[m], "threshold_rule": rule if m != "rrf" else rule.replace("scores", "RRF scores"),
                           "n_flagged": int(C[f"flag_{m}"].sum())} for m in ("if", "eif", "rrf")},
            "explanations": reports, "inputs": names, "versions": rec["versions"],
        }
        tables = {"contracts.csv.gz": C, "explanations.csv.gz": explanations,
                  "buyers.csv.gz": pd.read_csv(src / "buyers.csv.gz"),
                  "categories.csv.gz": pd.read_csv(src / "categories.csv.gz")}
        write_bundle(out, tables, manifest)
        print("saved", rel(out))
        return 0
    except Stop as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    sys.exit(build("dry" if p.parse_args().dry_run else "real"))
