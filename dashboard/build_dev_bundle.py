"""Build the development dashboard bundle from fold 5 (training period only).

Run from the repo root:  python -m dashboard.build_dev_bundle
Needs results/model_evaluation/explanations/ from model_evaluation.explain_eval.

Contracts: fold 5 validation (12,097). Models: IF 400 / 256 and EIF 400 / 256 / ndim 2, seed 0,
fitted on fold 5's training part. Threshold: 95th percentile of each model's fold 5 training scores.
Historical reference: fold 5's training part only. Attributions are reused from the section 8 run,
only after the integrity checks in dashboard.bundle.check_explanations pass.
This bundle is for development. It is not a deployment and no model is selected by it.
"""
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.bundle import check_explanations, contract_table, history_tables, write_bundle
from model_evaluation.explain_checks import anomaly_depth
from model_evaluation.fold5 import fit_models, fold5
from model_evaluation.harness import load_training_period
from src.baseline import git_state
from src.features import normalise_names

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dashboard" / "dev_fold5"
EXPL = ROOT / "results" / "model_evaluation" / "explanations"
SAVED = {"if": ("if_tree_s0.csv", "if_tree", "TreeExplainer (interventional), explains -E[h(x)]"),
         "eif": ("eif_perm_s0.csv", "eif_perm", "permutation SHAP, explains the EIF score")}
BANNER = ("Development data: fold 5 validation contracts from the training period (2018 to 2023). "
          "Not a deployment. No model has been selected; selection is frozen on 19 Nov.")


def main():
    if OUT.exists():
        print("STOPPED: bundle exists:", OUT.relative_to(ROOT), file=sys.stderr)
        return 1
    validation = json.loads((EXPL / "explain_validation.json").read_text())
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    d = fold5(X)
    models = fit_models(d["Ztr"])
    scores = {m: mod.score(d["Zva"]) for m, mod in models.items()}
    train_scores = {m: mod.score(d["Ztr"]) for m, mod in models.items()}
    thresholds = {m: float(np.quantile(s, 0.95)) for m, s in train_scores.items()}
    resolved = d["ib"].resolver_.transform(normalise_names(d["Fva"]["supplier_name"]))
    contracts = contract_table(d["Fva"], resolved, scores, train_scores, thresholds)
    outputs = {"if": anomaly_depth(models["if"].model, d["Zva"]), "eif": scores["eif"]}

    expl, reports = [], {}
    for m, (fname, key, method) in SAVED.items():
        saved = pd.read_csv(EXPL / fname)
        rows, rep = check_explanations(saved, contracts, m, d["names"], scores[m], outputs[m],
                                       validation[key]["validated"])
        rep["method"] = method
        flagged = set(contracts.loc[contracts[f"flag_{m}"], "row_id"])
        have = set(rows["row_id"]) if rows is not None else set()
        rep.update(n_flagged=len(flagged), n_flagged_with_attribution=len(flagged & have))
        reports[m] = rep
        print(f"{m}: integrity {'passed' if rep['passed'] else 'FAILED'} ({rep['reason']}) | "
              f"flagged {len(flagged)}, with attributions {len(flagged & have)}")
        if rows is not None:
            expl.append(rows)
    explanations = (pd.concat(expl, ignore_index=True) if expl
                    else pd.DataFrame(columns=["row_id", "model", "base", "explained_output"]))
    buyers, cats = history_tables(d["Ftr"])
    commit, dirty = git_state()
    manifest = {
        "bundle": OUT.name, "mode": "development", "banner": BANNER,
        "deployment_status": "development bundle: no deployed model",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "contracts": f"fold 5 validation, training period ({len(contracts):,} contracts)",
        "history_reference": f"fold 5 training part only ({len(d['Ftr']):,} contracts)",
        "queue_models": ["if", "eif"],
        "models": {m: {"threshold": thresholds[m], "threshold_rule": "95th percentile of fold 5 training scores",
                       "n_flagged": int(contracts[f"flag_{m}"].sum())} for m in models},
        "explanations": reports, "inputs": d["names"],
        "versions": {"python": platform.python_version(),
                     **{p: version(p) for p in ("numpy", "pandas", "scikit-learn", "isotree", "shap")}},
    }
    write_bundle(OUT, {"contracts.csv.gz": contracts, "explanations.csv.gz": explanations,
                       "buyers.csv.gz": buyers, "categories.csv.gz": cats}, manifest)
    print("saved", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
