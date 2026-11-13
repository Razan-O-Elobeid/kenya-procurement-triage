"""Training-period drift check (protocol 7.9(a)). Descriptive; changes no model, feature or threshold.

Run from the repo root:  python -m model_evaluation.drift_eval

For each of the five validation folds (training period only):
  1. share of clean validation contracts above the 95th percentile of the fold's own training
     scores, for the frozen baseline (IF 400 / 256, seed 0) and the selected EIF (400 / 256 / ndim 2, seed 0);
  2. standardized mean difference (SMD) of each model input, fold training part against validation fold;
  3. f4 and f5 coverage, the pre-closure share and the f3 distribution, in both parts.
SMD = (validation mean - training mean) / sqrt((training variance + validation variance) / 2).
An input with zero variance in both parts has no SMD (reported as missing).
Writes results/model_evaluation/drift_training_period/ and refuses to overwrite it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_evaluation" / "drift_training_period"
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def flag_share(train_scores, val_scores, q=0.95):
    """Threshold from training scores only; share of validation scores above it."""
    thr = float(np.quantile(train_scores, q))
    return thr, float(np.mean(np.asarray(train_scores) > thr)), float(np.mean(np.asarray(val_scores) > thr))


def smd(a, b):
    """Standardized mean difference of b against a, per column."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (b.mean(axis=0) - a.mean(axis=0)) / pooled
    return np.where(pooled > 0, out, np.nan)


def feature_summary(F):
    """f4 and f5 coverage, pre-closure share, f3 coverage and quantiles."""
    f3 = F["f3_decision_period"].astype(float)
    row = {"n": len(F),
           "f4_coverage": float(F["f4_supplier_concentration"].notna().mean()),
           "f5_coverage": float(F["f5_benford_deviation"].notna().mean()),
           "pre_closure_share": float(F["is_signed_pre_closure"].mean()),
           "f3_coverage": float(f3.notna().mean()),
           "f3_mean": float(f3.mean())}
    for q in QUANTILES:
        row[f"f3_q{int(q * 100):02d}"] = float(f3.quantile(q))
    return row


def run(folds, names, make_models):
    """One pass over the folds. Returns the three tables."""
    shares, smds, feats = [], [], []
    for f in folds:
        Ztr, Zva = np.asarray(f["Ztr"], dtype=float), np.asarray(f["Zva"], dtype=float)
        if not (np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zva)) and Ztr.shape[1] == Zva.shape[1] == len(names)):
            raise ValueError(f"fold {f['fold']}: inputs not finite or misshaped")
        for m, model in make_models().items():
            model.fit(Ztr)
            thr, s_tr, s_va = flag_share(model.score(Ztr), model.score(Zva))
            shares.append({"fold": f["fold"], "model": m, "threshold": thr, "share_train": s_tr,
                           "share_validation": s_va, "n_validation": len(Zva)})
        d = smd(Ztr, Zva)
        for j, n in enumerate(names):
            smds.append({"fold": f["fold"], "input": n, "mean_train": float(Ztr[:, j].mean()),
                         "mean_validation": float(Zva[:, j].mean()), "smd": float(d[j])})
        for part, F in (("train", f["Ftr"]), ("validation", f["Fva"])):
            feats.append({"fold": f["fold"], "part": part, **feature_summary(F)})
    return pd.DataFrame(shares), pd.DataFrame(smds), pd.DataFrame(feats)


def frozen_models():
    from model_evaluation.eif_eval import IsoTreeEIF
    from model_evaluation.harness import SklearnIF
    return {"if": SklearnIF(400, 256, seed=0), "eif": IsoTreeEIF(400, 256, 2, 0)}


def main():
    if OUT.exists():
        print("STOPPED: output exists:", OUT.relative_to(ROOT), file=sys.stderr)
        return 1
    from model_evaluation.harness import build_folds, load_training_period
    from src.pipeline import build_preprocessor
    X = load_training_period(ROOT)
    print(f"training period: {len(X):,} contracts (test period not loaded)")
    folds = build_folds(X)
    names = list(clone(build_preprocessor()).fit(folds[0]["Ftr"]).get_feature_names_out())
    try:
        shares, smds, feats = run(folds, names, frozen_models)
    except ValueError as e:
        print("STOPPED:", e, file=sys.stderr)
        return 1

    print("\nshare of validation contracts above the fold's training 95th percentile (5% expected without drift)")
    print(shares.pivot(index="fold", columns="model", values="share_validation").round(4).to_string())
    print("\nlargest |SMD| per fold (model inputs, validation against training)")
    for fold, g in smds.groupby("fold"):
        top = g.reindex(g["smd"].abs().sort_values(ascending=False).index).head(3)
        print(f"  fold {fold}: " + "; ".join(f"{r.input} {r.smd:+.2f}" for r in top.itertuples()))
    print("\nf4 and f5 coverage, pre-closure share, f3 median")
    print(feats.set_index(["fold", "part"])[["f4_coverage", "f5_coverage", "pre_closure_share", "f3_q50"]]
          .round(3).to_string())

    OUT.mkdir(parents=True)
    shares.to_csv(OUT / "flag_share.csv", index=False)
    smds.to_csv(OUT / "smd.csv", index=False)
    feats.to_csv(OUT / "features.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "protocol_section": "7.9(a)", "models": {"if": "IF 400 / 256, seed 0", "eif": "EIF 400 / 256 / ndim 2, seed 0"},
        "threshold_rule": "95th percentile of each fold's training scores",
        "mean_share_validation": shares.groupby("model")["share_validation"].mean().round(4).to_dict(),
        "inputs": names}, indent=2))
    print("\nsaved", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
