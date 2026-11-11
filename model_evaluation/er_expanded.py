"""Expanded entity-resolution evaluation (protocol 7.10). Training-period supplier names only.

Run from the repo root:
  python -m model_evaluation.er_expanded sample         # batch 1: 40 pairs per stratum (160)
  python -m model_evaluation.er_expanded sample --batch 2   # only if batch 1 gives under 141 decidable
  python -m model_evaluation.er_expanded evaluate       # after labelling

Fixed before any new label exists:
- candidates: Option A core pairs (core score 85+), as in Phase 2 notebook cell 21;
- strata: 85-90, 90-95, 95-100, exact core; development and Phase 2 held-out pairs excluded;
- each stratum's eligible pairs are shuffled once with seed 2 (Phase 2 used 0 and 1);
  batch 1 takes positions 0-39 of every stratum, batch b > 1 takes the next 5 per stratum;
- the file to label hides stratum, core score and original score; pair order is shuffled;
- the resolver is not changed; merge rule as in Phase 2: core score 95+ (90 and 100 as sensitivity).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.entity_resolution import evaluate

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels"
OUT = ROOT / "results" / "model_evaluation"
SEED = 2
FIRST, NEXT = 40, 5
MIN_DECIDABLE = 141
STRATA = ["85-90", "90-95", "95-100", "exact core"]
FILES = {
    "dev": LABELS / "entity_pairs_labelled.csv",
    "fresh": LABELS / "entity_pairs_fresh_labelled.csv",
    "fresh_sizes": LABELS / "entity_pair_fresh_strata.csv",
    "to_label": LABELS / "entity_pairs_new_to_label.csv",
    "key": LABELS / "entity_pairs_new_key.csv",
    "new_sizes": LABELS / "entity_pair_new_strata.csv",
    "labelled": LABELS / "entity_pairs_new_labelled.csv",
    "result": OUT / "entity_resolution_expanded.json",
}
BLIND_COLS = ["pair_id", "name_a", "name_b", "awards_a", "awards_b", "label"]


def pair_keys(df):
    return [frozenset(p) for p in zip(df["name_a"], df["name_b"])]


def eligible(pairs, exclude_frames):
    seen = set().union(*[set(pair_keys(f)) for f in exclude_frames])
    return pairs[[k not in seen for k in pair_keys(pairs)]].copy()


def shuffled_positions(pairs, seed=SEED):
    """Position of each pair in its stratum's fixed shuffled order."""
    p = pairs.sort_values(["stratum", "name_a", "name_b"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pos = np.empty(len(p), dtype=int)
    for s in STRATA:
        idx = np.flatnonzero(p["stratum"].to_numpy() == s)
        pos[idx[rng.permutation(len(idx))]] = np.arange(len(idx))
    return p.assign(position=pos)


def batch_range(b):
    return (0, FIRST) if b == 1 else (FIRST + NEXT * (b - 2), FIRST + NEXT * (b - 1))


def draw_batch(pos_pairs, b, first_id, seed=SEED):
    lo, hi = batch_range(b)
    s = pos_pairs[(pos_pairs["position"] >= lo) & (pos_pairs["position"] < hi)]
    s = s.sample(frac=1, random_state=seed + b).reset_index(drop=True)  # mix strata
    s.insert(0, "pair_id", range(first_id, first_id + len(s)))
    s["batch"] = b
    s["label"] = ""
    return s


def exact_ci(k, n):
    """Two-sided 95% Clopper-Pearson interval."""
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def read_labels(path):
    lab = pd.read_csv(path, dtype={"label": str})
    lab["label"] = lab["label"].fillna("").str.strip()
    return lab


def summarise(lab, sizes, name):
    """Phase 2 metrics on decidable pairs: weighted precision, recall, false-match rate, exact CIs."""
    dec = lab[lab["label"].isin(["0", "1"])]
    _, table, n = evaluate(dec.rename(columns={"core_score": "score"}), sizes,
                           thresholds=(90, 95, 100), use_acronym=False)
    exact = {}
    for part, m in [("identical cores", dec["core_score"] == 100),
                    ("fuzzy core 95 to 99", dec["core_score"].between(95, 99))]:
        k, nn = int((dec.loc[m, "label"] == "1").sum()), int(m.sum())
        lo, hi = exact_ci(k, nn) if nn else (np.nan, np.nan)
        exact[part] = {"same_business": k, "pairs": nn, "precision": k / nn if nn else None,
                       "ci95_low": lo, "ci95_high": hi}
    return {"sample": name, "decidable_pairs": int(n),
            "per_stratum": dec["stratum"].value_counts().reindex(STRATA, fill_value=0).to_dict(),
            "metrics": table.to_dict("records"), "exact_precision": exact}


def check_population(pairs, dev, fresh_sizes):
    """Phase 2 population (dev pairs excluded) must be reproduced exactly."""
    sizes = eligible(pairs, [dev])["stratum"].value_counts()
    return sizes.reindex(fresh_sizes.index).fillna(-1).astype(int).equals(fresh_sizes.astype(int))


def load_pairs():
    from model_evaluation.harness import load_training_period
    from src.entity_resolution import core_pairs
    from src.features import normalise_names
    X = load_training_period(ROOT)
    volume = normalise_names(X["supplier_name"]).dropna().value_counts()
    pairs = core_pairs(volume.index, min_score=85)
    pairs["awards_a"] = pairs["name_a"].map(volume).fillna(0).astype(int)
    pairs["awards_b"] = pairs["name_b"].map(volume).fillna(0).astype(int)
    return pairs


def cmd_sample(b):
    if b == 1 and any(FILES[k].exists() for k in ("to_label", "key", "new_sizes")):
        print("STOPPED: batch 1 files already exist", file=sys.stderr)
        return 1
    if b > 1 and not FILES["key"].exists():
        print("STOPPED: draw batch 1 first", file=sys.stderr)
        return 1
    dev, fresh = pd.read_csv(FILES["dev"]), pd.read_csv(FILES["fresh"])
    fresh_sizes = pd.read_csv(FILES["fresh_sizes"], index_col=0).iloc[:, 0]
    pairs = load_pairs()
    if not check_population(pairs, dev, fresh_sizes):
        print("STOPPED: candidate pairs do not reproduce the Phase 2 strata sizes", file=sys.stderr)
        return 1
    missing = set(pair_keys(fresh)) - set(pair_keys(pairs))
    if missing:
        print(f"STOPPED: {len(missing)} Phase 2 held-out pairs not among the candidates", file=sys.stderr)
        return 1
    pos = shuffled_positions(eligible(pairs, [dev, fresh]))
    if b == 1:
        first_id = 1
    else:
        key = pd.read_csv(FILES["key"])
        if b in set(key["batch"]) or b != key["batch"].max() + 1:
            print(f"STOPPED: batch {b} is not the next batch", file=sys.stderr)
            return 1
        first_id = int(key["pair_id"].max()) + 1
    s = draw_batch(pos, b, first_id)
    lo, hi = batch_range(b)
    if len(s) != (hi - lo) * len(STRATA):
        print("STOPPED: a stratum ran out of pairs", file=sys.stderr)
        return 1
    blind, key = s[BLIND_COLS], s[["pair_id", "stratum", "core_score", "orig_score", "batch"]]
    if b == 1:
        blind.to_csv(FILES["to_label"], index=False)
        key.to_csv(FILES["key"], index=False)
        pos["stratum"].value_counts().reindex(STRATA).rename("pairs_in_stratum").to_csv(FILES["new_sizes"])
    else:
        blind.to_csv(FILES["to_label"], mode="a", header=False, index=False)
        key.to_csv(FILES["key"], mode="a", header=False, index=False)
    print(f"batch {b}: {len(s)} pairs added to {FILES['to_label'].relative_to(ROOT)} "
          f"(columns: {', '.join(BLIND_COLS)}). Do not open {FILES['key'].name} until labelling is finished.")
    return 0


def cmd_evaluate():
    if FILES["result"].exists():
        print("STOPPED: result exists:", FILES["result"].relative_to(ROOT), file=sys.stderr)
        return 1
    blind, key = pd.read_csv(FILES["to_label"]), pd.read_csv(FILES["key"])
    lab = read_labels(FILES["labelled"])
    if not lab[BLIND_COLS[:-1]].equals(blind[BLIND_COLS[:-1]]):
        print("STOPPED: labelled file does not match the file to label (rows or names changed)", file=sys.stderr)
        return 1
    bad = set(lab["label"]) - {"0", "1", ""}
    if bad:
        print(f"STOPPED: unexpected label values {bad}; use 1, 0 or blank", file=sys.stderr)
        return 1
    new = lab.merge(key, on="pair_id", validate="one_to_one")
    n_dec = int(new["label"].isin(["0", "1"]).sum())
    if n_dec < MIN_DECIDABLE:
        print(f"STOPPED: {n_dec} decidable pairs, under {MIN_DECIDABLE}. "
              f"Draw the next batch with: python -m model_evaluation.er_expanded sample --batch "
              f"{int(key['batch'].max()) + 1}", file=sys.stderr)
        return 1
    fresh = read_labels(FILES["fresh"])
    new_sizes = pd.read_csv(FILES["new_sizes"], index_col=0).iloc[:, 0]
    fresh_sizes = pd.read_csv(FILES["fresh_sizes"], index_col=0).iloc[:, 0]
    cols = ["stratum", "core_score", "label"]
    result = {"protocol_section": "7.10", "rule": "Option A, core score 95+ (90 and 100 sensitivity)",
              "labels": new["label"].replace("", "blank").value_counts().to_dict(),
              "batches": sorted(int(b) for b in key["batch"].unique()),
              "new_alone": summarise(new[cols], new_sizes, "new pairs alone"),
              # one stratified sample without replacement from the Phase 2 population
              "combined": summarise(pd.concat([fresh[cols], new[cols]], ignore_index=True), fresh_sizes,
                                    "Phase 2 held-out plus new pairs"),
              "limitation": "single labeller"}
    OUT.mkdir(parents=True, exist_ok=True)
    FILES["result"].write_text(json.dumps(result, indent=2, default=float))
    for part in ("new_alone", "combined"):
        r = result[part]
        print(f"\n{r['sample']}: {r['decidable_pairs']} decidable pairs")
        print(pd.DataFrame(r["metrics"]).to_string(index=False))
        for k, v in r["exact_precision"].items():
            print(f"  {k}: {v['same_business']}/{v['pairs']}, exact 95% CI {v['ci95_low']:.2f} to {v['ci95_high']:.2f}")
    print("\nsaved", FILES["result"].relative_to(ROOT))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sample", "evaluate"])
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args(argv)
    return cmd_sample(a.batch) if a.command == "sample" else cmd_evaluate()


if __name__ == "__main__":
    sys.exit(main())
