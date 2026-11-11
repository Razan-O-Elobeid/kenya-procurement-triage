"""Checks for the expanded entity-resolution sample (protocol 7.10). Run: python tests/test_er_expanded.py

Synthetic pairs only. No project data is loaded.
"""
import sys, pathlib, tempfile, warnings, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import binomtest
warnings.simplefilter("ignore")
import model_evaluation.er_expanded as er

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []

# synthetic candidate pairs: 4 strata, 120 pairs each
rng = np.random.default_rng(0)
rows = []
for s, (lo, hi) in zip(er.STRATA, [(85, 90), (90, 95), (95, 100), (100, 101)]):
    for i in range(120):
        rows.append({"name_a": f"A {s} {i}", "name_b": f"B {s} {i}", "core_score": int(rng.integers(lo, hi)),
                     "orig_score": int(rng.integers(80, 101)), "stratum": s, "awards_a": 1, "awards_b": 2})
pairs = pd.DataFrame(rows)
dev = pairs.sample(30, random_state=1)[["name_a", "name_b"]]
fresh = pairs.drop(dev.index).groupby("stratum").head(15)
fresh = fresh.assign(name_a=fresh["name_b"], name_b=fresh["name_a"])  # reversed order still excluded

# 1 exclusion and fixed order
el = er.eligible(pairs, [dev, fresh])
keys = set(er.pair_keys(el))
results.append(ok(len(el) == 480 - 30 - 60 and not keys & set(er.pair_keys(dev)) and not keys & set(er.pair_keys(fresh)),
                  "development and Phase 2 held-out pairs excluded, in either name order"))
pos = er.shuffled_positions(el)
pos2 = er.shuffled_positions(el.sample(frac=1, random_state=5))
m = pos.merge(pos2, on=["name_a", "name_b"])
results.append(ok((m["position_x"] == m["position_y"]).all(), "positions do not depend on input row order"))
results.append(ok(all(sorted(g) == list(range(len(g))) for _, g in pos.groupby("stratum")["position"]),
                  "each stratum gets positions 0..n-1"))

# 2 batches: 40 per stratum, then 5 more, disjoint and in the fixed order
b1 = er.draw_batch(pos, 1, 1)
b2 = er.draw_batch(pos, 2, b1["pair_id"].max() + 1)
results.append(ok(b1["stratum"].value_counts().eq(40).all() and len(b1) == 160, "batch 1: 40 per stratum (160)"))
results.append(ok(b2["stratum"].value_counts().eq(5).all() and set(b2["position"]) == set(range(40, 45)),
                  "batch 2: next 5 per stratum (positions 40-44)"))
results.append(ok(not set(er.pair_keys(b1)) & set(er.pair_keys(b2)) and b2["pair_id"].min() == 161,
                  "batches disjoint, pair ids continue"))
results.append(ok(not (b1["stratum"].iloc[:40] == b1["stratum"].iloc[0]).all(), "pair order mixes strata"))
results.append(ok(set(er.BLIND_COLS) & {"stratum", "core_score", "orig_score", "position", "batch"} == set(),
                  "file to label hides stratum, scores, position and batch"))
results.append(ok(er.draw_batch(pos, 1, 1)[["name_a", "name_b"]].equals(b1[["name_a", "name_b"]]), "batch 1 reproducible"))

# 3 population check reproduces Phase 2 sizes exactly
sizes_dev_only = er.eligible(pairs, [dev])["stratum"].value_counts().reindex(er.STRATA)
results.append(ok(er.check_population(pairs, dev, sizes_dev_only), "population check passes on matching sizes"))
results.append(ok(not er.check_population(pairs, dev, sizes_dev_only + 1), "population check fails on any difference"))

# 4 exact intervals match scipy
lo, hi = er.exact_ci(37, 40)
ci = binomtest(37, 40).proportion_ci(method="exact")
results.append(ok(np.isclose(lo, ci.low) and np.isclose(hi, ci.high), "Clopper-Pearson interval matches scipy"))
results.append(ok(er.exact_ci(40, 40)[1] == 1.0 and er.exact_ci(0, 40)[0] == 0.0, "interval edges at 0 and 40 of 40"))

# 5 end to end in a temporary folder: sample command flow and evaluate with known labels
with tempfile.TemporaryDirectory() as tmp:
    t = pathlib.Path(tmp)
    er.FILES = {k: t / v.name for k, v in er.FILES.items()}
    er.ROOT = t
    dev.to_csv(er.FILES["dev"], index=False)
    fl = fresh.assign(label=np.where(fresh["stratum"].isin(["95-100", "exact core"]), "1", "0"))
    fl.to_csv(er.FILES["fresh"], index=False)
    sizes_dev_only.rename("pairs_in_stratum").to_csv(er.FILES["fresh_sizes"])
    er.load_pairs = lambda: pairs
    results.append(ok(er.main(["sample"]) == 0, "sample command runs"))
    results.append(ok(er.main(["sample"]) == 1, "batch 1 refused if files exist"))
    blind = pd.read_csv(er.FILES["to_label"])
    results.append(ok(list(blind.columns) == er.BLIND_COLS, "blind file columns as specified"))
    key = pd.read_csv(er.FILES["key"])
    truth = blind.merge(key, on="pair_id")
    # labels: same business when core score 95+, with 3 blanks (undecidable)
    lab = blind.copy()
    lab["label"] = np.where(truth["core_score"] >= 95, "1", "0")
    lab.loc[lab.index[:3], "label"] = ""
    lab.to_csv(er.FILES["labelled"], index=False)
    results.append(ok(er.main(["evaluate"]) == 0, "evaluate runs with 157 decidable pairs"))
    res = json.loads(er.FILES["result"].read_text())
    m95 = {r["metric"]: r["estimate"] for r in res["new_alone"]["metrics"] if r["threshold"] == 95}
    results.append(ok(m95["precision"] == 1.0 and m95["recall"] == 1.0 and m95["false-match rate"] == 0.0,
                      "perfect rule gives precision 1, recall 1, false-match rate 0 at 95"))
    results.append(ok(res["new_alone"]["decidable_pairs"] == 157 and res["combined"]["decidable_pairs"] == 217,
                      "new alone 157, combined with 60 Phase 2 pairs 217"))
    results.append(ok(res["new_alone"]["exact_precision"]["identical cores"]["pairs"] > 0, "exact CI reported for identical cores"))
    results.append(ok(er.main(["evaluate"]) == 1, "evaluate refuses to overwrite"))

    # under 141 decidable: stop and point to the next batch
    er.FILES["result"].unlink()
    lab.loc[lab.index[:20], "label"] = ""
    lab.to_csv(er.FILES["labelled"], index=False)
    results.append(ok(er.main(["evaluate"]) == 1 and not er.FILES["result"].exists(), "under 141 decidable: stopped"))
    results.append(ok(er.main(["sample", "--batch", "3"]) == 1, "batch 3 refused before batch 2"))
    results.append(ok(er.main(["sample", "--batch", "2"]) == 0 and len(pd.read_csv(er.FILES["to_label"])) == 180,
                      "batch 2 appends 20 pairs"))
    results.append(ok(er.main(["sample", "--batch", "2"]) == 1, "batch 2 cannot be drawn twice"))

    # edited names or bad labels are refused
    lab2 = pd.read_csv(er.FILES["to_label"], dtype={"label": str})
    lab2["label"] = "1"
    lab2.loc[0, "name_a"] = "CHANGED"
    lab2.to_csv(er.FILES["labelled"], index=False)
    results.append(ok(er.main(["evaluate"]) == 1, "edited names refused"))
    lab2 = pd.read_csv(er.FILES["to_label"], dtype={"label": str}); lab2["label"] = "yes"
    lab2.to_csv(er.FILES["labelled"], index=False)
    results.append(ok(er.main(["evaluate"]) == 1, "label values other than 1, 0 or blank refused"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
