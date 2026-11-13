"""Checks for reciprocal rank fusion (protocol 7.5). Run: python tests/test_rrf.py

Synthetic data only. No project data is loaded.
"""
import sys, pathlib, tempfile, warnings, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from model_evaluation import rrf
import model_evaluation.rrf_eval as re_
from model_evaluation.harness import SklearnIF, evaluate

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(0)


def raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


# 1 ranks against a fixed reference, strictly greater only
ref = np.array([1.0, 2.0, 2.0, 3.0])
r = rrf.reference_ranks(ref, [3.5, 3.0, 2.0, 1.5, 0.0])
results.append(ok(r.tolist() == [1, 1, 2, 4, 5], "rank = 1 + count of strictly greater reference scores"))
results.append(ok(rrf.reference_ranks(ref, ref).tolist() == [4, 2, 2, 1], "reference ranked against itself; ties share a rank"))
results.append(ok(rrf.reference_ranks(ref, [np.nan, 2.5]).tolist() == [5, 2], "missing score gets the worst rank, n + 1"))
results.append(ok(rrf.reference_ranks(ref[::-1], [2.0]).tolist() == [2], "reference order does not matter"))
results.append(ok(raises(lambda: rrf.reference_ranks([1.0, np.nan], [1.0])) and raises(lambda: rrf.reference_ranks([], [1.0]))
                  and raises(lambda: rrf.reference_ranks([1.0], [np.inf])), "bad reference or infinite score refused"))

# 2 brute force on random data with many ties
refb = np.round(rng.normal(size=500), 1); sb = np.round(rng.normal(size=300), 1)
brute = np.array([1 + (refb > x).sum() for x in sb])
results.append(ok(np.array_equal(rrf.reference_ranks(refb, sb), brute), "ranks match a brute-force count with ties"))

# 3 fusion by hand
refs = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([10.0, 20.0, 30.0])}
sc = {"a": np.array([2.5]), "b": np.array([5.0])}  # ranks 2 and 4
results.append(ok(np.isclose(rrf.fuse(refs, sc)[0], 1 / 62 + 1 / 64), "RRF = sum of 1 / (60 + rank)"))
results.append(ok(rrf.fuse(dict(reversed(list(refs.items()))), sc)[0] == rrf.fuse(refs, sc)[0], "component order does not matter"))
results.append(ok(raises(lambda: rrf.fuse(refs, {"a": sc["a"]})) and raises(lambda: rrf.fuse(refs, {"a": np.ones(2), "b": np.ones(3)})),
                  "mismatched components or lengths refused"))
results.append(ok(np.array_equal(rrf.reference_rrf(refs), rrf.fuse(refs, refs)), "reference RRF is the reference scored against itself"))

# 4 no batch dependence: one contract alone gets the same score as inside a batch
refs3 = {k: rng.normal(size=1000) for k in ("R", "IF", "EIF")}
batch = {k: rng.normal(size=200) for k in refs3}
full = rrf.fuse(refs3, batch)
alone = np.array([rrf.fuse(refs3, {k: v[i:i + 1] for k, v in batch.items()})[0] for i in range(200)])
results.append(ok(np.array_equal(full, alone), "each contract scored independently of the batch"))
results.append(ok(full.max() <= 3 / 61 + 1e-15 and full.min() >= 3 / (60 + 1001) - 1e-15, "scores within the theoretical bounds"))
one = {"m": refs3["R"]}
s_sorted = np.sort(rng.normal(size=100))
results.append(ok(np.all(np.diff(rrf.fuse(one, {"m": s_sorted})) >= 0), "single component: RRF never decreases as the score rises"))


# synthetic folds in the harness format
def make_fold(k, n_tr=1500, n_va=600, d=4, shift=8.0):
    Ztr = rng.normal(size=(n_tr, d)); Zva = rng.normal(size=(n_va, d))
    n_inj = 12
    Zinj = {t: rng.normal(size=(n_inj, d)) + (shift if t != "value_x5" else 1.0) for t in re_.TYPES}
    Ftr = pd.DataFrame({"R": rng.random(n_tr)}); Fva = pd.DataFrame({"R": rng.random(n_va)})
    inj = {t: pd.DataFrame({"R": rng.random(n_inj)}) for t in re_.TYPES}
    return dict(fold=k, Ztr=Ztr, Zva=Zva, Zinj=Zinj, Ftr=Ftr, Fva=Fva), inj


pairs = [make_fold(1), make_fold(2)]
folds = [p[0] for p in pairs]
injected = {p[0]["fold"]: p[1] for p in pairs}
models = {"IF": lambda s: SklearnIF(50, 128, seed=s)}

# 5 metric code identical to the harness
base = {f["fold"]: SklearnIF(50, 128, seed=0).fit(f["Ztr"]).score(f["Zva"]) for f in folds}
h = evaluate(lambda s: SklearnIF(50, 128, seed=s), folds, seeds=(0, 1), baseline=base)
mine = []
for f in folds:
    ms = [SklearnIF(50, 128, seed=s).fit(f["Ztr"]) for s in (0, 1)]
    mine.append(re_.fold_metrics(f["fold"], [m.score(f["Zva"]) for m in ms],
                                 {t: [m.score(f["Zinj"][t]) for m in ms] for t in re_.TYPES},
                                 f["Fva"]["R"].to_numpy(), base[f["fold"]]))
mine = pd.DataFrame(mine)
cols = ["stability", "recall_value", "recall_date", "recall_value_x5", "recall_primary", "spearman_R", "jaccard_top5_baseline"]
results.append(ok(np.allclose(mine[cols].to_numpy(), h[cols].to_numpy(), atol=0), "fold_metrics reproduces harness.evaluate exactly"))

# 6 RRF evaluation runs, is deterministic and flags about 5%
e1 = re_.evaluate_rrf(folds, injected, models, base)
e2 = re_.evaluate_rrf(folds, injected, models, base)
c = [x for x in e1.columns if x != "fit_seconds"]
results.append(ok(e1[c].equals(e2[c]), "two RRF runs give identical tables"))
results.append(ok(e1["share_above_cutoff"].between(0.04, 0.051).all(), "about 5% of clean contracts above the cut-off"))
results.append(ok((e1["recall_value"] > 0.8).all() and (e1["recall_date"] > 0.8).all(), "planted far-out copies are recalled"))
results.append(ok({"jaccard_top5_IF", "spearman_IF", "jaccard_top5_baseline"} <= set(e1.columns), "descriptive overlap columns present"))

# 7 seeds apply to model components only: R alone gives stability 1
eR = re_.evaluate_rrf(folds, injected, {}, None)
results.append(ok(np.allclose(eR["stability"], 1.0), "R(c) alone is deterministic across seeds"))

# 8 injected R(c) is read from the recomputed injected frames
hi = {k: {t: pd.DataFrame({"R": np.full(12, 2.0 if t == "date" else 0.0)}) for t in re_.TYPES} for k in injected}
lo = {k: {t: pd.DataFrame({"R": np.full(12, -1.0)}) for t in re_.TYPES} for k in injected}
folds_mild = [dict(f, Zinj={t: rng.normal(size=(12, 4)) for t in re_.TYPES}) for f in folds]
d_hi = re_.evaluate_rrf(folds_mild, hi, models)["recall_date"].mean()
d_lo = re_.evaluate_rrf(folds_mild, lo, models)["recall_date"].mean()
results.append(ok(d_hi > d_lo, f"higher injected R(c) raises date recall ({d_lo:.2f} -> {d_hi:.2f})"))

# 9 non-finite reference scores stop the run
bad = [dict(folds[0], Ftr=pd.DataFrame({"R": np.r_[np.nan, np.ones(1499)]}))]
try:
    re_.evaluate_rrf(bad, injected, models); stopped = False
except re_.Stop:
    stopped = True
results.append(ok(stopped, "missing R(c) in the training reference stops the run"))

# 10 ranking rule (7.5 clarification)
m = lambda st, v, d: pd.Series({"stability": st, "recall_value": v, "recall_date": d, "recall_primary": (v + d) / 2})
b, eif = m(0.99, 0.12, 0.84), m(0.99, 0.18, 0.85)
cases = {"win": m(0.98, 0.25, 0.86), "tie": m(0.98, 0.19, 0.85), "worse": m(0.98, 0.12, 0.80),
         "unstable": m(0.90, 0.40, 0.90), "guard": m(0.98, 0.60, 0.70)}
res = {k: re_.rank_rrf(v, b, eif) for k, v in cases.items()}
results.append(ok(res["win"]["ranks_best"] and res["win"]["scientific_result"] == "RRF ranks above EIF", "clear gain over EIF ranks RRF first"))
results.append(ok(not res["tie"]["ranks_best"] and "tie margin" in res["tie"]["scientific_result"], "within 0.02 of EIF: EIF keeps the rank"))
results.append(ok(not res["worse"]["ranks_best"] and "worse" in res["worse"]["scientific_result"], "worse than EIF by more than 0.02"))
results.append(ok(not res["unstable"]["ranks_best"] and "stability" in res["unstable"]["scientific_result"], "stability below 0.95 blocks ranking"))
results.append(ok(not res["guard"]["ranks_best"] and "guard" in res["guard"]["scientific_result"] and not res["guard"]["guard_date_ok"],
                  "date recall more than 0.05 below the baseline blocks ranking"))
results.append(ok(all("not deployment-eligible" in v["deployment"] for v in res.values()), "RRF never called deployment-eligible"))

# 11 eligibility check from saved results
good_sel = {"selected_within_family": {"ntrees": 400, "sample_size": 256, "ndim": 2},
            "cross_family": {"scientific_result": "EIF ranks above the baseline", "eif_stability_ok": True,
                             "guard_value_ok": True, "guard_date_ok": True},
            "deployment_eligibility": {"end_to_end_fit_under_10_minutes": True}}
good_exp = {"eif_perm": {"validated": True}}
results.append(ok(re_.check_eligibility(good_sel, good_exp)[0], "eligible EIF accepted"))
bad_sel = json.loads(json.dumps(good_sel)); bad_sel["selected_within_family"]["ndim"] = 4
bad_cf = json.loads(json.dumps(good_sel)); bad_cf["cross_family"]["scientific_result"] = "within the 0.02 tie margin; simpler model (baseline) kept"
results.append(ok(not re_.check_eligibility(bad_sel, good_exp)[0] and not re_.check_eligibility(bad_cf, good_exp)[0]
                  and not re_.check_eligibility(good_sel, {"eif_perm": {"validated": False}})[0]
                  and not re_.check_eligibility({}, {})[0], "wrong config, no win, unvalidated explanation or empty files rejected"))

# 12 alignment check with the real preprocessor on a synthetic indicator frame
from sklearn.base import clone
from src.pipeline import build_preprocessor, FLAGS, HISTORY, METHOD, CATEGORY


def frame(n):
    return pd.DataFrame({"value_amount": rng.lognormal(10, 2, n), CATEGORY: rng.choice(["goods", "works", "services"], n),
                         "days_to_signing": rng.normal(20, 10, n), METHOD: rng.choice(["open", "direct"], n),
                         "method_level": rng.integers(0, 3, n).astype(float),
                         **{h_: rng.random(n) for h_ in HISTORY}, **{fl: rng.integers(0, 2, n) for fl in FLAGS}})


Ftr = frame(400); inj = {t: frame(10) for t in re_.TYPES}
prep = clone(build_preprocessor()).fit(Ftr)
fa = dict(Ftr=Ftr, Zinj={t: prep.transform(g) for t, g in inj.items()})
results.append(ok(re_.check_alignment(fa, inj), "aligned injected inputs accepted"))
inj_bad = dict(inj); inj_bad["date"] = inj["date"].iloc[::-1].reset_index(drop=True)
results.append(ok(not re_.check_alignment(fa, inj_bad), "reordered injected copies detected"))

# 13 main stops before loading data if results are missing, and refuses to overwrite
with tempfile.TemporaryDirectory() as tmp:
    t = pathlib.Path(tmp)
    re_.OUT, re_.ROOT = t, t.parent
    re_.FILES = {"folds": t / "rrf_by_fold.csv", "result": t / "rrf_result.json"}
    re_.CP1, re_.EIF_GRID, re_.EIF_SEL, re_.EXPLAIN = (t / "cp1.csv", t / "grid.csv", t / "sel.json", t / "exp.json")
    results.append(ok(re_.main() == 1, "missing earlier results stop the run"))
    for p in (re_.CP1, re_.EIF_GRID):
        p.write_text("x\n")
    re_.EIF_SEL.write_text(json.dumps(bad_cf)); re_.EXPLAIN.write_text(json.dumps(good_exp))
    results.append(ok(re_.main() == 1, "ineligible EIF stops the run before any data is loaded"))
    re_.FILES["folds"].write_text("x")
    results.append(ok(re_.main() == 1, "refuses to overwrite existing output"))

# 14 main end to end on synthetic folds, real IF and EIF, data loading replaced
import model_evaluation.harness as hm
import model_evaluation.indicator_scores as isc
from model_evaluation.eif_eval import IsoTreeEIF
syn = [make_fold(k, n_tr=800, n_va=400)[0] for k in (1, 2, 3)]
syn_inj = {f["fold"]: {t: pd.DataFrame({"R": rng.random(12)}) for t in re_.TYPES} for f in syn}
hm.load_training_period = lambda root: pd.DataFrame(index=range(5))
hm.build_folds = lambda X: syn
isc.injected_frames = lambda f: syn_inj[f["fold"]]
re_.check_alignment = lambda f, inj: True
with tempfile.TemporaryDirectory() as tmp:
    t = pathlib.Path(tmp)
    re_.OUT, re_.ROOT = t, t.parent
    re_.FILES = {"folds": t / "rrf_by_fold.csv", "result": t / "rrf_result.json"}
    re_.CP1, re_.EIF_GRID, re_.EIF_SEL, re_.EXPLAIN = (t / "cp1.csv", t / "grid.csv", t / "sel.json", t / "exp.json")
    evaluate(lambda s: SklearnIF(400, 256, seed=s), syn, seeds=(0, 1)).to_csv(re_.CP1, index=False)
    g = evaluate(lambda s: IsoTreeEIF(400, 256, 2, s), syn, seeds=(0, 1))
    g.insert(0, "role", "grid"); g.insert(1, "ntrees", 400); g.insert(2, "sample_size", 256); g.insert(3, "ndim", 2)
    g.to_csv(re_.EIF_GRID, index=False)
    re_.EIF_SEL.write_text(json.dumps(good_sel)); re_.EXPLAIN.write_text(json.dumps(good_exp))
    code = re_.main()
    rec = json.loads(re_.FILES["result"].read_text()) if code == 0 else {}
    results.append(ok(code == 0 and rec.get("repeat_identical") and rec["component_reproduction_max_diff"]["EIF_vs_grid"] <= 1e-12,
                      "main runs end to end: components reproduce, repeat identical, outputs written"))
    byf = pd.read_csv(re_.FILES["folds"]) if code == 0 else pd.DataFrame()
    results.append(ok(len(byf) == 3 and {"jaccard_top5_IF", "jaccard_top5_EIF", "share_above_cutoff"} <= set(byf.columns)
                      and "scientific_result" in rec.get("ranking", {}), "per-fold table has both model components and a ranking"))
    results.append(ok(re_.main() == 1, "second run refuses to overwrite"))
    for p in re_.FILES.values():
        p.unlink()
    cp = pd.read_csv(re_.CP1); cp.loc[0, "recall_date"] += 1e-6; cp.to_csv(re_.CP1, index=False)
    results.append(ok(re_.main() == 1 and not re_.FILES["result"].exists(), "IF not reproducing CP1 stops the run, nothing saved"))

# 15 the script never touches the test period directly
src_text = (pathlib.Path(re_.__file__)).read_text()
results.append(ok("load_training_period" in src_text and "temporal_split" not in src_text and "test_scores" not in src_text,
                  "data enters only through load_training_period"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
