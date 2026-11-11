"""Smoke test of the explain_eval script on synthetic folds. Run: python tests/test_explain_eval_smoke.py

Project data, the harness folds and the preprocessor are replaced by small stand-ins,
so only the script's flow, outputs and stop rules are tested here.
"""
import sys, pathlib, json, tempfile, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore", UserWarning)
import model_evaluation.explain_eval as ee
import model_evaluation.explain_checks as ec

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []

rng = np.random.default_rng(1)
Ztr, Zva = rng.normal(size=(2000, 5)), rng.normal(size=(800, 5))
Zva[:8, 1] += 6
Fva = pd.DataFrame(Zva, columns=list("abcde")); Fva["ocid"] = [f"oc-{i}" for i in range(800)]


class Prep:
    def fit(self, F): return self
    def transform(self, F): return F[list("abcde")].to_numpy()
    def get_feature_names_out(self): return np.array(list("abcde"))


def setup(out):
    ee.OUT = pathlib.Path(out)
    ee.ROOT = pathlib.Path(out).parent
    ee.FOLD5 = dict(n_train=2000, n_val=800)
    ee.load_training_period = lambda root: pd.DataFrame({"x": range(10)})
    ee.build_folds = lambda X: [dict(fold=5, Ztr=Ztr, Zva=Zva, Ftr=None, Fva=Fva)]
    ee.build_preprocessor = lambda: Prep()
    ee.clone = lambda p: p
    ee.SklearnIF = lambda n, s, seed: __import__("model_evaluation.harness", fromlist=["x"]).SklearnIF(100, s, seed)
    ee.IsoTreeEIF = lambda nt, ss, nd, s: __import__("model_evaluation.eif_eval", fromlist=["x"]).IsoTreeEIF(100, ss, nd, s)


with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / "explanations"
    setup(out)
    code = ee.main()
    res = json.loads((out / "explain_validation.json").read_text())
    files = sorted(p.name for p in out.iterdir())
    results.append(ok(code == 0, "script runs end to end and exits 0"))
    results.append(ok(files == ["eif_perm_s0.csv", "eif_perm_s1.csv", "explain_validation.json", "if_perm_s0.csv",
                                "if_tree_s0.csv", "if_tree_s1.csv"], "expected output files only"))
    ct = res["if_tree"]["checks"]
    results.append(ok("not applicable" in ct["method_agreement"]["status"] and "method_agreement" not in ee.NEED_TREE,
                      "check 3 recorded as not applicable for TreeExplainer, not as passed"))
    results.append(ok(all(k in res["eif_perm"]["checks"] for k in ee.NEED_PERM), "all EIF checks recorded"))
    results.append(ok(res["if_tree"]["checks"]["target"]["passed"], "IF explained output has the ranking order"))
    df = pd.read_csv(out / "eif_perm_s0.csv")
    results.append(ok(df["contract_id"].str.startswith("oc-").all() and (df["n_permutations"] >= 10).all(),
                      "per-contract table has contract ids and at least 10 permutations"))
    results.append(ok(len(res["background_indices"]["0"]) == 100, "background row indices recorded"))
    results.append(ok(ee.main() == 1, "refuses to overwrite existing outputs"))

# feasibility stop: a timing limit of zero must stop before any full run
with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / "explanations"
    setup(out)
    ee.check_feasibility = lambda s, n, n_timed=20: {**ec.check_feasibility(s, n, n_timed), "passed": False}
    calls = []
    real = ee.permutation_explain
    ee.permutation_explain = lambda f, bg, X, s, me: calls.append(len(X)) or real(f, bg, X, s, me)
    code = ee.main()
    res = json.loads((out / "explain_validation.json").read_text())
    eif = res["eif_perm"]
    results.append(ok(code == 0 and not eif["validated"] and "not run" in eif["reason"],
                      "failed feasibility: EIF not validated, remaining checks reported as not run"))
    results.append(ok(set(eif["checks"]) == {"feasibility", "method_agreement"} and "full_s0" not in eif["seconds"],
                      "no full EIF run after a failed timing check"))
    results.append(ok(not (out / "eif_perm_s0.csv").exists() and not res["if_tree"]["validated"],
                      "no EIF attribution file; IF tree also stopped and not validated"))
    ee.permutation_explain, ee.check_feasibility = real, ec.check_feasibility

# feasibility fails only for the IF permutation run (the 2nd timing check): check 3 not completed
with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / "explanations"
    setup(out)
    n_checks, sizes = [0], []
    def fail_second(s, n, n_timed=20):
        n_checks[0] += 1
        r = ec.check_feasibility(s, n, n_timed)
        return {**r, "passed": False} if n_checks[0] == 2 else r
    ee.check_feasibility = fail_second
    real = ee.permutation_explain
    ee.permutation_explain = lambda f, bg, X, s, me: sizes.append(len(X)) or real(f, bg, X, s, me)
    code = ee.main()
    res = json.loads((out / "explain_validation.json").read_text())
    agree = res["eif_perm"]["checks"]["method_agreement"]
    results.append(ok(code == 0 and res["if_tree"]["validated"], "IF TreeExplainer unaffected"))
    results.append(ok(sizes[0] == 20 and sizes[1] == 20 and 40 not in sizes[:2],
                      "IF permutation: only the 20-contract timing run, no full run"))
    results.append(ok(not (out / "if_perm_s0.csv").exists() and not agree["passed"]
                      and "not completed" in agree["status"], "check 3 recorded as not completed (feasibility)"))
    results.append(ok(not res["eif_perm"]["validated"] and "method_agreement" in res["eif_perm"]["reason"],
                      "EIF permutation method not validated"))
    results.append(ok(res["if_perm_check3"]["feasibility"]["passed"] is False
                      and "additivity" not in res["if_perm_check3"], "IF permutation timing recorded, nothing else"))
    ee.permutation_explain, ee.check_feasibility = real, ec.check_feasibility

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
