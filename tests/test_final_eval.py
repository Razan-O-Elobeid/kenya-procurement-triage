"""Checks for the final evaluation and the final bundle builder. Run: python tests/test_final_eval.py

Synthetic data only. No project data is loaded. The real-run gate is tested in a throwaway Git repo.
"""
import sys, pathlib, tempfile, warnings, json, os, subprocess, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
import model_evaluation.final_eval as fe
import dashboard.build_final_bundle as bb
from dashboard.bundle import load_bundle
from model_evaluation.rrf import fuse
from src.tuning import DATE

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(0)
COLS = ["value_amount", "days_to_signing", "x3", "x4"]


def frame(n, start, shift=0.0):
    F = pd.DataFrame({"value_amount": rng.lognormal(10, 1, n) * (1 + shift), "days_to_signing": rng.normal(20, 8, n),
                      "x3": rng.normal(shift, 1, n), "x4": rng.normal(0, 1, n)})
    F["ocid"] = [f"ocds-{start + i}" for i in range(n)]
    F["main_ocid"] = F["ocid"]
    F["buyer_id"] = rng.choice(["b1", "b2", "b3"], n)
    F["buyer_name"] = F["buyer_id"].str.upper()
    F["supplier_name"] = rng.choice(["Acme Ltd", "Beta Co", "Gamma"], n)
    F["tender_mainProcurementCategory"] = rng.choice(["goods", "works"], n)
    F["tender_procurementMethod"] = rng.choice(["open", "direct"], n)
    F[DATE] = pd.date_range("2024-01-01", periods=n, freq="7h").astype(str)
    F["R"] = rng.random(n)
    return F


class Prep:
    def transform(self, G):
        Z = G[COLS].to_numpy(dtype=float).copy()
        Z[:, 0] = np.log1p(Z[:, 0])
        return Z

    def get_feature_names_out(self):
        return np.array([f"num__{c}" for c in COLS])


class Resolver:
    def transform(self, names):
        return np.asarray(names, dtype=object)


class IB:
    resolver_ = Resolver()


def make_d(n_tr=1500, n_te=600):
    Ftr, Fte = frame(n_tr, 0), frame(n_te, 10_000, shift=0.3)
    p = Prep()
    return dict(ib=IB(), prep=p, Ftr=Ftr, Fte=Fte, Ztr=p.transform(Ftr), Zte=p.transform(Fte),
                names=list(p.get_feature_names_out()))


def fake_injected(f, frac=0.02):
    F = f["Fva"]
    out = {}
    for k, s in zip(fe.TYPES, (0, 1, 2)):
        G = F.sample(n=max(1, int(frac * len(F))), random_state=s).copy()
        if k == "date":
            G["days_to_signing"] = -200.0; G["R"] = 0.99
        else:
            G["value_amount"] *= 50 if k == "value" else 5
        out[k] = G
    return out


import model_evaluation.indicator_scores as isc
isc.injected_frames = fake_injected

VALID = {"if_tree": {"validated": True}, "eif_perm": {"validated": True}}
GOOD = {"models": fe.FROZEN, "rrf": fe.RRF_SPEC, "threshold_rule": "p95_training_scores",
        "scientific_leader": "eif", "deployed_model": "eif", "queue_models": ["eif", "if"],
        "explained_models": ["if", "eif"], "explanation_settings": {"background_rows": 100, "background_seed": 0}}


def stops(fn):
    try:
        fn(); return False
    except fe.Stop:
        return True


def alt(**kw):
    return json.loads(json.dumps({**GOOD, **kw}))


# 1 freeze file validation
results.append(ok(fe.check_freeze(alt(), VALID) == GOOD, "valid freeze accepted"))
bad_models = alt(); bad_models["models"]["eif"]["ndim"] = 4
results.append(ok(stops(lambda: fe.check_freeze(bad_models, VALID)), "changed configuration refused"))
results.append(ok(stops(lambda: fe.check_freeze(alt(rrf={"components": ["R", "if"], "k": 60}), VALID)), "changed RRF refused"))
results.append(ok(stops(lambda: fe.check_freeze(alt(queue_models=["eif", "rrf"]), VALID)), "RRF in the queue refused"))
results.append(ok(stops(lambda: fe.check_freeze(alt(queue_models=["if", "eif"]), VALID)), "queue must start with the deployed model"))
results.append(ok(stops(lambda: fe.check_freeze(alt(), {"if_tree": {"validated": True}, "eif_perm": {"validated": False}})),
                  "deployed model without a validated explanation refused"))
results.append(ok(stops(lambda: fe.check_freeze(alt(explained_models=["if"]), VALID)), "deployed model missing from explained_models refused"))
results.append(ok(fe.check_freeze(alt(deployed_model=None), VALID)["deployed_model"] is None
                  and stops(lambda: fe.check_freeze(alt(deployed_model=None, queue_models=["if"]), VALID)),
                  "prototype: queue must start with the scientific leader"))
results.append(ok(stops(lambda: fe.check_freeze(alt(threshold_rule="p97"), VALID))
                  and stops(lambda: fe.check_freeze(alt(explanation_settings={"background_rows": 50, "background_seed": 0}), VALID)),
                  "threshold rule and explanation settings fixed"))

# 2 fitting, thresholds and RRF
d = make_d()
fitted, tr, te, thr, refs = fe.fit_score(d, fe.make_models())
results.append(ok(all(abs((tr[m] > thr[m]).mean() - 0.05) < 0.002 for m in ("if", "eif", "rrf")),
                  "thresholds are the training 95th percentiles"))
results.append(ok(set(refs) == {"R", "if", "eif"} and np.array_equal(tr["rrf"], fuse(refs, refs)), "training RRF = reference scored against itself"))
one = fuse(refs, {k: te[k][:1] for k in refs})[0]
results.append(ok(one == te["rrf"][0], "a test contract's RRF does not depend on the other test contracts"))
d_bad = make_d(); d_bad["Fte"].loc[0, "R"] = np.nan
results.append(ok(stops(lambda: fe.fit_score(d_bad, fe.make_models())), "missing R(c) stops the run"))

# 3 report
rep, by_month = fe.build_report(d, tr, te, thr, fitted, refs, DATE)
results.append(ok(rep["n_flagged"]["eif"] == int((te["eif"] > thr["eif"]).sum()) and set(rep["spearman"]) >= {"if_eif", "eif_rrf", "rrf_R"},
                  "report has flags, rank agreement and overlap"))
ir = rep["injection_recall"]
results.append(ok(set(ir) == set(fe.TYPES) and all(0 <= ir[k][m]["vs_training_threshold"] <= 1 for k in ir for m in ("if", "eif", "rrf")),
                  "injection recall for every type and model, both cut-offs"))
results.append(ok(int(by_month["n"].sum()) == len(te["if"]), "monthly table covers every scored contract"))
results.append(ok(rep["score_drift"]["if"]["ks_statistic"] > 0, "score drift computed against training"))

# 4 explanations on the real explainers
ids = d["Fte"]["ocid"].to_numpy()
for m in ("if", "eif"):
    df, r = fe.explanation_table(m, fitted, d, te, thr, ids)
    flagged = np.flatnonzero(te[m] > thr[m])
    results.append(ok(sorted(df["position"]) == sorted(flagged) and np.all(np.diff(df["ranking_score"]) <= 0)
                      and r["additivity"]["passed"] and r["target"]["passed"] and r["exact_repeat_first_20"],
                      f"{m}: flagged contracts explained, additive, correct target, exact repeat"))


# 5 real run in a throwaway Git repo
def setup_repo(tmp, freeze=GOOD, commit=True):
    root = pathlib.Path(tmp)
    (root / "model_evaluation").mkdir()
    (root / "results" / "model_evaluation" / "explanations").mkdir(parents=True)
    (root / "results" / "model_evaluation" / "explanations" / "explain_validation.json").write_text(json.dumps(VALID))
    (root / "model_evaluation" / "model_freeze.json").write_text(json.dumps(freeze))
    run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (root / ".gitignore").write_text("results/\n")
    run("add", ".gitignore")
    if commit:
        run("add", "model_evaluation/model_freeze.json")
    run("commit", "-q", "-m", "freeze")
    return root


def point_to(root):
    fe.ROOT = root
    fe.FREEZE = root / "model_evaluation" / "model_freeze.json"
    fe.VALIDATION = root / "results" / "model_evaluation" / "explanations" / "explain_validation.json"
    fe.PHASE2_REF = root / "results" / "test_scores.csv"
    fe.OUTS = {"real": root / "results" / "final_test", "dry": root / "results" / "final_test_dryrun"}
    bb.ROOT = root
    bb.SRC = dict(fe.OUTS)
    bb.OUT = {"real": root / "results" / "dashboard" / "final_test", "dry": root / "results" / "dashboard" / "final_dryrun"}
    bb.VALIDATION = fe.VALIDATION


loads = []
D0 = make_d()


def fake_load(root):
    loads.append(1)
    return D0["Ftr"], D0["Fte"]


fe.load_periods = fake_load
fe.prepare = lambda Xtr, Xte: dict(D0)
_, _, te0, thr0, _ = fe.fit_score(dict(D0), fe.make_models())
fe.EXPECTED = dict(n_train=1500, n_test=600, n_flagged_phase2=int((te0["if"] > thr0["if"]).sum()))


def write_ref(root, perturb=0.0):
    ref = pd.DataFrame({"main_ocid": D0["Fte"]["main_ocid"], "anomaly_score": te0["if"] + perturb, "top5": te0["if"] > thr0["if"]})
    ref.to_csv(root / "results" / "test_scores.csv", index=False)


with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp, commit=False); point_to(root); write_ref(root)
    results.append(ok(fe.main([]) == 1 and not loads, "untracked freeze: stopped before any data is loaded"))
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp); point_to(root); write_ref(root)
    (root / "model_evaluation" / "model_freeze.json").write_text(json.dumps(alt(queue_models=["eif"])))
    results.append(ok(fe.main([]) == 1 and not loads, "freeze changed after its commit: stopped before loading"))
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp); point_to(root); write_ref(root)
    results.append(ok(fe.main(["--freeze", str(root / "other.json")]) == 1 and not loads, "real run refuses any other freeze file"))
    code = fe.main([])
    log = json.loads((root / "results" / "final_test" / "RUN_LOG.json").read_text())["attempts"]
    results.append(ok(code == 0 and len(loads) == 1 and [a["status"] for a in log] == ["started", "completed"],
                      "committed freeze: one complete run, logged"))
    rec = json.loads((root / "results" / "final_test" / "final_report.json").read_text())
    results.append(ok(rec["freeze_commit"] and rec["report"]["thresholds"]["eif"] == thr0["eif"], "report records the freeze commit and frozen thresholds"))
    results.append(ok(fe.main([]) == 1 and len(loads) == 1, "second run refused: the test period is scored once"))
    man, T = load_bundle(root / "results" / "dashboard" / "final_test")
    results.append(ok(man["mode"] == "deployment" and man["queue_models"] == ["eif", "if"] and "rrf" in man["models"]
                      and not man["explanations"]["rrf"]["passed"], "bundle: deployment wording, RRF reported but not queued"))
    results.append(ok(all(man["explanations"][m]["passed"] and man["explanations"][m]["n_flagged_with_attribution"] == man["models"][m]["n_flagged"]
                          for m in ("if", "eif")), "every flagged contract has checked attributions"))
    results.append(ok(bb.build("real") == 1, "builder refuses to overwrite a bundle"))
    # builder on tampered results
    shutil.rmtree(root / "results" / "dashboard" / "final_test")
    C = pd.read_csv(root / "results" / "final_test" / "contracts.csv.gz")
    E = pd.read_csv(root / "results" / "final_test" / "explanations.csv.gz")
    E2 = E.copy(); E2.loc[E2["model"] == "eif", "ranking_score"] += 1e-6
    E2.to_csv(root / "results" / "final_test" / "explanations.csv.gz", index=False, compression="gzip")
    code = bb.build("real")
    man2, _ = load_bundle(root / "results" / "dashboard" / "final_test")
    results.append(ok(code == 0 and not man2["explanations"]["eif"]["passed"] and man2["explanations"]["if"]["passed"],
                      "mismatched attributions are not shown; scores still are"))
    shutil.rmtree(root / "results" / "dashboard" / "final_test")
    E3 = E.drop(index=E.index[E["model"] == "eif"][3])  # one otherwise-valid row removed
    E3.to_csv(root / "results" / "final_test" / "explanations.csv.gz", index=False, compression="gzip")
    code = bb.build("real")
    man3, _ = load_bundle(root / "results" / "dashboard" / "final_test")
    results.append(ok(code == 0 and not man3["explanations"]["eif"]["passed"] and "coverage" in man3["explanations"]["eif"]["reason"]
                      and man3["explanations"]["if"]["passed"], "one flagged contract without attribution: model's attributions hidden"))
    shutil.rmtree(root / "results" / "dashboard" / "final_test")
    E4 = pd.concat([E, E[E["model"] == "if"].iloc[[0]]], ignore_index=True)  # one row twice
    E4.to_csv(root / "results" / "final_test" / "explanations.csv.gz", index=False, compression="gzip")
    bb.build("real")
    man4, _ = load_bundle(root / "results" / "dashboard" / "final_test")
    results.append(ok(not man4["explanations"]["if"]["passed"] and man4["explanations"]["eif"]["passed"],
                      "a repeated explanation row is detected"))
    shutil.rmtree(root / "results" / "dashboard" / "final_test")
    C2 = C.copy(); C2.loc[0, "flag_eif"] = not C2.loc[0, "flag_eif"]
    C2.to_csv(root / "results" / "final_test" / "contracts.csv.gz", index=False, compression="gzip")
    results.append(ok(bb.build("real") == 1 and not (root / "results" / "dashboard" / "final_test").exists(),
                      "flags inconsistent with thresholds: no bundle"))

    # the app opens the final bundle
    C.to_csv(root / "results" / "final_test" / "contracts.csv.gz", index=False, compression="gzip")
    E.to_csv(root / "results" / "final_test" / "explanations.csv.gz", index=False, compression="gzip")
    bb.build("real")
    from streamlit.testing.v1 import AppTest
    os.environ["TRIAGE_BUNDLE"] = str(root / "results" / "dashboard" / "final_test")
    os.environ["TRIAGE_REVIEW_DB"] = str(root / "reviews.sqlite")
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=60).run()
    results.append(ok(not at.exception, "the app opens the final bundle without errors"))

# 6 integrity stop on the Phase 2 reproduction
loads.clear()
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp); point_to(root); write_ref(root, perturb=1e-6)
    out = root / "results" / "final_test"
    code = fe.main([])
    log = json.loads((out / "RUN_LOG.json").read_text())["attempts"]
    results.append(ok(code == 1 and log[-1]["status"] == "stopped_integrity" and not (out / "final_report.json").exists(),
                      "IF not reproducing Phase 2: stopped, logged, no results saved"))
    write_ref(root)
    results.append(ok(fe.main([]) == 1, "rerun after an integrity stop needs the explicit flag"))
    results.append(ok(fe.main(["--after-integrity-stop"]) == 0, "rerun with the flag completes"))
    results.append(ok(fe.main(["--after-integrity-stop"]) == 1, "after completion, even the flag cannot rerun"))
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp); point_to(root); write_ref(root)
    out = root / "results" / "final_test"; out.mkdir(parents=True)
    (out / "RUN_LOG.json").write_text(json.dumps({"attempts": [{"status": "started"}]}))
    results.append(ok(fe.main(["--after-integrity-stop"]) == 1, "an unfinished earlier attempt blocks any rerun"))

# 6b no EIF or RRF test scoring when the Phase 2 check fails
calls = {"eif_test": 0, "fuse": 0}
real_make, real_fuse = fe.make_models, fe.fuse


def spy_models():
    ms = real_make()
    sc = ms["eif"].score

    def score(Z):
        if Z.shape[0] == len(D0["Zte"]) and np.array_equal(Z, D0["Zte"]):
            calls["eif_test"] += 1
        return sc(Z)
    ms["eif"].score = score
    return ms


def spy_fuse(*a, **k):
    calls["fuse"] += 1
    return real_fuse(*a, **k)


fe.make_models, fe.fuse = spy_models, spy_fuse
loads.clear()
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp); point_to(root); write_ref(root, perturb=1e-6)
    code = fe.main([])
    log = json.loads((root / "results" / "final_test" / "RUN_LOG.json").read_text())["attempts"]
    results.append(ok(code == 1 and calls == {"eif_test": 0, "fuse": 0} and log[-1]["status"] == "stopped_integrity",
                      "failed Phase 2 check: EIF and RRF never score the test period"))
    write_ref(root)
    code = fe.main(["--after-integrity-stop"])
    results.append(ok(code == 0 and calls["eif_test"] >= 1 and calls["fuse"] >= 1, "after the check passes, EIF and RRF are scored"))
fe.make_models, fe.fuse = real_make, real_fuse

# 6c a stop after new test results blocks any rerun
loads.clear()
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp); point_to(root); write_ref(root)
    D0["Fte"].loc[5, "R"] = np.nan
    code = fe.main([])
    log = json.loads((root / "results" / "final_test" / "RUN_LOG.json").read_text())["attempts"]
    D0["Fte"].loc[5, "R"] = 0.5
    results.append(ok(code == 1 and log[-1]["status"] == "stopped_after_scoring", "stop after the Phase 2 check is logged as after scoring"))
    results.append(ok(fe.main(["--after-integrity-stop"]) == 1, "no rerun after a stop that followed new scoring"))

# 7 prototype wording when nothing is deployed
loads.clear()
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp, freeze=alt(deployed_model=None, explained_models=["if"])); point_to(root); write_ref(root)
    fe.main([])
    man, _ = load_bundle(root / "results" / "dashboard" / "final_test")
    results.append(ok(man["mode"] == "prototype" and "no validated model deployment" in man["deployment_status"]
                      and not man["explanations"]["eif"]["passed"], "prototype bundle when no model is deployed (8.1)"))

# 8 dry run uses the training period only and writes no run log
import model_evaluation.fold5 as f5mod
import model_evaluation.harness as hm
with tempfile.TemporaryDirectory() as tmp:
    root = setup_repo(tmp); point_to(root)
    (root / "fz.json").write_text(json.dumps(GOOD))
    hm.load_training_period = lambda r: pd.DataFrame(index=range(3))
    f5mod.fold5 = lambda X: {**make_d(), "Fva": D0["Fte"], "Zva": D0["Zte"]}
    loads.clear()
    code = fe.main(["--dry-run", "--freeze", str(root / "fz.json")])
    out = root / "results" / "final_test_dryrun"
    man, _ = load_bundle(root / "results" / "dashboard" / "final_dryrun")
    results.append(ok(code == 0 and not loads and not (out / "RUN_LOG.json").exists() and man["mode"] == "development",
                      "dry run: test period never loaded, no run log, development wording"))
    results.append(ok(fe.main(["--dry-run", "--freeze", str(root / "fz.json")]) == 1, "dry run refuses to overwrite"))

# 8b symlinked repo folder, as /var -> /private/var on macOS
loads.clear()
with tempfile.TemporaryDirectory() as tmp:
    real = pathlib.Path(tmp) / "real"; real.mkdir()
    alias = pathlib.Path(tmp) / "alias"; alias.symlink_to(real)
    root = setup_repo(str(alias)); point_to(root)  # ROOT stays unresolved, as in the earlier tests
    (root / "fz.json").write_text(json.dumps(GOOD))
    code = fe.main(["--dry-run", "--freeze", str((root / "fz.json").resolve())])
    rec = json.loads((root / "results" / "final_test_dryrun" / "final_report.json").read_text()) if code == 0 else {}
    results.append(ok(code == 0 and rec.get("freeze_file") == "fz.json", "dry run through a symlinked folder (macOS /var case)"))
    write_ref(root)
    code = fe.main(["--freeze", str(fe.FREEZE.resolve())])
    man, _ = load_bundle(root / "results" / "dashboard" / "final_test") if code == 0 else ({}, None)
    results.append(ok(code == 0 and len(loads) == 1 and man.get("created_from") == "results/final_test",
                      "real run accepts the committed freeze by its resolved path; bundle built"))

loads.clear()
with tempfile.TemporaryDirectory() as tmp:  # same, without --freeze: the default path must be accepted
    real = pathlib.Path(tmp) / "real"; real.mkdir()
    alias = pathlib.Path(tmp) / "alias"; alias.symlink_to(real)
    root = setup_repo(str(alias)); point_to(root); write_ref(root)
    code = fe.main([])
    log_file = root / "results" / "final_test" / "RUN_LOG.json"
    log = json.loads(log_file.read_text())["attempts"] if log_file.exists() else []
    results.append(ok(code == 0 and len(loads) == 1 and [a["status"] for a in log] == ["started", "completed"],
                      "real run through a symlinked folder without --freeze completes"))

# 9 order in the source: gate before data
src = pathlib.Path(fe.__file__).read_text()
body = src[src.index("def run(args)"):]
results.append(ok(body.index("check_committed(") < body.index("load_periods(ROOT)") and "load_periods" not in src[:src.index("def load_periods")],
                  "the test period is loaded only after the freeze gate"))
results.append(ok(body.index("check_phase2(") < body.index("score_new(d, fitted") and "fit_score(" not in body,
                  "new test scoring comes after the Phase 2 check in the real-run code"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
