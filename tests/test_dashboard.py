"""Checks for the dashboard bundle, the review log and the app. Run: python tests/test_dashboard.py

Synthetic data only. No project data is loaded; the app is run headless with Streamlit's AppTest.
"""
import sys, pathlib, tempfile, os, sqlite3, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from dashboard.bundle import check_explanations, contract_table, history_tables, load_bundle, write_bundle
from dashboard.review_log import add_review, latest_status, read_reviews
from dashboard.text import label
from model_evaluation.face_validity import build_sheet, top_k
from model_evaluation.fold5 import rank_desc

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
ROOT = pathlib.Path(__file__).resolve().parents[1]

# synthetic fold: 40 training rows, 30 validation rows
rng = np.random.default_rng(0)
def frame(n, start):
    return pd.DataFrame({
        "ocid": [f"ocds-{start + i}" for i in range(n)], "buyer_id": rng.integers(0, 4, n),
        "buyer_name": [f"BUYER {b}" for b in rng.integers(0, 4, n)], "supplier_name": [f"SUP {i % 7}" for i in range(n)],
        "tender_mainProcurementCategory": rng.choice(["goods", "works", "services"], n),
        "tender_procurementMethod": rng.choice(["open", "direct"], n), "value_amount": rng.lognormal(12, 1, n),
        "value_currency": "KES", "tender_tenderPeriod_endDate": "2023-01-01", "dateSigned": "2023-02-01",
        "status": "active", "days_to_signing": rng.integers(-5, 60, n).astype(float), "method_level": 0,
        "f1_direct_award": rng.integers(0, 2, n), "f3_decision_period": rng.random(n),
        "f4_supplier_concentration": rng.random(n), "f5_benford_deviation": rng.random(n), "benford_mad": rng.random(n),
        "n_prior_awards": rng.integers(0, 50, n), "is_signed_pre_closure": 0, "zero_value_award": 0, "above_p99": 0,
        "below_p1": 0, "is_terminated": 0, "supplier_missing": 0, "signing_date_missing": 0, "R": rng.random(n)})
Ftr, Fva = frame(40, 0), frame(30, 100)
names = ["value_amount", "days_to_signing", "above_p99"]
s_va, s_tr = {"if": rng.random(30), "eif": rng.random(30)}, {"if": rng.random(40), "eif": rng.random(40)}
thr = {m: float(np.quantile(s, 0.95)) for m, s in s_tr.items()}
C = contract_table(Fva, Fva["supplier_name"], s_va, s_tr, thr)

# 1 ranks, percentiles, flags
results.append(ok(list(rank_desc([0.5, 0.9, 0.5, 0.1])) == [2, 1, 3, 4], "rank 1 = highest; ties to the earlier row"))
results.append(ok(C["flag_if"].equals(pd.Series(s_va["if"] > thr["if"], name="flag_if")) and C["row_id"].tolist() == list(range(30)),
                  "flag = score above the training threshold; row ids are positions"))
results.append(ok(C["pct_if"].between(0, 1).all(), "training percentiles lie in [0, 1]"))

# 2 explanation integrity: matched by position and contract id, strict score match, input order, additivity
def saved_for(m, pos, out):
    v = rng.normal(size=(len(pos), 3))
    d = pd.DataFrame({"contract_id": C["ocid"].to_numpy()[pos], "position_in_fold5_validation": pos,
                      "ranking_score": s_va[m][pos], "explained_output": out[pos], "base": out[pos] - v.sum(1)})
    for j, n in enumerate(names):
        d[f"attr__{n}"] = v[:, j]
    return d
out = s_va["if"] * 2
pos = np.array([3, 7, 11])
good = saved_for("if", pos, out)
rows, rep = check_explanations(good, C, "if", names, s_va["if"], out, True)
results.append(ok(rep["passed"] and rows["row_id"].tolist() == [3, 7, 11] and (rows["model"] == "if").all(),
                  "matching attributions accepted, keyed by row and model"))
bad = good.copy(); bad.loc[0, "ranking_score"] += 1e-9
results.append(ok(not check_explanations(bad, C, "if", names, s_va["if"], out, True)[1]["passed"], "score off by 1e-9: refused"))
bad = good.copy(); bad.loc[0, "contract_id"] = "other"
results.append(ok(not check_explanations(bad, C, "if", names, s_va["if"], out, True)[1]["passed"], "wrong contract id: refused"))
bad = good[["contract_id", "position_in_fold5_validation", "ranking_score", "explained_output", "base",
            "attr__days_to_signing", "attr__value_amount", "attr__above_p99"]]
results.append(ok(not check_explanations(bad, C, "if", names, s_va["if"], out, True)[1]["passed"], "input order changed: refused"))
bad = good.copy(); bad.loc[1, "base"] += 1.0
results.append(ok(not check_explanations(bad, C, "if", names, s_va["if"], out, True)[1]["passed"], "not additive: refused"))
results.append(ok(not check_explanations(good, C, "eif", names, s_va["eif"], out, True)[1]["passed"],
                  "attributions checked against the wrong model: refused"))
results.append(ok(not check_explanations(good, C, "if", names, s_va["if"], out, False)[1]["passed"],
                  "method not validated under section 8: refused"))

# 3 history uses only the training frame
buyers, cats = history_tables(Ftr)
results.append(ok(buyers["buyer_awards"].sum() == (Ftr["value_amount"] > 0).sum(), "history built from training rows only"))

# 4 review log is append-only
with tempfile.TemporaryDirectory() as tmp:
    db = pathlib.Path(tmp) / "log.sqlite"
    add_review(db, "b", 3, "ocds-103", "if", "under review", "RO", "first look")
    add_review(db, "b", 3, "ocds-103", "if", "cleared", "RO", "explained by data error")
    r = read_reviews(db, "b")
    results.append(ok(len(r) == 2 and latest_status(r)[(3, "if")] == "cleared", "latest review gives the current status"))
    blocked = []
    for sql in ("UPDATE reviews SET status = 'new'", "DELETE FROM reviews"):
        try:
            con = sqlite3.connect(db); con.execute(sql); con.commit(); blocked.append(False)
        except sqlite3.DatabaseError:
            blocked.append(True)
        finally:
            con.close()
    results.append(ok(all(blocked) and len(read_reviews(db)) == 2, "UPDATE and DELETE are blocked by the database"))
    try:
        add_review(db, "b", 3, "ocds-103", "if", "approved", "RO"); bad_status = False
    except sqlite3.DatabaseError:
        bad_status = True
    try:
        add_review(db, "b", 3, "ocds-103", "if", "cleared", "  "); no_name = False
    except sqlite3.DatabaseError:
        no_name = True
    results.append(ok(bad_status and no_name, "unknown status and empty reviewer refused"))

# 5 bundle hashes: any edited file is refused
with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / "bundle"
    manifest = {"bundle": "test", "mode": "development", "banner": "Development data: synthetic test bundle.",
                "deployment_status": "development bundle: no deployed model", "git_commit": "abc1234",
                "contracts": "synthetic", "history_reference": "synthetic training rows", "queue_models": ["if", "eif"],
                "models": {m: {"threshold": thr[m], "threshold_rule": "95th percentile of training scores",
                               "n_flagged": int(C[f"flag_{m}"].sum())} for m in ("if", "eif")},
                "explanations": {"if": {**rep, "method": "TreeExplainer"},
                                 "eif": {"passed": False, "reason": "not validated", "method": "permutation SHAP"}},
                "inputs": names}
    write_bundle(out, {"contracts.csv.gz": C, "explanations.csv.gz": rows, "buyers.csv.gz": buyers,
                       "categories.csv.gz": cats}, manifest)
    m2, T = load_bundle(out)
    results.append(ok(len(T["contracts"]) == 30 and set(m2["files"]) == {"contracts.csv.gz", "explanations.csv.gz",
                      "buyers.csv.gz", "categories.csv.gz"}, "bundle round trip with file hashes"))

    # 6 the app runs headless on this bundle, read-only on the bundle files
    from streamlit.testing.v1 import AppTest
    os.environ["TRIAGE_BUNDLE"] = str(out)
    os.environ["TRIAGE_REVIEW_DB"] = str(pathlib.Path(tmp) / "reviews.sqlite")
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    results.append(ok(not at.exception and any("Development data" in w.value for w in at.warning),
                      "queue page loads with the development banner"))
    at.radio(key="page").set_value("Contract").run()
    with_attr = [int(x) for x in rows["row_id"]]
    at.session_state["row_id"] = with_attr[0]
    at.run()
    md = " ".join(m.value for m in at.markdown)
    results.append(ok(not at.exception and "the anomaly score" in md, "contract page shows validated attributions"))
    at.text_input(key="reviewer").input("RO"); at.selectbox(key="review_status").set_value("escalated")
    at.button(key="add_review").click().run()
    results.append(ok(not at.exception and len(read_reviews(os.environ["TRIAGE_REVIEW_DB"])) == 1,
                      "review form appends one row"))
    at.selectbox(key="model").set_value("eif").run()
    results.append(ok(not at.exception and any("No validated attribution" in i.value for i in at.info),
                      "model without validated explanations: message, no attributions"))
    for p in ("Review log", "About"):
        at.radio(key="page").set_value(p).run()
        results.append(ok(not at.exception, f"{p} page loads"))
    after = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    results.append(ok(before == after, "bundle files untouched by the app"))
    (out / "contracts.csv.gz").write_bytes(b"tampered")
    at2 = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    results.append(ok(any("does not match its hash" in e.value for e in at2.error), "edited bundle file refused by the app"))

# 7 face-validity sheet: ties to the earlier position, union without duplicates
top, ties = top_k(np.array([5, 9, 9, 1, 9, 3.0]), k=2)
results.append(ok(list(top) == [1, 2] and ties == {"tied_at_cut": 3, "tied_inside_top": 2},
                  "top k: ties to the earlier contract, tie count reported"))
lists = {"if": top_k(s_va["if"], 10), "rc": top_k(Fva["R"].to_numpy(), 10)}
sheet = build_sheet(Fva, lists)
n_union = len(set(lists["if"][0]) | set(lists["rc"][0]))
results.append(ok(len(sheet) == n_union and sheet["position_in_fold5_validation"].is_unique, "one row per contract in either list"))
results.append(ok(sheet["rank_if"].notna().sum() == 10 and sheet["rank_rc"].notna().sum() == 10
                  and {"code", "reason", "sources", "notes"} <= set(sheet.columns), "rank columns and empty coding columns"))
results.append(ok(label("num__value_amount") == "Contract value (log scale, clipped)" and label("unknown") == "unknown",
                  "plain labels; unknown names left as they are"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
