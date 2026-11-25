"""Synthetic checks for the scoring API and its input schema. Run: python tests/test_serving_api.py"""
import sys, pathlib, copy, json, hashlib, logging, tempfile, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore", message="Attempting to use more than 1 thread")
from fastapi.testclient import TestClient

from model_evaluation.final_eval import fit_train, make_models, prepare
from serving import core
from serving.api import StartupError, create_app, load_bundle
from serving.schema import MAX_BATCH
from dashboard.bundle import contract_table
from src.features import normalise_names

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(1)
KEY = "test-key-0123456789abcdef"
H = {"X-API-Key": KEY}


def periods(n=3000):
    end = pd.Timestamp("2018-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 3100, n), "D")
    df = pd.DataFrame({
        "ocid": [f"ocds-{i:06d}" for i in range(n)],
        "buyer_id": rng.choice([f"KE-B{i}" for i in range(25)], n),
        "buyer_name": rng.choice(["County Government of X", "Ministry of Y", "Authority Z"], n),
        "supplier_name": rng.choice([f"FIRM {i} LTD" for i in range(200)] + [None], n),
        "value_amount": np.where(rng.random(n) < .01, 0, rng.lognormal(13, 2, n)),
        "tender_mainProcurementCategory": rng.choice(["goods", "works", "services"], n),
        "tender_procurementMethod": rng.choice(["open", "selective", "direct"], n),
        "status": rng.choice(np.array(["terminated", "active", None], dtype=object), n, p=[.02, .38, .6]),
        "tender_tenderPeriod_endDate": end,
    })
    df["dateSigned"] = df["tender_tenderPeriod_endDate"] + pd.to_timedelta(rng.normal(25, 60, n).round(), "D")
    df.loc[rng.random(n) < .05, "dateSigned"] = pd.NaT
    pad = rng.random(n)  # names with surrounding spaces, as in the real release
    df.loc[pad < .08, "buyer_name"] = " " + df["buyer_name"] + "  "
    df.loc[(pad > .9) & df["supplier_name"].notna(), "supplier_name"] = df["supplier_name"] + " "
    blank = rng.random(n)  # blank supplier text, which the frozen pipeline treats as reported
    df.loc[blank < .01, "supplier_name"] = " "
    df.loc[(blank >= .01) & (blank < .02), "supplier_name"] = "  "
    df.loc[(blank >= .02) & (blank < .03), "supplier_name"] = ""
    df = df.sort_values("tender_tenderPeriod_endDate", kind="stable").reset_index(drop=True)
    train = (df["tender_tenderPeriod_endDate"].dt.year <= 2023).to_numpy()
    return df[train], df[~train]


Xtr, Xte = periods()
d = prepare(Xtr, Xte)
fitted, tr, thr, _ = fit_train(d, make_models())
bundle = core.build_bundle(d["ib"], d["prep"], fitted, thr, tr, Xte, d["Ztr"][:100], d["names"])
te_scores = {m: fitted[m].score(d["Zte"]) for m in core.MODELS}
frozen_tab = contract_table(d["Fte"], d["ib"].resolver_.transform(normalise_names(d["Fte"]["supplier_name"])),
                        te_scores, tr, thr)  # the frozen batch run's table
meta = {"deployed_model": "eif", "comparison_model": "if", "artifact_sha256": "x", "git_commit": "abc"}
client = TestClient(create_app(bundle, meta, KEY))


def as_json(row):
    """One contract as a client would send it."""
    r = row.iloc[0]
    out = {c: r[c] for c in core.REQUIRED + ["ocid"]}
    for c in core.DATES:
        out[c] = None if pd.isna(r[c]) else r[c].isoformat()
    out["supplier_name"] = None if pd.isna(r["supplier_name"]) else r["supplier_name"]
    out["status"] = None if pd.isna(r["status"]) else r["status"]
    out["value_amount"] = float(r["value_amount"])
    return out


good = as_json(Xte.iloc[[5]])


def post(contracts, headers=H):
    return client.post("/score", json={"contracts": contracts}, headers=headers)


def errors(r):
    return {e["field"]: e["message"] for e in r.json().get("errors", [])}


# 1 health and auth
r = client.get("/health")
results.append(ok(r.status_code == 200 and r.json() == {"status": "ok"}, "/health open, says nothing about the model"))
results.append(ok(client.get("/model").status_code == 401, "/model needs a key"))
results.append(ok(client.get("/model", headers={"X-API-Key": "wrong"}).status_code == 401, "wrong key rejected"))
results.append(ok(post([good], headers={}).status_code == 401, "/score needs a key"))
results.append(ok(post([{"bad": 1}], headers={}).status_code == 401, "key checked before the body is read"))
results.append(ok(client.get("/nowhere").status_code == 401, "unknown paths give nothing away without a key"))
m = client.get("/model", headers=H)
results.append(ok(m.status_code == 200 and m.json()["thresholds"] == bundle["thresholds"]
                  and m.json()["deployed_model"] == "eif", "/model shows the frozen thresholds and deployed model"))
results.append(ok(not any(x in json.dumps(m.json()) for x in ("ocds-", "FIRM ", "KE-B")),
                  "/model holds no contract records"))
try:
    create_app(bundle, meta, "short")
    results.append(ok(False, "short API key refused at startup"))
except StartupError:
    results.append(ok(True, "short API key refused at startup"))

# 2 scoring matches the core, which matches the frozen batch run
r = post([good])
body = r.json()
direct = core.score_rows(bundle, Xte.iloc[[5]])
results.append(ok(r.status_code == 200 and len(body["results"]) == 1, "/score returns one result per contract"))
res = body["results"][0]
results.append(ok(all(res[f"score_{m}"] == float(direct[f"score_{m}"].iloc[0]) for m in core.MODELS),
                  "API scores equal the serving core exactly"))
results.append(ok(res["flag_eif"] == (res["score_eif"] > thr["eif"]) and res["ocid"] == good["ocid"],
                  "flag follows the frozen threshold; ocid echoed"))
results.append(ok("not a corruption finding" in body["note"] and body["latency_ms"] > 0,
                  "response carries the note and latency"))
results.append(ok(set(res) & {"benford_mad", "n_prior_awards", "supplier_resolved"} == set(),
                  "no raw history counts or resolved names returned"))
missing = dict(good, supplier_name=None, dateSigned=None)
r2 = post([missing]).json()["results"][0]
results.append(ok(r2["f4_supplier_concentration"] is None and r2["days_to_signing"] is None,
                  "missing values come back as null, not NaN"))

# 3 dates
naive = dict(good, tender_tenderPeriod_endDate=good["tender_tenderPeriod_endDate"][:19],
             dateSigned=good["dateSigned"][:19] if good["dateSigned"] else None)
results.append(ok(post([naive]).json()["results"][0]["score_eif"] == res["score_eif"],
                  "dates without a timezone read as UTC"))
results.append(ok(post([good]).json()["results"] == body["results"], "same request, same results"))

# 4 contracts in one request are scored independently
many = [as_json(Xte.iloc[[i]]) for i in range(0, 40, 4)]
together = post(many).json()["results"]
alone = [post([c]).json()["results"][0] for c in many]
results.append(ok(together == alone, "batch results equal one-by-one results"))

# 4b status: required key, null allowed, kept as missing
null_pos = int(np.flatnonzero(Xte["status"].isna().to_numpy())[0])
term_pos = int(np.flatnonzero((Xte["status"] == "terminated").to_numpy())[0])
for label, p_ in [("null", null_pos), ("terminated", term_pos)]:
    c = as_json(Xte.iloc[[p_]])
    r = post([c])
    got = r.json()["results"][0] if r.status_code == 200 else {}
    frozen = int(d["Fte"]["is_terminated"].iloc[p_])
    direct = core.score_rows(bundle, Xte.iloc[[p_]])
    results.append(ok(r.status_code == 200 and got["is_terminated"] == frozen == (1 if label == "terminated" else 0)
                      and got["score_eif"] == float(direct["score_eif"].iloc[0]),
                      f"{label} status accepted; is_terminated {got.get('is_terminated')} equals the frozen pipeline"))
c = as_json(Xte.iloc[[null_pos]])
results.append(ok(c["status"] is None, "the null-status test really sends null"))
seen = {}
orig = core.score_rows
core.score_rows = lambda b, df: seen.setdefault("df", df.copy()) is None or orig(b, df)
post([c])
core.score_rows = orig
results.append(ok(seen["df"]["status"].isna().all(), "null status reaches the pipeline as missing, not as text"))
r = post([dict(good, status="active")])
results.append(ok(r.status_code == 200 and r.json()["results"][0]["is_terminated"] == 0, "other status text accepted"))

# 4c names keep their spaces and score exactly like the frozen run
padded = ((Xte["buyer_name"] != Xte["buyer_name"].str.strip()) |
          (Xte["supplier_name"].notna() & (Xte["supplier_name"] != Xte["supplier_name"].str.strip()))).to_numpy()
pp = np.flatnonzero(padded)[:30]
seen = []
orig = core.score_rows
core.score_rows = lambda b, df: seen.append(df.copy()) or orig(b, df)
got = [post([as_json(Xte.iloc[[i]])]).json()["results"][0] for i in pp]
core.score_rows = orig
kept = all(df_["buyer_name"].iloc[0] == Xte["buyer_name"].iloc[i] and
           (df_["supplier_name"].iloc[0] == Xte["supplier_name"].iloc[i] or pd.isna(Xte["supplier_name"].iloc[i]))
           for df_, i in zip(seen, pp))
results.append(ok(len(pp) >= 10 and kept, f"padded names reach the pipeline unchanged ({len(pp)} contracts)"))
cols = ["score_eif", "score_if", "R", "f3_decision_period", "f4_supplier_concentration", "f5_benford_deviation",
        "days_to_signing"]
def same(a, b):
    return (a is None and pd.isna(b)) or (a is not None and not pd.isna(b) and abs(a - b) <= 1e-12)
results.append(ok(all(same(g[c], frozen_tab[c].iloc[i]) for g, i in zip(got, pp) for c in cols) and
                  all(g["flag_eif"] == bool(frozen_tab["flag_eif"].iloc[i]) for g, i in zip(got, pp)),
                  "padded names: scores, flags and features equal the frozen run"))
stripped = [dict(as_json(Xte.iloc[[i]]), buyer_name=Xte["buyer_name"].iloc[i].strip()) for i in pp[:10]]
results.append(ok(all(post([c]).status_code == 200 for c in stripped), "the same names without spaces are also valid"))
nosup = np.flatnonzero(Xte["supplier_name"].isna().to_numpy())[:5]
g2 = [post([as_json(Xte.iloc[[i]])]).json()["results"][0] for i in nosup]
results.append(ok(len(nosup) > 0 and all(g["f4_supplier_concentration"] is None and same(g["score_eif"], frozen_tab["score_eif"].iloc[i])
                                         for g, i in zip(g2, nosup)),
                  "missing supplier: f4 null and scores equal the frozen run"))

# 4d every kind of supplier name is kept as sent and matches the frozen run on all 14 inputs
from serving.api import request_frame
kinds = {"blank ' '": Xte["supplier_name"].eq(" "), "whitespace '  '": Xte["supplier_name"].eq("  "),
         "empty ''": Xte["supplier_name"].eq(""), "null": Xte["supplier_name"].isna(),
         "padded": Xte["supplier_name"].notna() & Xte["supplier_name"].str.strip().ne("")
                   & Xte["supplier_name"].ne(Xte["supplier_name"].str.strip()),
         "ordinary": Xte["supplier_name"].notna() & Xte["supplier_name"].str.strip().ne("")
                     & Xte["supplier_name"].eq(Xte["supplier_name"].str.strip())}
check = ["score_eif", "score_if", "pct_eif", "pct_if", "R", "f3_decision_period", "f4_supplier_concentration",
         "f5_benford_deviation", "days_to_signing", "is_terminated"]
for label, mask in kinds.items():
    idx = np.flatnonzero(mask.to_numpy())[:3]
    good_all = len(idx) > 0
    for i in idx:
        sent = as_json(Xte.iloc[[i]])
        if label != "null":
            sent["supplier_name"] = Xte["supplier_name"].iloc[i]  # exactly the original text
        cap = []
        orig = core.score_rows
        core.score_rows = lambda b, df: cap.append(df.copy()) or orig(b, df)
        r = post([sent])
        core.score_rows = orig
        if r.status_code != 200:
            good_all = False
            break
        g = r.json()["results"][0]
        kept = (pd.isna(cap[0]["supplier_name"].iloc[0]) if label == "null"
                else cap[0]["supplier_name"].iloc[0] == Xte["supplier_name"].iloc[i])
        F = bundle["ib"].transform(core.prepare_rows(cap[0], bundle["buyer_kind"]))
        Z = np.asarray(bundle["prep"].transform(F), dtype=float)
        z_ok = np.allclose(Z[0], d["Zte"][i], rtol=0, atol=1e-12, equal_nan=True)
        f_ok = all(same(g[c], frozen_tab[c].iloc[i]) for c in check)
        flags = all(g[k] == bool(frozen_tab[k].iloc[i]) for k in ("flag_eif", "flag_if"))
        good_all &= kept and z_ok and f_ok and flags
    results.append(ok(good_all, f"supplier {label}: kept as sent; 14 inputs, scores, percentiles, flags and "
                                f"features equal the frozen run ({len(idx)} contracts)"))
r = post([dict(good, supplier_name=" ")]).json()["results"][0]
results.append(ok(r["f4_supplier_concentration"] is not None, "blank supplier text is treated as reported, not missing"))

# 5 validation: clear field-level errors, no input echoed
cases = [
    ("missing field", [{k: v for k, v in good.items() if k != "buyer_name"}], "contracts.0.buyer_name"),
    ("negative value", [dict(good, value_amount=-5)], "contracts.0.value_amount"),
    ("value as text", [dict(good, value_amount="lots")], "contracts.0.value_amount"),
    ("infinite value", [dict(good, value_amount="Infinity")], "contracts.0.value_amount"),
    ("bad date", [dict(good, dateSigned="31/02/2024")], "contracts.0.dateSigned"),
    ("closing before 2018", [dict(good, tender_tenderPeriod_endDate="2015-06-01")], "contracts.0.tender_tenderPeriod_endDate"),
    ("null closing date", [dict(good, tender_tenderPeriod_endDate=None)], "contracts.0.tender_tenderPeriod_endDate"),
    ("unknown method", [dict(good, tender_procurementMethod="lottery")], "contracts.0.tender_procurementMethod"),
    ("unknown category", [dict(good, tender_mainProcurementCategory="consultancy")], "contracts.0.tender_mainProcurementCategory"),
    ("extra field", [dict(good, risk_label=1)], "contracts.0.risk_label"),
    ("empty buyer id", [dict(good, buyer_id="  ")], "contracts.0.buyer_id"),
    ("second contract bad", [good, dict(good, status="")], "contracts.1.status"),
    ("blank buyer name", [dict(good, buyer_name="   ")], "contracts.0.buyer_name"),
    ("empty buyer name", [dict(good, buyer_name="")], "contracts.0.buyer_name"),
    ("buyer name as a number", [dict(good, buyer_name=42)], "contracts.0.buyer_name"),
    ("buyer name as a list", [dict(good, buyer_name=["Ministry"])], "contracts.0.buyer_name"),
    ("supplier name as a number", [dict(good, supplier_name=7)], "contracts.0.supplier_name"),
    ("status key missing", [{k: v for k, v in good.items() if k != "status"}], "contracts.0.status"),
    ("status as a number", [dict(good, status=5)], "contracts.0.status"),
    ("status as a list", [dict(good, status=["terminated"])], "contracts.0.status"),
    ("status as true", [dict(good, status=True)], "contracts.0.status"),
]
for label, contracts, field in cases:
    r = post(contracts)
    e = errors(r)
    shape = all(set(x) == {"field", "message"} for x in r.json().get("errors", []))  # no input echoed
    results.append(ok(r.status_code == 422 and field in e and shape,
                      f"422 for {label}: {e.get(field, r.text)[:70]}"))
r = post([])
results.append(ok(r.status_code == 422 and "contracts" in errors(r), "422 for an empty list"))
r = post([good] * (MAX_BATCH + 1))
results.append(ok(r.status_code == 422 and "contracts" in errors(r), f"422 above {MAX_BATCH} contracts"))
r = client.post("/score", content=b"not json", headers={**H, "Content-Type": "application/json"})
results.append(ok(r.status_code == 422, "422 for a body that is not JSON"))
r = client.post("/score", content=b"x" * 70_000, headers={**H, "Content-Type": "application/json"})
results.append(ok(r.status_code == 413, "413 for an oversized body"))

# 6 numeric buyer ids are accepted and cast
results.append(ok(post([dict(good, buyer_id=12345)]).status_code == 200, "numeric buyer id accepted"))

# 7 resolver cache is capped and results do not change
memo = bundle["ib"].resolver_._memo
before = post([good]).json()["results"]
memo.update({f"FAKE NAME {i}": f"~new:FAKE NAME {i}" for i in range(25_000)})
post([dict(good, supplier_name="BRAND NEW SUPPLIER XYZ")])
results.append(ok(len(bundle["ib"].resolver_._memo) < 25_000, "resolver cache reset above the cap"))
results.append(ok(post([good]).json()["results"] == before, "results unchanged after the cache reset"))

# 8 internal errors return no traceback
broken = copy.copy(bundle)
broken["models"] = dict(bundle["models"])
class Boom:
    def score(self, Z):
        raise RuntimeError("secret internal detail")
broken["models"]["eif"] = Boom()
logging.getLogger("triage.api").disabled = True  # the expected traceback stays out of the test output
c2 = TestClient(create_app(broken, meta, KEY), raise_server_exceptions=False)
r = c2.post("/score", json={"contracts": [good]}, headers=H)
results.append(ok(r.status_code == 500 and r.json() == {"detail": "internal error"}, "500 hides internal details"))

# 9 the artefact must match its metadata
with tempfile.TemporaryDirectory() as t:
    p = pathlib.Path(t) / "serving_bundle.joblib"
    joblib.dump(bundle, p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    p.with_suffix(".json").write_text(json.dumps({"artifact_sha256": h}))
    b2, _ = load_bundle(p)
    results.append(ok(b2["thresholds"] == bundle["thresholds"], "artefact loads when its hash matches"))
    p.with_suffix(".json").write_text(json.dumps({"artifact_sha256": "0" * 64}))
    try:
        load_bundle(p)
        results.append(ok(False, "artefact refused when its hash differs"))
    except StartupError:
        results.append(ok(True, "artefact refused when its hash differs"))

# 10 docs can be switched off
c3 = TestClient(create_app(bundle, meta, KEY, docs=False))
results.append(ok(c3.get("/docs").status_code in (401, 404) and c3.get("/openapi.json", headers=H).status_code == 404,
                  "docs and schema can be switched off"))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
