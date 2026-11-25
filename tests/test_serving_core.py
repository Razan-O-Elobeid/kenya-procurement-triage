"""Synthetic checks for the serving core and the export gates. Run: python tests/test_serving_core.py"""
import sys, pathlib, copy, tempfile, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore", message="Attempting to use more than 1 thread")

from model_evaluation.final_eval import fit_train, make_models, prepare
from model_evaluation.explain_checks import background
from dashboard.bundle import contract_table
from src.features import IndicatorBuilder, normalise_names
from serving import core, export_artifacts as ex

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(0)


def periods(n=5000, buyers=30):
    """Synthetic release split like temporal_split: sorted by closing date, train indices first."""
    end = pd.Timestamp("2018-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 3100, n), "D")
    df = pd.DataFrame({
        "ocid": [f"ocds-{i:06d}" for i in range(n)],
        "buyer_id": rng.choice([f"KE-B{i}" for i in range(buyers)], n),
        "buyer_name": rng.choice(["County Government of X", "Ministry of Y", "Authority Z"], n),
        "supplier_name": rng.choice([f"FIRM {i} LTD" for i in range(300)] + [None], n),
        "value_amount": np.where(rng.random(n) < .01, 0, rng.lognormal(13, 2, n)),
        "tender_mainProcurementCategory": rng.choice(["goods", "works", "services"], n),
        "tender_procurementMethod": rng.choice(["open", "selective", "direct"], n),
        "status": rng.choice(np.array(["terminated", "active", None], dtype=object), n, p=[.02, .38, .6]),
        "tender_tenderPeriod_endDate": end,
    })
    df["dateSigned"] = df["tender_tenderPeriod_endDate"] + pd.to_timedelta(rng.normal(25, 60, n).round(), "D")
    df.loc[rng.random(n) < .05, "dateSigned"] = pd.NaT
    df = df.sort_values("tender_tenderPeriod_endDate", kind="stable").reset_index(drop=True)
    train = (df["tender_tenderPeriod_endDate"].dt.year <= 2023).to_numpy()
    return df[train], df[~train]


Xtr, Xte = periods()

# frozen-style reference run, as final_eval did it
d = prepare(Xtr, Xte)
fitted, tr, thr, _ = fit_train(d, make_models())
te = {m: fitted[m].score(d["Zte"]) for m in core.MODELS}
resolved = d["ib"].resolver_.transform(normalise_names(d["Fte"]["supplier_name"]))
with tempfile.TemporaryDirectory() as t:  # csv round trip, like the real files
    contract_table(d["Fte"], resolved, te, tr, thr).to_csv(f"{t}/c.csv.gz", index=False)
    pd.DataFrame({m: tr[m] for m in core.MODELS}).to_csv(f"{t}/t.csv.gz", index=False)
    contracts, train_scores = pd.read_csv(f"{t}/c.csv.gz"), pd.read_csv(f"{t}/t.csv.gz")
bundle = core.build_bundle(d["ib"], d["prep"], fitted, thr, tr, Xte, background(d["Ztr"], 0), d["names"])
ib_ext = bundle["ib"]

# 1 prepare_rows
row = Xte.iloc[[0]]
def raises(f):
    try:
        f()
        return False
    except (ValueError, TypeError):
        return True
results.append(ok(raises(lambda: core.prepare_rows(row.drop(columns="buyer_name"))), "missing column rejected"))
bad = row.copy(); bad["dateSigned"] = "not a date"
results.append(ok(raises(lambda: core.prepare_rows(bad)), "unparseable date rejected"))
bad = row.copy(); bad["tender_tenderPeriod_endDate"] = pd.NaT
results.append(ok(raises(lambda: core.prepare_rows(bad)), "missing closing date rejected"))
bad = row.copy(); bad["value_amount"] = np.nan
results.append(ok(raises(lambda: core.prepare_rows(bad)), "missing value rejected"))
results.append(ok(raises(lambda: core.prepare_rows(row.iloc[[]])), "empty request rejected"))
naive = row.copy()
for c in core.DATES:
    naive[c] = naive[c].dt.tz_localize(None)
iso = row.copy()
for c in core.DATES:
    iso[c] = iso[c].dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
a, b, c_ = core.prepare_rows(row), core.prepare_rows(naive), core.prepare_rows(iso)
results.append(ok(a["dateSigned"].equals(b["dateSigned"]) and str(a["dateSigned"].dt.tz) == "UTC",
                  "naive dates read as UTC"))
results.append(ok((c_["tender_tenderPeriod_endDate"] - a["tender_tenderPeriod_endDate"]).iloc[0]
                  == pd.Timedelta(hours=-3), "offset strings converted to UTC"))
p = core.prepare_rows(Xte.iloc[:5])
results.append(ok(list(p.index) == [-1, -2, -3, -4, -5], "incoming rows get negative indices"))
results.append(ok(bundle["buyer_kind"] == "str", "string buyer ids recorded"))
num = row.copy(); num["buyer_id"] = "17"
results.append(ok(core.prepare_rows(num, "int64")["buyer_id"].dtype == np.int64, "numeric buyer ids cast to the history type"))

# 2 history extension
results.append(ok(core.fitted_state_hash(ib_ext) == core.fitted_state_hash(d["ib"]), "fitted references unchanged"))
results.append(ok(len(d["ib"].history_) == len(Xtr), "original builder not changed"))
results.append(ok(len(ib_ext.history_) == len(Xtr) + len(Xte), "history holds training and test-period awards"))
results.append(ok(raises(lambda: core.extend_history(d["ib"], Xtr.iloc[:3])), "overlapping indices rejected"))
results.append(ok(raises(lambda: core.extend_history(d["ib"], Xte.iloc[:3].set_axis([-1, -2, -3]))),
                  "negative history indices rejected"))

# 3 single-contract serving path equals the frozen batch run
pos = np.arange(0, len(Xte), 7)
single = pd.concat([core.score_rows(bundle, Xte.iloc[[i]]) for i in pos], ignore_index=True)
ref = contracts.iloc[pos].reset_index(drop=True)
results.append(ok(all(ex.max_diff(single[f"score_{m}"], ref[f"score_{m}"]) <= 1e-12 for m in core.MODELS),
                  f"single-contract scores equal the frozen batch ({len(pos)} contracts)"))
results.append(ok(all(ex.max_diff(single[c], ref[c]) <= 1e-12 for c in ex.FEATURES),
                  "single-contract history features equal the frozen batch"))
nulls = Xte["status"].iloc[pos].isna().to_numpy()
results.append(ok(nulls.any() and (~nulls).any() and np.array_equal(single["is_terminated"], ref["is_terminated"]),
                  f"is_terminated equals the frozen batch ({int(nulls.sum())} null and {int((~nulls).sum())} non-null status)"))
results.append(ok(all(np.array_equal(single[f"flag_{m}"], ref[f"flag_{m}"].astype(bool)) and
                      ex.max_diff(single[f"pct_{m}"], ref[f"pct_{m}"]) <= 1e-12 for m in core.MODELS),
                  "flags and percentiles equal the frozen batch"))
Fs = pd.concat([ib_ext.transform(core.prepare_rows(Xte.iloc[[i]])) for i in pos[:40]])
Zs = np.asarray(d["prep"].transform(Fs))
results.append(ok(np.allclose(Zs, d["Zte"][pos[:40]], rtol=0, atol=1e-12), "all 14 model inputs equal"))

# 4 contracts in one request do not see each other; order does not matter
new = Xte.iloc[[10, 20, 30, 40, 50]].copy()
new["ocid"] = [f"new-{i}" for i in range(5)]
new["buyer_id"] = "KE-NEW"  # a buyer with no history
new["tender_tenderPeriod_endDate"] = pd.Timestamp("2027-01-01", tz="UTC") + pd.to_timedelta([0, 10, 20, 30, 40], "D")
together = core.score_rows(bundle, new)
alone = pd.concat([core.score_rows(bundle, new.iloc[[i]]) for i in range(5)], ignore_index=True)
cols = [f"score_{m}" for m in core.MODELS] + ex.FEATURES
results.append(ok(all(ex.max_diff(together[c], alone[c]) == 0 for c in cols), "batch result equals one-by-one"))
results.append(ok((together["n_prior_awards"] == 0).all(), "later contracts in a request do not count earlier ones"))
shuf = core.score_rows(bundle, new.iloc[::-1]).iloc[::-1].reset_index(drop=True)
results.append(ok(all(ex.max_diff(shuf[c], together[c]) == 0 for c in cols), "request order does not matter"))

# 5 scoring changes no state
h0, s0 = core.history_hash(ib_ext), core.fitted_state_hash(ib_ext)
first = core.score_rows(bundle, Xte.iloc[[3]])
core.score_rows(bundle, Xte.iloc[100:130])
second = core.score_rows(bundle, Xte.iloc[[3]])
results.append(ok(core.history_hash(ib_ext) == h0 and core.fitted_state_hash(ib_ext) == s0,
                  "history and fitted state unchanged after scoring"))
results.append(ok(first.equals(second), "same contract, same result, whatever was scored before"))

# 6 no future information
target = Xte[Xte["value_amount"] > 0].iloc[[200]]
base = core.score_rows(bundle, target)
later = target.copy(); later.index = [10**8]
later["tender_tenderPeriod_endDate"] += pd.Timedelta(days=30)
earlier = target.copy(); earlier.index = [10**8 + 1]
earlier["tender_tenderPeriod_endDate"] -= pd.Timedelta(days=30)
b_later = copy.copy(bundle); b_later["ib"] = core.extend_history(ib_ext, later)
b_earlier = copy.copy(bundle); b_earlier["ib"] = core.extend_history(ib_ext, earlier)
results.append(ok(core.score_rows(b_later, target)[cols].equals(base[cols]), "a later award changes nothing"))
results.append(ok(core.score_rows(b_earlier, target)["n_prior_awards"].iloc[0] == base["n_prior_awards"].iloc[0] + 1,
                  "an earlier award is counted"))
same_day = target.copy(); same_day.index = [10**8 + 2]
b_same = copy.copy(bundle); b_same["ib"] = core.extend_history(ib_ext, same_day)
results.append(ok(core.score_rows(b_same, target)[cols].equals(base[cols]), "a same-day award is not counted"))

# 7 threshold and percentile logic
b_thr = copy.copy(bundle); b_thr["thresholds"] = dict(bundle["thresholds"], eif=float(base["score_eif"].iloc[0]))
results.append(ok(not core.score_rows(b_thr, target)["flag_eif"].iloc[0], "flag is strictly above the threshold"))
results.append(ok(base["pct_eif"].iloc[0] == (np.asarray(tr["eif"]) <= base["score_eif"].iloc[0]).mean(),
                  "percentile is against training scores"))

# 8 joblib round trip
with tempfile.TemporaryDirectory() as t:
    joblib.dump(bundle, f"{t}/b.joblib")
    back = joblib.load(f"{t}/b.joblib")
r1, r2 = core.score_rows(bundle, Xte.iloc[:25]), core.score_rows(back, Xte.iloc[:25])
results.append(ok(r1.equals(r2), "reloaded bundle gives identical results"))
results.append(ok(ex.model_hash(back["models"]["eif"]) == ex.model_hash(bundle["models"]["eif"]),
                  "EIF pickles to the same bytes after reload"))

# 9 export gates on synthetic data
expected = {"n": (len(Xtr), len(Xte)), "thr": thr, "flags": {m: int((te[m] > thr[m]).sum()) for m in core.MODELS}}
refs = {"thresholds": thr, "train_scores": train_scores, "contracts": contracts}
with tempfile.TemporaryDirectory() as t:
    out = {"model": pathlib.Path(t) / "s.joblib", "meta": pathlib.Path(t) / "s.json"}
    code = ex.run(out, Xtr, Xte, refs, expected, n_random=30, n_top=10)
    results.append(ok(code == 0 and out["model"].exists() and out["meta"].exists(), "export saves when every gate passes"))
    stamp = out["model"].stat().st_mtime_ns
    results.append(ok(ex.run(out, Xtr, Xte, refs, expected, 5, 5) == 1 and out["model"].stat().st_mtime_ns == stamp,
                      "export refuses to overwrite"))
def tamper_score(r, e):
    r["contracts"].loc[0, "score_eif"] += 1e-9

def tamper_threshold(r, e):
    e["thr"]["eif"] += 1e-15

def tamper_train(r, e):
    r["train_scores"].loc[:, "if"] *= 1 + 1e-9

def tamper_split(r, e):
    e["n"] = (len(Xtr) + 1, len(Xte))

print("\nfour deliberate failures follow; each FAIL and STOPPED line below is expected")
for label, change in [("tampered test score", tamper_score), ("wrong threshold", tamper_threshold),
                      ("wrong training scores", tamper_train), ("wrong split size", tamper_split)]:
    r = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in refs.items()}
    e = copy.deepcopy(expected)
    change(r, e)
    with tempfile.TemporaryDirectory() as t:
        out = {"model": pathlib.Path(t) / "s.joblib", "meta": pathlib.Path(t) / "s.json"}
        code = ex.run(out, Xtr, Xte, r, e, n_random=5, n_top=5)
        left = list(pathlib.Path(t).iterdir())
    results.append(ok(code == 1 and not left, f"export stops and saves nothing: {label}"))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
