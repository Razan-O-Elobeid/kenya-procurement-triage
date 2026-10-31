"""Synthetic checks for the supplier resolver. Run: python tests/test_resolver.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, copy, warnings
warnings.simplefilter("error", FutureWarning)
from rapidfuzz import fuzz
from src.features import normalise_names, IndicatorBuilder
from src.entity_resolution import SupplierResolver, core_name

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []

# chain: 25-letter core, one change per step -> X0~X1 (96), X1~X2 (96), X0 vs X2 (92)
x0 = "MAKAMBALAWESTERNTRANSPORT"; x1 = x0[:5] + "Q" + x0[6:]; x2 = x1[:15] + "Z" + x1[16:]
print("chain core scores:", fuzz.ratio(x0, x1), fuzz.ratio(x1, x2), fuzz.ratio(x0, x2))
train_names = (["AAR INSURANCE"] * 5 + ["AAR INSURANCE LTD"] * 2 + ["M/S AAR Insurance"] +
    ["KENYA TONER AND INK SUPPLIERS LTD"] * 3 + ["KENYA TONNER AND INK SUPPLIERS LTD"] +
    ["KEVCY ENTERPRISES"] * 2 + ["KEVICK ENTERPRISES LTD"] +
    ["STIMAL OFFICE SUPPLIES LTD"] * 2 + ["SAI OFFICE SUPPLIES LTD"] +
    ["ACME HARDWARE"] + ["ACME PHARMACY"] +
    [x0] * 4 + [x1] + [x2] * 2 + ["CO LTD"])
r = SupplierResolver().fit(normalise_names(pd.Series(train_names)))
cl = lambda n: r.transform(normalise_names(pd.Series([n])))[0]

# 1 exact core
results.append(ok(cl("AAR INSURANCE") == cl("AAR INSURANCE LTD") == cl("M/S AAR Insurance") == "AAR INSURANCE",
                  "exact core: AAR INSURANCE / LTD / M S merged, canonical = most frequent name"))
# 2 fuzzy at 95
results.append(ok(cl("KENYA TONER AND INK SUPPLIERS LTD") == cl("KENYA TONNER AND INK SUPPLIERS LTD"),
                  "fuzzy >= 95: TONER / TONNER merged"))
results.append(ok(cl("KEVCY ENTERPRISES") != cl("KEVICK ENTERPRISES LTD"),
                  "fuzzy below 95 (core 91): KEVCY / KEVICK kept apart"))
# 3 non-matches
results.append(ok(cl("STIMAL OFFICE SUPPLIES LTD") != cl("SAI OFFICE SUPPLIES LTD"), "non-match: STIMAL / SAI apart"))
results.append(ok(cl("ACME HARDWARE") != cl("ACME PHARMACY"), "non-match: ACME HARDWARE / PHARMACY apart"))
results.append(ok(cl("CO LTD") == "CO LTD", "empty core keeps its own name"))
# 4 chaining
results.append(ok(cl(x0) == cl(x1) == cl(x2) == x0,
                  "chaining: X0-X1 and X1-X2 link, so X0 and X2 join one cluster (direct score 92)"))
print("cluster sizes:", r.clusters_.groupby("id").size().sort_values(ascending=False).to_dict())

# 5 leakage
before = (copy.deepcopy(r.name_map_), copy.deepcopy(r.key_map_), r.n_clusters_)
test = pd.Series(["AAR INSURANCES", "NEWCO VENTURES", "NEWCO VENTURES LTD", "ZEBRA LOGISTICS NAIROBI",
                  "ZEBRA LOGISTICX NAIROBI", "KENYA TONERR AND INK SUPPLIERS", None])
out = r.transform(normalise_names(test))
results.append(ok((r.name_map_, r.key_map_, r.n_clusters_) == before, "transform leaves training identities unchanged"))
results.append(ok(out[0] == "AAR INSURANCE", "unseen test name with a training core joins that training cluster"))
results.append(ok(out[1] == out[2] and out[1].startswith("~new:"), "test-only exact-core variants share an identity (no fitting)"))
results.append(ok(out[3] != out[4], "test-only fuzzy variants are NOT merged with each other"))
results.append(ok(out[5] == cl("KENYA TONER AND INK SUPPLIERS LTD"), "unseen test name fuzzy-matched (>=95) to a training cluster"))
results.append(ok(pd.isna(out[6]), "missing stays missing"))
r2 = SupplierResolver().fit(normalise_names(pd.Series(train_names)))
r2.transform(normalise_names(pd.Series(["SOMETHING ELSE", "AAR INSURANCES"])))
results.append(ok(r2.name_map_ == r.name_map_, "training identities identical whatever test data is seen later"))

# 6 IndicatorBuilder end to end
def contracts(rows, start_id):
    df = pd.DataFrame(rows, columns=["buyer_id", "supplier_name", "date", "value_amount"])
    df.index = range(start_id, start_id + len(df))
    df["tender_tenderPeriod_endDate"] = pd.to_datetime(df["date"])
    df["dateSigned"] = df["tender_tenderPeriod_endDate"] + pd.Timedelta(days=10)
    df["buyer_name"] = "County Government of X"
    df["tender_procurementMethod"] = "open"; df["status"] = "active"
    return df.drop(columns="date")
tr = contracts([("B1", "AAR INSURANCE", "2020-01-01", 100), ("B1", "OTHER CO", "2020-02-01", 100),
                ("B1", "AAR INSURANCE LTD", "2020-03-01", 50), ("B1", None, "2020-04-01", 10)], 0)
te = contracts([("B1", "M/S AAR INSURANCE", "2024-01-01", 70), ("B1", "OTHER CO", "2024-02-01", 30)], 100)
fa = IndicatorBuilder().fit(tr); fb = IndicatorBuilder(resolve_suppliers=False).fit(tr)
f4a = fa.transform(tr)["f4_supplier_concentration"]; f4b = fb.transform(tr)["f4_supplier_concentration"]
print("train f4 resolved:", f4a.round(4).tolist(), "| unresolved:", f4b.round(4).tolist())
results.append(ok(abs(f4a[2] - 100 / 200) < 1e-9 and f4b[2] == 0,
                  "f4 hand check: AAR LTD contract sees 100 of 200 prior value (unresolved: 0)"))
results.append(ok(pd.isna(f4a[3]), "f4 NaN for a missing supplier"))
t4 = fa.transform(te)["f4_supplier_concentration"]
results.append(ok(abs(t4[100] - 150 / 260) < 1e-9, "test contract (M/S AAR) uses training history of its cluster: 150/260"))
base = fa.transform(tr)["f4_supplier_concentration"]
fa.transform(te); again = fa.transform(tr)["f4_supplier_concentration"]
results.append(ok(base.equals(again), "training features unchanged after transforming test data"))
te2 = pd.concat([te, contracts([("B1", "AAR INSURANCE", "2025-06-01", 999)], 200)])
results.append(ok(np.allclose(fa.transform(te2)["f4_supplier_concentration"].loc[[100, 101]], t4.loc[[100, 101]]),
                  "earlier test contracts unaffected by a later test contract"))

# 7 identity collision: an unseen name must not reuse a training name as its id
rc = SupplierResolver().fit(normalise_names(pd.Series(["NEW ABC TRADER", "OTHER FIRM"])))
oc = rc.transform(normalise_names(pd.Series(["NEW ABC TRADER", "ABC TRADER"])))
results.append(ok(oc[0] != oc[1], "unseen ABC TRADER does not collide with training NEW ABC TRADER"))

# 8 later test rows cannot change an earlier test row's identity or f4
early = contracts([("B1", "ZEBRA LOGISTICS", "2024-03-01", 40), ("B1", "AAR INSURANCES", "2024-04-01", 20)], 300)
later = contracts([("B1", "ZEBRA LOGISTICX", "2025-01-01", 500), ("B1", "ZEBRA LOGISTICS LTD", "2025-02-01", 500),
                   ("B1", "AAR INSURANCE", "2025-03-01", 500)], 400)
fresh = IndicatorBuilder().fit(tr)
id_alone = fresh.resolver_.transform(normalise_names(early["supplier_name"])).tolist()
f4_alone = fresh.transform(early)["f4_supplier_concentration"]
fresh2 = IndicatorBuilder().fit(tr)  # new instance, empty cache, sees later names first
fresh2.resolver_.transform(normalise_names(later["supplier_name"]))
id_with = fresh2.resolver_.transform(normalise_names(early["supplier_name"])).tolist()
f4_with = fresh2.transform(pd.concat([early, later]))["f4_supplier_concentration"].loc[early.index]
results.append(ok(id_alone == id_with, "earlier test row keeps its identity whatever later rows (or order) bring"))
results.append(ok(np.allclose(f4_alone, f4_with, equal_nan=True), "earlier test row keeps its f4 when later rows are added"))

# 9 deterministic tie-breaking: equal scores go to the more frequent training name
ta = "KAMBALAWESTERNTRANSPORTER"            # 25 letters
tb = ta[:3] + "X" + ta[4:12] + "Y" + ta[13:]   # two letters different (92), stays separate
tq = ta[:3] + "X" + ta[4:]                   # one letter from each (96 and 96)
rt = SupplierResolver().fit(normalise_names(pd.Series([ta] * 3 + [tb])))
print("tie scores:", fuzz.ratio(tq, ta), fuzz.ratio(tq, tb), "| training pair:", round(fuzz.ratio(ta, tb), 1))
results.append(ok(rt.n_clusters_ == 2 and rt.transform(pd.Series([tq]))[0] == ta,
                  "tie between two training firms goes to the more frequent one"))

print(f"\n{sum(results)}/{len(results)} tests passed")
