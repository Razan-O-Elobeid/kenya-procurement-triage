"""Synthetic checks for the transformers and the full pipeline. Run: python tests/test_pipeline.py"""
import sys, pathlib, copy, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("error", FutureWarning)
from src.transformers import ClipGroupImputer, GroupModeImputer, signed_log
from src.pipeline import build_pipeline, FLAGS

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(0)


def contracts(n, start, y0, y1, methods=("open", "selective", "direct")):
    cat = rng.choice(["goods", "works", "services", None], n, p=[.5, .3, .19, .01])
    df = pd.DataFrame({
        "buyer_id": rng.choice([f"B{i}" for i in range(40)], n),
        "buyer_name": rng.choice(["County Government of X", "Ministry of Y", "Authority Z"], n),
        "supplier_name": rng.choice([f"FIRM {i} LTD" for i in range(400)] + [None], n),
        "value_amount": np.where(rng.random(n) < .01, 0, rng.lognormal(13, 2, n)),
        "tender_mainProcurementCategory": cat,
        "tender_procurementMethod": rng.choice(list(methods), n),
        "status": np.where(rng.random(n) < .003, "terminated", "active"),
        "tender_tenderPeriod_endDate": pd.to_datetime(y0) + pd.to_timedelta(rng.integers(0, (pd.Timestamp(y1) - pd.Timestamp(y0)).days, n), "D"),
    })
    df["dateSigned"] = df["tender_tenderPeriod_endDate"] + pd.to_timedelta(rng.normal(25, 60, n).round(), "D")
    df.loc[rng.random(n) < .06, "dateSigned"] = pd.NaT
    df.index = range(start, start + n)
    return df


# 1 ClipGroupImputer
v = pd.DataFrame({"v": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, np.nan], "g": list("aaaaabbbbbb") + ["a"]})
cgi = ClipGroupImputer(0.1, 0.9, zero_as_missing=True).fit(v)
obs = v["v"].where(v["v"] != 0).dropna()
results.append(ok(np.isclose(cgi.lo_, obs.quantile(.1)) and np.isclose(cgi.hi_, obs.quantile(.9)),
                  "clip thresholds = train quantiles of observed non-zero values"))
out = cgi.transform(v).ravel()
results.append(ok(out[0] == cgi.medians_["a"] and out[-1] == cgi.medians_["a"], "zero and NaN filled with group median"))
results.append(ok(out[10] == cgi.hi_, "extreme value clipped to upper threshold"))
new = cgi.transform(pd.DataFrame({"v": [np.nan], "g": ["zzz"]})).ravel()
results.append(ok(new[0] == cgi.overall_, "unseen group falls back to overall median"))

# 2 GroupModeImputer
c = pd.DataFrame({"c": ["x", "x", "y", None, "y", "y", None], "g": ["a", "a", "a", "a", "b", "b", "zz"]})
gmi = GroupModeImputer().fit(c.iloc[:6])
o = gmi.transform(c).ravel()
results.append(ok(o[3] == "x" and o[6] == gmi.overall_, "category filled with group mode, unseen group with overall mode"))
results.append(ok(np.allclose(signed_log([-9, 0, 9]), [-np.log(10), 0, np.log(10)]), "signed log is symmetric"))

# 3 full pipeline
tr = contracts(4000, 0, "2018-07-01", "2023-12-31")
te = contracts(1500, 10000, "2024-01-01", "2026-06-30", methods=("open", "selective", "direct", "NewMethodSpelling"))
te.loc[te.index[:5], "tender_mainProcurementCategory"] = "consultancy"  # unseen category
pipe = build_pipeline(n_estimators=50).fit(tr)
prep = pipe.named_steps["prep"]
names = list(prep.get_feature_names_out())
Ztr = prep.transform(pipe.named_steps["indicators"].transform(tr))
Zte = prep.transform(pipe.named_steps["indicators"].transform(te))
results.append(ok(Ztr.shape[0] == len(tr) and Zte.shape[0] == len(te), "rows preserved for train and test"))
results.append(ok(np.isfinite(Ztr).all() and np.isfinite(Zte).all(), "no NaN or inf after preprocessing (incl. unseen method and category)"))
results.append(ok(not any("f1_" in n or "f3_" in n for n in names), "f1 and f3 not among model inputs"))
results.append(ok(all(any(f in n for n in names) for f in FLAGS), "all five flags among model inputs"))
results.append(ok(np.allclose(Ztr[:, :2].mean(0), 0, atol=1e-8) and np.allclose(Ztr[:, :2].std(0), 1, atol=1e-6),
                  "value and days standardised on train"))

# 4 thresholds match step 4 and IndicatorBuilder
ib = pipe.named_steps["indicators"]
vc = prep.named_transformers_["value"].steps[0][1]
pos = tr["value_amount"][tr["value_amount"] > 0]
results.append(ok(np.isclose(vc.lo_, ib.p1_) and np.isclose(vc.hi_, ib.p99_) and np.isclose(vc.hi_, pos.quantile(.99)),
                  "value clip thresholds equal the below_p1 / above_p99 thresholds (positive train values)"))

# 5 leakage
before = copy.deepcopy((vc.lo_, vc.hi_, vc.medians_, prep.named_transformers_["days"].steps[0][1].medians_,
                        prep.named_transformers_["value"].steps[2][1].mean_.copy()))
pipe.score_samples(te)
after = (vc.lo_, vc.hi_, vc.medians_, prep.named_transformers_["days"].steps[0][1].medians_,
         prep.named_transformers_["value"].steps[2][1].mean_)
results.append(ok(before[:4] == after[:4] and np.allclose(before[4], after[4]), "fitted parameters unchanged by scoring test data"))
F_te = ib.transform(te)
part = prep.transform(F_te.iloc[:300])
results.append(ok(np.allclose(part, prep.transform(F_te)[:300]), "preprocessing is row-wise: a test row does not depend on other test rows"))
s = pipe.score_samples(te)
results.append(ok(len(s) == len(te) and np.isfinite(s).all(), "Isolation Forest scores every test contract"))

print(f"\n{sum(results)}/{len(results)} tests passed")
