"""Checks for the relative-value feature (protocol 7.3). Run: python tests/test_relative_value.py"""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore", UserWarning)
from sklearn.base import clone
from src.features import IndicatorBuilder
from src.pipeline import build_preprocessor
from src.tuning import inject
from model_evaluation.harness import inject_value
from model_evaluation.relative_value import RelativeValue, build_preprocessor_relative, COLUMNS

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(0)
LOG50, LOG5 = np.log(50), np.log(5)


def table(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def contracts(n, y0, y1):
    df = pd.DataFrame({
        "buyer_id": rng.choice([f"B{i}" for i in range(40)] + [None], n),
        "buyer_name": rng.choice(["County Government of X", "Ministry of Y"], n),
        "supplier_name": rng.choice([f"FIRM {i} LTD" for i in range(400)] + [None], n),
        "value_amount": np.where(rng.random(n) < .01, 0, rng.lognormal(13, 1.5, n)),
        "tender_mainProcurementCategory": rng.choice(["goods", "works", "services", None], n, p=[.4, .3, .25, .05]),
        "tender_procurementMethod": rng.choice(["open", "selective", "direct"], n, p=[.9, .05, .05]),
        "status": "active",
        "tender_tenderPeriod_endDate": pd.to_datetime(y0) + pd.to_timedelta(
            rng.integers(0, (pd.Timestamp(y1) - pd.Timestamp(y0)).days, n), "D"),
    })
    df["dateSigned"] = df["tender_tenderPeriod_endDate"] + pd.to_timedelta(np.abs(rng.normal(25, 20, n)).round(), "D")
    return df


# 1 references come from training rows only, and transform never changes them
tr = table([[np.exp(10 + i % 3), "B1", "goods"] for i in range(10)])
va = table([[1e12, "B1", "goods"], [1e12, "B9", "works"]])
rv = RelativeValue("buyer").fit(tr)
before = (rv.overall_median_, rv.buyers_.copy(), rv.categories_.copy(), rv.lo_, rv.hi_)
rv.transform(va)
after = (rv.overall_median_, rv.buyers_, rv.categories_, rv.lo_, rv.hi_)
results.append(ok(np.isclose(rv.buyers_.loc["B1", "median"], 11.0), "buyer median fitted on training rows"))
results.append(ok(before[0] == after[0] and before[1].equals(after[1]) and before[2].equals(after[2])
                  and before[3:] == after[3:], "transform leaves fitted references unchanged"))

# 2 buyer fallbacks: >= 5 awards -> buyer, else category group, else overall; missing buyer id -> category
tr = table([[np.exp(10), "BIG", "goods"]] * 6 + [[np.exp(20), "SMALL", "goods"]] * 2
           + [[np.exp(12), "OTHER", "goods"]] * 4 + [[np.exp(30), "X", "works"]] * 2)
rv = RelativeValue("buyer").fit(tr)
q = table([[np.exp(15), "BIG", "goods"], [np.exp(15), "SMALL", "goods"], [np.exp(15), None, "goods"],
           [np.exp(15), "X", "works"], [np.exp(15), "NEW", "unseen"]])
lvl = list(rv.reference_level(q))
goods_med = np.median([10] * 6 + [20] * 2 + [12] * 4)
overall = np.median([10] * 6 + [20] * 2 + [12] * 4 + [30] * 2)
r = rv.raw(q)
results.append(ok(lvl == ["buyer", "category", "category", "overall", "overall"], f"buyer fallback levels {lvl}"))
results.append(ok(np.allclose(r, [15 - 10, 15 - goods_med, 15 - goods_med, 15 - overall, 15 - overall]),
                  "buyer fallback values"))

# 3 missing category is its own group (>= 5 awards), otherwise overall
tr = table([[np.exp(8), "A", None]] * 5 + [[np.exp(14), "A", "goods"]] * 10)
rv = RelativeValue("category").fit(tr)
results.append(ok("__missing__" in rv.categories_.index, "missing category kept as its own group"))
results.append(ok(rv.reference_level(table([[np.exp(9), "A", None]]))[0] == "category", "missing category uses its group"))
tr4 = table([[np.exp(8), "A", None]] * 4 + [[np.exp(14), "A", "goods"]] * 10)
rv4 = RelativeValue("category").fit(tr4)
results.append(ok(rv4.reference_level(table([[np.exp(9), "A", None]]))[0] == "overall",
                  "missing category with < 5 awards falls back to overall"))

# 4 category z-score: own MAD, zero group MAD -> overall MAD, all zero -> denominator 1
vals = [10, 11, 12, 13, 14, 15]
tr = table([[np.exp(v), "A", "goods"] for v in vals] + [[np.exp(20), "A", "works"]] * 6)
rv = RelativeValue("category").fit(tr)
mad_g = np.median(np.abs(np.array(vals) - np.median(vals)))
r = rv.raw(table([[np.exp(16), "A", "goods"], [np.exp(22), "A", "works"]]))
all_vals = vals + [20] * 6
mad_all = np.median(np.abs(np.array(all_vals) - np.median(all_vals)))
results.append(ok(np.isclose(r[0], (16 - np.median(vals)) / (1.4826 * mad_g)), "category z-score with own MAD"))
results.append(ok(np.isclose(r[1], (22 - 20) / (1.4826 * mad_all)), "zero group MAD uses overall MAD and own median"))
rv0 = RelativeValue("category").fit(table([[np.exp(7), "A", "goods"]] * 6))
results.append(ok(np.isclose(rv0.raw(table([[np.exp(9), "A", "goods"]]))[0], 2.0), "all MADs zero gives denominator 1"))

# 5 invalid values give 0
rv = RelativeValue("buyer").fit(table([[np.exp(10 + i), "A", "goods"] for i in range(6)]))
bad = table([[0, "A", "goods"], [-5, "A", "goods"], [np.nan, "A", "goods"], [np.inf, "A", "goods"], ["x", "A", "goods"]])
results.append(ok(np.all(rv.transform(bad) == 0), "zero, negative, missing, infinite and text values give 0"))

# 6 unclipped input: two values above the Phase 2 P99 still differ; clipping at training P0.1 / P99.9
tr = table([[v, b, "goods"] for v, b in zip(rng.lognormal(13, 1.5, 5000), rng.choice(["A", "B", "C"], 5000))])
rv = RelativeValue("buyer").fit(tr)
p99 = tr["value_amount"].quantile(0.99)
two = rv.transform(table([[p99 * 1.5, "A", "goods"], [p99 * 2.5, "A", "goods"]])).ravel()
results.append(ok(two[1] > two[0], "values above the Phase 2 P99 are not capped together"))
raw_tr = rv.raw(tr)
results.append(ok(np.isclose(rv.lo_, np.quantile(raw_tr, 0.001)) and np.isclose(rv.hi_, np.quantile(raw_tr, 0.999)),
                  "clip limits are the training P0.1 / P99.9 of the feature"))
results.append(ok(rv.transform(table([[p99 * 1e6, "A", "goods"]]))[0, 0] == rv.hi_, "extreme values clipped at P99.9"))

# 7 pipeline: 14 vs 15 inputs, first 14 identical; injected copies recomputed against unchanged references
train, valid = contracts(3000, "2018-01-01", "2021-12-31"), contracts(1000, "2022-01-01", "2022-12-31")
ib = IndicatorBuilder().fit(train)
Ftr, Fva = ib.transform(train), ib.transform(valid)
base = clone(build_preprocessor()).fit(Ftr)
for variant in ("buyer", "category"):
    ext = build_preprocessor_relative(variant).fit(Ftr)
    Zb, Ze = base.transform(Fva), ext.transform(Fva)
    names = ext.get_feature_names_out()
    results.append(ok(Zb.shape[1] == 14 and Ze.shape[1] == 15, f"{variant}: 14 base inputs, 15 extended"))
    results.append(ok(np.array_equal(Zb, Ze[:, :14]), f"{variant}: first 14 columns identical to the base pipeline"))
    results.append(ok(names[-1] == f"relvalue__relative_value_{variant}", f"{variant}: last column {names[-1]}"))

    tv = ext.named_transformers_["relvalue"]
    rel, scaler = tv.steps[0][1], tv.steps[1][1]
    refs = (rel.buyers_.copy(), rel.categories_.copy(), rel.lo_, rel.hi_, scaler.mean_.copy(), scaler.scale_.copy())
    for kind, factor, G in (("x50", LOG50, inject(Fva, ib, "value", 0.05, seed=0)),
                            ("x5", LOG5, inject_value(Fva, ib, 5, 0.05, seed=0))):
        orig = rel.raw(Fva.loc[G.index, COLUMNS])
        new = rel.raw(G[COLUMNS])
        if variant == "buyer":
            good = np.allclose(new - orig, factor)
        else:
            # same group reference, so within a group the shift is log(factor) over one fixed denominator
            d = pd.Series(new - orig)
            grp = G[COLUMNS[2]].fillna("__missing__").reset_index(drop=True)
            good = bool((d > 0).all() and d.groupby(grp).agg(lambda s: np.ptp(s) < 1e-9).all())
        results.append(ok(good, f"{variant}: {kind} copies shifted against the training reference"))
        Zg = ext.transform(G)
        inside = (new > rel.lo_) & (new < rel.hi_) & (orig > rel.lo_) & (orig < rel.hi_)
        dz = (Zg[:, 14] - ext.transform(Fva.loc[G.index])[:, 14]) * scaler.scale_[0]
        results.append(ok(np.allclose(dz[inside], (new - orig)[inside]), f"{variant}: {kind} shift reaches the model input"))
    same = (rel.buyers_.equals(refs[0]) and rel.categories_.equals(refs[1]) and (rel.lo_, rel.hi_) == refs[2:4]
            and np.array_equal(scaler.mean_, refs[4]) and np.array_equal(scaler.scale_, refs[5]))
    results.append(ok(same, f"{variant}: references unchanged after scoring copies"))

# 8 committed adoption rule and variant tie-breaker (relative_value_eval.adopt / choose)
from model_evaluation.relative_value_eval import adopt, choose

base = {"recall_value": 0.10, "recall_value_x5": 0.06, "recall_primary": 0.48, "stability": 0.99}


def row(value=0.15, x5=0.10, primary=0.50, stability=0.99):
    return {"recall_value": value, "recall_value_x5": x5, "recall_primary": primary, "stability": stability}


def decision(passes, primary):
    return {"passes": passes, "primary": primary}


results.append(ok(adopt(row(), base)["passes"], "rule: all requirements met is adopted"))
results.append(ok(not adopt(row(x5=0.07), base)["passes"], "rule: fails only the x5 gain, not adopted"))
results.append(ok(not adopt(row(value=0.11), base)["passes"], "rule: fails only the value gain, not adopted"))
results.append(ok(not adopt(row(primary=0.47), base)["passes"], "rule: primary recall falls, not adopted"))
results.append(ok(not adopt(row(stability=0.94), base)["passes"], "rule: stability below 0.95, not adopted"))
results.append(ok(adopt(row(primary=0.48), base)["passes"], "rule: primary recall unchanged still allowed"))
results.append(ok(choose({"buyer": decision(False, 0.5), "category": decision(False, 0.6)}) is None,
                  "choice: neither passes, none adopted"))
results.append(ok(choose({"buyer": decision(False, 0.6), "category": decision(True, 0.5)}) == "category",
                  "choice: only category passes, category"))
results.append(ok(choose({"buyer": decision(True, 0.5), "category": decision(False, 0.6)}) == "buyer",
                  "choice: only buyer passes, buyer"))
results.append(ok(choose({"buyer": decision(True, 0.50), "category": decision(True, 0.515)}) == "buyer",
                  "choice: both pass, category ahead by 0.015, buyer"))
results.append(ok(choose({"buyer": decision(True, 0.50), "category": decision(True, 0.53)}) == "category",
                  "choice: both pass, category ahead by 0.03, category"))
results.append(ok(choose({"buyer": decision(True, 0.53), "category": decision(True, 0.50)}) == "buyer",
                  "choice: both pass, buyer ahead, buyer"))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
