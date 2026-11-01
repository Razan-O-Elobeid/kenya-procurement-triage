"""Synthetic checks for tuning and anomaly injection. Run: python tests/test_tuning.py"""
import sys, pathlib, copy, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
warnings.simplefilter("ignore", UserWarning)
from sklearn.ensemble import IsolationForest
from src.features import IndicatorBuilder
from src.pipeline import build_preprocessor
from src.tuning import inject, run_grid, summarise

ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []
rng = np.random.default_rng(0)


def contracts(n, y0, y1):
    df = pd.DataFrame({
        "buyer_id": rng.choice([f"B{i}" for i in range(40)], n),
        "buyer_name": rng.choice(["County Government of X", "Ministry of Y"], n),
        "supplier_name": rng.choice([f"FIRM {i} LTD" for i in range(400)] + [None], n),
        "value_amount": np.where(rng.random(n) < .01, 0, rng.lognormal(13, 1.5, n)),
        "tender_mainProcurementCategory": rng.choice(["goods", "works", "services"], n),
        "tender_procurementMethod": rng.choice(["open", "selective", "direct"], n, p=[.9, .05, .05]),
        "status": "active",
        "tender_tenderPeriod_endDate": pd.to_datetime(y0) + pd.to_timedelta(rng.integers(0, (pd.Timestamp(y1) - pd.Timestamp(y0)).days, n), "D"),
    })
    df["dateSigned"] = df["tender_tenderPeriod_endDate"] + pd.to_timedelta(np.abs(rng.normal(25, 20, n)).round(), "D")
    df.loc[rng.random(n) < .05, "dateSigned"] = pd.NaT
    return df


X = contracts(6000, "2018-07-01", "2023-12-31")
tr, va = X.iloc[:4000], X.iloc[4000:]
ib = IndicatorBuilder().fit(tr)
Fva = ib.transform(va)
before = copy.deepcopy(Fva)
params = (ib.p1_, ib.p99_, len(ib.history_))

# 1 value injection
G = inject(Fva, ib, "value", 0.02, seed=0)
orig = Fva.loc[G.index]
results.append(ok(len(G) == int(0.02 * len(Fva)), "2% of the validation fold injected"))
results.append(ok(np.allclose(G["value_amount"], orig["value_amount"] * 50) and (orig["value_amount"] > 0).all(),
                  "value injection multiplies non-zero values by 50"))
results.append(ok((G["above_p99"] == (G["value_amount"] > ib.p99_).astype(int)).all()
                  and (G["below_p1"] == ((G["value_amount"] > 0) & (G["value_amount"] < ib.p1_)).astype(int)).all(),
                  "value flags recomputed with the fitted thresholds"))
same = ["f4_supplier_concentration", "f5_benford_deviation", "days_to_signing", "method_level"]
results.append(ok(G[same].equals(orig[same]), "f4, f5, days and method unchanged by value injection"))

# 2 date injection
D = inject(Fva, ib, "date", 0.02, seed=1)
results.append(ok((D["days_to_signing"] == -200).all() and (D["is_signed_pre_closure"] == 1).all()
                  and Fva.loc[D.index, "days_to_signing"].notna().all(),
                  "date injection: signed 200 days before close, only rows that had a signing date"))
results.append(ok(D[["value_amount", "f4_supplier_concentration", "f5_benford_deviation"]].equals(
                  Fva.loc[D.index, ["value_amount", "f4_supplier_concentration", "f5_benford_deviation"]]),
                  "value, f4 and f5 unchanged by date injection"))

# 3 no contamination
results.append(ok(Fva.equals(before), "clean validation features untouched by injection"))
results.append(ok((ib.p1_, ib.p99_, len(ib.history_)) == params, "fitted thresholds and histories untouched by injection"))

# 4 threshold comes from clean scores only
prep = build_preprocessor().fit(ib.transform(tr))
m = IsolationForest(n_estimators=50, random_state=0).fit(prep.transform(ib.transform(tr)))
clean = -m.score_samples(prep.transform(Fva))
thr = np.quantile(clean, 0.95)
inj = -m.score_samples(prep.transform(D))
mixed_thr = np.quantile(np.r_[clean, inj], 0.95)
results.append(ok(np.isclose((clean > thr).mean(), 0.05, atol=0.002), "clean threshold flags 5% of the clean fold"))
print(f"     (clean threshold {thr:.4f} vs threshold if copies were mixed in {mixed_thr:.4f})")

# 5 grid and selection rule
res = run_grid(X, trees=(50,), samples=(256, 1024), seeds=(0, 1), n_splits=3, log=lambda s: None)
results.append(ok(len(res) == 2 * 3 and res["stability"].between(-1, 1).all()
                  and res[["recall_value", "recall_date"]].stack().between(0, 1).all(), "grid returns one row per setting and fold, values in range"))
print(f"     date recall {res['recall_date'].mean():.2f}, value recall {res['recall_value'].mean():.2f} (synthetic)")
fake = pd.DataFrame({"trees": [100, 200, 400], "max_samples": [256] * 3, "fold": [1] * 3,
                     "stability": [0.950, 0.947, 0.900], "recall_value": [0.2, 0.6, 0.9], "recall_date": [0.2, 0.6, 0.9]})
_, pick = summarise(fake)
results.append(ok(pick["trees"] == 200, "within 0.005 stability, higher mean recall wins"))
fake.loc[1, "stability"] = 0.940
_, pick = summarise(fake)
results.append(ok(pick["trees"] == 100, "outside 0.005, highest stability wins regardless of recall"))

print(f"\n{sum(results)}/{len(results)} tests passed")
