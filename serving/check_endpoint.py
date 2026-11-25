"""Posts held-out contracts to a running scoring API and compares the replies with the
official bundle (results/final_test/contracts.csv.gz). A consistency check, not an evaluation.

    TRIAGE_API_KEY=<key> python -m serving.check_endpoint --url http://127.0.0.1:8000 --n 20

The key is read from the environment and never printed. Exits 1 on any mismatch or error.
"""
import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "results" / "final_test" / "contracts.csv.gz"
FIELDS = ["ocid", "buyer_id", "buyer_name", "supplier_name", "value_amount", "tender_mainProcurementCategory",
          "tender_procurementMethod", "status", "tender_tenderPeriod_endDate", "dateSigned"]
TOL = 1e-12
FEATURES = ["R", "f3_decision_period", "f4_supplier_concentration", "f5_benford_deviation", "days_to_signing",
            "is_terminated"]


def call(url, key, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method="POST" if body else "GET",
                                 headers={"X-API-Key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def payload(row):
    out = {}
    for c in FIELDS:
        v = row[c]
        if isinstance(v, float) and math.isnan(v):
            v = None
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[c] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--contracts", default=str(CONTRACTS), help="official contracts table")
    ap.add_argument("--affected", action="store_true", help="only contracts whose names carry surrounding spaces")
    a = ap.parse_args()
    key = os.environ.get("TRIAGE_API_KEY", "")
    if not key:
        print("STOPPED: set TRIAGE_API_KEY", file=sys.stderr)
        return 1

    ref = pd.read_csv(a.contracts, low_memory=False)
    pool = np.arange(len(ref))
    if a.affected:
        pad = np.zeros(len(ref), bool)
        for f in ("buyer_name", "supplier_name"):
            t = ref[f].astype("string")
            pad |= (t.notna() & (t != t.str.strip())).to_numpy(dtype=bool)
        pool = np.flatnonzero(pad)
    a.n = min(a.n, len(pool))
    pos = np.sort(np.random.default_rng(a.seed).choice(pool, size=a.n, replace=False))
    status, health = call(a.url, key, "/health")
    print("health:", status, health)
    bad = 0
    for p in pos:
        row = ref.iloc[p]
        status, body = call(a.url, key, "/score", {"contracts": [payload(row)]})
        if status != 200:
            print(f"FAIL {row['ocid']}: HTTP {status} {body}")
            bad += 1
            continue
        r = body["results"][0]
        diffs = [abs(r[f"score_{m}"] - row[f"score_{m}"]) for m in ("eif", "if")]
        for k in FEATURES:
            if (r[k] is None) != pd.isna(row[k]):
                diffs.append(1.0)
            elif r[k] is not None:
                diffs.append(abs(float(r[k]) - float(row[k])))
        same = max(diffs) <= TOL and r["flag_eif"] == bool(row["flag_eif"]) and r["flag_if"] == bool(row["flag_if"])
        bad += not same
        print(f"{'PASS' if same else 'FAIL'} {row['ocid']}: eif {r['score_eif']:.6f} "
              f"(official {row['score_eif']:.6f}), flag {r['flag_eif']}, features {'equal' if same else 'differ'}, "
              f"{body['latency_ms']} ms")
    print(f"\n{a.n - bad}/{a.n} contracts match the official bundle")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
