"""Latency benchmark for a running scoring API. Sequential requests from one client, a new
connection per request. Uses held-out contracts from the official bundle as inputs.

    TRIAGE_API_KEY=<key> python -m serving.benchmark --url http://127.0.0.1:8000 \
        --label "docker-arm64-macbook-air-1cpu-2gb" --server "<where the server ran>"

Two scenarios: one contract per request, and 20 contracts per request. Reports client
round-trip time and the server's own scoring time (latency_ms). Writes a JSON summary to
benchmarks/latency_<label>.json with no contract records, and refuses to overwrite it.
Exits 1 if any request fails. The key is read from the environment and never printed.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from serving.check_endpoint import CONTRACTS, call, payload

ROOT = Path(__file__).resolve().parents[1]


def stats(ms):
    a = np.asarray(ms, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 1), "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "p99": round(float(np.percentile(a, 99)), 1),
            "max": round(float(a.max()), 1)}


def scenario(url, key, batches, warmup):
    client, server, fails = [], [], 0
    for i, batch in enumerate(batches):
        t0 = time.perf_counter()
        status, body = call(url, key, "/score", {"contracts": batch})
        dt = (time.perf_counter() - t0) * 1000
        if status != 200:
            fails += 1
            continue
        if i >= warmup:
            client.append(dt)
            server.append(body["latency_ms"])
    return client, server, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True, help="short name, used in the output file name")
    ap.add_argument("--server", required=True, help="where the server ran: machine, container limits, region")
    ap.add_argument("--n", type=int, default=100, help="timed single-contract requests")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--contracts", default=str(CONTRACTS))
    a = ap.parse_args()
    key = os.environ.get("TRIAGE_API_KEY", "")
    out = ROOT / "benchmarks" / f"latency_{a.label}.json"
    if not key:
        print("STOPPED: set TRIAGE_API_KEY", file=sys.stderr)
        return 1
    if out.exists():
        print("STOPPED: output exists:", out, file=sys.stderr)
        return 1

    ref = pd.read_csv(a.contracts, low_memory=False)
    rng = np.random.default_rng(a.seed)
    n_batch = max(a.n // 5, 10)
    pos = rng.choice(len(ref), size=a.n + a.warmup + 20 * (n_batch + a.warmup), replace=False)
    rows = [payload(ref.iloc[p]) for p in pos]
    singles = [[r] for r in rows[:a.n + a.warmup]]
    rest = rows[a.n + a.warmup:]
    batches = [rest[i:i + 20] for i in range(0, 20 * (n_batch + a.warmup), 20)]

    status, model = call(a.url, key, "/model")
    if status != 200:
        print(f"STOPPED: /model returned {status}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    c1, s1, f1 = scenario(a.url, key, singles, a.warmup)
    c20, s20, f20 = scenario(a.url, key, batches, a.warmup)
    wall = time.perf_counter() - t0

    record = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": a.label, "server": a.server, "url_kind": "local" if "127.0.0.1" in a.url or "localhost" in a.url else "remote",
        "client": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
                   "cpu_count": os.cpu_count()},
        "method": "sequential requests, one client, new connection per request, first "
                  f"{a.warmup} requests per scenario not timed; inputs are held-out contracts (seed {a.seed})",
        "model": {k: model.get(k) for k in ("deployed_model", "artifact_sha256", "git_commit")},
        "single_contract": {"client_round_trip_ms": stats(c1), "server_scoring_ms": stats(s1), "failures": f1},
        "batch_of_20": {"client_round_trip_ms": stats(c20), "server_scoring_ms": stats(s20),
                        "client_ms_per_contract_p50": round(float(np.percentile(c20, 50)) / 20, 1), "failures": f20},
        "wall_seconds": round(wall, 1),
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    for name in ("single_contract", "batch_of_20"):
        r = record[name]
        print(f"{name}: client p50 {r['client_round_trip_ms']['p50']} ms, p95 {r['client_round_trip_ms']['p95']} ms; "
              f"server p50 {r['server_scoring_ms']['p50']} ms; failures {r['failures']}")
    print("written", out)
    return 0 if f1 == 0 and f20 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
