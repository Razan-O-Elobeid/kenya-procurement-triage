"""Checks the API schema against every held-out contract in the official bundle. No server needed.

    python -m serving.check_schema

Reports:
  - contracts rejected, with every error counted by field and type
  - values the schema would change on the way in (should be none)
  - is_terminated rebuilt from the validated status against the official column
  - names that are blank or carry surrounding spaces in the release
  - with --score-affected N: every blank-name contract plus up to N with padded names, scored through
    the API path (schema, request_frame, serving core) and compared with the official bundle on the
    14 model inputs (rebuilt from the bundle's feature columns), scores, percentiles, flags and features
Reads results/final_test/contracts.csv.gz and the serving artefact. Writes nothing.
Exits 1 if any contract is rejected, any value changes or is_terminated differs.
"""
import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd
from pydantic import ValidationError

from serving import core
from serving.api import DEFAULT_BUNDLE, load_bundle, request_frame, vocabulary
from serving.check_endpoint import CONTRACTS, FIELDS, payload
from serving.schema import ScoreRequest, vocab_errors

COMPARE = ["score_eif", "score_if", "pct_eif", "pct_if", "R", "f3_decision_period", "f4_supplier_concentration",
           "f5_benford_deviation", "days_to_signing", "is_terminated", "zero_value_award", "above_p99", "below_p1",
           "is_signed_pre_closure"]
TEXT = ["ocid", "buyer_name", "supplier_name", "tender_mainProcurementCategory", "tender_procurementMethod", "status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--contracts", default=str(CONTRACTS))
    ap.add_argument("--score-affected", type=int, default=0, metavar="N")
    a = ap.parse_args()
    bundle, _ = load_bundle(a.bundle)
    vocab = vocabulary(bundle)
    ref = pd.read_csv(a.contracts, low_memory=False)
    print(f"{len(ref):,} held-out contracts; buyer_id stored as {ref['buyer_id'].dtype}, "
          f"artefact expects {bundle['buyer_kind']}")

    errors, changed, term_diff, rejected, accepted = Counter(), Counter(), 0, 0, 0
    for i in range(len(ref)):
        raw = payload(ref.iloc[i])
        try:
            c = ScoreRequest(contracts=[raw]).contracts[0]
        except ValidationError as e:
            rejected += 1
            for x in e.errors():
                errors[(".".join(str(p) for p in x["loc"][2:]), x["type"])] += 1
            continue
        v = vocab_errors(ScoreRequest.model_construct(contracts=[c]), vocab)
        if v:
            rejected += 1
            for x in v:
                errors[(x["field"].split(".")[-1], "vocabulary")] += 1
            continue
        accepted += 1
        for f in TEXT:
            if getattr(c, f) != raw[f]:
                changed[f] += 1
        if str(c.buyer_id) != str(raw["buyer_id"]):
            changed["buyer_id"] += 1
        if c.value_amount != raw["value_amount"]:
            changed["value_amount"] += 1
        for f in ("tender_tenderPeriod_endDate", "dateSigned"):
            want = pd.to_datetime(raw[f], utc=True, format="ISO8601") if raw[f] is not None else None
            got = pd.Timestamp(getattr(c, f)) if getattr(c, f) is not None else None
            if (want is None) != (got is None) or (want is not None and want != got):
                changed[f] += 1
        term_diff += int(c.status == "terminated") != int(ref["is_terminated"].iloc[i])

    print(f"\naccepted {accepted:,}, rejected {rejected:,}")
    for (field, kind), n in errors.most_common():
        print(f"  rejected  {n:>6,}  {field}  {kind}")
    print("\nvalues the schema would change:", dict(changed) or "none")
    print(f"is_terminated differences on accepted contracts: {term_diff}")
    print("\nmissing values per field:")
    for f in FIELDS:
        print(f"  {f:<32} {int(ref[f].isna().sum()):>6,}")
    print("status values:", ref["status"].value_counts(dropna=False).to_dict())
    padded = pd.Series(False, index=ref.index)
    blanks = pd.Series(False, index=ref.index)
    for f in ("buyer_name", "supplier_name"):
        t = ref[f].dropna().astype(str)
        blank, pad = t.str.strip() == "", t != t.str.strip()
        padded |= pad.reindex(ref.index, fill_value=False)
        blanks |= blank.reindex(ref.index, fill_value=False)
        print(f"{f}: {int(blank.sum())} blank, {int(pad.sum()):,} with surrounding spaces")

    mismatch = 0
    if a.score_affected:
        pool = np.flatnonzero((padded & ~blanks).to_numpy())
        pick = np.random.default_rng(0).choice(pool, size=min(a.score_affected, len(pool)), replace=False)
        pos = np.sort(np.union1d(np.flatnonzero(blanks.to_numpy()), pick))
        for p in pos:
            c = ScoreRequest(contracts=[payload(ref.iloc[p])]).contracts
            frame = request_frame(c)
            r = core.score_rows(bundle, frame).iloc[0]
            o = ref.iloc[p]
            F = bundle["ib"].transform(core.prepare_rows(frame, bundle["buyer_kind"]))
            z = np.asarray(bundle["prep"].transform(F), dtype=float)[0]
            z_ref = np.asarray(bundle["prep"].transform(ref.iloc[[p]]), dtype=float)[0]
            bad = [] if np.allclose(z, z_ref, rtol=0, atol=1e-12, equal_nan=True) else ["14 model inputs"]
            bad += [k for k in COMPARE if not ((pd.isna(r[k]) and pd.isna(o[k])) or
                                              (not pd.isna(r[k]) and not pd.isna(o[k]) and abs(float(r[k]) - float(o[k])) <= 1e-12))]
            bad += [k for k in ("flag_eif", "flag_if") if bool(r[k]) != bool(o[k])]
            if bad:
                mismatch += 1
                print(f"  MISMATCH {o['ocid']}: {bad}")
            elif blanks.iloc[p]:
                print(f"  blank-name contract {o['ocid']} (supplier {o['supplier_name']!r}): 14 inputs, scores, "
                      f"percentiles, flags and features equal the official bundle")
        print(f"\naffected contracts scored through the API path ({int(blanks.sum())} blank, {len(pick)} padded): "
              f"{len(pos) - mismatch}/{len(pos)} equal the official bundle on all checks")
    return 0 if rejected == 0 and not changed and term_diff == 0 and mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
