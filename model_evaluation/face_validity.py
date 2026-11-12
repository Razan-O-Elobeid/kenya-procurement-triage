"""Top-50 face-validity sheet (protocol 7.1). Training period only, fold 5.

  python -m model_evaluation.face_validity sheet             # IF baseline config and R(c), as committed
  python -m model_evaluation.face_validity sheet --with-eif  # adds the EIF top 50 (needs a protocol note first)
  python -m model_evaluation.face_validity summary           # after coding

Lists: the 50 highest-scoring fold 5 validation contracts of the baseline configuration
(IF 400 / 256, seed 0, fitted on fold 5's training part) and the 50 highest by R(c).
Ties at the cut go to the earlier contract (date order); the number of tied contracts is printed.
A contract in both lists appears once. Codes are descriptive, not corruption labels,
and are never used to choose between models.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "data" / "labels" / "face_validity_fold5_to_code.csv"
CODED = ROOT / "data" / "labels" / "face_validity_fold5_coded.csv"
K = 50
CODES = ["likely data error", "plausible legitimate reason", "unexplained", "publicly reported concern"]
SHOW = ["ocid", "buyer_name", "supplier_name", "tender_title", "tender_mainProcurementCategory",
        "tender_procurementMethod", "value_amount", "value_currency", "tender_tenderPeriod_endDate",
        "dateSigned", "status", "days_to_signing", "f1_direct_award", "f3_decision_period",
        "f4_supplier_concentration", "f5_benford_deviation", "R", "is_signed_pre_closure",
        "zero_value_award", "above_p99", "below_p1", "is_terminated"]
CODE_COLS = ["code", "reason", "sources", "notes"]


def top_k(scores, k=K):
    """Positions of the k highest scores (ties to the earlier position) and the tie count at the cut."""
    scores = np.asarray(scores, dtype=float)
    order = np.lexsort((np.arange(len(scores)), -scores))
    top = order[:k]
    cut = scores[top[-1]]
    return top, {"tied_at_cut": int((scores == cut).sum()), "tied_inside_top": int((scores[top] == cut).sum())}


def build_sheet(F, lists, show=SHOW):
    """One row per contract in any list; rank columns per list; empty coding columns."""
    pos = sorted(set().union(*[set(p) for p, _ in lists.values()]))
    sheet = F.iloc[pos][[c for c in show if c in F.columns]].copy()
    sheet.insert(0, "position_in_fold5_validation", pos)
    for name, (p, _) in lists.items():
        r = {int(q): i + 1 for i, q in enumerate(p)}
        sheet.insert(1, f"rank_{name}", [r.get(q, pd.NA) for q in pos])
    first = sheet[[c for c in sheet.columns if c.startswith("rank_")]].min(axis=1)
    sheet = sheet.assign(_o=first).sort_values(["_o", "position_in_fold5_validation"]).drop(columns="_o")
    sheet.insert(0, "review_id", range(1, len(sheet) + 1))
    for c in CODE_COLS:
        sheet[c] = ""
    return sheet.reset_index(drop=True)


def cmd_sheet(with_eif):
    if SHEET.exists():
        print("STOPPED: sheet exists:", SHEET.relative_to(ROOT), file=sys.stderr)
        return 1
    from model_evaluation.fold5 import fit_models, fold5
    from model_evaluation.harness import load_training_period
    d = fold5(load_training_period(ROOT))
    models = fit_models(d["Ztr"])
    lists = {"if": top_k(models["if"].score(d["Zva"])),
             "rc": top_k(d["Fva"]["R"].fillna(-np.inf).to_numpy())}
    if with_eif:
        lists["eif"] = top_k(models["eif"].score(d["Zva"]))
    sheet = build_sheet(d["Fva"], lists)
    missing = [c for c in SHOW if c not in d["Fva"].columns]
    sheet.to_csv(SHEET, index=False)
    for name, (p, ties) in lists.items():
        print(f"top {K} by {name}: contracts tied at the cut {ties['tied_at_cut']} "
              f"({ties['tied_inside_top']} inside the top {K})")
    sets = {n: set(p) for n, (p, _) in lists.items()}
    print("overlap IF and R(c):", len(sets["if"] & sets["rc"]), "| contracts to code:", len(sheet))
    if missing:
        print("columns not in the data (left out):", ", ".join(missing))
    print("saved", SHEET.relative_to(ROOT), "| code into a copy named", CODED.name)
    return 0


def cmd_summary():
    sheet = pd.read_csv(SHEET)
    coded = pd.read_csv(CODED, dtype={"code": str})
    # spreadsheets may reformat numbers and dates, so only the identifying columns are compared
    keep = ["review_id", "position_in_fold5_validation", "ocid"]
    if not coded[keep].astype(str).equals(sheet[keep].astype(str)):
        print("STOPPED: coded file does not match the sheet (rows reordered or ids changed)", file=sys.stderr)
        return 1
    code = coded["code"].fillna("").str.strip()
    bad = sorted(set(code) - set(CODES) - {""})
    if bad:
        print(f"STOPPED: unexpected codes {bad}; use one of {CODES}", file=sys.stderr)
        return 1
    print(f"coded {int((code != '').sum())} of {len(coded)}")
    for c in [c for c in coded.columns if c.startswith("rank_")]:
        in_list = coded[c].notna()
        print(f"\n{c.replace('rank_', 'top 50 by ')}:")
        print(code[in_list].replace("", "not coded").value_counts().reindex(CODES + ["not coded"], fill_value=0).to_string())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sheet", "summary"])
    ap.add_argument("--with-eif", action="store_true")
    a = ap.parse_args(argv)
    return cmd_sheet(a.with_eif) if a.command == "sheet" else cmd_summary()


if __name__ == "__main__":
    sys.exit(main())
