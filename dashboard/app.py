"""Auditor dashboard. Reads a frozen bundle; never fits, scores or explains anything.

Run from the repo root:  streamlit run dashboard/app.py
Bundle:     TRIAGE_BUNDLE (default results/dashboard/dev_fold5)
Review log: TRIAGE_REVIEW_DB (default results/dashboard/review_log.sqlite)
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.bundle import load_bundle  # noqa: E402
from dashboard.review_log import add_review, latest_status, read_reviews  # noqa: E402
from dashboard.text import LIMITATIONS, MODEL_NAMES, STATUSES, label  # noqa: E402

BUNDLE = Path(os.environ.get("TRIAGE_BUNDLE", ROOT / "results" / "dashboard" / "dev_fold5"))
DB = Path(os.environ.get("TRIAGE_REVIEW_DB", ROOT / "results" / "dashboard" / "review_log.sqlite"))
TOP_N = 5

st.set_page_config(page_title="Procurement risk triage", layout="wide")


@st.cache_data(show_spinner="Loading bundle")
def get_bundle(path, stamp):
    return load_bundle(path)


def stamp(path):
    """Changes whenever any bundle file changes, so the hashes are checked again."""
    return tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in Path(path).iterdir()))


try:
    manifest, T = get_bundle(str(BUNDLE), stamp(BUNDLE))
except (FileNotFoundError, ValueError) as e:
    st.error(f"Cannot open the bundle at {BUNDLE}: {e}")
    st.stop()

C, E, B, K = T["contracts"], T["explanations"], T["buyers"], T["categories"]
bundle = manifest["bundle"]
(st.warning if manifest["mode"] != "deployment" else st.info)(manifest["banner"])

page = st.sidebar.radio("Page", ["Queue", "Contract", "Review log", "About"], key="page")
models = manifest["queue_models"]
model = st.sidebar.selectbox("Ranking model", models, format_func=lambda m: MODEL_NAMES.get(m, m),
                             disabled=len(models) == 1, key="model")
st.sidebar.caption(f"Bundle: {bundle} | commit {str(manifest.get('git_commit'))[:7]}")
st.sidebar.caption(f"Deployment status: {manifest['deployment_status']}")
DB.parent.mkdir(parents=True, exist_ok=True)
reviews = read_reviews(DB, bundle)
status_now = latest_status(reviews)
exp_rep = manifest["explanations"].get(model, {})


def money(v):
    return "missing" if pd.isna(v) else f"{v:,.0f}"


def contract_label(row_id):
    r = C.loc[C["row_id"] == row_id].iloc[0]
    return f"#{int(r[f'rank_{model}'])} | {r['ocid']} | {str(r.get('buyer_name'))[:40]}"


def open_contract(row_id):
    st.session_state["row_id"] = int(row_id)
    st.session_state["page"] = "Contract"


def queue_frame():
    q = C.sort_values(f"rank_{model}").copy()
    q["review_status"] = [status_now.get((int(r), model), "new") for r in q["row_id"]]
    return q


if page == "Queue":
    st.header(f"Review queue: {MODEL_NAMES.get(model, model)}")
    m = manifest["models"][model]
    st.caption(f"{manifest['contracts']}. Flagged: score above {m['threshold']:.4f} "
               f"({m['threshold_rule']}), {m['n_flagged']:,} contracts.")
    q = queue_frame()
    c1, c2, c3, c4 = st.columns(4)
    only = c1.checkbox("Flagged only", value=True)
    cats = c2.multiselect("Category", sorted(q["tender_mainProcurementCategory"].dropna().unique()))
    meths = c3.multiselect("Method", sorted(q["tender_procurementMethod"].dropna().unique()))
    buyer = c4.text_input("Buyer contains")
    if only:
        q = q[q[f"flag_{model}"]]
    if cats:
        q = q[q["tender_mainProcurementCategory"].isin(cats)]
    if meths:
        q = q[q["tender_procurementMethod"].isin(meths)]
    if buyer:
        q = q[q["buyer_name"].astype(str).str.contains(buyer, case=False, regex=False)]
    cols = [f"rank_{model}", "ocid", "buyer_name", "supplier_name", "tender_mainProcurementCategory",
            "tender_procurementMethod", "value_amount", "tender_tenderPeriod_endDate", f"score_{model}",
            "R", "review_status"]
    shown = q[cols].rename(columns={f"rank_{model}": "Rank", "ocid": "Contract ID", "buyer_name": "Buyer",
                                    "supplier_name": "Supplier", "tender_mainProcurementCategory": "Category",
                                    "tender_procurementMethod": "Method", "value_amount": "Value",
                                    "tender_tenderPeriod_endDate": "Tender closed", f"score_{model}": "Score",
                                    "R": "R(c)", "review_status": "Review status"})
    shown["Supplier"] = shown["Supplier"].fillna("(missing)")
    shown["Tender closed"] = shown["Tender closed"].astype(str).str[:10]
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"Value": st.column_config.NumberColumn(format="localized"),
                                "Score": st.column_config.NumberColumn(format="%.4f"),
                                "R(c)": st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Export this list (CSV)", q[cols].to_csv(index=False), f"shortlist_{bundle}_{model}.csv")
    if len(q):
        pick = st.selectbox("Open contract", q["row_id"].tolist(), format_func=contract_label, key="queue_pick")
        st.button("Open", on_click=open_contract, args=(pick,))

elif page == "Contract":
    q = queue_frame()
    flagged = q[q[f"flag_{model}"]]
    choices = flagged["row_id"].tolist() or q["row_id"].head(50).tolist()
    current = st.session_state.get("row_id", choices[0])
    if current not in choices:
        choices = [current] + choices
    row_id = st.selectbox("Contract", choices, index=choices.index(current), format_func=contract_label)
    st.session_state["row_id"] = int(row_id)
    r = C.set_index("row_id").loc[row_id]
    st.header(f"{r['ocid']}")
    st.caption(f"Status: {status_now.get((int(row_id), model), 'new')} | rank {int(r[f'rank_{model}'])} by "
               f"{MODEL_NAMES.get(model, model)} | {'flagged' if r[f'flag_{model}'] else 'not flagged'}")

    left, right = st.columns(2)
    with left:
        st.subheader("Contract")
        facts = {"Buyer": r.get("buyer_name"), "Supplier (as published)": r.get("supplier_name"),
                 "Supplier (resolved identity)": r.get("supplier_resolved"),
                 "Category": r.get("tender_mainProcurementCategory"), "Method": r.get("tender_procurementMethod"),
                 "Value": f"{money(r.get('value_amount'))} {r.get('value_currency', '')}",
                 "Tender closed": r.get("tender_tenderPeriod_endDate"), "Signed": r.get("dateSigned"),
                 "Status": r.get("status")}
        st.table(pd.Series(facts, name="").astype(str))
        st.subheader("Compared with the training history")
        b = B[B["buyer_id"] == r.get("buyer_id")]
        k = K[K["tender_mainProcurementCategory"] == r.get("tender_mainProcurementCategory")]
        v = r.get("value_amount")
        lines = []
        if len(b):
            bm = b.iloc[0]["buyer_median_value"]
            lines.append(f"Buyer's median award value: {money(bm)} ({int(b.iloc[0]['buyer_awards'])} awards)"
                         + (f"; this contract is {v / bm:.1f} times that" if pd.notna(v) and bm > 0 else ""))
        else:
            lines.append("This buyer has no awards in the training history.")
        if len(k):
            lines.append(f"Median award in this category: {money(k.iloc[0]['category_median_value'])}")
        st.markdown("\n".join(f"- {x}" for x in lines))

    with right:
        st.subheader("Red-flag scorecard")
        sc = []
        if r.get("f1_direct_award") == 1:
            sc.append("Direct or specially permitted award (f1).")
        if pd.notna(r.get("f3_decision_period")):
            sc.append(f"Decision period (f3): {r['f3_decision_period']:.2f}; signed {r['days_to_signing']:.0f} days "
                      f"after close, quicker than {r['f3_decision_period']:.0%} of training contracts.")
        if pd.notna(r.get("f4_supplier_concentration")):
            sc.append(f"Supplier concentration (f4): this supplier won {r['f4_supplier_concentration']:.0%} "
                      "of the buyer's earlier award value.")
        if pd.notna(r.get("f5_benford_deviation")):
            sc.append(f"Digit pattern (f5): the buyer's earlier awards deviate from Benford more than "
                      f"{r['f5_benford_deviation']:.0%} of training buyers.")
        for flag, text in [("is_signed_pre_closure", "Signed before the tender closed."),
                           ("zero_value_award", "Zero-value award."),
                           ("above_p99", "Value above the training 99th percentile."),
                           ("below_p1", "Value below the training 1st percentile."),
                           ("is_terminated", "Contract terminated."),
                           ("supplier_missing", "Supplier name missing."),
                           ("signing_date_missing", "Signing date missing.")]:
            if r.get(flag) == 1:
                sc.append(text)
        st.markdown("\n".join(f"- {x}" for x in sc) or "No indicator fired.")
        st.caption(f"R(c) = {r['R']:.3f} (rank {int(r['rank_R'])}).")

        st.subheader("Why the model ranked it here")
        ex = E[(E["row_id"] == row_id) & (E["model"] == model)]
        if exp_rep.get("passed") and len(ex) == 1:
            attr = [c for c in E.columns if c.startswith("attr__")]
            a = ex.iloc[0][attr].astype(float)
            a.index = [label(c[len("attr__"):]) for c in attr]
            top = a.reindex(a.abs().sort_values(ascending=False).index).head(TOP_N)
            st.bar_chart(top.rename("contribution"), horizontal=True)
            st.markdown("\n".join(f"- {n}: {'raises' if x > 0 else 'lowers'} the anomaly score" for n, x in top.items()))
            st.caption(f"{exp_rep['method']}; validated under protocol section 8. Contributions describe the "
                       "model's score, not wrongdoing.")
        else:
            reason = exp_rep.get("reason", "no explanation in this bundle") if not exp_rep.get("passed") \
                else "not computed for this contract in this bundle"
            st.info(f"No validated attribution is shown for this contract ({reason}). "
                    "Use the scorecard and the historical comparison.")

    with st.expander("Audit trail"):
        trail = {"bundle": bundle, "git_commit": manifest.get("git_commit"), "model": model,
                 "threshold": manifest["models"][model]["threshold"],
                 "threshold_rule": manifest["models"][model]["threshold_rule"],
                 "explanation": {k: exp_rep.get(k) for k in ("method", "section8_validated", "passed", "reason")},
                 "contract": {k: (None if pd.isna(x) else x) for k, x in r.items()},
                 "file_hashes": manifest["files"]}
        st.json(trail, expanded=False)
        st.download_button("Download audit record (JSON)", json.dumps(trail, indent=2, default=str),
                           f"audit_{bundle}_{row_id}.json")

    st.subheader("Review")
    with st.form("review", clear_on_submit=True):
        status = st.selectbox("Status", STATUSES, key="review_status")
        reviewer = st.text_input("Reviewer", key="reviewer")
        note = st.text_area("Note", key="review_note")
        if st.form_submit_button("Add review", key="add_review"):
            if not reviewer.strip():
                st.error("Reviewer is required.")
            else:
                add_review(DB, bundle, row_id, r["ocid"], model, status, reviewer, note)
                st.success("Review added.")
                st.rerun()
    hist = reviews[(reviews["row_id"] == row_id) & (reviews["model"] == model)]
    st.dataframe(hist[["created_utc", "status", "reviewer", "note"]] if len(hist) else pd.DataFrame(
        columns=["created_utc", "status", "reviewer", "note"]), hide_index=True, width="stretch")

elif page == "Review log":
    st.header("Review log (append-only)")
    st.caption("Every review is a new row; earlier rows cannot be changed or deleted through the database. "
               "Reviews are not labels and never change scores.")
    st.dataframe(reviews, hide_index=True, width="stretch")
    st.download_button("Export review log (CSV)", reviews.to_csv(index=False), f"review_log_{bundle}.csv")

else:
    st.header("About this bundle")
    st.markdown(f"- **Contracts:** {manifest['contracts']}\n- **Historical reference:** {manifest['history_reference']}"
                f"\n- **Deployment status:** {manifest['deployment_status']}")
    rows = [{"model": MODEL_NAMES.get(m, m), "threshold": v["threshold"], "flagged": v["n_flagged"],
             "explanation": manifest["explanations"].get(m, {}).get("method"),
             "validated (section 8)": manifest["explanations"].get(m, {}).get("section8_validated"),
             "integrity": manifest["explanations"].get(m, {}).get("reason")}
            for m, v in manifest["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Limitations")
    st.markdown(LIMITATIONS)
