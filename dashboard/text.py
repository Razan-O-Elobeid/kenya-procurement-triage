"""Plain-language labels used by the bundle and the app."""

INPUT_LABELS = {
    "value_amount": "Contract value (log scale, clipped)",
    "days_to_signing": "Days from tender close to signing",
    "method_level": "Procurement method (competition level)",
    "tender_mainProcurementCategory_services": "Category: services",
    "tender_mainProcurementCategory_works": "Category: works",
    "f4_supplier_concentration": "Supplier's share of the buyer's earlier awards",
    "f5_benford_deviation": "Buyer's digit pattern (Benford deviation)",
    "missingindicator_f4_supplier_concentration": "Supplier share missing",
    "missingindicator_f5_benford_deviation": "Digit pattern missing (short buyer history)",
    "zero_value_award": "Zero-value award",
    "above_p99": "Value above the training 99th percentile",
    "below_p1": "Value below the training 1st percentile",
    "is_signed_pre_closure": "Signed before the tender closed",
    "is_terminated": "Contract terminated",
}


def label(name):
    """Label for an input; the raw name for anything unexpected (never guessed)."""
    for prefix in ("num__", "cat__", "bin__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return INPUT_LABELS.get(name, name)


MODEL_NAMES = {"if": "Isolation Forest 400 / 256 (baseline)", "eif": "Extended Isolation Forest 400 / 256 / ndim 2",
               "R": "Composite red-flag score R(c)"}

STATUSES = ["new", "under review", "cleared", "escalated"]

LIMITATIONS = """
- **No ground truth.** Scores rank contracts for review. A high score is not evidence of wrongdoing.
- **Synthetic recall only.** Detection was tested on artificially distorted copies (value and date
  distortions), which measures sensitivity, not corruption detection. Value inflation is weakly detected.
- **Explanations** show which inputs pushed this model's score up or down. They describe the model,
  not the contract's legality. SHAP treats inputs as independent, which does not hold for
  days to signing and signing before closure.
- **Supplier names** are grouped by Option A. Merged names are almost always the same business
  (precision about 0.99), but about a third of same-business name variants are not merged (recall about
  0.69 on candidate pairs), which can split a supplier's history and understate its concentration.
  The effect on rankings has not been quantified.
- **Reviews** are stored in an append-only log (enforced by database rules, not cryptographically
  tamper-proof). Reviews are not labels and never change scores.
"""
