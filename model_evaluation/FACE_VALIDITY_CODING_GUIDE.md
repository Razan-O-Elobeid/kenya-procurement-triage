# Face-validity coding guide (protocol 7.1)

Written 16 Nov 2026, before any contract was coded. Committed before coding starts, so the rules cannot follow the results.

The 91 contracts are the IF top 50 and the R(c) top 50 on fold 5 validation (training period), 9 in both. Codes are descriptive. They are not corruption labels and are never used to choose models.

## Setup

1. Copy `data/labels/face_validity_fold5_to_code.csv` to `data/labels/face_validity_fold5_coded.csv`. Code only the copy.
2. Hide the `rank_if` and `rank_rc` columns while coding, so the code does not depend on which list a contract came from.
3. Do not sort, delete or insert rows, and do not edit `review_id`, `position_in_fold5_validation` or `ocid`. The summary checks them.
4. Fill only `code`, `reason`, `sources` and `notes`. Save as CSV (UTF-8).

## One code per contract, applied in this order

1. **publicly reported concern.** A credible public source names this contract, this tender, or this buyer and supplier for this procurement: Auditor-General report, EACC, PPRA or the Review Board, a court decision, or established media. `sources` must give the URL and the source's date. A concern about the buyer in general, not this procurement, does not qualify; put it in `notes`. This code is not evidence of wrongdoing.
2. **likely data error.** The record contradicts itself or is implausible on its face. Examples: signed years before the tender closed; a value implausible for what was bought (works for a few KES); zero value; currency or unit mismatch; an obvious duplicate. Decided from the record alone, without a web search.
3. **plausible legitimate reason.** The record is coherent, and the unusual feature has a recognised ordinary explanation. Examples: emergency or relief supply; a sole publisher or proprietary product; a framework or repeat order; a large, genuine infrastructure job. `reason` must name the explanation. "Could be fine" is not enough.
4. **unexplained.** None of the above applies within the time cap.

`reason` is one short sentence on what made the contract unusual and why the code fits.

## Procedure and time cap

- **Pass 1, record only (about 1 to 2 min each):** read the row, and code every likely data error. Do not search.
- **Pass 2, web search for the rest, at most 5 min per contract:** searches for "buyer supplier", "tender title Kenya", and the buyer with "Auditor General" or "Review Board". Stop at 5 min. If nothing qualifies, code from the record alone (plausible legitimate reason or unexplained).
- Record the date coded in `notes` when a search was made.
- Expected total: about 4 to 5 hours over 16 and 17 Nov.

## After coding

```bash
python -m model_evaluation.face_validity summary
```

Copy the two count tables into section 7 of `MODEL_SELECTION_FREEZE.md`, then commit the coded file and the record on 18 Nov.
