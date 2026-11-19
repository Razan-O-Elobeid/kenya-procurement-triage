# Model selection and freeze (protocol sections 6.3 and 10)

**Status: FROZEN on 19 Nov 2026. The freeze commit is the commit that adds model_evaluation/model_freeze.json.**

New model comparisons and this selection used only the training period (2018 to 2023, 72,585 contracts): five chronological validation folds, seeds {0, 1}. The test period (2024 to Aug 2026) is not entirely unseen: the Phase 2 baseline's held-out test results were already known from Phase 2. No new model was selected or tuned using them. The run planned for 20 Nov is the one-time held-out evaluation of the newly frozen comparison. Synthetic recall measures sensitivity to two artificial distortions, not detection of real corruption.

## 1. Families in the comparison

| Family                                              | Status                                      | Where                                                               |
| --------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| Isolation Forest (IF), 400 trees, `max_samples` 256 | frozen Phase 2 baseline                     | `src/baseline.py` (`d773018`), CP1 reproduced (`e78b5f1`)           |
| Extended Isolation Forest (EIF)                     | evaluated, grid of 18 plus `ndim` 1 control | `model_evaluation/eif_eval.py` (`b9d8c4d`, seed fix `21df75a`)      |
| Reciprocal rank fusion (RRF) of R(c), IF and EIF    | evaluated (7.5)                             | `model_evaluation/rrf_eval.py` (`a566c1c`), clarification `99fb2b4` |
| Autoencoder, Deep SVDD, buyer-supplier graph        | deferred, not evaluated, not ranked         | scope decision of 11 Nov (7.6), deviations log                      |

Feature variants (7.3 relative value, 7.4 f4 percentile) were evaluated and not adopted; the features are those of Phase 2.

## 2. Selection evidence (fold means)

| Method                                          | Stability | Value x 50 |  Date | Value x 5 | Primary |
| ----------------------------------------------- | --------: | ---------: | ----: | --------: | ------: |
| IF 400 / 256 (frozen baseline)                  |    0.9916 |      0.117 | 0.842 |     0.060 |   0.480 |
| EIF 400 / 256 / ndim 2 (selected within family) |    0.9893 |      0.181 | 0.852 |     0.070 |   0.516 |
| RRF (R(c), IF, EIF)                             |    0.9968 |      0.116 | 0.753 |     0.062 |   0.434 |

EIF within-family selection (Phase 2 rule: highest stability, within 0.005 tied, ties to primary recall): tie set 400/256/2, 400/4096/2, 400/1024/2, 400/256/4; selected 400/256/2. The `ndim` 1 control (stability 0.9921, primary 0.486) is reported only.

## 3. Applying the pre-registered rules (6.2, 6.3, 7.5 clarification)

| Rule                                                       | EIF against the baseline          | RRF                       |
| ---------------------------------------------------------- | --------------------------------- | ------------------------- |
| Stability at least 0.95                                    | 0.9893: passes                    | 0.9968: passes            |
| Per-type guard (no type more than 0.05 below the baseline) | value +0.064, date +0.010: passes | date -0.089: **fails**    |
| Primary recall more than 0.02 above the comparison         | +0.037 above the baseline: passes | -0.082 against EIF: fails |
| Result                                                     | **EIF ranks above the baseline**  | not ranked above EIF      |

EIF has higher primary recall than the baseline in all five folds and higher value recall in all five; date recall is slightly lower in folds 3 to 5 (at most -0.037), within the guard.

**Scientific ranking:** 1. EIF 400 / 256 / ndim 2; 2. IF 400 / 256. RRF is reported and not ranked above either single family (guard failure, 7.5 clarification).

## 4. Deployment eligibility (6.3)

| Target                                              | IF                                                      | EIF                                                                                                                    | RRF            |
| --------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------- |
| Full training-period fit under 10 min               | 3.94 s                                                  | 3.65 s                                                                                                                 | not applicable |
| Validated explanation method (section 8, `b04ca10`) | TreeExplainer, interventional: validated                | permutation SHAP: validated                                                                                            | none           |
| Explains the review set under 30 min (fold 5)       | measured 8.6 s (extrapolated from 20 contracts: 19.6 s) | measured 7.4, 7.5 and 6.9 min in three full runs (extrapolated: 7.4 min); rehearsal measured 5.1 min for 415 contracts | not applicable |

EIF passed all section 8 checks: additivity 2.72e-15, correct target, stability 0.917 (requirement 0.9, the smallest margin), exact repeat. Method agreement (2.98 of 3) was measured on the IF, because EIF has no exact reference method.

## 5. Decision

- **Deployed model: EIF 400 / 256 / ndim 2**, the best-ranked family meeting both deployment targets.
- **IF 400 / 256** stays in the dashboard queue as the comparison model, with its own validated attributions.
- **RRF** is scored once at the final test for reporting only, never queued and never deployed.
- No threshold, feature or rule was changed by this selection.

## 6. Configurations selected for the 19 Nov freeze

These become frozen only when `model_freeze.json` is committed on 19 Nov.

| Model | Library                               | Settings                                                                   | Seed at the final test       |
| ----- | ------------------------------------- | -------------------------------------------------------------------------- | ---------------------------- |
| IF    | scikit-learn 1.6.1                    | `n_estimators` 400, `max_samples` 256                                      | `random_state` 0             |
| EIF   | isotree 0.6.1.post10, single-threaded | `ntrees` 400, `sample_size` 256, `ndim` 2                                  | `random_seed` 0 (100000 x 0) |
| RRF   | `model_evaluation/rrf.py`             | components R(c), IF, EIF; equal weights; k = 60; training-referenced ranks | inherits                     |

- **Features and preprocessing:** Phase 2 (`src/features.py`, `src/pipeline.py`), fitted on the full training period.
- **Thresholds:** 95th percentile of each model's training-period scores, set before the test period is scored. RRF: 95th percentile of training-period RRF scores.
- **Explanations (8.1):** 100 training rows, background seed 0; TreeExplainer interventional for the IF; permutation SHAP with `max_evals = 10 x (2d + 1)` for the EIF.
- **Machine-readable copy:** `model_evaluation/model_freeze.json`, prepared but not yet committed. It is read by `model_evaluation/final_eval.py` (`9cb30b5`), which stops unless that file is committed and unchanged.
- **Final run:** once, on 20 Nov 2026, under the section 10 clarification (`635738b`). The rehearsal on 16 Nov passed every integrity check on the training period.

## 7. Face validity (7.1), descriptive only

Not used to choose between models. Sheet `data/labels/face_validity_fold5_to_code.csv` (`50fd93c`), coded copy `data/labels/face_validity_fold5_coded.csv`, coding guide `model_evaluation/FACE_VALIDITY_CODING_GUIDE.md`. 91 selected contract records from the union of the IF top 50 and R(c) top 50 on fold 5 validation, with nine overlapping records. They cover 78 distinct OCIDs: 17 records share an OCID with another record (reviews 1, 22, 24, 26, 44, 46, 48, 50, 52, 54; 40 and 43; 68 and 70; 25, 27 and 29), so the 91 records are not all distinct procurements.

Coded by the project owner on 17 and 18 Nov 2026 under the guide committed at `bc7b6d6`, before coding. All 91 records coded: 44 likely data error, 47 unexplained, 0 plausible legitimate reason, 0 publicly reported concern. These are provisional human-review classifications, not confirmed data errors or corruption labels. The two lists overlap, so their columns below must not be added to obtain the 91-record totals.

| Code                        | IF top 50 | R(c) top 50 |
| --------------------------- | --------: | ----------: |
| likely data error           |        36 |          13 |
| plausible legitimate reason |         0 |           0 |
| unexplained                 |        14 |          37 |
| publicly reported concern   |         0 |           0 |
| not coded                   |         0 |           0 |

- **The IF top-50 shortlist contains many apparent data inconsistencies:** signing dates years before the tender closed (for example, the 2018 to 2019 ASAL bean awards), values of a few KES (The Meru National Polytechnic) and zero-value awards.
- **The R(c) top-50 shortlist contains many direct awards whose justification could not be established** from the available records and limited searches, consistent with R(c)'s design (f1 direct award). This does not mean the awards were unjustified or improper.
- **Rule for negative signing intervals:** intervals of months or years, or contradicted by a source, are coded as data errors; short intervals (1 to 7 days) with no other contradiction are coded as unexplained (reviews 59, 66, 69 and 73, recoded on 18 Nov and then given the same capped search, before this summary was committed).
- Air tickets recorded as "works" are coded as data errors; vehicle maintenance recorded as "works" is not, as that category is borderline.
- **Data-quality limitation:** reviews 25 and 27 appear to be duplicate records (same OCID, title, amount and dates), so one record may be counted twice.
- **Searches:** the 47 unexplained records received limited AI-assisted public-web searches, capped at five minutes per record. No sufficiently supported contract-specific evidence was found to justify changing those classifications. The searches were not exhaustive, and not every external source was independently verified.
- No record was coded as plausible legitimate reason, because the guide requires a named explanation rather than "could be fine".
- An unexplained contract does not imply corruption or misconduct.
- The review is descriptive. It did not influence, and must not influence, the pre-registered model ranking.

## 8. Limitations carried into the final test

- The EIF margin is modest: +0.037 against a 0.02 tie margin, with no formal significance test; the evidence is consistency across five time-ordered folds. The selected configuration is the best of a four-way tie set.
- Value inflation is weakly detected by every method (value x 5 recall about chance).
- EIF explanation stability (0.917) is close to its 0.9 requirement, and EIF has no exact reference explanation.
- Three pre-registered families were deferred, not evaluated.
- The final comparison is not fully blind: the Phase 2 baseline's test results were seen in Phase 2. The baseline is fixed, so this does not bias it.
- Training-period drift (7.9(a)) suggests fewer than 5% of new contracts may cross a training threshold; thresholds are not recalibrated.
