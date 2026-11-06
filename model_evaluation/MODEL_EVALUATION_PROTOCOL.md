# Model Evaluation and Selection Protocol

Once committed, this file is the pre-registration record for model evaluation and selection.
Changes made after a result has been seen go in the deviations log (section 12), never as silent edits.
Amendments made before the experiment they govern (for example the exact relative-value definition, section 7.3) are committed as separate commits before that experiment runs.

## 1. Purpose and limits

This work compares anomaly-detection models and feature variants for ranking Kenyan public contracts by review priority, then deploys the selected configuration in an auditor dashboard.

There are no verified corruption labels.

- The composite score R(c) is a rule-based index, not ground truth.
- Synthetic anomaly recall measures sensitivity to artificial perturbations, not corruption detection.
- Manual inspection notes and auditor reviews are not labels and are never used for supervised training.

## 2. Data roles

| Period | Contracts | Role |
|---|---:|---|
| 2018 to 2023 (training period) | 72,585 | All model comparison, feature decisions, hyperparameter tuning, ensemble choices and selection of the deployed model, using time-ordered validation folds only |
| 2024 to Aug 2026 (held-out test) | 36,399 | One final evaluation of frozen configurations, for reporting generalization only |

Rules:

1. No decision may use test-period data or results. This covers features, models, hyperparameters, thresholds, ensemble components and weights.
2. The previously reported Phase 2 test results (1,131 of 36,399 flagged; score distribution) must not be used to choose between new models or features.
3. All selection criteria are written in this file and committed before the experiment they govern is run. The git commit time is the record.
4. Selected configurations are frozen (section 10) before the test period is scored.
5. The test period is scored once for all frozen configurations. Its results never trigger further tuning.
6. Any model or feature changed after the final test run is labeled **exploratory**, not independent confirmation.
7. The Phase 2 Isolation Forest test results were seen during Phase 2. The baseline is fixed, so this does not bias it, but the final comparison is not fully blind. This is stated in the report.

## 3. File layout and naming

- New code, protocol and selection records go in `model_evaluation/`. New result files go in `results/model_evaluation/`.
- `model_evaluation/MODEL_SELECTION_FREEZE.md` is the final model-selection record (section 10).
- Files created before this naming convention are kept unchanged to preserve reproducibility and commit references:
  - `src/baseline.py`;
  - `models/baseline_iforest_400_256.json`;
  - the local output `results/phase3/baseline_test_scores.csv`;
  - the commit messages of `d773018` and `6fb13b5`.

## 4. Frozen baseline

- Configuration: scikit-learn `IsolationForest`, 400 trees, `max_samples=256`, `random_state=0`, inside the Phase 2 pipeline (`src/pipeline.py::build_pipeline`), fitted on the full training period.
- Threshold: 95th percentile of training scores, 0.5637224653931433.
- Script: `src/baseline.py` at commit `d773018`. Metadata: `models/baseline_iforest_400_256.json` (commit `6fb13b5`).
- Verification: reproduces Phase 2 exactly (sorted test scores identical within 1e-12, 1,131 flagged, identical flagged OCIDs, identical reload).
- Run: `python -m src.baseline`. It refuses to overwrite existing outputs. `--overwrite` moves old outputs to `models/superseded/<timestamp>/` and must not be used without a deviations-log entry.
- The model file (`*.joblib`) and score files (`*.csv`) contain release data and are kept out of the public repository.
- Phase 2 files are read, never written, by the new code.

## 5. Validation procedure (same for every model)

- Training period sorted by `tender_tenderPeriod_endDate` (stable sort).
- `TimeSeriesSplit(n_splits=5)`. Fold training sizes 12,100 / 24,197 / 36,294 / 48,391 / 60,488. Validation 12,097 each.
- Inside each fold, everything fitted is fitted on that fold's training part only: `IndicatorBuilder` (thresholds, ECDF references, supplier resolver, histories), preprocessing, model, and any early-stopping data or graph.
- Inputs: the 14 Phase 2 model inputs, unless an entry in section 7 pre-registers a feature variant.
- Seeds: {0, 1} for every model.

**Synthetic anomaly injection** (from `src/tuning.py::inject`):

- 2% copies of clean validation rows per type. The seed is the fold index used by `run_grid` (0 to 4).
- Types:
  - `value`: value x 50, non-zero rows only; value flags recomputed with fitted thresholds.
  - `date`: signing 200 days before close, rows with a signing date only; pre-closure flag set.
  - `value_x5` (new, secondary): value x 5, same handling as `value`.
- Copies never enter histories, fitted parameters or the clean ranking.
- **Score functions that use f3 or R(c)** (R(c), the naive count, RRF): `inject` changes `days_to_signing` but not f3 or R. For these scores, f3 of date copies is recomputed as `1 - ECDF(d_ref_, -200)`, using the fold's fitted `IndicatorBuilder`, and R(c) is recomputed as the mean of the available indicators. Nothing else changes. Value copies need no recomputation, because f1, f3, f4 and f5 do not depend on a contract's own value.
- Threshold: 95th percentile of clean validation scores for that model, fold and seed. Chance recall is 0.05.

## 6. Metrics and selection

### 6.1 Metrics (reported for every configuration, per fold and as fold means)

1. **Seed stability:** Spearman correlation between seed 0 and seed 1 scores on the clean validation fold.
2. **Injection recall by type:** `value`, `date` and `value_x5`, always reported separately.
3. **Primary recall:** the mean of `value` and `date` recall (the Phase 2 definition, kept for continuity).
4. **Top-5% overlap with the frozen baseline:** Jaccard index on the clean validation fold.
5. **Spearman with R(c):** descriptive only, never a selection criterion.
6. **Computational cost:** fit time on the full training period, scoring time per 10,000 contracts, and explanation time (section 8), measured on the MacBook Air (CPU).

### 6.2 Interpreting the primary metric

Primary recall is an average of two synthetic perturbation types. A gain in one type can hide a loss in the other, so per-type recall is always reported next to it.

**Per-type guard:** a challenger whose recall on either `value` or `date` is more than 0.05 below the frozen baseline's on the same folds cannot be selected over the baseline on primary recall alone. The comparison is reported as a trade-off, and the baseline is kept.

None of these numbers measures detection of real corruption.

### 6.3 Scientific ranking and deployment eligibility (kept separate)

**Scientific ranking.** Every configuration is ranked by the validation metrics regardless of computational cost.

- Within a family (hyperparameters), the Phase 2 rule is kept for continuity: highest mean seed stability; settings within 0.005 of the best are tied; ties are broken by higher primary recall.
- Across families: the primary metric is primary recall, subject to the per-type guard (6.2). Families within 0.02 of the best are tied. A tie goes to the simpler family in this order: Isolation Forest, Extended Isolation Forest, autoencoder, Deep SVDD, graph model.
- Mean seed stability of at least 0.95 is a scientific requirement for any family to be ranked as a candidate.

**Deployment eligibility.** These are practical targets for the dashboard on the MacBook Air. They are not measures of detection quality.

- Full training-period fit in under 10 minutes.
- A validated explanation method (section 8) that explains the review set in under 30 minutes.

A model that misses a practical target is not judged a worse detector. Its scientific results are reported in full, together with its computational limits.

**Deployed model.** The deployed model is the best-ranked family that meets both deployment targets. If the best-ranked family misses a target, the report states that it ranked best scientifically but was not deployable under the project's constraints. If no new family beats the baseline outside the tie margin and the guard, the frozen baseline stays the deployed model.

All results are reported, including families that are not selected.

## 7. Pre-registered experiments

Each entry is completed and committed before its experiment runs.

### 7.1 Evaluation commitments from the proposal (training period)

**Bootstrap stability, and why raw-row resampling is not used.**

Resampling raw contract rows with replacement and refitting the full pipeline would break the indicator assumptions in `src/features.py`:

- `IndicatorBuilder` stores the fitted rows as `history_`, indexed by row id. Duplicated rows would count the same award several times in a buyer's history, inflating f4 totals and the f5 digit counts.
- Duplicated index labels make `.loc[idx]` return several rows per contract, which misaligns or breaks the history features.
- f4 and f5 sum awards dated strictly before each contract's day. Duplicates of an earlier award would enter later contracts' histories as if they were separate awards.

Two temporally valid alternatives are used instead.

- **(a) Model-stage bootstrap (primary).**
  - In each fold, `IndicatorBuilder` and the preprocessing are fitted once on the full fold-training part, in date order, exactly as in section 5.
  - Only the model stage is resampled: 20 bootstrap resamples of the rows of the transformed fold-training matrix (with replacement, seed = resample number 0 to 19). Each refits the anomaly model and scores the clean validation fold.
  - The features of every contract are unchanged, so chronology and histories are untouched.
  - Reported: mean pairwise Spearman correlation and mean top-5% Jaccard against the non-resampled fit.
  - Measures: how much the ranking depends on which training contracts the model happens to see, with feature construction held fixed.
- **(b) Time-block jackknife of the full pipeline (secondary, fold 5 only).**
  - Each calendar quarter of fold 5's training part is dropped once. The full pipeline is refitted on the rest, in date order, and scores fold 5's clean validation set.
  - Dropping rows creates history gaps but never duplicates an award.
  - Reported: Spearman correlation and top-5% Jaccard against the full fit, per dropped quarter.
  - Measures: sensitivity of the whole pipeline, including the history features, to losing one period of data.

**R(c) reweighting stability.** 200 weight vectors from a flat Dirichlet over the available indicators (f1, f3, f4, f5), seed 0. Report the Spearman correlation and top-5% Jaccard of each reweighted R(c) against equal weights, on each validation fold.

**Naive red-flag count baseline.** Count of:

- f1, `is_signed_pre_closure`, `above_p99`, `below_p1`, `zero_value_award` and `is_terminated`;
- plus 1 for each of f3, f4 and f5 at or above its 95th percentile on the fold's training part (missing counts as 0).

Report rank agreement and top-5% overlap with R(c) and the baseline, and its injection recall under section 5. Ties in the count are broken at random, seed 0.

**Top-50 face validity.**

- The 50 highest-scoring contracts of the frozen baseline configuration fitted on fold 5's training part, on fold 5's validation set (training period). The top 50 by R(c) are inspected the same way.
- Each contract is coded with a fixed sheet: likely data error / plausible legitimate reason / unexplained / publicly reported concern, with sources.
- The results are descriptive. They may expose data errors, which would be fixed only through a deviations-log entry. They are not used to choose between models.

### 7.2 Extended Isolation Forest (first comparison)

- **Library:** `isotree` (version pinned in `requirements.txt` once installed; installation on Python 3.13 still to be checked).
- **Grid:** `ntrees` {100, 200, 400} x `sample_size` {256, 1024, 4096} x `ndim` {2, 4}. Seeds {0, 1}.
- **Implementation control:** `isotree` with `ndim=1`, 400 trees, `sample_size=256`, reported only, not selectable. It separates implementation differences from the effect of oblique splits.
- **Settings:** all other parameters left at the installed defaults. The full parameter dictionary is saved with the results.
- **Score:** isotree's standardized outlier score (higher = more anomalous), used as is.
- **Explanation:** see section 8. TreeExplainer does not support isotree.
- **Reproducibility check (CP2):** two identical runs must give identical metric tables.

### 7.3 Feature variants (with the Isolation Forest baseline configuration)

**Relative value.**

- The exact definition is written and committed as an amendment before this experiment runs. It is not changed after its validation results are seen. The amendment must state:
  - the formula;
  - which training-only reference it uses (for example the median log value per procurement category) and that the reference is fitted on each fold's training part only;
  - the handling of zero, missing and clipped values, using the same P1/P99 and imputation rules as the Phase 2 value input;
  - the handling of a missing category or a category unseen in the training part;
  - where the feature enters the preprocessing.
- **Decision rule (fixed now):** adopt only if `value` recall and `value_x5` recall each rise by at least 0.02, primary recall does not fall, and stability stays at least 0.95.
- **Caveat:** the feature is designed to detect exactly the injected value distortion, so a gain is weak evidence.

**Clip-only days** (no signed log). Adopt only if primary recall rises by at least 0.02, stability falls by no more than 0.005, and the per-type guard (6.2) holds. Otherwise the Phase 2 transformation stays.

### 7.4 R(c) sensitivity: f4 as a percentile

- f4 is replaced by its training-part percentile in R(c).
- Report the Spearman correlation and top-5% Jaccard against the current R(c), and mean R(c) by number of available indicators.
- **Decision rule:** adopt only if the range of mean R(c) across the 2-, 3- and 4-indicator groups falls by at least half. Otherwise R(c) stays as in Phase 2, and the result is reported as a sensitivity analysis.

### 7.5 Reciprocal rank fusion (RRF)

**Components (fixed now):**

1. R(c);
2. the frozen Isolation Forest configuration;
3. the best-ranked eligible new family from section 6.3, if one exists.

The third component is determined exclusively from the 2018 to 2023 validation results, under the rules in section 6.3. It is fixed and recorded in `model_evaluation/MODEL_SELECTION_FREEZE.md` before any test-period scoring. If no new family is eligible, RRF has two components.

Equal weights, k = 60 (Cormack, Clarke and Buettcher, 2009). Weights and k are fixed now and are not chosen using any results. No RRF component, weight or k may be selected or changed using the 2024 to 2026 test results.

**Training-referenced ranks.** Plain RRF ranks contracts within the list being scored. That would make a contract's fused score depend on which other contracts are in the batch. Ranks are therefore taken against a fixed training reference:

- For component m, the reference is the scores of the training part the component was fitted on: the fold's training part during validation, the full training period at the final test.
- The rank of contract c is `rank_m(c) = 1 + #{reference contracts with score > s_m(c)}`, where higher scores mean more anomalous.
- RRF(c) = sum over components of `1 / (60 + rank_m(c))`.
- Reference contracts are ranked against the reference itself, counting strictly greater scores only.
- A missing component score receives the worst possible rank, which is the reference size plus 1. R(c) is never missing, because f1 is always available.

**In validation (each fold):**

- Components are fitted on the fold's training part.
- Clean validation contracts and injected copies are scored with the same fitted components, then ranked against the same training reference.
- For injected copies, f3 and R(c) are recomputed as in section 5.
- Recall threshold: 95th percentile of the clean validation RRF scores, as for every other model.
- Seed stability: the two seeds apply to the model components (R(c) is deterministic).

**At the final test:**

- Components are refitted on the full training period with their frozen settings.
- Reference ranks use full training-period scores.
- The flagging threshold is the 95th percentile of the training-period RRF scores. It is fixed before the test period is scored.
- Test contracts are scored individually against the training reference, so no test contract affects another's RRF score.

RRF is reported with the same metrics. It is deployed only if it wins under the rules in section 6.3.

### 7.6 Autoencoder, 7.7 Deep SVDD, 7.8 buyer-supplier graph

Entries are written and committed before each experiment, after the previous checkpoint passes (section 9). Planned constraints:

- early stopping on the last 10% (by date) of each fold's training part, never on the validation fold;
- training on all fold-training contracts (not filtered by R(c));
- fixed torch seeds and deterministic CPU execution;
- the Deep SVDD centre computed from fold-training data, with no bias terms;
- the graph built per fold from fold-training awards and the frozen resolver, with coverage reported.

## 8. Explainability validation

An attribution method is called **validated** for a model only if it passes all of the checks below, on the top 5% of fold 5's clean validation set (about 605 contracts).

1. **Additivity:** for every explained contract, the sum of attributions plus the base value equals the explained model output, within 1e-6 relative error.
2. **Correct target:** the explained output is the model score used for ranking, or a strictly monotonic transformation of it. This is checked by rank correlation of 1.0 between the explained output and the ranking score on the explained contracts. For the scikit-learn Isolation Forest, TreeExplainer explains a path-length based output, so this check is required, not assumed.
3. **Method agreement (model-agnostic methods only):** the model-agnostic method is also run on the frozen Isolation Forest, where TreeExplainer is available. For each contract, its three largest features are compared with TreeExplainer's. The mean top-3 overlap is reported. A mean below 2 of 3 means the method is not validated.
4. **Stability:** two runs with different background samples (seeds 0 and 1) give a mean Spearman correlation of at least 0.9 between the per-contract attribution vectors.
5. **Feasibility:** time is first measured on 20 contracts and extrapolated to the review set. The full review set must finish in under 30 minutes (section 6.3).

Fixed settings:

- background data: 100 rows of the fold-training matrix, sampled with the stated seed;
- inputs: the 14 model inputs;
- method for isotree: SHAP permutation explainer.

Known limitation: model-agnostic SHAP with an independent background treats features as independent. That assumption does not hold for, for example, `days_to_signing` and `is_signed_pre_closure`. This is reported with the results.

If a method fails any check, the report and the dashboard say that no validated attribution exists for that model. The dashboard then shows only the itemized red-flag scorecard and historical comparisons for it. A failing method is never shown as if it were validated, and no other method is substituted silently.

## 9. Checkpoints

| ID | Condition to pass | If it fails |
|---|---|---|
| CP0 | Baseline reproduces Phase 2 | Passed 7 Nov 2026 |
| CP1 | Evaluation harness reproduces the Phase 2 grid result for 400 / 256 in `results/if_tuning_summary.csv` (stability 0.9916, value 0.117, date 0.842, rounded) | Fix the harness before any new model |
| CP2 | EIF results reproducible across two runs | EIF reported as not reproducible; continue with the baseline |
| CP3 | Autoencoder reproducible across two runs, stability at least 0.95 | Skip Deep SVDD |
| CP4 | Deep SVDD reproducible, stability at least 0.95 | Report as attempted |
| CP5 | Graph experiment working by 16 Nov | Stop; write a feasibility note |
| Hard stop | 18 Nov: unfinished experiments are reported as attempted | Dashboard work starts regardless |
| Freeze | 19 Nov: `model_evaluation/MODEL_SELECTION_FREEZE.md` written and committed | No test scoring before this |

## 10. Freeze and final test

- `model_evaluation/MODEL_SELECTION_FREEZE.md` lists each selected configuration, its commit, its settings and the selection evidence (fold means).
- Each frozen configuration is refitted on the full training period, and its threshold is set as the 95th percentile of training scores.
- The test period is scored once.
- Reported:
  - share above threshold;
  - score drift against training;
  - rank agreement and top-5% overlap between models;
  - injection recall on the test period (sensitivity only).
- The test-period results are added to the report as they come out, including unfavorable results.

## 11. Use of outputs in deployment

- The dashboard shows frozen scores only. It does not refit or rescore single contracts.
- Auditor reviews are stored in an append-only log, separate from contract records and scores.
- Reviews are not labels and do not trigger retraining.
- External sources (debarment lists, Kenya Gazette, OpenSanctions), if added, are context only:
  - shown as unverified name matches;
  - limited to notices dated before the contract signing date;
  - never used as model inputs or validation labels.

## 12. Deviations log

| Date | Section | Change | Reason | Seen results before change? |
|---|---|---|---|---|
| | | | | |
