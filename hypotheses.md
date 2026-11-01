# Pre-registered hypotheses

**Status: final, committed before the test block was scored.** The commit that adds this file is its timestamp. The test block (2010-03 to 2010-11, 2,058 clients) is scored only after this file is committed unchanged; `scripts/run_ladder.py --block test` refuses otherwise.

Written 1 November 2026, after phase 0 (`docs/phase0_evidence.md`) and before the full ladder was run on any block. What had been seen before writing: the demo's pilot run of an unconstrained and a factor-grouped LightGBM on the pilot block (AUC 0.606 and 0.627, without directions). After writing, the ladder is run once on the pilot block as a check of the code; any change made to this file after that run is listed under "Changes" with its reason.

## Fixed design

- **Data:** UCI Bank Marketing, `bank-additional-full.csv`, after the availability audit (`duration`, `campaign`, `month`, `day_of_week` removed).
- **Blocks:** train 2008-05 to 2009-05; calibration 2009-06 to 2009-08; pilot 2009-09 to 2009-12; test 2010-03 to 2010-11 (`config/splits.yaml`).
- **Reasoning model:** `reasoning_model.yaml` version 0.2: five factors; one direction (`prev_success` increasing).
- **Models** (`src/dsp/models.py`, fixed settings, no tuning): rule baseline; logistic regression; LightGBM unconstrained, with directions, and structured (directions plus one interaction group per factor, `monotone_constraints_method = advanced`); EBM unconstrained and factor-grouped (pairs only within a factor, directions by `monotonize`).
- **Calibration:** isotonic, fitted on the calibration block, for every model.
- **Statistics:** paired bootstrap over test clients, 1,000 resamples, seed 20261101, 95% percentile intervals. No correction for multiple comparisons; H1b is the primary hypothesis and the others are secondary.

## Primary metric

**AUC.** Precision in the top 10% is reported as the secondary metric that matters to an advisor, but on 2,058 test clients it rests on 206 clients and its interval is too wide (several percentage points) to decide a 2-point margin. This is a change from the research plan, made before any test result was seen.

## Hypotheses

| # | Comparison | Hypothesis | Decision rule |
| --- | --- | --- | --- |
| H1a | `lgbm_directions` minus `lgbm_unconstrained` | One direction on one binary feature costs nothing measurable: difference in AUC between −0.005 and +0.005 | Supported if the 95% interval lies inside [−0.005, +0.005]; rejected if it lies outside; otherwise inconclusive |
| **H1b (primary)** | `lgbm_structured` minus `lgbm_directions` | Factor grouping costs little: AUC no more than 0.02 lower (non-inferiority margin 0.02), with an expected point difference between −0.010 and 0 | Non-inferior if the interval's lower bound is above −0.02; inferior if the upper bound is below −0.02; otherwise inconclusive |
| H1c | `ebm_structured` minus `ebm_unconstrained` | Same as H1b for EBMs: margin 0.02 | As H1b |
| H2a | Directional test, `prev_success`, structured models | Zero clients whose score falls when `prev_success` goes from 0 to 1 (sanity check of the constraint) | Supported if the violation share is 0 for `lgbm_structured`, `lgbm_directions` and `ebm_structured` |
| H2b | Directional test, `prev_success`, unconstrained models | No threshold: the share of clients whose score falls is reported, for `lgbm_unconstrained`, `ebm_unconstrained` and `logistic` | Descriptive |
| H2c | Realistic directional test, every model | For clients contacted in an earlier campaign: 'contacted, no success' against 'contacted, success', both with a contact date (own, or the training median for successes). No threshold; reported because the H2b move combines values never seen together in training | Descriptive |
| H3 | Separation, `lgbm_structured` | No tree mixes factors, and shuffling one factor's features across test clients moves no other factor's share (within 1e-9) for any client, for every factor whose own share moves. Sanity check, not the hypothesis: factor shares plus base equal the raw score within 1e-6 (true for any tree model) | Supported, or a bug that is fixed and reported before any other result. Reported for contrast: the share of clients whose other-factor shares move in `lgbm_unconstrained` and `lgbm_directions` |

### RQ3: stability of the main reason (H4)

Main reason = the factor (or single feature) that raises a client's score most, leaving out market context, which the brief shows as context only. The model is refitted on 30 bootstrap resamples of the training months (seed 20261102). For each test client, agreement = the share of refits that give its most common main reason. Differences are averaged over clients, with a 1,000-resample bootstrap over clients (seed 20261101).

| # | Comparison | Hypothesis | Decision rule |
| --- | --- | --- | --- |
| H4a | `lgbm_structured`: factor-level minus feature-level agreement | The main factor is more stable than the main feature | Supported if the interval's lower bound is above 0; rejected if its upper bound is below 0; otherwise inconclusive |
| H4b | `lgbm_unconstrained`: domain factors minus random groups of the same sizes (mean of 5 random partitions of the same 12 features) | Domain factors give more stable main reasons than arbitrary groups, so the gain is not only from having fewer categories than features | As H4a |

Also reported for RQ3, without hypotheses: the same factor-minus-feature difference for `lgbm_unconstrained`, and `lgbm_structured` minus `lgbm_unconstrained` factor-level agreement.

**Why H4b:** there are 4 reason factors and 12 reason features, so factor-level agreement is higher partly by construction. H4b compares domain factors with random groups of the same sizes to separate that effect from the grouping itself.

## Also reported, without hypotheses

- Average precision, precision and lift in the top 10%, Brier score and expected calibration error for every model.
- `lgbm_structured` minus `logistic`, with interval.
- `lgbm_structured` without the life-stage factor: the accuracy cost of dropping it.
- Precision and selection rate in the top 10% by age band (under 30, 30 to 59, 60 and over) and by job, for `lgbm_structured` and `lgbm_unconstrained`.

## Known limits, stated in advance

- One test block of 2,058 clients in one period; the yes-rate rises from 6.7% in training to about 50% in the test months, so absolute accuracy and calibration will be poor for every model, and the comparisons are between models facing the same shift.
- The market indicators stand in for the bank's changing offers and strategy; their effects are context, not client behaviour.
- One direction only: H1a tests very little and is reported for completeness.
- The settings are fixed rather than tuned, the same for every LightGBM model.

## Changes

**1 November 2026, after running the full ladder on the pilot block** (twice: the second run only added the change-size columns below) (`results/ladder/pilot/`, summarised in `docs/pilot_results.md`). The run was a check of the code. No hypothesis, margin, metric or model setting was changed after it. Three changes, none of which changes a margin, metric or model setting:

- The directional test now also reports the average and smallest score change, so a large wrong sign can be told apart from a near-zero one.
- The pilot run showed that logistic regression violates the `prev_success` direction for every previously contacted client (−0.22 log-odds on average). In the training months a previous success almost always comes with a recorded contact date (`contacted_before` = 1; only 14 contacted clients without a success, none with a success and no date), so the two features are nearly identical (correlation 0.97). Logistic regression puts the effect on `contacted_before`, and the test's move (`prev_success` from 0 to 1 with `contacted_before` held at 0) is a combination never seen in training. H2b stays descriptive; this caveat is reported with its result.

- H3 was reworded. It first tested that factor shares add up to the score, but TreeSHAP shares add up for any tree model, so that is only a sanity check. What factor grouping guarantees is separation, which H3 now tests (`records.separation_test`). On the pilot block the structured model moves no other factor's share; the unconstrained model moves them for 99.8% to 100% of clients when capacity, relationship, reachability or life stage is shuffled. Shuffling market context tests nothing in either model, because the pilot months' market values all lie outside the training range.

**1 November 2026, review before committing** (no test result had been seen):

- **Margin kept at 0.02.** It was set from the literature on the accuracy cost of additive and constrained models (0 to 2 AUC points) before any test data was seen. The pilot showed intervals of about ±0.035, so on the test block H1b may well be inconclusive; the margin states what loss is acceptable, not what the data can detect, so it was not widened.
- **H2c added** (realistic directional test), descriptive only.
- **RQ3 (H4a, H4b) added**, written before any stability result on any block. After writing, the stability code is run once on the pilot block as a check of the code; any change after that run is listed below.
- **Code check of H2c and H4 on the pilot block** (`results/ladder/pilot/`): both run as intended. No hypothesis, margin or setting was changed after it.
- **Observation from that check, no change:** logistic regression still lowers the score for every client in the realistic variant (−0.22 log-odds). For a linear model the change is just the coefficient, whatever the other values, so the "never seen together" explanation above does not apply to it; the cause is the near-duplicate `contacted_before` (correlation 0.97), which takes the positive weight.
