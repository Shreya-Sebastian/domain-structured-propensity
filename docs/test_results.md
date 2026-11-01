# Test-block results

Scored once on 1 November 2026, after `hypotheses.md` was committed (commit `f7ac82a`). Test block: 2010-03 to 2010-11, 2,058 clients, yes-rate 52.1% (training: 6.7%). Raw outputs: `results/ladder/test/`.

## Verdicts

| # | Question | Result | Verdict |
| --- | --- | --- | --- |
| H1a | Cost of the one direction (AUC, directions minus unconstrained) | +0.008 [+0.004, +0.013] | **Inconclusive** by the rule (interval not inside ±0.005). It is a small gain, not a cost |
| **H1b (primary)** | Cost of factor grouping (AUC, structured minus directions; margin 0.02) | +0.016 [−0.004, +0.036] | **Non-inferior**: lower bound above −0.02 |
| H1c | Same for EBMs (structured minus unconstrained) | +0.016 [+0.001, +0.032] | **Non-inferior** (and better) |
| H2a | Structured models lower anyone's score when a previous success is added | 0 of 1,215 clients for all three | **Supported** (sanity check) |
| H2b, H2c | Same for unconstrained models | Trees and EBM: 0%. Logistic regression: 100% (−0.22 log-odds), also in the realistic variant | Descriptive |
| H3 | Separation in the structured model | Shuffling any factor moves no other factor's share for any client; unconstrained models: 81% to 100% of clients | **Supported** |
| H4a | Main reason more stable at factor level than feature level (structured) | Agreement 0.817 vs 0.682; difference +0.135 [+0.128, +0.143] | **Supported** |
| H4b | Domain factors more stable than random groups of the same sizes (unconstrained) | 0.720 vs 0.665; difference +0.056 [+0.050, +0.061] | **Supported** |

Intervals are 95% paired bootstrap intervals over test clients.

## Accuracy of every model

| Model | AUC | Average precision | Precision in top 10% | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| Rule baseline | 0.666 | 0.650 | 77.9% | 0.235 | 0.117 |
| Logistic regression | 0.613 | 0.604 | 62.6% | 0.243 | 0.062 |
| LightGBM unconstrained | 0.632 | 0.633 | 68.0% | 0.273 | 0.179 |
| LightGBM with direction | 0.640 | 0.634 | 66.5% | 0.272 | 0.180 |
| **LightGBM structured** | **0.656** | 0.647 | 68.9% | 0.261 | 0.152 |
| LightGBM structured, no life stage | 0.666 | 0.651 | 71.4% | 0.261 | 0.171 |
| EBM unconstrained | 0.604 | 0.625 | 74.3% | 0.269 | 0.165 |
| EBM structured | 0.621 | 0.634 | 73.3% | 0.265 | 0.162 |

Other differences (not pre-registered as hypotheses):

- Structured minus unconstrained LightGBM (the whole structure): AUC +0.024 [+0.003, +0.045].
- Structured LightGBM minus logistic regression: AUC +0.043 [+0.029, +0.057].
- Dropping life stage from the structured model: AUC +0.010 [+0.001, +0.020].
- Precision in the top 10% differences all have intervals of about ±6 points and include zero.

## What the results say

1. **Following the reasoning model cost nothing on these months.** The structured models were at least as accurate as the unconstrained ones, for LightGBM and for EBM. The most likely reason is the shift between training (2008 to mid-2009, 6.7% yes) and test (2010, 52% yes): forbidding interactions across factors stops the model from learning combinations, such as life stage with 2008 market conditions, that no longer hold. This is a reading, not something the study tested.
2. **A simple rule did as well as the best model.** "Previous success first, then most recent contact" matched the best AUC and had the best precision in the top 10%. Under this much drift, the strongest signal is the client's own history with the bank, which is also the one factor with an external direction. Any claim for the models has to be made against this rule.
3. **Reasons are steadier at the factor level, and more so with domain factors.** Across 30 refits, the structured model gave the same main factor 82% of the time, against 68% for the main single feature. Part of that comes from having 4 factors rather than 12 features; the random-group control shows domain grouping adds about 6 points on top.
4. **Separation holds exactly.** In the structured model, changing a client's age can never move their relationship share; in the unconstrained model it does for almost every client.
5. **Life stage is not needed and changes who is selected.** Dropping it improved AUC slightly. With it, the structured model put 22.9% of clients aged 60 and over in the top 10%, against 3.8% for the unconstrained model, with similar precision (68% and 69%). The two models select very different people for almost the same accuracy, which is a reason to monitor selection by group, not only accuracy.
6. **Probabilities are poorly calibrated for every model** (ECE 0.06 to 0.18), because calibration was fitted on mid-2009 months with a 36% yes-rate. Ranking is usable; probabilities would need recalibration on recent months.

## Limits

- One test block of 2,058 clients from one bank, one product and one period of strong change.
- The settings were fixed, not tuned; a tuned unconstrained model might close the gap.
- Market indicators stand in for the bank's own offers and strategy; their share is context.
- The structure encodes assumptions, not causes.
- Logistic regression's wrong sign on previous success comes from a near-duplicate feature (`contacted_before`, correlation 0.97), not from the data contradicting the direction.
