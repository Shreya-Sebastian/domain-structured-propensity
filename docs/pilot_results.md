# Pilot results: the full ladder on validation months

Run on 1 November 2026, `scripts/run_ladder.py --block pilot`. Outputs in `results/ladder/pilot/`.

**These are validation results, not the study's result.** The pilot block (September to December 2009, 1,243 clients) checks the code and shows how wide the intervals are. The test block (2010) has not been scored.

## Headline

**On the pilot months the structured model is not worse than the unconstrained one, and on average precision it is better (+0.041, interval +0.004 to +0.081).** No model ranks clients well: AUC is 0.59 to 0.64 for all of them. The yes-rate rises from 6.7% in the training months to 43.6% here, and a simple rule (previous success first, then most recent contact) has the best precision in the top 10%.

## All models

| Model | AUC | Average precision | Precision in top 10% | Lift in top 10% | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Rule baseline | 0.622 | 0.541 | 75.3% | 1.73 | 0.230 | 0.069 |
| Logistic regression | 0.640 | 0.596 | 72.6% | 1.66 | 0.237 | 0.079 |
| LightGBM unconstrained | 0.605 | 0.549 | 66.9% | 1.54 | 0.253 | 0.117 |
| LightGBM with direction | 0.613 | 0.554 | 65.3% | 1.50 | 0.251 | 0.113 |
| **LightGBM structured** | **0.629** | **0.590** | **74.2%** | **1.70** | 0.247 | 0.111 |
| EBM unconstrained | 0.591 | 0.540 | 65.3% | 1.50 | 0.252 | 0.118 |
| EBM structured | 0.608 | 0.579 | 71.0% | 1.63 | 0.244 | 0.100 |
| LightGBM structured, no life stage | 0.638 | 0.574 | 72.0% | 1.65 | 0.250 | 0.120 |

Precision in the top 10% rests on 124 clients, so differences of a few points in that column are noise.

## Comparisons (paired bootstrap, 1,000 resamples, 95% intervals)

| Comparison | AUC difference | Average precision difference | Top-10% precision difference |
| --- | --- | --- | --- |
| Direction minus unconstrained (H1a) | +0.008 [−0.001, +0.016] | +0.005 [−0.004, +0.016] | −1.6 pp [−5.7, +4.0] |
| **Structured minus direction (H1b, primary)** | **+0.015 [−0.022, +0.054]** | +0.036 [−0.003, +0.072] | +8.9 pp [−2.4, +17.7] |
| Structured minus unconstrained | +0.023 [−0.015, +0.061] | +0.041 [+0.004, +0.081] | +7.3 pp [−2.4, +17.7] |
| EBM structured minus EBM unconstrained (H1c) | +0.017 [−0.015, +0.047] | +0.039 [+0.009, +0.068] | +5.6 pp [−4.0, +13.7] |
| Structured minus logistic | −0.011 [−0.029, +0.007] | −0.005 [−0.027, +0.019] | +1.6 pp [−5.7, +8.9] |
| No life stage minus structured | +0.009 [−0.010, +0.030] | −0.016 [−0.040, +0.010] | −2.2 pp [−8.9, +5.0] |

**If the decision rules in `hypotheses.md` were applied to these pilot numbers** (they are not: they apply to the test block):

- H1a would be inconclusive: the interval is not inside ±0.005. A single direction on one flag moved AUC by +0.008, because the constraint changes which splits each tree makes.
- H1b would be inconclusive: the lower bound (−0.022) is just below the −0.02 margin, although the point estimate favours the structured model.
- H1c would be non-inferior: the lower bound (−0.015) is above −0.02.

## Directional test: previous success

Moving `prev_success` from 0 to 1 for the 560 clients contacted in an earlier campaign:

| Model | Clients whose score falls | Average change (log-odds) |
| --- | --- | --- |
| Logistic regression | 100% | −0.22 |
| LightGBM unconstrained | 0% | +0.18 |
| LightGBM with direction | 0% | +0.17 |
| LightGBM structured | 0% | +0.39 |
| EBM unconstrained | 0% | +0.63 |
| EBM structured | 0% | +0.81 |

The tree models follow the direction even without the constraint. Logistic regression violates it for everyone, because of how the data are coded: in the training months a previous success almost always comes with a recorded contact date (`contacted_before` = 1), so the two features are nearly identical (correlation 0.97). The model puts the effect of success on `contacted_before`, and the test's move (success switched on, contact date left missing) is a combination never seen in training. That makes this test partly a test of extrapolation for unconstrained models.

## Separation (H3)

Shuffle one factor's features across pilot clients and count clients whose **other** factors' shares move (`results/ladder/pilot/separation.csv`):

| Shuffled factor | Structured | Unconstrained | With direction only |
| --- | --- | --- | --- |
| Capacity | 0% | 100% | 100% |
| Relationship | 0% | 99.8% | 99.6% |
| Reachability | 0% | 100% | 100% |
| Life stage | 0% | 99.9% | 100% |
| Market context | not informative | not informative | not informative |

In the unconstrained model, other factors' shares move by up to 1.2 log-odds when life stage is shuffled. Market context tests nothing: every pilot month's market values lie outside the training range, so no model's own market share moves either.

None of the structured model's 300 trees mixes factors; all 300 trees in the other two LightGBM models do. Sanity check: for all pilot clients the factor shares plus base equal the structured model's raw score (largest gap 7e-15), as TreeSHAP shares do for any tree model.

## Groups in the top 10%

| Age band | Clients | Selected, structured | Selected, unconstrained | Yes-rate of those selected, structured |
| --- | --- | --- | --- | --- |
| Under 30 | 215 | 19.5% | 14.0% | 76% |
| 30 to 59 | 734 | 4.1% | 12.8% | 63% |
| 60 and over | 294 | 17.7% | 0.0% | 79% |

The unconstrained model puts nobody aged 60 or over in its top 10%, while the structured model does, and they say yes more often than the average selected client. The same holds for retired clients (18.6% selected against 0%). This is a description, not an explanation: the data do not show why the unconstrained model ranks older clients down.

## What this shows and what it does not

**Shows:**
- The code runs end to end for every model, with intervals, directional tests, group checks and the separation test.
- Factor grouping gives separation; the unconstrained models do not have it.
- On these months, factor grouping costs nothing measurable and may help.
- Dropping the life-stage factor costs nothing measurable in AUC.

**Does not show:**
- Anything about the test months: the shift continues (about 50% yes in 2010).
- Whether the structured model's advantage is real. A plausible reading is that blocking cross-factor interactions stops the model from learning combinations with the market indicators that held in 2008 and broke later. That is a hypothesis, not a finding.
- Anything about tuned models: every model uses fixed settings.

## Points for the review of `hypotheses.md`

1. **The margins may be too tight for the data.** The AUC intervals here are about ±0.035 wide on 1,243 clients; on 2,058 test clients they will be roughly ±0.03. With a margin of 0.02, H1b can come out inconclusive even if the structured model is as good. Options: keep the margin and accept that "inconclusive" is a likely result; widen it before the test run, with the reason recorded; or add the pilot block to the test evaluation (not recommended: it has been seen).
2. **H1a tests almost nothing** (one constraint on one flag). Keep it, but report it briefly.
3. **The directional test's move** for `prev_success` is out of distribution for unconstrained models (above). Decide whether to add a realistic variant that sets `prev_success` and `contacted_before` together.
