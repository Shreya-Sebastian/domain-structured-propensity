# Domain-structured propensity models

A feasibility study on public bank data: what does it cost to make a propensity model follow a written reasoning model, and what do you get back? What you get back is reasons that are computed exactly from the model, never generated, and that can be checked.

**Status: demo element only.** The full study (accuracy cost of directions and of factor grouping, conformance tests, a pre-registered test-block result) is designed but not run.

## What the demo shows

A gradient-boosted model (LightGBM) is fitted twice on the same data with the same settings:

- **unconstrained**
- **factor-grouped**: each tree may only split on features from one factor of the reasoning model (`interaction_constraints`)

Because every tree in the factor-grouped model stays inside one factor, the model's score is a sum of one function per factor. Each client's score therefore splits **exactly** into five factor contributions. Those contributions go into a decision record (JSON), and fixed templates turn the record into an advisor brief. No language model is involved.

## Data

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), `bank-additional-full.csv`: 41,188 phone contacts of a Portuguese bank, May 2008 to November 2010 (Moro, Cortez and Rita, 2014). Licence CC BY 4.0. Not included in the repository; download the zip from the link and unpack `bank-additional.zip` into `data/raw/`.

Availability audit (scoring moment: before the first call of a campaign): `duration`, `campaign`, `month` and `day_of_week` are removed, and a test fails if any reaches a model. Call dates are rebuilt from row order. `pdays = 999` (never contacted) becomes a flag plus a missing value.

## Reasoning model

`reasoning_model.yaml` assigns every feature to one of five factors: capacity to save, relationship and past response, market context, reachability, life stage. It also marks which factors may be raised with a client (life stage never) and which are context only (market context). No directions are enforced yet.

## Time blocks

Fixed in `config/splits.yaml` before any model was fitted:

| Block | Months | Rows | Yes-rate |
| --- | --- | --- | --- |
| Train | 2008-05 to 2009-05 | 36,224 | 6.7% |
| Calibration | 2009-06 to 2009-08 | 1,663 | 35.7% |
| Pilot | 2009-09 to 2009-12 | 1,243 | 43.6% |
| Test | 2010-03 to 2010-11 | 2,058 | not looked at |

The test block is never scored by the demo.

## Demo results

Checks, all passing (`tests/`, 14 tests):

| Check | Result |
| --- | --- |
| Banned fields reach no model | Pass |
| Trees in the factor-grouped model that mix factors | 0 of 300 |
| Trees in the unconstrained model that mix factors | 300 of 300 |
| Largest gap between factor sum plus base and model score, over 1,243 pilot clients | below 1e-14 |
| Every number in a brief appears in its record | Pass |
| Life stage never appears as a talking point | Pass |

Pilot metrics on the pilot block (**indicative only**: one seed, untuned, no intervals, validation months):

| Model | AUC | Average precision | Precision in top 10% |
| --- | --- | --- | --- |
| Unconstrained | 0.606 | 0.547 | 66.9% |
| Factor-grouped | 0.627 | 0.593 | 71.8% |

The factor-grouped model is not worse on these months, and is slightly ahead. That should not be read as a finding: the difference has no interval, settings were not tuned, and the yes-rate shifts strongly between training (6.7%) and pilot (43.6%), which can favour simpler models. Accuracy is modest for both, as expected when training on 2008 and scoring late 2009.

Example briefs for three clients chosen by rule (top-ranked, at the top-10% boundary, median) are in `results/demo/briefs.md`, their records in `results/demo/records/`.

## Run it

```
py -3 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/run_demo.py
.venv/Scripts/python -m pytest -q
```

## Limits

- The factors encode assumptions, not causes. The model can be shown to follow them, not that they drive uptake.
- Contributions are exact for this model and relative to the average training client; they describe the model, not the client's motives.
- Not a bank's own data or system; one product, one country, 2008 to 2010; no client id, so a client can repeat across campaigns.
- Probabilities are calibrated on 2009-06 to 2009-08 and applied to later months, while the yes-rate keeps rising.

## Next

Phase 0: sources for each direction in the reasoning model and `hypotheses.md`, committed before the test block is scored. Then the full ladder (unconstrained, directions only, directions plus factor groups), bootstrap intervals and directional tests.
