# Domain-structured propensity models

A feasibility study on public bank data: what does it cost to make a propensity model follow a written reasoning model, and what do you get back? What you get back is reasons that are separate by factor: each factor's share of a client's score depends only on that factor's own facts. The reasons are computed, never generated, and can be checked.

**Status:** demo built; phase 0 (evidence for the reasoning model) done; full model ladder run on the pilot (validation) months only. Hypotheses are drafted in `hypotheses.md` and not yet committed, so the test block has not been scored.

## What the demo shows

A gradient-boosted model (LightGBM) is fitted twice on the same data with the same settings:

- **unconstrained**
- **factor-grouped**: each tree may only split on features from one factor of the reasoning model (`interaction_constraints`), plus the one supported direction (`monotone_constraints`)

Because every tree in the factor-grouped model stays inside one factor, the model's score is a sum of one function per factor. Each factor's share of a client's score is therefore **separate**: it depends only on that factor's own facts, so changing, say, age can never move the relationship share. In an unconstrained model the shares also add up to the score (TreeSHAP shares always do), but interactions spread across factors, so changing age moves other factors' shares too. The shares go into a decision record (JSON), and fixed templates turn the record into an advisor brief. No language model is involved.

## Data

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), `bank-additional-full.csv`: 41,188 phone contacts of a Portuguese bank, May 2008 to November 2010 (Moro, Cortez and Rita, 2014). Licence CC BY 4.0. Not included in the repository; download the zip from the link and unpack `bank-additional.zip` into `data/raw/`.

Availability audit (scoring moment: before the first call of a campaign): `duration`, `campaign`, `month` and `day_of_week` are removed, and a test fails if any reaches a model. Call dates are rebuilt from row order. `pdays = 999` (never contacted) becomes a flag plus a missing value. The previous campaign outcome becomes a 0/1 flag `prev_success`, because LightGBM cannot enforce a direction on a categorical feature.

## Reasoning model

`reasoning_model.yaml` assigns every feature to one of five factors: capacity to save, relationship and past response, market context, reachability, life stage. It also marks which factors may be raised with a client (life stage never) and which are context only (market context).

**Phase 0** (`docs/phase0_evidence.md`) looked for evidence outside this bank's data for each direction. Only one is supported: a previous campaign success raises uptake (moderate, by analogy with repeat-purchase and donor data). Everything else is left free inside its factor: the evidence on debts is offsetting, on interest rates contradictory, and on recency and contact type absent. The market indicators also stand in for the bank's own changing offers and strategy.

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

Checks, all passing (`tests/`, 21 tests):

| Check | Result |
| --- | --- |
| Banned fields reach no model | Pass |
| Trees in the factor-grouped model that mix factors | 0 of 300 |
| Trees in the unconstrained model that mix factors | 300 of 300 |
| **Separation:** shuffle one factor's facts across pilot clients; clients whose other factors' shares move, factor-grouped model | 0% for every factor |
| Same, unconstrained model | 99.8% to 100% (capacity, relationship, reachability, life stage) |
| Sanity check: factor shares plus base equal the model score (true for any tree model) | gap below 1e-14 |
| Every number in a brief appears in its record | Pass |
| Life stage never appears as a talking point | Pass |

Pilot metrics on the pilot block (**indicative only**: one seed, untuned, no intervals, validation months):

| Model | AUC | Average precision | Precision in top 10% |
| --- | --- | --- | --- |
| Unconstrained | 0.605 | 0.549 | 66.9% |
| Factor-grouped | 0.629 | 0.590 | 74.2% |

The factor-grouped model is not worse on these months, and is slightly ahead. That should not be read as a finding: the difference has no interval, settings were not tuned, and the yes-rate shifts strongly between training (6.7%) and pilot (43.6%), which can favour simpler models. Accuracy is modest for both, as expected when training on 2008 and scoring late 2009.

Example briefs for three clients chosen by rule (top-ranked, at the top-10% boundary, median) are in `results/demo/briefs.md`, their records in `results/demo/records/`.

## Model ladder (pilot months only)

`scripts/run_ladder.py` fits every model in the study with fixed settings, calibrates on the calibration months and evaluates on one block, with paired bootstrap intervals and directional tests. On the pilot months (1,243 clients) the structured LightGBM is not worse than the unconstrained one (AUC 0.629 against 0.605; difference +0.023, interval −0.015 to +0.061). Details and caveats: `docs/pilot_results.md`. The test block is refused until `hypotheses.md` is committed.

## Run it

```
py -3 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/run_demo.py
.venv/Scripts/python scripts/run_ladder.py --block pilot
.venv/Scripts/python -m pytest -q
```

## Limits

- The factors encode assumptions, not causes. The model can be shown to follow them, not that they drive uptake.
- Factor shares are relative to the average training client; they describe the model, not the client's motives. Separation holds for the factor-grouped model only.
- Within the pilot and test months the market indicators take values the training months never reached, so neither model can tell those months apart: the market-context share is the same for every client in a period.
- Not a bank's own data or system; one product, one country, 2008 to 2010; no client id, so a client can repeat across campaigns.
- Probabilities are calibrated on 2009-06 to 2009-08 and applied to later months, while the yes-rate keeps rising.

## Next

Review and commit `hypotheses.md`, then score the test block once. After that: stability of factor-level against feature-level reasons (RQ3) and validity over time (RQ4).
