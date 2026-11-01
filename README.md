# Domain-structured propensity models

A feasibility study on public bank data: what does it cost to make a propensity model follow a written reasoning model, and what do you get back? What you get back is reasons that are separate by factor: each factor's share of a client's score depends only on that factor's own facts. The reasons are computed, never generated, and can be checked.

**Status:** done for the core questions. Phase 0 (evidence for the reasoning model), the model ladder, separation and stability tests are built; the hypotheses were committed (`f7ac82a`) before the 2010 test months were scored once. Full results: `docs/test_results.md`.

## Results in short (2010 test months, 2,058 clients)

- **Factor grouping cost no accuracy.** Primary hypothesis (H1b, margin 0.02 AUC): structured minus directions-only LightGBM +0.016 [−0.004, +0.036], so non-inferior. The EBMs agree (+0.016 [+0.001, +0.032]). The whole structure against the unconstrained LightGBM: +0.024 [+0.003, +0.045].
- **A simple rule did as well as the best model:** previous success first, then most recent contact (AUC 0.666, best precision in the top 10%). Under strong drift (6.7% yes in training, 52% in test), the client's own history carries most of the signal.
- **Reasons are steadier by factor:** across 30 refits, the structured model's main factor stayed the same for 82% of clients against 68% for the main single feature (+0.135 [+0.128, +0.143]); domain factors beat random groups of the same sizes by +0.056 [+0.050, +0.061].
- **Separation holds:** in the structured model, changing one factor's facts never moved another factor's share; in unconstrained models it did for 81% to 100% of clients.
- **Life stage is not needed:** dropping it raised AUC by +0.010 [+0.001, +0.020], and with it the structured model selected far more clients aged 60 and over (22.9% against 3.8%) at similar precision.
- **Probabilities need recalibration** on recent months (ECE 0.06 to 0.18 for every model).

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
| Test | 2010-03 to 2010-11 | 2,058 | 52.1% |

The test block was scored once, after the hypotheses were committed. The demo never scores it.

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

## Model ladder

`scripts/run_ladder.py` fits every model in the study with fixed settings, calibrates on the calibration months and evaluates on one block: paired bootstrap intervals, directional tests, the separation test and the stability test (30 bootstrap refits). Pilot-month results, used only to check the code: `docs/pilot_results.md`. Test-month results: `docs/test_results.md`. The script refuses the test block unless `hypotheses.md` is committed and unchanged.

## Run it

```
py -3 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/run_demo.py
.venv/Scripts/python scripts/run_ladder.py --block pilot
.venv/Scripts/python scripts/run_ladder.py --block test
.venv/Scripts/python -m pytest -q
```

## Limits

- The factors encode assumptions, not causes. The model can be shown to follow them, not that they drive uptake.
- Factor shares are relative to the average training client; they describe the model, not the client's motives. Separation holds for the factor-grouped model only.
- In the pilot months the market indicators take values the training months never reached, so the market-context share is the same for every client there. In the 2010 test months they vary and the share differs between clients, but it still mostly reflects the period and the bank's own strategy, not the client.
- Not a bank's own data or system; one product, one country, 2008 to 2010; no client id, so a client can repeat across campaigns.
- Probabilities are calibrated on 2009-06 to 2009-08 and applied to later months, while the yes-rate keeps rising.
- One test block, fixed rather than tuned settings: a tuned unconstrained model might close the gap.

## Next

Validity over time (RQ4: rolling refits and counterfactuals), a version without the life-stage factor as the default, and the planned RAG layer that may only retrieve and phrase computed reasons (not built).
