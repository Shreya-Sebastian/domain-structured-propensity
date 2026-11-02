# Decision audit and robustness checks

Done on 1 November 2026, after the test block had been scored once. Every design decision (47 in total) was checked against primary sources; the full register with verdicts and sources is in the results document's "Decisions and evidence" tab. Decisions that were only partly supported, or unsupported, were rerun with the better-supported alternative. **These are robustness checks decided after the result was seen. They are reported next to the pre-registered result in `docs/test_results.md`, never in its place.**

Scripts: `scripts/run_robustness.py`, `scripts/run_robustness_models.py`, `scripts/run_robustness_settings.py`. Outputs: `results/robustness/`.

## Verdicts

- 27 decisions supported, 18 partly supported, 2 wrong or unsupported.
- **Wrong:** `pdays = 999` was read as "never contacted"; 4,110 rows have 999 with earlier contacts. The feature `contacted_before` therefore means "has a recorded contact date".
- **Unsupported as the only evidence:** fixed, untuned settings shared by every model.

## What the checks found

1. **Settings matter most.** Across 30 random settings shared by both models, the structured model's test AUC stayed between 0.628 and 0.651; the directions-only model's ranged from 0.561 to 0.669. Structured was ahead in 23 of 30 (median +0.023). With each model's setting chosen on the pilot months, structured was behind by 0.034 [−0.049, −0.020]. The best less constrained setting beat the best structured one by 0.019. With LightGBM's library defaults, structured was ahead by 0.068 [+0.046, +0.091]. Reading: factor grouping acts as strong regularisation. With fixed settings there is no measurable cost; with each model tuned properly on the pilot months the cost is about 0.03 AUC (0.034 [−0.049, −0.020]). The 0.019 best-against-best gap chooses settings on the test block, so it understates the fair comparison. The benefit is much lower sensitivity to settings.
2. **Data fixes support the result.** Dropping 12 duplicates, removing `contact` (a last-call attribute), lagging the market indicators by one month, removing them, or all fixes together: the structured model stayed ahead of the directions-only model in every case (+0.013 to +0.108).
3. **Market context hurts every model.** Every test client's Euribor and employment level lies outside the training range. Without the market factor, all LightGBM models improved; structured reached 0.677, above the rule baseline (0.666).
4. **The pdays fix changed no tree** (`previous` carries the same information) but lowered the rule baseline from 0.666 to 0.639.
5. **`campaign` adds almost nothing:** 0.634 with it against 0.632 without, so the availability audit cost no accuracy.
6. **Statistics agree.** DeLong and BCa (10,000 resamples) match the percentile bootstrap: primary difference +0.016, DeLong [−0.004, +0.036], p = 0.11. Non-inferior at every margin from 0.005 to 0.03. A different seed gives identical models.
7. **Calibration needs recent data.** Platt on the calibration block was no better than isotonic (ECE 0.153). Platt refitted on the pilot months gave ECE 0.042 [0.034, 0.070].
8. **Month by month,** the structured model beat the unconstrained one in all 9 test months.
9. **Stability holds with 100 refits** (main factor 81%, main feature 67%). Domain factors (72%) beat the average of 50 random groupings (68%, range 63% to 72%) and a correlation-based grouping (65%), but only tie the best random grouping. Agreement is 73% for clients whose top two factors are close and 96% where they are far apart.
10. **Without life stage,** selection in the top 10% is much more even across age bands (9.7%, 9.3%, 13.0% against 11.5%, 5.8%, 22.9%), with slightly higher AUC.

## Decisions changed for future work

| Was | Now |
| --- | --- |
| `contacted_before` described as "contacted before" | "has a recorded contact date"; use `previous > 0` as the contact flag |
| Separation test presented as a finding | A check that the code builds what the design says (separation holds by construction) |
| "Structure costs nothing" | "No measurable cost with fixed settings; about 0.03 AUC with proper tuning; much less sensitive to settings" |
| Life stage in the default model | Drop it |
| Market context in the model | Drop or lag it |
| Calibrate once on mid-2009 | Recalibrate on the most recent months |
| H4b "supported" | Supported against the average, not the best, random grouping |
