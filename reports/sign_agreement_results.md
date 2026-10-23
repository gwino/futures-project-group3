# Overnight/Daytime Sign Agreement and Last-Half-Hour Momentum, CL 2021–2026 (#33)

This note reports the third new hypothesis: does agreement or disagreement between the overnight (ON, previous 13:30 → 08:00 CT) and daytime (DAY, 08:00 → 13:00 CT) return directions change the last-half-hour (LH, 13:00 → 13:30 CT) momentum that the paper's rest-of-day predictor is meant to capture? Following the instructor's check-in note, coherence is treated both as four sign categories and as the two-dimensional vector (ON, DAY). One script produces every number here:

```bash
python -m src.analyze_sign_agreement --config config/sign_agreement.yaml
python -m pytest tests/test_sign_agreement.py
```

Tables are in `reports/tables/sign_agreement_*.csv`, figures in `reports/figures/sign_agreement_*.png`, and `analysis/07_sign_agreement_results.ipynb` collects them. The final project report, `reports/final_report.md`, places this result next to the replication (#29), the strategy backtest (#30), the overnight/daytime decomposition (#31), the volatility regimes (#32) and the subperiods (#38).

## Summary

- **When ON and DAY agree, the last half hour reverses rather than continues.** On the 720 agreement days the paper's sign-of-ROD strategy earns −4.5 basis points per day (standard error 1.7); on the 704 disagreement days it earns +1.9 (1.7). The difference, −6.4 (2.3), has p = 0.007, below the exploratory Bonferroni level of 0.010 for the five defined comparisons. In LH ~ ROD + Agree + ROD×Agree the interaction slope × 100 is −4.9 (1.6): the ROD slope is +2.3 (1.2) on disagreement days and −2.6 on agreement days. The project plan expected agreement to strengthen momentum; the data say the opposite.
- **The two-dimensional view agrees, but the product term is not a directional statistic.** ON×DAY is positive on every agreement day whatever the common direction, so its slope (108 (47) × 100) measures a sign-symmetric, nonlinear dependence of LH on the (ON, DAY) vector, not a reversal. The directional claim rests on sign(ROD)·LH and on the agreement interactions; on the tercile grid the strategy return is most negative in the ON mid/DAY high cell (−7.1 (3.1)) and in the two agreement corners (−6.9 (4.7), −5.6 (3.8)).
- **Out of sample the agreement term helps only with 10 March 2026 in the sample.** Adding Agree and ROD×Agree to the paper's ROD model raises the expanding-window OOS R² from 0.43% to 0.82% against the historical mean (Clark-West p = 0.051) and gives 0.38% against the ROD model it nests (p = 0.059). Without 10 March 2026, a both-negative agreement day with a +4.6% last-half-hour reversal, the OOS R² is −0.47% (p = 0.11). The in-sample difference survives that exclusion (−5.7 (2.3), p = 0.011) but not the exclusion of 2022 and 2026 together (−1.4 (2.3), p = 0.54).
- **It is a 2022 and 2026 result.** The agree-minus-disagree difference is −21 (8) basis points in 2022 and −12 (7) in 2026, and within ±6 of zero in the other years. Both are high-volatility years, which links this result to the volatility finding in #32, where agreement was not separated out.
- **No sign filter makes the strategy profitable after costs.** Trading the sign of ROD only on disagreement days gives a net mean of −0.1 (0.8) basis points and USD +3,055 over the sample, against −3.1 (1.3) and −USD 31,095 for the unfiltered baseline. A disagreement filter removes the losses but does not create a profit, and it is chosen with hindsight.

## Definitions and methods

| Item | Rule |
|---|---|
| Sample | The 1,436 trade dates of the return features (#28), 2021-01-04 to 2026-10-07. The 12 dates on which ON or DAY is exactly zero are reported separately and take no part in any group comparison or regression; the `include_zero_return_days` variant shows the alternative coding (agree = 0) |
| Categories | ON+/DAY+ (415 days), ON−/DAY− (305), ON+/DAY− (377), ON−/DAY+ (327). The first two are agreement, the other two disagreement; both signs are known at 13:00 CT |
| Group statistics | Mean LH and the paper's strategy return η = sign(ROD)·LH per group, with Newey-West standard errors (7 lags, the rule of #29) from one regression of the chronological series on the group indicators, so the lags run over consecutive sample dates rather than over the dates of one group. The agree-minus-disagree difference is the slope of η on the agreement indicator in the same chronological sample. Hit rates are the share of days on which LH has the sign of ROD, among days with LH ≠ 0 (LH is exactly zero on 21 days, which count as neither hit nor miss), with binomial standard errors |
| Two-dimensional view | ON and DAY cut into terciles; mean LH, η and their standard errors per cell from the same indicator regression. Regressions LH ~ ON + DAY + ON×DAY (a nonlinear term), LH ~ ROD + Agree + ROD×Agree and LH ~ ON + DAY + Agree + ON×Agree + DAY×Agree, with a Wald test of the interaction terms from the HAC covariance |
| Out of sample | Expanding-window forecasts after 500 observations (924 forecasts from 2023-01-06 on the 1,424 regression dates), against the historical mean and against the nested model, with the Clark-West test, as in #29 |
| Trading | Sign-of-ROD positions on all days, on agreement days only, on disagreement days only, plus sign-of-DAY and sign-of-ON, priced with the #30 baseline execution variant and costs (first trade at or after 13:00 and 13:30 CT, USD 2.50 commission and 0.5 tick per side) |
| Multiple testing | Five defined comparisons (the agree-minus-disagree difference, ROD×Agree, ON×DAY, ON×Agree, DAY×Agree), fixed in the configuration when the script was written and before its results were read, but not registered outside this repository. The Bonferroni level of 0.010 is reported as an exploratory correction |

## Results

### Sign categories (`sign_agreement_categories.csv`)

Mean LH and the strategy return η = sign(ROD)·LH in basis points with Newey-West standard errors; the hit rate is the share of decided days on which LH has the sign of ROD.

| Group | n | Mean LH (SE) | η = sign(ROD)·LH (SE) | 95% interval for η | LH follows ROD |
|---|---|---|---|---|---|
| ON+/DAY+ | 415 | −4.11 (1.94) | −4.11 (1.94) | [−7.9, −0.3] | 0.477 |
| ON−/DAY− | 305 | 5.00 (3.01) | −5.00 (3.01) | [−10.9, 0.9] | 0.470 |
| ON+/DAY− | 377 | −3.69 (2.30) | 2.10 (2.34) | [−2.5, 6.7] | 0.515 |
| ON−/DAY+ | 327 | −1.07 (2.42) | 1.62 (2.39) | [−3.1, 6.3] | 0.476 |
| Agreement | 720 | −0.25 (1.66) | −4.48 (1.72) | [−7.9, −1.1] | 0.474 |
| Disagreement | 704 | −2.47 (1.64) | 1.88 (1.72) | [−1.5, 5.3] | 0.497 |
| Agreement − disagreement | 1,424 | | −6.36 (2.34), p = 0.007 | [−11.0, −1.8] | −0.023 (0.025) |

![Categories](figures/sign_agreement_categories.png)

On agreement days the close moves against the day: ON+/DAY+ days end with a negative last half hour and ON−/DAY− days with a positive one. On disagreement days the mean LH is negative in both cells, so it carries no information about the sign of ROD, and the strategy return is small and positive. The hit rates stay near one half everywhere (0.47 on agreement days, 0.50 on disagreement days, difference −0.02 (0.03)); the difference is in the size of the moves, not their frequency.

### The two-dimensional view (`sign_agreement_grid.csv`)

![Grid](figures/sign_agreement_grid.png)

The figure shows the mean LH in basis points on the ON × DAY tercile grid (standard errors 3.0 to 4.7, 147 to 170 dates per cell). Read directionally through η = sign(ROD)·LH, the strategy loses most in the ON mid/DAY high cell (−7.1 (3.1)) and in the two agreement corners, ON low/DAY low (−6.9 (4.7)) and ON high/DAY high (−5.6 (3.8)); it gains most in the ON high/DAY low cell (+7.9 (4.0)), a strong disagreement cell. The middle row and column are close to zero.

| Model | Term | Slope × 100 (SE) | 95% interval | p | Adj. R² (%) | Wald p (interactions) |
|---|---|---|---|---|---|---|
| LH ~ ROD | ROD | −1.49 (0.97) | [−3.4, 0.4] | 0.13 | 0.55 | |
| LH ~ ROD + Agree + ROD×Agree | ROD | 2.29 (1.25) | [−0.2, 4.7] | 0.067 | 1.69 | 0.002 |
| | ROD×Agree | −4.93 (1.61) | [−8.1, −1.8] | 0.002 | | |
| LH ~ ON + DAY | ON | −1.29 (1.01) | [−3.3, 0.7] | 0.20 | 0.55 | |
| | DAY | −1.87 (1.44) | [−4.7, 0.9] | 0.19 | | |
| LH ~ ON + DAY + ON×DAY | ON×DAY | 108 (47) | [16, 200] | 0.021 | 1.91 | 0.021 |
| LH ~ ON + DAY + Agree + ON×Agree + DAY×Agree | ON×Agree | −1.12 (2.07) | [−5.2, 2.9] | 0.59 | 2.46 | 0.005 |
| | DAY×Agree | −9.99 (3.45) | [−16.8, −3.2] | 0.004 | | |

Regressions use the 1,424 dates with a sign category. The ON×DAY slope says that LH is more negative when the product is large and positive, which happens on agreement days of either direction; it does not say which way LH moves relative to the day, which is what the agreement interactions measure. The agreement effect sits in the daytime leg: on agreement days the DAY slope moves from +3.5 to −6.5, while the ON slope hardly changes.

### Out of sample (`sign_agreement_oos.csv`, `sign_agreement_robustness_models.csv`)

924 expanding-window forecasts from 2023-01-06. OOS R² in percent.

| Model | vs historical mean | Clark-West p | vs nested model | Clark-West p |
|---|---|---|---|---|
| LH ~ ROD | 0.43 | 0.14 | | |
| LH ~ ROD + Agree + ROD×Agree | 0.82 | 0.051 | 0.38 (vs ROD) | 0.059 |
| LH ~ ON + DAY | −0.21 | 0.32 | | |
| LH ~ ON + DAY + ON×DAY | −0.42 | 0.24 | −0.21 (vs ON + DAY) | 0.22 |
| LH ~ ON + DAY + Agree + ON×Agree + DAY×Agree | −0.22 | 0.15 | −0.01 (vs ON + DAY) | 0.11 |

The agreement interaction is the only addition in the project that improves on the paper's ROD model out of sample, and the improvement depends on one day. Without 10 March 2026 the agreement model's OOS R² is −0.47% against the historical mean (Clark-West p = 0.11) and −0.45% against ROD (p = 0.12); without 2022 and 2026 it is −1.26% and −0.89%. The product term fits better in sample but not out of it under any variant.

### Stability (`sign_agreement_by_year.csv`)

![By year](figures/sign_agreement_by_year.png)

| Year | Agreement η (SE) | Disagreement η (SE) | Difference (SE) |
|---|---|---|---|
| 2021 | −0.6 (3.9) | −0.7 (3.3) | 0.2 (5.3) |
| 2022 | −7.6 (5.2) | 13.5 (6.5) | −21.1 (7.7) |
| 2023 | 0.4 (3.9) | −3.2 (3.8) | 3.6 (5.0) |
| 2024 | −2.3 (2.8) | 1.3 (2.6) | −3.7 (3.3) |
| 2025 | −2.3 (2.9) | 3.4 (2.9) | −5.7 (4.6) |
| 2026 (to Oct 7) | −16.3 (5.7) | −4.4 (4.5) | −11.9 (6.9) |

The sign of the difference is negative in four of six years, but the magnitude comes from 2022 and 2026. Both were high-volatility years (#32), and in both the agreement days are the days on which oil moved a lot in one direction before the close.

### Trading comparison (`sign_agreement_strategies.csv`)

Net of the #30 baseline costs (USD 15 per round trip). Means over all 1,436 dates, days without a position counting as zero.

| Strategy | Traded days | Net mean (SE) | Sharpe (SE) | Hit rate | Net P&L | Gross P&L |
|---|---|---|---|---|---|---|
| Sign of ROD (baseline, #30) | 1,435 | −3.15 (1.26) | −1.07 (0.42) | 0.466 | −31,095 | −9,570 |
| Sign of ROD, agreement days only | 720 | −3.19 (0.87) | −1.44 (0.42) | 0.457 | −35,650 | −24,850 |
| Sign of ROD, disagreement days only | 703 | −0.08 (0.84) | −0.04 (0.42) | 0.471 | 3,055 | 13,600 |
| Sign of DAY | 1,432 | −3.41 (1.14) | −1.15 (0.42) | 0.468 | −36,870 | −15,390 |
| Sign of ON | 1,428 | −4.82 (1.26) | −1.64 (0.42) | 0.463 | −54,050 | −32,630 |

![Cumulative](figures/sign_agreement_cumulative.png)

The baseline's losses come entirely from agreement days. Skipping them leaves a strategy that is flat after costs (USD +3,055 over almost six years, interval well around zero). The filter is a hindsight choice: it was selected because the agreement result was found in the same sample, so the number is a description of the past rather than a forecast. The strategy hit rates follow the #30 convention and count a zero return as a miss.

### Robustness (`sign_agreement_robustness*.csv`)

Agree-minus-disagree difference in η, the two main interaction slopes (× 100) and the agreement model's out-of-sample R² against the historical mean, with standard errors.

| Variant | Difference (SE), p | ROD×Agree (SE) | DAY×Agree (SE) | OOS R² of ROD + Agree model (%), Clark-West p |
|---|---|---|---|---|
| Baseline (zero-return days set aside) | −6.36 (2.34), 0.007 | −4.93 (1.61) | −9.99 (3.45) | 0.82, 0.051 |
| Zero-return days coded as agree = 0 | same | −5.00 (1.60) | −10.06 (3.43) | 0.75, 0.052 |
| Without 10 March 2026 | −5.74 (2.27), 0.011 | −3.93 (1.39) | −7.51 (2.52) | −0.47, 0.11 |
| Without 2022 and 2026 | −1.41 (2.32), 0.54 | −0.75 (1.63) | −3.49 (2.74) | −1.26, 0.90 |
| Without roll dates | −7.18 (2.37), 0.002 | −5.34 (1.62) | −10.92 (3.53) | 0.73, 0.056 |
| Roll 3 sessions before expiry (127 days on another contract) | −6.55 (2.33), 0.005 | −4.98 (1.61) | −9.99 (3.48) | 0.81, 0.052 |
| Without degraded dates | −6.89 (2.38), 0.004 | −5.10 (1.64) | −9.90 (3.44) | 1.04, 0.043 |
| Without sessions after a weekend or holiday | −6.74 (2.69), 0.012 | −6.18 (1.84) | −11.36 (4.09) | 1.34, 0.015 |
| Log returns | −6.36 (2.34), 0.007 | −5.01 (1.63) | −10.07 (3.45) | 0.91, 0.050 |

The in-sample result does not depend on a single day, on the zero-day coding, on the roll convention (both the exclusion of roll dates and the pre-expiry roll rule, which rebuilds the reference prices from the minute bars as in #29) or on the return convention. It does depend on 2022 and 2026 together, and the out-of-sample gain depends on 10 March 2026 alone. Without that day the ON×DAY product slope falls from 108 (47) to 46 (38), so the continuous version leans on it too.

## Interpretation and caveats

- **Direction.** The hedging-demand story of the paper predicts stronger momentum when the day's move is unambiguous. On CL after 2020 the opposite holds: unambiguous days reverse into the close. One reading is that in a market dominated by two crises (2022 and 2026), large one-directional days attract profit-taking or inventory unwinding before the 13:30 reference close rather than hedging flows in the direction of the move.
- **Statistical weight.** One of five defined comparisons clears the exploratory Bonferroni level in sample, the out-of-sample Clark-West test is at p ≈ 0.05 with 10 March 2026 in the sample and p = 0.11 without it, and the effect disappears without 2022 and 2026. This is a lead tied to two crisis years, not an established effect.
- **Overlap with #32.** High-volatility days and agreement days overlap in 2022 and 2026. The volatility interactions alone were not significant (#32); the sign interaction is. A joint model is a natural follow-up and is not attempted here to avoid stacking tests.
- **Trading.** No version of the strategy earns a positive net return with a confidence interval that excludes zero.

## Checks

- Tests cover the classification and the zero-return days, the strategy return and hit indicators (a zero LH is neither hit nor miss), the indicator-regression group means (equal to the plain group means), the agree-minus-disagree difference on a constructed sample, the grid's partition of the dates, recovery of a planted agreement effect with its Wald test and nested OOS comparison, and the sign-filtered positions. Every sign is from a 13:00 CT price.
- The trading comparison reuses the #30 fills, baseline execution variant and costs, so the baseline row reproduces the #30 net result exactly (−3.15 bps, −USD 31,095).
- 87 tests pass (`python -m pytest`), 7 of them for this module.
