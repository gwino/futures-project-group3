# Last-Half-Hour Directional Strategy on CL, 2021–2026 (#30)

This note reports the trading-strategy test of Baltussen, Da, Lammers & Martens (2021), Eq. (12), on NYMEX WTI crude oil (CL) from 4 January 2021 to 7 October 2026. The paper's timing strategy is long over the last half hour of the reference session when the rest-of-day return is positive and short otherwise. The paper reports it without transaction costs. Here the same signal is traded with executable fills and explicit costs, and the paper's frictionless return, the feasible gross P&L and the net P&L are reported separately. The results are produced by one script, read from `config/lh_strategy.yaml`:

```bash
python -m src.backtest_lh_strategy --config config/lh_strategy.yaml
python -m pytest tests/test_backtest_lh_strategy.py
```

Tables are in `reports/tables/lh_strategy_*.csv`, figures in `reports/figures/lh_strategy_*.png`, and `reports/tables/lh_strategy.manifest.json` records the settings, checks and output hashes. The notebook `analysis/05_lh_strategy_results.ipynb` collects them.

## Summary

- **The sign-of-ROD strategy did not make money on CL after 2020, before or after costs.** The mean last-half-hour return of the position is −1.2 basis points of notional per day with a standard error of 1.3, so the 95% interval runs from −3.7 to +1.3 basis points. The annualized Sharpe ratio is −0.40 with a standard error of 0.42. The hit rate is 48.0% with a standard error of 1.3%. The paper's commodity portfolio earned a Sharpe ratio of 1.42 on this signal over 1974–2020 (Table 6, Panel C), but CL alone was never significant in the paper's regressions (Appendix Table B3).
- **Execution delay costs nothing measurable; the spread does.** Entering at the first trade at or after 13:00 CT and exiting at the first trade at or after 13:30 CT changes the mean by 0.03 basis points relative to the paper's reference prices. With a half-tick of spread and USD 2.50 commission per side (USD 15 per round trip), the mean falls to −3.1 basis points and the interval excludes zero. There is no breakeven cost: the gross P&L is already negative, at −USD 6.67 per round trip on average.
- **Nothing in the robustness checks changes the sign.** Skipping roll dates, expiry-adjacent sessions, degraded dates, sessions after a weekend or holiday, or the single worst day moves the gross mean between −1.2 and −0.2 basis points, always inside the baseline interval. The 3-sessions-before-expiry roll rule gives the same result as the paper's volume-crossover rule.
- **2026 is the only year with a clearly negative estimate.** Gross means by calendar year sit within ±2.4 basis points of zero with intervals that include zero, except 2026, at −10.5 basis points (interval −17.9 to −3.1). The single worst day, 10 March 2026, is a short position on a −15.4% rest-of-day return that reversed by +4.6% in the last half hour; removing it leaves 2026 negative.

## Strategy and execution

| Item | Rule |
|---|---|
| Sample | 1,436 trade dates that pass the paper's filters (`paper_sample`), 2021-01-04 to 2026-10-07; 1,435 are traded (ROD was exactly zero on 2024-06-13) |
| Signal | Sign of ROD = P(13:00 CT) / P(previous 13:30 CT) − 1, from the held contract's reference prices (#27). Both prices are the close of the last 1-minute bar starting before the time, so the signal uses no trade from 13:00 CT onward |
| Position | +1 contract when ROD > 0 (770 days), −1 when ROD < 0 (665 days), flat when ROD = 0 |
| Entry | Open of the first held-contract 1-minute bar starting at or after 13:00 CT, that is the first trade at or after 13:00. The 13:00 bar traded on all 1,436 days, so the delay is always 0 minutes; a day is skipped if no bar starts within 5 minutes |
| Exit | Open of the first bar starting at or after 13:30 CT; the 13:30 bar also traded on every day |
| Contract | The contract held under the paper's volume-crossover rule; both fills are on that contract, so a roll gap never enters a trade |
| Size | 1 contract, USD 1,000 per 1.00 price move; the daily return is the P&L divided by the entry price × 1,000 |
| Costs | USD 2.50 commission per side and 0.5 tick (USD 5) spread and slippage per side: USD 15 per round trip. The data has no quotes, so the spread cost is a proxy: the CL front month usually quotes one tick wide, so half a tick per side is paying the spread with marketable orders. The sensitivity grid runs from 0 to 3 ticks per side and 0 to USD 5 commission |
| Frictionless return | The paper's convention: position × (P(13:30) / P(13:00) − 1) from the reference prices, no delay, no costs |
| Trade-at-settlement exit (variant) | CL settles on the volume-weighted average price of outright trades from 13:28:00 to 13:30:00 CT, and a TAS order (CLT) placed with the 13:00 entry clears at that settlement plus its differential, which is zero or a few ticks. The variant exits at a proxy for the settlement, the volume-weighted average of the 13:28 and 13:29 bars (each at its OHLC average), with no exit spread at zero differential or 1 tick of differential. The paper's roll rule never holds a contract on its expiration day, so the different final-settlement window does not apply |

**Metrics.** Means are in basis points of notional per trade date, with Newey-West standard errors (7 lags, the automatic rule ⌊4(n/100)^(2/9)⌋ used in #29; the plain standard error is 1.24 against 1.26 with the lags, and the lag-1 autocorrelation of daily returns is 0.05). Annualized return is the mean × 252 and annualized volatility the standard deviation × √252; the Sharpe ratio is their ratio, with the Lo (2002) standard error. The hit rate is the share of traded days with a positive return. Drawdowns are on the cumulative P&L per contract. Days without a position count as zero. The signal has no fitted parameter, so there is no in-sample/out-of-sample split: every day is out of sample.

## Results

### Summary table (`lh_strategy_summary.csv`)

Mean daily return in basis points with the HAC standard error; Sharpe ratio annualized; P&L in USD per contract over the sample.

| Signal | Accounting | Mean (SE) | 95% interval | Sharpe (SE) | Hit rate | Total P&L | Max drawdown |
|---|---|---|---|---|---|---|---|
| Sign of ROD | paper, frictionless | −1.18 (1.26) | [−3.7, 1.3] | −0.40 (0.42) | 0.480 |  |  |
| Sign of ROD | feasible, gross | −1.15 (1.26) | [−3.6, 1.3] | −0.39 (0.42) | 0.481 | −9,570 | 23,380 |
| Sign of ROD | feasible, net | −3.15 (1.26) | [−5.6, −0.7] | −1.07 (0.42) | 0.466 | −31,095 | 39,235 |
| Sign of ONFH | paper, frictionless | −3.15 (1.21) | [−5.5, −0.8] | −1.07 (0.42) | 0.469 |  |  |
| Sign of ONFH | feasible, net | −5.08 (1.21) | [−7.4, −2.7] | −1.73 (0.42) | 0.455 | −59,790 | 61,850 |
| ONFH and ROD agree (1,083 days) | paper, frictionless | −2.15 (1.11) | [−4.3, 0.0] | −0.83 (0.42) | 0.470 |  |  |
| ONFH and ROD agree | feasible, net | −3.61 (1.11) | [−5.8, −1.4] | −1.39 (0.42) | 0.455 | −40,005 | 42,650 |
| Always Long | paper, frictionless | −1.42 (1.17) | [−3.7, 0.9] | −0.48 (0.42) | 0.483 |  |  |
| Always Long | feasible, net | −3.33 (1.17) | [−5.6, −1.0] | −1.13 (0.42) | 0.467 | −41,470 | 52,160 |

The last half hour itself was slightly negative on average (Always Long, −1.4 basis points), and the sign-of-ROD position captured none of the variation: 770 long and 665 short days gave the same mean as being always long. In annualized terms the gross strategy returned −2.9% at 7.4% volatility; the paper's commodity portfolio returned +4.3% at 3.1% volatility over 1974–2020.

### Cost sensitivity (`lh_strategy_cost_sensitivity.csv`, figure `lh_strategy_cost_sensitivity.png`)

Each tick per side costs 2.7 basis points per day (USD 20 per round trip over an average notional of about USD 77,000). With USD 2.50 commission per side:

| Spread and slippage per side | Round trip | Mean net (SE) | Sharpe | Hit rate | Total net P&L |
|---|---|---|---|---|---|
| 0 ticks | USD 5 | −1.82 (1.26) | −0.62 | 0.481 | −16,745 |
| 0.5 tick (baseline) | USD 15 | −3.15 (1.26) | −1.07 | 0.466 | −31,095 |
| 1 tick | USD 25 | −4.48 (1.26) | −1.52 | 0.451 | −45,445 |
| 2 ticks | USD 45 | −7.15 (1.26) | −2.42 | 0.417 | −74,145 |

With no costs at all the mean is −1.15 basis points, so no cost level in the grid makes the strategy profitable.

### By year (`lh_strategy_by_year.csv`, figure `lh_strategy_by_year.png`)

Gross mean daily return in basis points with the HAC standard error and 95% interval.

| Year | Days | Gross mean (SE) | 95% interval | Hit rate | Gross P&L | Net P&L |
|---|---|---|---|---|---|---|
| 2021 | 250 | −0.68 (2.38) | [−5.3, 4.0] | 0.472 | −2,170 | −5,920 |
| 2022 | 249 | 2.40 (4.39) | [−6.2, 11.0] | 0.510 | 8,970 | 5,235 |
| 2023 | 248 | −0.94 (2.98) | [−6.8, 4.9] | 0.484 | −1,030 | −4,750 |
| 2024 | 249 | −0.13 (2.15) | [−4.3, 4.1] | 0.496 | 140 | −3,580 |
| 2025 | 248 | 0.79 (1.81) | [−2.8, 4.3] | 0.496 | 1,270 | −2,450 |
| 2026 (to Oct 7) | 192 | −10.48 (3.78) | [−17.9, −3.1] | 0.411 | −16,750 | −19,630 |

2022 is the only positive year (gross +USD 8,970), with a wide interval from the volatility after February 2022. The rolling 252-day mean (`lh_strategy_rolling.png`) stays within ±4 basis points of zero until 2026 and then falls to −10 basis points. 2026 includes the March 2026 Strait of Hormuz closure, during which the last half hour repeatedly reversed the day's move; the hit rate that year is 41%.

### Roll rules (`lh_strategy_roll_rules.csv`)

The 3-sessions-before-expiry rule (#17/#18) holds a different contract from the paper's volume-crossover rule on 127 of the 1,436 days, the one or two sessions before each of the 69 paper rolls. Its reference prices and fills are rebuilt from the held and next-listed contracts' minute bars.

| Roll rule | Gross mean (SE) | Net mean (SE) | Sharpe net | Gross P&L | Net P&L |
|---|---|---|---|---|---|
| Paper, volume crossover | −1.15 (1.26) | −3.15 (1.26) | −1.07 | −9,570 | −31,095 |
| 3 sessions before expiry | −1.20 (1.25) | −3.20 (1.25) | −1.09 | −10,470 | −31,995 |

### Timing and blackout robustness (`lh_strategy_robustness.csv`)

Gross and net mean daily return of the sign-of-ROD strategy in basis points, with the HAC standard error.

| Variant | Days skipped | Gross mean (SE) | Net mean (SE) | Gross P&L |
|---|---|---|---|---|
| Baseline: next bar open at 13:00 and 13:30 | 1 | −1.15 (1.26) | −3.15 (1.26) | −9,570 |
| Paper's reference prices (not executable) | 1 | −1.18 (1.26) | −3.18 (1.26) | −9,750 |
| Exit at the 13:30 reference price | 1 | −1.22 (1.26) | −3.22 (1.26) | −10,240 |
| Enter one minute later, at 13:01 | 1 | −0.34 (1.24) | −2.34 (1.24) | 100 |
| Exit by trade-at-settlement, zero differential (settlement proxy, no exit spread) | 1 | −0.92 (1.22) | −2.26 (1.23) | −6,874 |
| Exit by trade-at-settlement, 1-tick differential | 1 | −0.92 (1.22) | −3.59 (1.23) | −6,874 |
| Skip the 69 roll dates | 69 | −0.52 (1.18) | −2.42 (1.18) | −3,030 |
| Skip sessions within 3 of expiry | 127 | −1.12 (1.20) | −2.94 (1.20) | −10,550 |
| Skip the 12 Databento degraded dates | 12 | −1.18 (1.26) | −3.16 (1.26) | −9,710 |
| Skip sessions after a weekend or holiday | 304 | −0.15 (1.13) | −1.72 (1.13) | 1,890 |
| Skip 10 March 2026 | 1 | −0.83 (1.23) | −2.83 (1.23) | −5,900 |

The 13:01 entry and the weekend-skip variants are the least negative, but each sits within one standard error of the baseline and neither has a net mean above −1.7 basis points. The signal blackout on roll dates improves the gross mean by 0.6 basis points, which is 0.5 standard errors. The trade-at-settlement exit is the most realistic improvement: the settlement proxy is on average slightly better than the first trade after 13:30 (gross −0.92 against −1.15), and paying no spread on the exit saves USD 5 per round trip, so the net mean rises to −2.26 (1.23) at zero differential; at a 1-tick differential it is −3.59. The gross mean stays negative, so the exit convention does not change the conclusion. The settlement itself can be taken from Databento's statistics schema and the TAS differential from CLT trades in the same data; both are left for a follow-up.

### Adverse days (`lh_strategy_worst_days.csv`)

94 of the 1,435 traded days lost more than USD 500 gross per contract (103 net). The ten worst net days lost between USD 1,325 and USD 3,685; seven of them were short positions on days when a large negative rest-of-day return reversed in the last half hour, and five fell in March–June 2026.

| Date | Position | ROD | Entry → exit | Last-half-hour return | Net P&L |
|---|---|---|---|---|---|
| 2026-03-10 | short | −15.4% | 80.18 → 83.85 | +4.6% | −3,685 |
| 2022-03-08 | long | +5.3% | 125.67 → 123.63 | −1.6% | −2,055 |
| 2026-04-06 | long | +2.3% | 114.07 → 112.25 | −1.6% | −1,835 |
| 2022-07-05 | short | −9.8% | 97.81 → 99.54 | +1.8% | −1,745 |
| 2026-04-30 | short | −3.1% | 103.67 → 105.15 | +1.4% | −1,495 |

## Checks

- **No look-ahead.** The config loader rejects any signal price observed after 13:00 CT. Tests change every price from 13:00 CT onward and confirm that the positions do not move, and change every price before 13:00 CT and confirm that the fills do not move.
- **Fills.** Tests cover the first bar at or after the time, the 5-minute delay limit, the daylight-saving change, the Sunday-evening and holiday sessions that fold into the next trade date (their 13:00 bar is not the fill), and the roll date, on which both fills are on the newly held contract.
- **Accounting.** A hand-calculated long and short day (USD 250 gross, USD 15 of costs) matches the ledger; untraded days carry no P&L and no costs and record the reason.
- **Signal cross-check.** The script compares its ROD with `r_rod` from the #28 feature file on all 1,436 shared dates; the largest absolute difference is 0, recorded in the manifest.
- 53 tests pass (`python -m pytest`), 19 of them for this module (the settlement proxy and the exit-side slippage have their own tests).

## Limitations

- **No quotes.** The spread cost is a proxy, and the grid is the sensitivity analysis. The trade-at-settlement variant shows what removing the exit spread is worth; limit orders or other order types might do better or worse than the next-trade fill.
- **One contract, no margin.** Returns are on the full notional; a margin-based return would scale the mean and volatility together and leave the Sharpe ratio unchanged.
- **Thin expiring contract.** On the last session before a paper roll the held contract trades around 40k contracts rather than 218k. It still traded in the 13:00 and 13:30 minutes on every such day, and skipping those sessions does not change the result.
- **One market, one signal.** The paper's Sharpe ratios come from portfolios of many contracts; this is a single market whose historical regression was not significant either.

## Points for review

- The spread assumption (0.5 tick per side as the baseline) and the commission (USD 2.50 per side) are judgment calls; both are config settings. The trade-at-settlement exit (Josie's suggestion on the PR) removes the exit spread and is reported as a variant; whether it should become the baseline is open.
- The settlement proxy uses the 13:28 and 13:29 bars' OHLC averages weighted by volume. The exact settlement is available from Databento's statistics schema if the team wants to replace the proxy.
- Whether to show the paper's frictionless series or only the feasible gross series in the final report. They differ by 0.03 basis points, so one of them may be enough.
- Once #35 merges, the signal can be read from `cl_return_features.parquet` instead of recomputed; the cross-check in the manifest will confirm that the two agree.
