# Methodology Note: Data, Sessions, and Rolls

This note is a draft of the methodology section for the final report. It covers the daily horizon. The intraday horizon will be added once 1-minute data is available. Settings marked *proposed* await confirmation in issue #4.

## Data

We use daily OHLCV bars for NYMEX WTI crude oil futures (CL) from Databento's `GLBX.MDP3` dataset. The sample runs from 7 June 2010 to 7 October 2026. The query uses the parent symbol `CL.FUT`, which returns every listed outright contract and every spread. We keep the 248 outright contracts and identify each one by its Databento `instrument_id`. CME symbols use a one-digit year that repeats every decade (for example, `CLZ9` was the December 2019 contract and is now the December 2029 contract). We therefore map each instrument to a full contract month: the earliest year with the right final digit whose expiry falls on or after the instrument's last observed session.

## Trading sessions and timestamps

Databento's daily bars cover UTC calendar days, while CME trading sessions run from 17:00 to 16:00 Central Time (CT). We assign each bar to the CME trade date with the same calendar date. Bars dated on a weekend or a CME holiday are folded into the next trade date: their open becomes the session open, their volume is added, and the session high and low cover both bars. The weekend bars come from the Sunday-evening reopen, and the holidays include early-halt sessions whose trades settle on the next day. We build the holiday calendar from CME rules: New Year's Day, Martin Luther King Jr. Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day, Thanksgiving, and Christmas. Each processed bar is timestamped at 16:00 CT on its trade date.

**Limitation.** A weekday UTC bar ends at midnight UTC, so its close is not the settlement price, and it includes the first one to two hours of the next session. As a result, the gap between one bar's close and the next bar's open is almost zero (median 0.013%). In the daily backtest, executing at the next open is therefore close to executing at the signal bar's close.

## Validation

We check that timestamps are in order within each contract, that there are no duplicate records, and that OHLC values are consistent (low ≤ open, close ≤ high). We also check for negative volume and non-positive prices, and flag outliers: close-to-close changes larger than 12 times the contract's rolling 60-session median absolute change. Finally, we count missing sessions against the trade calendar and flag dates Databento reports as degraded. Flagged records are kept and reported, never silently removed. In the held series, four trade dates in 2014 are missing because of a Databento outage. All four outliers are market-wide moves, confirmed by the next contract moving the same way on the same day.

## Contract expiry and roll rule

The last trade date of each contract follows the CME rule: three business days before the 25th calendar day of the month before delivery, or four business days before it if the 25th is not a business day. For all 169 contracts that expired within the sample, the computed date matches the last session in the data.

*Proposed rule:* we hold the nearest contract until three trade dates before its last trade date. On that session we switch to the next listed month. The rule depends only on the calendar, so it is known in advance and introduces no look-ahead. We chose three sessions from the data: the expiring contract still carries a median 61% of front-two volume three sessions before expiry, but only 26% two sessions before. The alternative rule (`volume_crossover`, which rolls the session after the next contract's volume first exceeds the held contract's) selects the same date for 195 of 196 rolls. As cross-checks, Databento's calendar series `CL.c.0` switches on the session after our computed expiry for 194 of 196 contracts. The two exceptions are Monday expiries, where `CL.c.0` switches at the Sunday-evening open. Databento's volume series `CL.v.0` rolls a median of three sessions after our rule.

## Signal prices and tradable prices

On each roll we measure the gap between the new contract's close and the old contract's close on the last session before the roll where both traded. Signal prices (`signal_*`) are additively back-adjusted: every earlier price is shifted by the sum of all later gaps, so the adjusted series has no jumps at rolls and its latest prices equal the raw prices. Additive adjustment preserves price differences in USD per barrel, which is what moving-average crossovers use and what one contract earns. Ratio adjustment would not work with the negative prices of April 2020. Because the cumulative adjustment reaches −50 USD/bbl, adjusted prices can be negative and must not be used for percentage returns.

Tradable prices are the raw prices of the contract actually held. P&L and returns are measured within one contract. On a roll date, the old position is closed and the new one opened at that session's prices: the roll calendar records both contracts' opens on the roll date.

## Intraday horizon

This section covers the intraday data used for the 15-minute strategy and for the common-window daily comparison. The full rules are in `data/README.md` (*Intraday data*), and the exploration is in `analysis/02.5_intraday_data_exploration.ipynb`.

### Data and sessions

We use 1-minute OHLCV bars for the same outright CL contracts from Databento's `GLBX.MDP3` dataset (`ohlcv-1m`, parent `CL.FUT`), covering 2 January 2025 to 7 October 2026 (443 trade dates). Each minute is assigned to the CME trade date whose session it belongs to: a session runs from 17:00 CT on the prior day to 16:00 CT, and holiday sessions fold into the next trade date, as in the daily data. Two one-lot trades stamped during the 16:00 to 17:00 CT halt, and the partial evening session of 8 October 2026, are dropped.

### 15-minute bars and the trading window

Databento stamps each 1-minute bar at the start of its minute, so 15-minute bars are left-closed and labeled by their end time: the bar stamped 08:15 covers 08:00:00 to 08:14:59. Bars are cut on the UTC clock, so the repeated hour at the end of daylight saving time is never merged, and empty bars are dropped rather than forward-filled. Signals use the full session, but positions may be held only from 08:00 to 13:30 CT on the trade date (22 bars), and all positions are closed at the close of the last bar in that window (13:30 CT, or earlier on early-close days such as 24 December 2025).

### Rolls and prices

The intraday series rolls on the same trade dates as the daily series (3 trade dates before the last trade date; all 21 rolls in the sample match). The held contract changes at the 17:00 CT open of the roll date, so both horizons hold the same contract on every trade date. Signal prices are additively back-adjusted using the price gap at the last 15-minute bar before the roll session where both contracts traded; for every roll this equals the gap between session closes. Because intraday positions are flat at every session end, the intraday strategy never trades a roll.

### Common-window daily bars

For the daily-versus-intraday comparison, daily bars are rebuilt from the same 1-minute data on CME sessions, so their close is the last trade before 16:00 CT. These session bars hold the same contract as the UTC-day bars on every date, but their daily changes have a correlation of only 0.88 with the UTC-day series, because a UTC-day bar closes at midnight UTC and includes the start of the next session. Using session bars for both horizons removes that mismatch from the comparison.

### Limitations

- **Short sample.** The common window covers about 21 months, so a 70/30 split leaves a holdout of about six months (from 30 March 2026). Results over this window are sensitive to the 2026 volatility regime.
- **No settlement prices.** The intraday close at the cutoff bar is a traded price, not the official settlement price, although both fall in the 13:28 to 13:30 CT settlement window.
- **Costs dominate the intraday horizon.** With the cost settings in #4, the intraday parameter grid implies roughly 10% to 13% of capital per year in trading costs, compared with about 0.4% for the daily grid, so the intraday results depend heavily on the cost assumptions.
