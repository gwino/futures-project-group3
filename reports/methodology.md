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
