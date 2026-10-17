# Data

This folder holds the raw Databento files and the processed datasets used by the analysis. Market data is licensed from Databento, so only files the team has agreed to share are committed. The processed outputs in `processed/` are committed, with parquet files stored through Git LFS (run `git lfs pull` after cloning). Everything else can be rebuilt with the steps below. To build the return variables (#28), start from `reports/minute_dataset_handoff.md`.

## Provenance

| Item | Value |
|---|---|
| Vendor | [Databento](https://databento.com) Historical API |
| Dataset | `GLBX.MDP3` (CME Globex MDP 3.0) |
| Schema | `ohlcv-1d` (daily open, high, low, close, volume) |
| Symbols | Parent `CL.FUT` (`stype_in: parent`): all NYMEX WTI crude oil futures and their spreads |
| Cross-check symbols | `CL.v.0`, `CL.v.1` (volume roll), `CL.c.0`, `CL.c.1` (calendar roll), `stype_in: continuous` |
| Requested range | 2010-06-06 to 2026-10-08 (end exclusive); data runs to 2026-10-07 |
| Cost | Databento reported an estimated cost of $0.00 for each file |

## Files

| File | Created by | Description |
|---|---|---|
| `raw/crude_oil_all_daily.dbn` | `src/download_data.py --config config/crude_oil_daily_all.yaml` | Raw daily bars for every `CL.FUT` instrument (777,383 rows). Committed. |
| `raw/crude_oil_all_daily.dbn.manifest.json` | same | Query, download time, SHA256. Committed. |
| `raw/crude_oil_continuous_daily.dbn` | `src/download_data.py --config config/crude_oil_continuous_daily.yaml` | Databento continuous series; only its symbol mappings are used, to cross-check roll dates. |
| `processed/cl_daily_contracts.parquet` | `src/prepare_daily_data.py` | Validated session bars for each of the 248 outright contracts. |
| `processed/cl_daily_continuous.parquet` | same | One row per trade date for the contract held under the roll rule, with raw and back-adjusted prices. |
| `processed/cl_daily_roll_calendar.csv` | same | One row per roll (196 rolls). |
| `processed/cl_daily_quality_summary.csv` | same | Data-quality checks and their results. |
| `processed/cl_daily_contract_quality.csv` | same | Sessions, missing sessions, and the longest gap for each contract. |
| `processed/cl_daily_roll_crosscheck.csv` | same | Our roll dates compared with Databento `CL.v.0` and `CL.c.0`. |
| `processed/cl_daily_prepare.manifest.json` | same | Settings plus SHA256 of the input and every output. |

Re-downloading the same query returns identical records, but the file's SHA256 can differ because DBN headers vary between downloads. To check the data against the committed manifest, use the committed `.dbn` file.

## How to rebuild

```bash
conda activate futures-group3          # or any Python 3.10+ environment with requirements.txt
export DATABENTO_API_KEY="YOUR_API_KEY" # only needed for downloads
python src/download_data.py --config config/crude_oil_daily_all.yaml            # prints the cost estimate
python src/download_data.py --config config/crude_oil_daily_all.yaml --confirm  # downloads
python src/download_data.py --config config/crude_oil_continuous_daily.yaml --confirm  # optional cross-check
python src/prepare_daily_data.py --config config/prepare_daily.yaml
```

The preparation step needs no API key. If the continuous cross-check file is missing, it skips the cross-check.

## Processed schema

`cl_daily_continuous.parquet` (also see `cl_daily_contracts.parquet`, which has the same price columns for every contract):

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | datetime, America/Chicago | Trade date at 16:00 CT (Globex daily halt) |
| `trade_date` | date | CME trade date |
| `contract_id` | string | Full contract code, e.g. `CLZ2019` |
| `instrument_id` | int | Databento instrument ID (unique per contract life) |
| `symbol` | string | Raw CME symbol, e.g. `CLZ9` (one-digit years are reused every decade) |
| `open`, `high`, `low`, `close` | float, USD/bbl | Raw prices of the held contract; use for fills and P&L |
| `volume` | int, contracts | Volume of the held contract |
| `signal_open` … `signal_close` | float, USD/bbl | Additively back-adjusted prices; use only for signals such as moving averages |
| `back_adjustment` | float, USD/bbl | `signal_* − raw`: the sum of all later roll gaps |
| `roll_flag` | bool | True on the first session held in a new contract |
| `sessions_to_expiry` | int | Trade dates until the held contract's last trade date |
| `source_bars` | int | Number of UTC daily bars folded into this session (2 after a weekend or holiday) |
| `outlier_flag` | bool | Close-to-close change exceeds 12× the contract's rolling 60-session median absolute change |
| `degraded_date` | bool | Session includes a date Databento reports as degraded |

Units: prices in USD per barrel; one contract is 1,000 barrels, so 1.00 USD/bbl = 1,000 USD per contract. Tick size is 0.01 USD/bbl (10 USD per contract).

`cl_daily_roll_calendar.csv`: `roll_date` (first session in the new contract), `from_contract`, `to_contract`, instrument IDs, `from_last_trade_date`, `sessions_before_expiry`, `gap_date` (last session before the roll where both contracts traded), `from_close`, `to_close`, `gap` (`to_close − from_close`), `ratio`, `from_open_on_roll`, `to_open_on_roll`.

## Processing summary

The steps are described in full in `reports/methodology.md`.

1. Keep outright contracts only (`^CL[FGHJKMNQUVXZ][0-9]{1,2}$`). This drops 667,421 spread and strategy rows.
2. Assign each UTC daily bar to a CME trade date, and fold Sunday and holiday fragments into the next trade date.
3. Resolve each `instrument_id` to a full contract month and compute its last trade date using the CME rule. The computed date matches the last session in the data for all 169 contracts that expired in the sample.
4. Run validation checks: duplicates, timestamp order, OHLC consistency, volume, non-positive prices, outliers, missing sessions, and degraded dates.
5. Roll 3 trade dates before each contract's last trade date and build the back-adjusted series.

## Limitations

- **UTC daily bars are not CME sessions.** A bar's close is the price at 00:00 UTC (19:00 or 20:00 New York time), not the 14:30 ET settlement price. Each weekday bar also includes the first one to two hours of the next session, from the 18:00 ET reopen to midnight UTC. Folding fixes weekend and holiday fragments but cannot split weekday bars.
- **Missing dates.** 2014-06-13 and 2014-09-23 to 25 have no data for any contract. Databento lists 32 degraded dates in the sample (see `config/prepare_daily.yaml`); 27 held-contract sessions fall on them.
- **No settlement prices or open interest.** The `ohlcv-1d` schema has neither, so the roll rule uses expiry dates and is checked against volume.
- **Assumed holiday calendar.** The CME holiday calendar is built from rules, including Juneteenth from 2022. The fact that computed expiries match the data for all 169 contracts supports it, but it is not an official CME file.
- **Negative prices.** CLK2020 traded below zero on 2020-04-20 (close −2.67, low −40.32). The held series rolled out of that contract on 2020-04-16, so it is unaffected. The per-contract file keeps those sessions.
- **Daily data only.** Intraday analysis (15-minute bars, volume by time of day) needs `ohlcv-1m` data, which has not been downloaded yet.

## Intraday data

### Provenance

| Item | Value |
|---|---|
| Dataset | `GLBX.MDP3`, schema `ohlcv-1m`, parent `CL.FUT` (`stype_in: parent`) |
| Requested range | 2025-01-01 to 2026-10-08 (end exclusive); complete trade dates run from 2025-01-02 to 2026-10-07 |
| Raw file | `raw/crude_oil_conflict_1m.dbn` (Git LFS), created by `src/download_data.py --config config/intraday.yaml` |

### Files

| File | Created by | Description |
|---|---|---|
| `processed/cl_intraday_15m_contracts.parquet` | `python -m src.prepare_intraday_data` | Validated 15-minute bars for each of the 71 outright contracts (456,260 bars). |
| `processed/cl_intraday_15m_continuous.parquet` | same | The held contract's 15-minute bars (41,740 bars), with raw and back-adjusted prices and the trading-window flags. |
| `processed/cl_session_daily_contracts.parquet` | same | CME session bars (17:00 to 16:00 CT) for each contract, built from the 1-minute data. |
| `processed/cl_session_daily_continuous.parquet` | same | One row per trade date for the held contract (443 trade dates); the daily horizon for the common window. |
| `processed/cl_intraday_roll_calendar.csv` | same | One row per roll (21 rolls), with the session-close gap (`gap`) and the 15-minute gap (`intraday_gap`). |
| `processed/cl_intraday_window_coverage.csv` | same | Number of held-contract bars inside the trading window on each trade date. |
| `processed/cl_intraday_session_coverage.csv` | same | One row per contract and trade date: expected, observed, and missing bars, the longest run of missing bars, the same counts for the trading window, and whether the contract is held. |
| `processed/cl_intraday_contract_coverage.csv` | same | The session coverage totalled per contract, overall and for the sessions in which the contract is held. |
| `processed/cl_intraday_quality_summary.csv` | same | Data-quality checks and their results. |
| `processed/cl_intraday_prepare.manifest.json` | same | Settings plus SHA256 of the input and every output. |
| `../reports/figures/intraday_roll_examples.png` | same | Raw and back-adjusted prices around three rolls. |

Settings are in `config/prepare_intraday.yaml`. The contract pattern, holiday calendar, roll rule, and degraded dates are read from `config/prepare_daily.yaml`, so both horizons use the same trade dates and roll on the same sessions.

Run the daily step first: it writes `cl_daily_roll_calendar.csv`, which the intraday step uses to cross-check its roll dates (the check is reported as skipped if the file is missing). Neither step needs an API key.

```bash
python -m src.prepare_daily_data --config config/prepare_daily.yaml
python -m src.prepare_intraday_data --config config/prepare_intraday.yaml
```

### Session and bar rules (#4)

1. Keep outright contracts only (same pattern as the daily pipeline). Symbols are mapped from the file's instrument mappings, which loads the 15.2 million rows in seconds.
2. Assign each minute to the CME trade date whose session it belongs to: a session runs from 17:00 CT on the prior day to 16:00 CT. Holiday sessions fold into the next trade date. Minutes stamped in the 16:00 to 17:00 CT halt (2 one-lot trades) and the partial evening session of 2026-10-08 are dropped.
3. Aggregate to 15-minute bars. Databento's 1-minute `ts_event` marks the start of each minute, so bins are left-closed and labeled by their end time (`closed="left", label="right"`): the bar stamped 08:15 covers 08:00:00 to 08:14:59. Bins are cut on the UTC clock, so the repeated hour at the end of daylight saving time is not merged. Empty bars are dropped, not forward-filled.
4. Mark the trading window: `in_trade_window` is true for bars from 08:00 to 13:30 CT on the trade date itself (22 bars), and `is_cutoff_bar` marks the last of them, where intraday positions are flattened. Evening-session bars and folded holiday bars are outside the window.
5. Roll on the same trade dates as the daily pipeline (3 trade dates before expiry; all 21 rolls match `cl_daily_roll_calendar.csv`). The held contract changes at the 17:00 CT open of the roll date. Signal prices are additively back-adjusted using the gap at the last 15-minute bar before the roll session where both contracts traded; it equals the session-close gap for every roll.

### Intraday schema

`cl_intraday_15m_continuous.parquet` uses the same columns as `cl_daily_continuous.parquet`, with these differences:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | datetime, America/Chicago | End of the 15-minute bar |
| `trade_date` | date | CME trade date of the session the bar belongs to |
| `in_trade_window` | bool | Bar lies inside 08:00 to 13:30 CT on its trade date |
| `is_cutoff_bar` | bool | Last window bar of the trade date (13:30 CT, or earlier on early-close days) |
| `minutes` | int | Number of 1-minute bars in the 15-minute bar |
| `outlier_flag` | bool | Close-to-close change exceeds 15× the contract's rolling median absolute change over the previous 920 bars |

`cl_session_daily_continuous.parquet` has the same columns as `cl_daily_continuous.parquet`, except that `source_minutes` replaces `source_bars` and there are no outlier or degraded flags. Its `timestamp` (16:00 CT) is the true end of the session, and its `close` is the last trade before 16:00 CT.

### Intraday findings for strategy design

The full exploration, including liquidity and volatility by time of day, autocorrelation, and trading costs, is in `analysis/02.5_intraday_data_exploration.ipynb`.

- **Quality.** No missing values, duplicates, OHLC inconsistencies, zero-volume bars, or non-positive prices. Every trade date in the window has outright data. 4,614 bars fall on Databento degraded dates and are flagged.
- **Window coverage.** 442 of 443 trade dates have all 22 window bars. The exception is 2025-12-24 (early close, 19 bars), so its cutoff bar is at 12:45 CT.
- **Missing bars per contract and session.** A bar is expected whenever any outright contract traded in it on that trade date, so early closes and holiday sessions are handled without a separate calendar (a normal session has 92 bars). The held contract has no missing bars in any of its 443 sessions. Deferred months trade sparsely: across all 71 contracts, 456,260 of 1,700,563 expected bars are observed. See `cl_intraday_session_coverage.csv` and `cl_intraday_contract_coverage.csv`.
- **Liquidity.** The window holds 53.4% of the held contract's volume, so most trading happens in the hours where the intraday strategy can trade.
- **Outliers.** 125 held-contract bars are flagged, and all of them are matched by a same-direction move in the next listed month at the same time, so they are market moves rather than bad prints. The most common bar is 17:15 CT (30 of 125), whose close-to-close change spans the daily halt or the weekend; 70% of the flagged bars fall outside the trading window.
- **Common-window split.** A 70/30 split of the 443 common trade dates puts the cutoff at 2026-03-27 (last development date) / 2026-03-30 (first holdout date).

## 1-minute research dataset, 2021–2026 (#25–#27)

This is the dataset for the Baltussen, Da, Lammers & Martens (2021) replication. It joins two contiguous `ohlcv-1m` downloads, keeps the full electronic session, and records, for every trade date, the held contract's prices at the paper's reference times. The 15-minute outputs above are unchanged.

### Inventory and provenance (#25, #26)

| File | Query (`GLBX.MDP3`, `ohlcv-1m`, parent `CL.FUT`) | Rows | Created by |
|---|---|---|---|
| `raw/crude_oil_2021_2024_1m.dbn` | 2020-12-01 to 2025-01-01 (end exclusive) | 26,343,989 (7,501,614 outright) | `src/download_data.py --config config/intraday_2021_2024.yaml --confirm` |
| `raw/crude_oil_conflict_1m.dbn` | 2025-01-01 to 2026-10-08 (end exclusive) | 15,196,660 (3,641,915 outright) | `src/download_data.py --config config/intraday.yaml --confirm` |

- Each raw file has a `.manifest.json` next to it with the query, download time, and SHA256. Databento reported an estimated cost of $0.00. Raw `.dbn` files are licensed market data and are not committed.
- The 2021–2024 file starts in December 2020 so that the first 2021 session has a previous 13:30 close and a roll before it. The two query ranges meet exactly at 2025-01-01 00:00 UTC; the pipeline checks this and found no duplicate minutes.
- **Frequency and session.** Both files are true 1-minute bars covering the full Globex session (17:00 to 16:00 CT), including overnight. There are no bars in the 16:00–17:00 CT halt, apart from 3 one-lot prints, which are dropped.
- **Timestamps.** `ts_event` is UTC and marks the **start** of the minute. A bar stamped 13:29 CT holds trades from 13:29:00 to 13:29:59, so its close is the last trade before 13:30.
- **Coverage.** 1,469 trade dates from 2020-12-02 to 2026-10-07, with no trade date missing; 1,448 of them are from 2021-01-04 on. 121 outright contracts; all 51 that expired in the sample last traded on their computed last trade date. The partial sessions at the start and end of the query range (2020-12-01 and 2026-10-08) are dropped.
- **Quality.** No missing values, duplicates, OHLC inconsistencies, zero or negative volume, or non-positive prices. 13 held sessions fall on Databento degraded dates; they are flagged, not removed.
- **Monthly inventory.** `reports/tables/cl_intraday_coverage.csv` has one row per month. Columns: source file, held contracts, rolls, sessions with complete reference prices, paper-sample sessions, early closes, overnight and window coverage, longest gaps, and volume. Every one of the 71 months has an overnight and a daytime minute path on every trade date, so realized overnight volatility can be computed for the whole sample.

### Sessions, reference prices and missing-point rules (#27)

Settings are in `config/prepare_minute.yaml`. Contract pattern, holiday calendar, and degraded dates come from `config/prepare_daily.yaml`.

```bash
python -m src.prepare_daily_data --config config/prepare_daily.yaml    # optional: roll cross-check
python -m src.prepare_minute_data --config config/prepare_minute.yaml
python -m pytest                                                        # 16 tests
```

1. **Trade dates.** Timestamps are converted from UTC to America/Chicago, so daylight saving is handled by the time zone. Each minute is assigned to the CME trade date whose session it belongs to (17:00 CT on the previous trade date to 16:00 CT). Sunday-evening and holiday sessions fold into the next trade date, using the same rule as the 15-minute pipeline.
2. **Reference times** follow the paper's CL window, 08:00–13:30 CT:
   - previous trade date: 13:30 (`p_prev_1330`)
   - trade date: 08:00, 08:30, 12:30, 13:00, 13:30 (`p_0800` … `p_1330`)

   13:30 is the reference close, not the 16:00 CT end of the electronic session.
3. **Price at time T.** The close of the held contract's last 1-minute bar that starts before T. Only trades before T are used, so `p_1300` contains nothing from 13:00–13:30 CT. If the minute just before T had no trade, a bar up to 5 minutes older is used, and `stale_*` records how many minutes older it is. Beyond that the price is missing. Prices are never forward-filled further. In the sample no reference price needed a stale bar.
4. **Missing points.** 13 trade dates are incomplete (`complete = False`), all because of holidays:
   - 6 early closes at 12:45 CT have no 13:00 or 13:30 price.
   - The 6 trade dates after them, plus 2020-12-02 (the first date in the sample), have no previous 13:30 price.
5. **Paper sample filters** (Section 2 of the paper). Non-positive prices are removed (there were none). Early-close days (8, detected when the market stops 30+ minutes before 16:00 CT) and sessions under 100 contracts (none) are flagged. `paper_sample` is true for complete, full-length sessions with enough volume: 1,454 of 1,469 trade dates, or 1,436 from 2021-01-04 on.

### Rolls (#27)

- **Paper rule (used here).** Hold the most liquid contract. Roll at the 17:00 CT open after the first session in which the next contract's session volume exceeds the held contract's. That session's volume is known at its 16:00 close, so there is no look-ahead. A backstop of 1 trade date before expiry was never needed.
- **Result.** The first crossover came 2 sessions before expiry in 59 of 70 rolls and 3 sessions before in 11. The paper rule therefore rolls 1–2 sessions later than the existing 3-sessions-before-expiry rule (#17/#18).
- **Comparison.** `cl_minute_roll_comparison.csv` lists both calendars side by side. Under the existing rule, all 70 rolls match `cl_daily_roll_calendar.csv`.
- **On the last session before a paper roll,** the expiring contract trades a median of about 40k contracts (range 20k–145k), against a median of about 218k on all sessions. It still trades in 326–330 of the 330 window minutes, so its reference prices exist.
- **Roll-safe returns.** Every reference price for trade date t, including `p_prev_1330` on the previous trade date, comes from the contract held on t. A roll gap therefore never enters an overnight, daytime, or closing return. Raw prices are kept; there is no back-adjustment.
- **Realized volatility.** Use the same contract over the same interval. On a roll date the overnight path starts on the previous trade date in the new contract, whose minutes are stored with `held = False`.

### Files

| File | Description |
|---|---|
| `processed/cl_minute_front_contracts.parquet` | 3.9 million validated 1-minute bars of the held and the next listed contract on every trade date. Columns: `timestamp` (CT, minute start), `ts_utc`, `trade_date`, `contract_id`, `instrument_id`, `symbol`, OHLC, `volume`, `held`, `folded`, `ohlc_ok`, `outlier_flag`, `outlier_corroborated`, `degraded_date`, `source`. |
| `processed/cl_minute_reference_prices.parquet` | One row per trade date: held contract, `p_*` and `stale_*` for each reference time, `session_volume`, `roll_flag`, `early_close`, `low_volume`, `folded_holiday`, `degraded_date`, `complete`, `paper_sample`. Input to #28. |
| `processed/cl_minute_session_quality.csv` | Per trade date: minutes traded by the held contract overnight (previous 13:30 to 08:00) and in the window (08:00 to 13:30), compared with the minutes in which any outright traded. Also missing minutes, longest gap, early-close time, volume, and flags. Gaps are counted only in minutes when the market traded, so scheduled halts, weekends, and holidays are not gaps. |
| `processed/cl_minute_roll_calendar.csv`, `processed/cl_minute_roll_comparison.csv` | Paper-rule rolls, and the comparison with the existing pre-expiry rule. |
| `processed/cl_minute_quality_summary.csv`, `processed/cl_minute_prepare.manifest.json` | All checks, and settings with input and output SHA256 hashes. |
| `../reports/tables/cl_intraday_coverage.csv` | Monthly coverage inventory (#25). |

### Checks and limitations

- **Cross-check.** The 08:00, 13:00 and 13:30 prices match the closes of the existing 15-minute bars exactly, for the same contract, on every overlapping 2025–2026 session.
- **Outliers.** A minute is flagged when its change exceeds 15 times the rolling median 1-minute change. 3,645 held-contract minutes are flagged, and 3,616 of them are matched by the next listed month in the same minute. Most fall at the 17:00 CT reopen, or are real shocks (June 2025, March and April 2026). Flags are reported, never removed.
- **Fallback.** No fallback was needed, because the full overnight session was available. With endpoint prices alone, overnight returns could still be computed, but overnight realized volatility could not.
