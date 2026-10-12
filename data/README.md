# Data

This folder holds the raw Databento files and the processed datasets used by the analysis. Market data is licensed from Databento, so only files the team has agreed to share are committed. Everything else can be rebuilt with the steps below.

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
| `processed/cl_intraday_quality_summary.csv` | same | Data-quality checks and their results. |
| `processed/cl_intraday_prepare.manifest.json` | same | Settings plus SHA256 of the input and every output. |
| `../reports/figures/intraday_roll_examples.png` | same | Raw and back-adjusted prices around three rolls. |

Settings are in `config/prepare_intraday.yaml`. The contract pattern, holiday calendar, roll rule, and degraded dates are read from `config/prepare_daily.yaml`, so both horizons use the same trade dates and roll on the same sessions. Run `src/prepare_daily_data.py` first to enable the roll cross-check against `cl_daily_roll_calendar.csv`.

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

- **Quality.** No missing values, duplicates, OHLC inconsistencies, zero-volume bars, or non-positive prices. Every trade date in the window has outright data. 4,614 bars fall on Databento degraded dates and are flagged.
- **Window coverage.** 442 of 443 trade dates have all 22 window bars. The exception is 2025-12-24 (early close, 19 bars), so its cutoff bar is at 12:45 CT.
- **Liquidity.** The window holds 53.4% of the held contract's volume, so most trading happens in the hours where the intraday strategy can trade.
- **Outliers.** 125 held-contract bars are flagged, and all of them are matched by a same-direction move in the next listed month at the same time, so they are market moves rather than bad prints. The most common bar is 17:15 CT (30 of 125), whose close-to-close change spans the daily halt or the weekend; 70% of the flagged bars fall outside the trading window.
- **Common-window split.** A 70/30 split of the 443 common trade dates puts the cutoff at 2026-03-27 (last development date) / 2026-03-30 (first holdout date).
