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
