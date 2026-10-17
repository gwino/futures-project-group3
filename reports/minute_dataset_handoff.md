# Minute Dataset Handoff for Return Construction (#28)

This note is for whoever builds the return variables in #28. It explains which file to start from, how each paper variable maps to its columns, which contract is used on each date (the roll rule), and which sessions belong in the sample. The data work behind it is #25–#27. Full provenance and the quality checks are in `data/README.md`, section "1-minute research dataset, 2021–2026".

## What is ready

| Item | Value |
|---|---|
| Source | Databento `GLBX.MDP3`, `ohlcv-1m`, parent `CL.FUT`; two contiguous downloads, 2020-12-01 to 2026-10-08 (UTC, end exclusive) |
| Coverage | 1,469 trade dates, 2020-12-02 to 2026-10-07; full Globex session (17:00–16:00 CT), overnight included |
| Replication sample | **1,436 trade dates, 2021-01-04 to 2026-10-07** (`paper_sample` true, from 2021) |
| First out-of-sample date | With the paper's minimum 500-observation initial window, observation 500 is 2023-01-03, so OOS forecasts start on **2023-01-04** |
| Rolls | 70, under the paper's volume-crossover rule (below) |

The processed files are committed. Parquet files use Git LFS, so run `git lfs pull` after cloning. Nobody needs an API key or needs to rerun the pipeline to start #28.

## Start from `data/processed/cl_minute_reference_prices.parquet`

One row per trade date. Every price is the **raw price of the contract held on that trade date**.

| Column | Meaning |
|---|---|
| `trade_date`, `prev_trade_date` | CME trade date t and the previous trade date (Friday for a Monday; skips holidays) |
| `contract_id`, `instrument_id` | Contract held on t, e.g. `CLK2022` |
| `p_prev_1330` | Price at 13:30 CT on `prev_trade_date`: the paper's previous close P(c, t−1) |
| `p_0800`, `p_0830`, `p_1230`, `p_1300`, `p_1330` | Prices at those Chicago times on t |
| `stale_*` | Minutes between the bar used and the minute just before the reference time (0 means that minute traded; always 0 in this sample) |
| `session_volume` | Held contract's volume over the full 17:00–16:00 session |
| `roll_flag` | t is the first session in a new contract |
| `early_close`, `low_volume`, `folded_holiday`, `degraded_date` | Session flags (see "Sample selection") |
| `complete` | All six prices exist |
| `paper_sample` | `complete` and not an early close and volume ≥ 100: **use this filter** |

**Price at time T.** The close of the held contract's last 1-minute bar that starts before T. Databento stamps each bar at the start of its minute, so the price at 13:00 is the close of the 12:59 bar, which is the last trade before 13:00. If that minute had no trade, a bar up to 5 minutes older is used; beyond that the price is missing. Nothing is forward-filled further.

## Return variables

All times are America/Chicago (daylight saving is already handled). The paper uses simple returns.

| Variable | Interval (CT) | Simple return from the file |
|---|---|---|
| ON | previous 13:30 → 08:00 | `p_0800 / p_prev_1330 - 1` |
| FH | 08:00 → 08:30 | `p_0830 / p_0800 - 1` |
| ONFH | previous 13:30 → 08:30 | `p_0830 / p_prev_1330 - 1` |
| M | 08:30 → 12:30 | `p_1230 / p_0830 - 1` |
| SLH | 12:30 → 13:00 | `p_1300 / p_1230 - 1` |
| ROD | previous 13:30 → 13:00 | `p_1300 / p_prev_1330 - 1` |
| DAY | 08:00 → 13:00 | `p_1300 / p_0800 - 1` |
| LH (target) | 13:00 → 13:30 | `p_1330 / p_1300 - 1` |

- **Log versions.** Log returns add up exactly: log ON + log DAY = log ROD, and log ON + log FH + log M + log SLH = log ROD (checked on the data; the error is about 1e-16). Simple returns compound and do not add, so label log versions clearly.
- **No look-ahead.** Every predictor uses only prices up to `p_1300`, which contains no trade from 13:00 onward. `p_1330` belongs only to the target LH. A unit test (`test_1300_price_uses_no_trade_from_1300_onward`) checks this.

Loading example:

```python
import pandas as pd

ref = pd.read_parquet("data/processed/cl_minute_reference_prices.parquet")
ref = ref[ref["paper_sample"] & ref["trade_date"].ge("2021-01-01")].reset_index(drop=True)

ref["r_onfh"] = ref["p_0830"] / ref["p_prev_1330"] - 1
ref["r_rod"] = ref["p_1300"] / ref["p_prev_1330"] - 1
ref["r_lh"] = ref["p_1330"] / ref["p_1300"] - 1   # target
```

On this sample, ROD has a daily standard deviation of 2.48% and LH 0.47%.

## Rollover rule used

This follows the paper (Section 2): "We consider the most liquid contract … and roll it over when the daily tick volume of the next back-month contract exceeds the current contract."

1. **Daily volume.** Each contract's volume over the CME session (17:00 CT on the previous trade date to 16:00 CT), summed from the minute bars.
2. **Crossover.** Starting from the last roll, find the first session d in which the next listed contract's volume exceeds the held contract's volume.
3. **Switch.** The roll takes effect at the 17:00 CT open that starts the next trade date, d + 1 (`roll_date`). Session d's volume is known at its 16:00 close, so the rule has no look-ahead.
4. **Backstop.** The roll happens no later than 1 trade date before the held contract's last trade date. It was never needed: every roll came from a real crossover.

**Results.** The first crossover happened 2 sessions before expiry in 59 of 70 rolls and 3 sessions before in 11. The held contract is therefore the expiring one until 1–2 sessions before its last trade date.

**Comparison with the existing pipelines (#17/#18).** The earlier rule rolls 3 trade dates before expiry, so the paper rule rolls 1–2 sessions later on every roll. On the last session before a paper roll, the expiring contract trades a median of about 40k contracts (range 20k–145k), against a median of about 218k on all sessions. It still trades in 326–330 of the 330 window minutes, so its reference prices exist. Both calendars are in `data/processed/cl_minute_roll_comparison.csv`. To switch rules, set `roll.rule: days_before_expiry` and `days_before_expiry: 3` in `config/prepare_minute.yaml` and rerun the pipeline.

**Why rolls do not distort returns.** Every price for trade date t, including `p_prev_1330` on the previous day, comes from the contract held on t. On a roll date the overnight return therefore starts from the **new** contract's previous 13:30 price. For example, on the roll date 2022-03-21, `p_prev_1330` is CLK2022's 103.12, not CLJ2022's 104.73. No back-adjustment is needed, and none is applied.

## Sample selection

The paper drops sessions with non-positive prices, non-business days, early-close days, and days with total volume under 100 contracts. `paper_sample` applies the same rules:

- **Non-positive prices:** none in 2020–2026.
- **Non-business days:** weekend and holiday trading is folded into the next trade date (`folded_holiday`, 35 sample sessions), so the holiday itself is never a row.
- **Early closes:** 8 sessions whose market close is 30 or more minutes before 16:00 CT: Christmas Eve and the day after Thanksgiving, closing at 12:45 or 13:45 CT. These are excluded.
- **Missing previous close:** the session after a 12:45 CT early close has no `p_prev_1330`, so it is excluded too. The same applies to 2020-12-02, the first date in the data.
- **Low volume:** 0 sessions.
- **Degraded dates:** Databento flags data quality on these dates (12 sample sessions). They are kept; run a robustness check without them if needed.

Excluded trade dates from 2021 on: 2021-11-26, 2021-11-29, 2022-11-25, 2022-11-28, 2023-11-24, 2023-11-27, 2024-11-29, 2024-12-24, 2024-12-26, 2025-11-28, 2025-12-24, 2025-12-26.

Weekend and holiday gaps: of the 1,436 sample sessions, 255 have a 3-day gap since the previous trade date and 43 a 4-day gap. Their ON and ROD returns span the weekend, as in the paper.

## Decisions left to #28

- Simple versus log returns for the new ON + DAY decomposition (the paper's three core regressions use simple returns).
- Units for reported coefficients: match the paper's tables (Appendix Table B3 for CL).
- Whether to report a robustness check that drops degraded dates, roll dates (69 in the sample), or the expiry-adjacent sessions described under the rollover rule.

## For realized volatility (#32)

`data/processed/cl_minute_front_contracts.parquet` holds every 1-minute bar of the held contract (`held = True`) and the next listed contract on each trade date.

- **Use the contract held on t for the whole interval.** On a roll date, the overnight path from the previous 13:30 to 16:00 sits in the new contract's rows from the previous trade date, which have `held = False`. Select by `contract_id` and time, not by `held` alone.
- **Exclude reopening returns from the variance sum.** The 16:00–17:00 CT halt and weekends have no bars, so the first return after a halt is a reopening jump.
- **Coverage.** Every trade date has an overnight path. The longest run of market minutes without a held-contract trade is 10 minutes overnight and 1 minute in the 08:00–13:30 window (`data/processed/cl_minute_session_quality.csv`).
- **Outliers.** `outlier_flag` marks large 1-minute moves; they are real market moves and are kept.

## Rebuilding

```bash
export DATABENTO_API_KEY="YOUR_API_KEY"   # downloads only
python src/download_data.py --config config/intraday_2021_2024.yaml --confirm
python src/download_data.py --config config/intraday.yaml --confirm
python -m src.prepare_daily_data --config config/prepare_daily.yaml      # optional roll cross-check
python -m src.prepare_minute_data --config config/prepare_minute.yaml
python -m pytest
```
