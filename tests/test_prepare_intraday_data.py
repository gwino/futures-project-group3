"""Checks for the session, bar-labeling, and trading-window rules in #4.

Run from the repository root: python -m pytest
"""

import pandas as pd
import pytest

from src.prepare_daily_data import cme_holidays, trade_calendar
from src.prepare_intraday_data import aggregate_bars, assign_trade_dates, flag_trade_window, session_coverage

TZ = "America/Chicago"


@pytest.fixture(scope="module")
def calendar() -> pd.DatetimeIndex:
	holidays = cme_holidays(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31"), "2022-06-19")
	return trade_calendar(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31"), holidays)


def minutes_at(*local_times: str, prices=None) -> pd.DataFrame:
	"""One-minute bars for one instrument, stamped at the start of each minute (as Databento does)."""
	ts = pd.DatetimeIndex([pd.Timestamp(t, tz=TZ) for t in local_times]).tz_convert("UTC")
	prices = prices if prices is not None else [70.0 + i / 100 for i in range(len(ts))]
	return pd.DataFrame(
		{
			"ts_utc": ts,
			"instrument_id": 1,
			"symbol": "CLZ5",
			"open": prices,
			"high": [p + 0.05 for p in prices],
			"low": [p - 0.05 for p in prices],
			"close": prices,
			"volume": 1,
		}
	)


def test_session_boundary_at_1700(calendar):
	frame = assign_trade_dates(minutes_at("2025-11-04 16:59", "2025-11-04 17:00"), calendar, TZ, "17:00", "16:00")
	assert frame["trade_date"].dt.date.astype(str).tolist() == ["2025-11-04", "2025-11-05"]


def test_daily_halt_is_after_close(calendar):
	frame = assign_trade_dates(minutes_at("2025-11-04 15:59", "2025-11-04 16:00", "2025-11-04 16:59"), calendar, TZ, "17:00", "16:00")
	assert frame["after_close"].tolist() == [False, True, True]


def test_sunday_open_and_holiday_fold_into_next_trade_date(calendar):
	# Sunday 2025-11-02 opens Monday's session; Sunday 2025-01-19 and MLK Day 2025-01-20 fold into Tuesday.
	frame = assign_trade_dates(
		minutes_at("2025-11-02 18:00", "2025-01-19 18:00", "2025-01-20 09:00"), calendar, TZ, "17:00", "16:00"
	)
	assert frame["trade_date"].dt.date.astype(str).tolist() == ["2025-11-03", "2025-01-21", "2025-01-21"]
	assert frame["folded"].tolist() == [False, True, True]


def test_bars_are_left_closed_and_labeled_by_end_time(calendar):
	times = [f"2025-11-04 08:{minute:02d}" for minute in range(16)]
	frame = assign_trade_dates(minutes_at(*times), calendar, TZ, "17:00", "16:00")
	bars = aggregate_bars(frame, "15min")
	assert bars["bar_end"].dt.tz_convert(TZ).dt.strftime("%H:%M").tolist() == ["08:15", "08:30"]
	first = bars.iloc[0]
	assert first["minutes"] == 15
	assert first["open"] == pytest.approx(70.00)
	assert first["close"] == pytest.approx(70.14)
	assert first["high"] == pytest.approx(70.19)
	assert first["volume"] == 15
	assert bars.iloc[1]["open"] == pytest.approx(70.15)


def test_repeated_hour_at_end_of_daylight_saving_is_not_merged(calendar):
	# 01:05 occurs twice on 2025-11-02 (CDT, then CST); the two minutes must land in different bars.
	ts = pd.DatetimeIndex(
		[pd.Timestamp("2025-11-02 06:05", tz="UTC"), pd.Timestamp("2025-11-02 07:05", tz="UTC")]
	)
	frame = minutes_at("2025-11-03 08:00", "2025-11-03 08:01").assign(ts_utc=ts)
	frame = assign_trade_dates(frame, calendar, TZ, "17:00", "16:00")
	bars = aggregate_bars(frame, "15min")
	assert len(bars) == 2
	assert bars["trade_date"].dt.date.astype(str).unique().tolist() == ["2025-11-03"]


def test_trade_window_covers_0800_to_1330_on_the_trade_date():
	ends = ["2025-11-04 08:00", "2025-11-04 08:15", "2025-11-04 13:30", "2025-11-04 13:45", "2025-11-03 08:15"]
	bars = pd.DataFrame(
		{
			"bar_end": pd.DatetimeIndex([pd.Timestamp(t, tz=TZ) for t in ends]).tz_convert("UTC"),
			"trade_date": pd.Timestamp("2025-11-04"),
		}
	)
	assert flag_trade_window(bars, TZ, "08:00", "13:30", "15min").tolist() == [False, True, True, False, False]


def test_session_coverage_counts_missing_bars_and_longest_gap():
	# The market trades in six bars on day 1 and two on day 2; bars 3 and 4 are inside the window.
	day1, day2 = pd.Timestamp("2025-11-04"), pd.Timestamp("2025-11-05")
	ends = pd.date_range("2025-11-04 14:00", periods=6, freq="15min", tz="UTC")
	market = [(1, day1, end, 2 <= i <= 3) for i, end in enumerate(ends)]
	sparse = [(2, day1, ends[0], False), (2, day1, ends[3], True), (2, day1, ends[5], False)]
	next_day = pd.date_range("2025-11-05 14:00", periods=2, freq="15min", tz="UTC")
	market += [(1, day2, end, False) for end in next_day]
	bars = pd.DataFrame(market + sparse, columns=["instrument_id", "trade_date", "bar_end", "in_trade_window"])
	contracts = pd.DataFrame(
		{"instrument_id": [1, 2], "contract_id": ["CLZ2025", "CLF2026"], "first_date": [day1, day1], "last_date": [day2, day2]}
	)
	coverage = session_coverage(bars, contracts).set_index(["contract_id", "trade_date"])

	full = coverage.loc[("CLZ2025", day1)]
	assert (full["expected_bars"], full["observed_bars"], full["missing_bars"], full["longest_gap_bars"]) == (6, 6, 0, 0)
	sparse_day = coverage.loc[("CLF2026", day1)]
	assert (sparse_day["observed_bars"], sparse_day["missing_bars"], sparse_day["longest_gap_bars"]) == (3, 3, 2)
	assert (sparse_day["expected_window_bars"], sparse_day["missing_window_bars"]) == (2, 1)
	# A trade date inside the contract's life with no bars at all is fully missing.
	empty_day = coverage.loc[("CLF2026", day2)]
	assert (empty_day["expected_bars"], empty_day["observed_bars"], empty_day["longest_gap_bars"]) == (2, 0, 2)
