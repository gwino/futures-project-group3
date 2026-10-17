"""Checks for the reference-price, roll, and session rules of the minute pipeline (#27).

Run from the repository root: python -m pytest
"""

import pandas as pd
import pytest

from src.prepare_daily_data import cme_holidays, trade_calendar
from src.prepare_minute_data import at_clock, complete_sessions, held_contracts, period_coverage, reference_prices

TZ = "America/Chicago"
POINTS = ["08:00", "08:30", "12:30", "13:00", "13:30"]


@pytest.fixture(scope="module")
def calendar() -> pd.DatetimeIndex:
	holidays = cme_holidays(pd.Timestamp("2019-01-01"), pd.Timestamp("2026-12-31"), "2022-06-19")
	return trade_calendar(pd.Timestamp("2019-01-01"), pd.Timestamp("2026-12-31"), holidays)


def minutes(*rows: tuple[int, str, float]) -> pd.DataFrame:
	"""1-minute bars from (instrument_id, Chicago minute start, close) tuples."""
	return pd.DataFrame(
		{
			"ts_utc": [pd.Timestamp(local, tz=TZ).tz_convert("UTC") for _, local, _ in rows],
			"instrument_id": [instrument for instrument, _, _ in rows],
			"close": [close for _, _, close in rows],
		}
	)


def held_on(*pairs: tuple[str, int]) -> pd.Series:
	return pd.Series([instrument for _, instrument in pairs], index=pd.DatetimeIndex([date for date, _ in pairs]), name="instrument_id")


def full_day(instrument: int, date: str, close: float) -> list[tuple[int, str, float]]:
	"""One trade in the minute before each reference time."""
	return [(instrument, f"{date} {clock}", close) for clock in ("07:59", "08:29", "12:29", "12:59", "13:29")]


def test_reference_times_follow_daylight_saving():
	# 08:00 CT is 14:00 UTC under CST and 13:00 UTC under CDT (clocks moved on 2025-03-09).
	instants = at_clock(pd.DatetimeIndex(["2025-03-07", "2025-03-10"]), "08:00", TZ)
	assert instants.dt.strftime("%H:%M").tolist() == ["14:00", "13:00"]


def test_reference_prices_across_the_daylight_saving_change(calendar):
	bars = minutes(*full_day(1, "2025-03-07", 66.0), *full_day(1, "2025-03-10", 67.0))
	refs = reference_prices(bars, held_on(("2025-03-07", 1), ("2025-03-10", 1)), calendar, TZ, "13:30", POINTS, 5).set_index("trade_date")
	monday = refs.loc["2025-03-10"]
	assert (monday["p_prev_1330"], monday["p_0800"], monday["p_1330"]) == (66.0, 67.0, 67.0)
	assert monday["stale_0800"] == 0 and monday["complete"]


def test_monday_uses_friday_close_and_sunday_reopen_is_overnight(calendar):
	bars = minutes(
		(1, "2025-11-07 13:29", 60.0), (1, "2025-11-07 13:30", 60.1), (1, "2025-11-07 15:59", 60.2),
		(1, "2025-11-09 17:00", 61.0), (1, "2025-11-10 07:59", 62.0),
	)
	refs = reference_prices(bars, held_on(("2025-11-10", 1)), calendar, TZ, "13:30", POINTS, 5).iloc[0]
	assert refs["prev_trade_date"] == pd.Timestamp("2025-11-07")
	assert (refs["p_prev_1330"], refs["p_0800"]) == (60.0, 62.0)

	periods = pd.DataFrame(
		{
			"trade_date": [pd.Timestamp("2025-11-10")], "instrument_id": [1],
			"start": at_clock(["2025-11-07"], "13:30", TZ), "end": at_clock(["2025-11-10"], "08:00", TZ),
		}
	)
	market = pd.DatetimeIndex(bars["ts_utc"]).sort_values()
	coverage = period_coverage(bars, market, periods, "overnight").iloc[0]
	# The 13:29 minute belongs to the previous day's trading window. The Friday halt and the weekend are not gaps.
	assert (coverage["overnight_minutes"], coverage["overnight_missing_minutes"], coverage["overnight_longest_gap_minutes"]) == (4, 0, 0)


def test_held_contract_changes_on_the_roll_date():
	rolls = pd.DataFrame({"roll_date": [pd.Timestamp("2025-11-18")], "from_instrument_id": [1], "to_instrument_id": [2]})
	held = held_contracts(pd.DatetimeIndex(["2025-11-17", "2025-11-18", "2025-11-19"]), rolls)
	assert held.tolist() == [1, 2, 2]


def test_roll_date_previous_close_comes_from_the_new_contract(calendar):
	# On the roll date the overnight return starts at the new contract's previous 13:30 price, so the
	# 0.50 roll gap between the contracts does not enter it.
	bars = minutes((1, "2025-11-17 13:29", 60.00), (2, "2025-11-17 13:29", 60.50), *full_day(2, "2025-11-18", 61.0))
	refs = reference_prices(bars, held_on(("2025-11-17", 1), ("2025-11-18", 2)), calendar, TZ, "13:30", POINTS, 5).set_index("trade_date")
	assert refs.loc["2025-11-17", "p_1330"] == 60.00
	assert refs.loc["2025-11-18", "p_prev_1330"] == 60.50


def test_1300_price_uses_no_trade_from_1300_onward(calendar):
	early = [(1, "2025-11-18 07:59", 70.0), (1, "2025-11-18 08:29", 70.0), (1, "2025-11-18 12:29", 70.0), (1, "2025-11-18 12:59", 71.0)]
	late = [(1, f"2025-11-18 13:{minute:02d}", 90.0 + minute) for minute in range(30)]
	held = held_on(("2025-11-18", 1))
	refs = reference_prices(minutes(*early, *late), held, calendar, TZ, "13:30", POINTS, 5).iloc[0]
	assert refs["p_1300"] == 71.0
	assert refs["p_1330"] == 119.0
	# Changing every price from 13:00 on leaves all features known at 13:00 unchanged.
	shocked = [(instrument, local, -close) for instrument, local, close in late]
	again = reference_prices(minutes(*early, *shocked), held, calendar, TZ, "13:30", POINTS, 5).iloc[0]
	known = ["p_prev_1330", "p_0800", "p_0830", "p_1230", "p_1300"]
	assert again[known].equals(refs[known])


def test_stale_prices_are_used_up_to_the_limit_and_never_beyond(calendar):
	base = [(1, "2025-11-18 07:59", 70.0), (1, "2025-11-18 08:29", 70.0), (1, "2025-11-18 12:29", 70.0), (1, "2025-11-18 13:29", 70.0)]
	held = held_on(("2025-11-18", 1))
	within = reference_prices(minutes(*base, (1, "2025-11-18 12:54", 71.0)), held, calendar, TZ, "13:30", POINTS, 5).iloc[0]
	assert (within["p_1300"], within["stale_1300"]) == (71.0, 5)
	beyond = reference_prices(minutes(*base, (1, "2025-11-18 12:53", 71.0)), held, calendar, TZ, "13:30", POINTS, 5).iloc[0]
	assert pd.isna(beyond["p_1300"]) and not beyond["complete"]


def test_sessions_cut_by_the_query_range_are_dropped(calendar):
	# A query from 2020-12-01 00:00 UTC starts during the 2020-12-01 session (opened 2020-11-30 17:00 CT);
	# a query ending 2026-10-08 00:00 UTC stops during the 2026-10-08 session.
	frame = pd.DataFrame({"trade_date": pd.DatetimeIndex(["2020-12-01", "2020-12-02", "2026-10-07", "2026-10-08"])})
	whole = complete_sessions(frame, calendar, TZ, "17:00", "16:00", pd.Timestamp("2020-12-01", tz="UTC"), pd.Timestamp("2026-10-08", tz="UTC"))
	assert whole.tolist() == [False, True, True, False]


def test_window_gaps_count_only_minutes_in_which_the_market_traded():
	# Instrument 2 trades in a minute where the held contract does not, so one market minute is missing.
	held = [(1, f"2025-11-18 08:{minute:02d}", 70.0) for minute in (0, 1, 3)]
	bars = minutes(*held)
	market = pd.DatetimeIndex(minutes(*held, (2, "2025-11-18 08:02", 71.0))["ts_utc"]).sort_values()
	periods = pd.DataFrame(
		{
			"trade_date": [pd.Timestamp("2025-11-18")], "instrument_id": [1],
			"start": at_clock(["2025-11-18"], "08:00", TZ), "end": at_clock(["2025-11-18"], "08:04", TZ),
		}
	)
	coverage = period_coverage(bars, market, periods, "window").iloc[0]
	assert (coverage["window_minutes"], coverage["window_market_minutes"], coverage["window_missing_minutes"], coverage["window_longest_gap_minutes"]) == (3, 4, 1, 1)
