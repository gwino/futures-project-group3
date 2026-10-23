"""Checks for the signal timing, fills, costs and metrics of the last-half-hour strategy backtest (#30).

Run from the repository root: python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest_lh_strategy import (
	alternative_references, apply_costs, blackout_mask, build_ledger, check_signals, evaluate, hac_standard_error, max_drawdown,
	next_bar_fills, positions, reference_fills, settlement_proxy_fills, skip_days,
)

TZ = "America/Chicago"
EXECUTION = {"entry_time": "13:00", "exit_time": "13:30", "max_delay_minutes": 5, "contracts": 1, "multiplier": 1000, "tick_size": 0.01}
EVALUATION = {"trading_days_per_year": 252, "confidence": 0.95, "hac_lags": 5, "rolling_window": 252, "worst_days": 10, "adverse_loss_usd": 500}
ROD = {"long_if_positive": ["p_prev_1330", "p_1300"]}


def minutes(*rows: tuple[int, str, float, float]) -> pd.DataFrame:
	"""1-minute bars from (instrument_id, Chicago minute start, open, close) tuples."""
	return pd.DataFrame(
		{
			"ts_utc": [pd.Timestamp(local, tz=TZ).tz_convert("UTC") for _, local, _, _ in rows],
			"instrument_id": [instrument for instrument, _, _, _ in rows],
			"contract_id": [f"C{instrument}" for instrument, _, _, _ in rows],
			"open": [open_ for _, _, open_, _ in rows],
			"close": [close for _, _, _, close in rows],
		}
	)


def ohlcv(*rows: tuple[int, str, float, float, float, float, int]) -> pd.DataFrame:
	"""1-minute bars from (instrument_id, Chicago minute start, open, high, low, close, volume) tuples."""
	frame = pd.DataFrame(rows, columns=["instrument_id", "local", "open", "high", "low", "close", "volume"])
	frame["ts_utc"] = [pd.Timestamp(local, tz=TZ).tz_convert("UTC") for local in frame["local"]]
	return frame.drop(columns="local")


def held_on(*pairs: tuple[str, int]) -> pd.Series:
	return pd.Series([instrument for _, instrument in pairs], index=pd.DatetimeIndex([date for date, _ in pairs]), name="instrument_id")


def references(*rows: dict) -> pd.DataFrame:
	"""Reference-price rows; prices default to 100 and flags to a normal sample day."""
	defaults = {
		"contract_id": "C1", "instrument_id": 1, "p_prev_1330": 100.0, "p_0800": 100.0, "p_0830": 100.0, "p_1230": 100.0, "p_1300": 100.0, "p_1330": 100.0,
		"stale_prev_1330": 0.0, "stale_0800": 0.0, "stale_0830": 0.0, "stale_1230": 0.0, "stale_1300": 0.0, "stale_1330": 0.0,
		"session_volume": 100_000,
		"roll_flag": False, "early_close": False, "low_volume": False, "folded_holiday": False, "degraded_date": False, "complete": True, "paper_sample": True,
	}
	frame = pd.DataFrame([{**defaults, **row} for row in rows])
	frame["trade_date"] = pd.to_datetime(frame["trade_date"])
	frame["prev_trade_date"] = pd.to_datetime(frame["prev_trade_date"]) if "prev_trade_date" in frame else frame["trade_date"] - pd.Timedelta(days=1)
	return frame


def ledger_for(refs: pd.DataFrame, bars: pd.DataFrame, spec: dict = ROD, execution: dict = EXECUTION) -> pd.DataFrame:
	held = refs.set_index("trade_date")["instrument_id"]
	position, predictor = positions(refs.set_index("trade_date"), spec)
	entry = next_bar_fills(bars, held, execution["entry_time"], TZ, execution["max_delay_minutes"])
	exit = next_bar_fills(bars, held, execution["exit_time"], TZ, execution["max_delay_minutes"])
	return build_ledger(refs, position, predictor, entry, exit, "rod", execution)


# --- Signal -----------------------------------------------------------------------------------


def test_signal_prices_must_be_known_by_the_decision_time():
	check_signals({"rod": ROD}, "13:00")
	with pytest.raises(ValueError, match="look-ahead"):
		check_signals({"bad": {"long_if_positive": ["p_prev_1330", "p_1330"]}}, "13:00")
	with pytest.raises(ValueError, match="before"):
		check_signals({"bad": {"long_if_positive": ["p_1300", "p_0800"]}}, "13:00")


def test_position_follows_the_sign_of_the_predictor_and_is_flat_at_zero():
	refs = references(
		{"trade_date": "2024-01-02", "p_prev_1330": 100.0, "p_1300": 101.0},
		{"trade_date": "2024-01-03", "p_prev_1330": 100.0, "p_1300": 99.0},
		{"trade_date": "2024-01-04", "p_prev_1330": 100.0, "p_1300": 100.0},
		{"trade_date": "2024-01-05", "p_prev_1330": np.nan, "p_1300": 100.0},
	).set_index("trade_date")
	position, predictor = positions(refs, ROD)
	assert position.tolist() == [1, -1, 0, 0]
	assert predictor.iloc[0] == pytest.approx(0.01)


def test_agreement_signal_trades_only_when_both_returns_share_a_sign():
	refs = references(
		{"trade_date": "2024-01-02", "p_prev_1330": 100.0, "p_0830": 101.0, "p_1300": 102.0},
		{"trade_date": "2024-01-03", "p_prev_1330": 100.0, "p_0830": 99.0, "p_1300": 101.0},
		{"trade_date": "2024-01-04", "p_prev_1330": 100.0, "p_0830": 99.0, "p_1300": 98.0},
	).set_index("trade_date")
	position, _ = positions(refs, {"long_if_all_positive": [["p_prev_1330", "p_0830"], ["p_prev_1330", "p_1300"]]})
	assert position.tolist() == [1, 0, -1]


# --- Fills ------------------------------------------------------------------------------------


def test_entry_is_the_open_of_the_first_bar_at_or_after_1300():
	bars = minutes((1, "2024-01-02 12:59", 99.0, 99.5), (1, "2024-01-02 13:00", 100.0, 100.2), (1, "2024-01-02 13:01", 101.0, 101.1))
	fill = next_bar_fills(bars, held_on(("2024-01-02", 1)), "13:00", TZ, 5).loc["2024-01-02"]
	assert fill["price"] == 100.0 and fill["delay"] == 0
	assert fill["time"].strftime("%H:%M") == "13:00"


def test_missing_1300_bar_uses_the_next_bar_within_the_delay_limit_or_nothing():
	bars = minutes((1, "2024-01-02 12:59", 99.0, 99.5), (1, "2024-01-02 13:02", 100.5, 100.6), (1, "2024-01-03 12:59", 99.0, 99.5), (1, "2024-01-03 13:06", 100.0, 100.0))
	fills = next_bar_fills(bars, held_on(("2024-01-02", 1), ("2024-01-03", 1)), "13:00", TZ, 5)
	assert fills.loc["2024-01-02", "price"] == 100.5 and fills.loc["2024-01-02", "delay"] == 2
	assert np.isnan(fills.loc["2024-01-03", "price"])


def test_fills_follow_daylight_saving():
	# Clocks moved on 2025-03-09: 13:00 CT is 19:00 UTC before and 18:00 UTC after.
	bars = minutes((1, "2025-03-07 13:00", 60.0, 60.0), (1, "2025-03-10 12:00", 61.0, 61.0), (1, "2025-03-10 13:00", 62.0, 62.0))
	fills = next_bar_fills(bars, held_on(("2025-03-07", 1), ("2025-03-10", 1)), "13:00", TZ, 5)
	assert fills["price"].tolist() == [60.0, 62.0]
	assert fills["time"].dt.tz_convert("UTC").dt.strftime("%H:%M").tolist() == ["19:00", "18:00"]


def test_folded_holiday_bars_are_not_fills_for_the_next_trade_date():
	# Martin Luther King Day 2022-01-17 traded and folds into trade date 2022-01-18; its 13:00 bar is not the 2022-01-18 entry.
	bars = minutes((1, "2022-01-17 13:00", 84.0, 84.0), (1, "2022-01-18 13:00", 85.0, 85.0))
	fills = next_bar_fills(bars, held_on(("2022-01-18", 1)), "13:00", TZ, 5)
	assert fills.loc["2022-01-18", "price"] == 85.0


def test_roll_date_fills_use_the_newly_held_contract():
	bars = minutes((1, "2022-03-21 13:00", 104.0, 104.0), (2, "2022-03-21 13:00", 109.0, 109.0))
	fills = next_bar_fills(bars, held_on(("2022-03-21", 2)), "13:00", TZ, 5)
	assert fills.loc["2022-03-21", "price"] == 109.0


def test_reference_fill_is_the_pipeline_price_from_the_bar_before_the_time():
	refs = references({"trade_date": "2024-01-02", "p_1330": 101.5, "stale_1330": 0.0})
	fill = reference_fills(refs, "13:30", TZ).loc["2024-01-02"]
	assert fill["price"] == 101.5 and fill["delay"] == -1
	assert fill["time"].strftime("%H:%M") == "13:29"
	with pytest.raises(ValueError, match="13:01"):
		reference_fills(refs, "13:01", TZ)


def test_settlement_proxy_is_the_volume_weighted_average_of_the_two_bars_before_1330():
	bars = ohlcv(
		(1, "2024-01-02 13:27", 90.0, 90.0, 90.0, 90.0, 1000),            # before the window: ignored
		(1, "2024-01-02 13:28", 100.0, 100.4, 99.8, 100.2, 300),          # OHLC average 100.1
		(1, "2024-01-02 13:29", 100.2, 100.6, 100.2, 100.6, 100),         # OHLC average 100.4
		(1, "2024-01-02 13:30", 120.0, 120.0, 120.0, 120.0, 1000),        # at the time: ignored
		(2, "2024-01-02 13:29", 50.0, 50.0, 50.0, 50.0, 500),             # other contract: ignored
	)
	fill = settlement_proxy_fills(bars, held_on(("2024-01-02", 1)), "13:30", TZ, 2).loc["2024-01-02"]
	assert fill["price"] == pytest.approx((100.1 * 300 + 100.4 * 100) / 400)
	assert fill["time"].strftime("%H:%M") == "13:30" and fill["delay"] == 0
	missing = settlement_proxy_fills(bars, held_on(("2024-01-03", 1)), "13:30", TZ, 2).loc["2024-01-03"]
	assert np.isnan(missing["price"]) and pd.isna(missing["time"])


def test_exit_side_can_carry_its_own_slippage():
	refs = references({"trade_date": "2024-01-02", "p_prev_1330": 100.0, "p_1300": 101.0, "p_1330": 101.4})
	bars = minutes((1, "2024-01-02 13:00", 101.00, 101.1), (1, "2024-01-02 13:30", 101.25, 101.3))
	both = apply_costs(ledger_for(refs, bars), 2.50, 0.5, EXECUTION).iloc[0]
	tas = apply_costs(ledger_for(refs, bars), 2.50, 0.5, EXECUTION, exit_slippage_ticks_per_side=0).iloc[0]
	assert both["slippage_usd"] == pytest.approx(10.0) and tas["slippage_usd"] == pytest.approx(5.0)
	assert tas["commission_usd"] == 5.0 and tas["pnl_net_usd"] == pytest.approx(240.0)


def test_signal_ignores_prices_from_1300_on_and_fills_ignore_prices_before_1300():
	refs = references({"trade_date": "2024-01-02", "p_prev_1330": 100.0, "p_1300": 101.0, "p_1330": 103.0})
	bars = minutes((1, "2024-01-02 12:59", 100.9, 101.0), (1, "2024-01-02 13:00", 101.2, 101.3), (1, "2024-01-02 13:30", 103.1, 103.0))
	base = ledger_for(refs, bars)
	later = ledger_for(refs.assign(p_1330=50.0), bars.assign(open=np.where(bars["ts_utc"] >= pd.Timestamp("2024-01-02 13:00", tz=TZ), 50.0, bars["open"])))
	earlier = ledger_for(refs, bars.assign(open=np.where(bars["ts_utc"] < pd.Timestamp("2024-01-02 13:00", tz=TZ), 50.0, bars["open"])))
	assert later["position"].tolist() == base["position"].tolist() == [1]
	assert earlier["entry_price"].iloc[0] == base["entry_price"].iloc[0] == 101.2
	assert earlier["exit_price"].iloc[0] == 103.1


# --- Ledger and costs -------------------------------------------------------------------------


def test_pnl_matches_a_hand_calculation_for_long_and_short_days():
	refs = references(
		{"trade_date": "2024-01-02", "p_prev_1330": 100.0, "p_1300": 101.0, "p_1330": 101.4},
		{"trade_date": "2024-01-03", "p_prev_1330": 100.0, "p_1300": 99.0, "p_1330": 99.3},
	)
	bars = minutes(
		(1, "2024-01-02 13:00", 101.00, 101.1), (1, "2024-01-02 13:30", 101.25, 101.3),
		(1, "2024-01-03 13:00", 99.00, 99.1), (1, "2024-01-03 13:30", 99.25, 99.2),
	)
	ledger = apply_costs(ledger_for(refs, bars), 2.50, 0.5, EXECUTION)
	long_day, short_day = ledger.iloc[0], ledger.iloc[1]
	assert long_day["position"] == 1 and long_day["pnl_gross_usd"] == pytest.approx(250.0)
	assert long_day["r_lh"] == pytest.approx(0.4 / 101) and long_day["r_frictionless"] == pytest.approx(0.4 / 101)
	assert long_day["r_gross"] == pytest.approx(0.25 / 101)
	assert long_day["commission_usd"] == 5.0 and long_day["slippage_usd"] == pytest.approx(10.0)
	assert long_day["pnl_net_usd"] == pytest.approx(235.0) and long_day["r_net"] == pytest.approx(235.0 / 101_000)
	assert short_day["position"] == -1 and short_day["pnl_gross_usd"] == pytest.approx(-250.0)
	assert short_day["r_frictionless"] == pytest.approx(-0.3 / 99) and short_day["pnl_net_usd"] == pytest.approx(-265.0)


def test_untraded_days_carry_no_pnl_no_costs_and_a_reason():
	refs = references(
		{"trade_date": "2024-01-02", "p_prev_1330": 100.0, "p_1300": 100.0},
		{"trade_date": "2024-01-03", "p_prev_1330": 100.0, "p_1300": 101.0},
	)
	bars = minutes((1, "2024-01-02 13:00", 100.0, 100.0), (1, "2024-01-02 13:30", 100.0, 100.0), (1, "2024-01-03 13:00", 101.0, 101.0))
	ledger = apply_costs(ledger_for(refs, bars), 2.50, 0.5, EXECUTION)
	assert not ledger["traded"].any()
	assert ledger["skip_reason"].tolist() == ["no position (signal zero or missing)", "no exit fill within 5 minutes"]
	assert ledger[["pnl_gross_usd", "cost_usd", "pnl_net_usd", "r_net"]].eq(0).all().all()


def test_blackout_skips_the_masked_days_and_keeps_the_rest():
	refs = references(
		{"trade_date": "2024-01-02", "p_prev_1330": 100.0, "p_1300": 101.0, "roll_flag": True},
		{"trade_date": "2024-01-03", "p_prev_1330": 100.0, "p_1300": 101.0},
	)
	refs["sessions_to_expiry"] = [2, 10]
	bars = minutes(*[(1, f"{date} {clock}", 101.0, 101.0) for date in ("2024-01-02", "2024-01-03") for clock in ("13:00", "13:30")])
	ledger = ledger_for(refs, bars)
	assert blackout_mask(refs, {"column": "roll_flag"}).tolist() == [True, False]
	assert blackout_mask(refs, {"column": "sessions_to_expiry", "at_most": 3}).tolist() == [True, False]
	assert blackout_mask(refs, {"dates": ["2024-01-03"]}).tolist() == [False, True]
	assert blackout_mask(refs, {}).tolist() == [False, False]
	skipped = skip_days(ledger, blackout_mask(refs, {"column": "roll_flag"}), "blackout roll")
	assert skipped["traded"].tolist() == [False, True] and skipped["position"].tolist() == [0, 1]
	assert skipped["skip_reason"].iloc[0] == "blackout roll"


# --- Metrics ----------------------------------------------------------------------------------


def test_max_drawdown_of_a_cumulative_path():
	drawdown, trough = max_drawdown(np.array([1.0, -2.0, 1.0, -1.0, 3.0]))
	assert drawdown == 2.0 and trough == 1


def test_hac_standard_error_with_no_lags_is_the_plain_standard_error():
	values = np.array([1.0, -1.0, 2.0, 0.5, -0.5, 1.5])
	assert hac_standard_error(values, 0) == pytest.approx(values.std(ddof=0) / np.sqrt(len(values)))


def test_evaluate_reports_hit_rate_sharpe_and_turnover():
	dates = pd.bdate_range("2024-01-01", periods=4)
	ledger = pd.DataFrame(
		{
			"trade_date": dates, "position": [1, -1, 1, 0], "traded": [True, True, True, False], "sides": [2, 2, 2, 0],
			"r_net": [0.002, -0.001, 0.001, 0.0], "pnl_net_usd": [200.0, -100.0, 100.0, 0.0], "cost_usd": [15.0, 15.0, 15.0, 0.0],
		}
	)
	result = evaluate(ledger, "r_net", "pnl_net_usd", EVALUATION)
	assert result["days"] == 4 and result["traded_days"] == 3 and result["long_days"] == 2 and result["short_days"] == 1
	assert result["hit_rate"] == pytest.approx(2 / 3)
	assert result["mean_bps"] == pytest.approx(5.0)
	assert result["sharpe"] == pytest.approx(0.0005 / np.std([0.002, -0.001, 0.001, 0.0], ddof=1) * np.sqrt(252))
	assert result["total_pnl_usd"] == 200.0 and result["max_drawdown_usd"] == 100.0 and result["worst_day"] == dates[1].date()
	assert result["round_trips_per_year"] == pytest.approx(3 / (4 / 252)) and result["contracts_traded_per_year"] == pytest.approx(6 / (4 / 252))
	assert result["adverse_days"] == 0


# --- Roll rules -------------------------------------------------------------------------------


def test_alternative_roll_calendar_switches_the_held_contract_and_its_previous_close():
	# The paper rule holds contract 1 through 2024-01-04; the pre-expiry rule switches to contract 2 on 2024-01-03.
	days = ["2024-01-02", "2024-01-03", "2024-01-04"]
	bars = minutes(
		*[(instrument, f"{day} {clock}", price, price) for instrument, price in ((1, 100.0), (2, 110.0)) for day in days for clock in ("07:59", "08:29", "12:29", "12:59", "13:29")]
	)
	refs = references(*[{"trade_date": day, "prev_trade_date": prev} for day, prev in zip(days, ["2024-01-01", *days[:-1]])])
	comparison = pd.DataFrame({"from_contract": ["C1"], "to_contract": ["C2"], "roll_date_paper": ["2024-01-05"], "roll_date_pre_expiry": ["2024-01-03"]})
	minute_cfg = {"session": {"timezone": TZ}, "reference": {"previous_close": "13:30", "points": ["08:00", "08:30", "12:30", "13:00", "13:30"], "max_stale_minutes": 5}}
	rebuilt = alternative_references(refs, bars, comparison, "roll_date_pre_expiry", minute_cfg).set_index("trade_date")
	assert rebuilt["contract_id"].tolist() == ["C1", "C2", "C2"]
	assert rebuilt["roll_flag"].tolist() == [False, True, False]
	assert rebuilt.loc["2024-01-03", "p_prev_1330"] == 110.0 and rebuilt.loc["2024-01-03", "p_1300"] == 110.0
	assert rebuilt.loc["2024-01-02", "paper_sample"] == False and rebuilt.loc["2024-01-03", "paper_sample"]  # noqa: E712  (no 2024-01-01 bars)
