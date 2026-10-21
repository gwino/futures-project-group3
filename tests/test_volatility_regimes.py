"""Checks for the realized-variance features and the volatility-regime tests (#32).

Run from the repository root: python -m pytest
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.analyze_volatility_regimes import fit_model, interaction_design, nested_oos, regime_labels, wald_p_value
from src.build_volatility_features import build_features, check_windows, realized_variance, window_paths

TZ = "America/Chicago"
WINDOWS = {"on": ["prev_13:30", "08:00"], "day": ["08:00", "13:00"]}


def minutes(*rows: tuple[int, object, float]) -> pd.DataFrame:
	"""1-minute bars from (instrument_id, Chicago minute start, close) tuples."""
	stamps = [local if isinstance(local, pd.Timestamp) else pd.Timestamp(local, tz=TZ) for _, local, _ in rows]
	return pd.DataFrame(
		{
			"ts_utc": [stamp.tz_convert("UTC") for stamp in stamps],
			"instrument_id": [instrument for instrument, _, _ in rows],
			"close": [close for _, _, close in rows],
		}
	)


def ladder(instrument: int, start: str, end: str, step: int, first_close: float, increment: float = 0.1) -> list[tuple[int, pd.Timestamp, float]]:
	"""Bars every `step` minutes from start to end (Chicago time), closes rising by `increment` per bar."""
	stamps = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq=f"{step}min")
	return [(instrument, stamp, round(first_close + index * increment, 6)) for index, stamp in enumerate(stamps)]


def squared_log_returns(closes: list[float]) -> list[float]:
	return [math.log(later / earlier) ** 2 for earlier, later in zip(closes, closes[1:])]


def references(*rows: dict) -> pd.DataFrame:
	defaults = {
		"contract_id": "C1", "instrument_id": 1, "p_prev_1330": 100.0, "p_0800": 100.0, "p_1300": 100.0,
		"roll_flag": False, "folded_holiday": False, "degraded_date": False, "early_close": False, "low_volume": False, "complete": True, "paper_sample": True,
	}
	frame = pd.DataFrame([{**defaults, **row} for row in rows])
	frame["trade_date"], frame["prev_trade_date"] = pd.to_datetime(frame["trade_date"]), pd.to_datetime(frame["prev_trade_date"])
	return frame


def quality(*dates: str) -> pd.DataFrame:
	return pd.DataFrame(
		{"trade_date": pd.to_datetime(dates), "overnight_minutes": 1000, "overnight_missing_minutes": 0, "overnight_longest_gap_minutes": 0, "window_minutes": 330, "window_missing_minutes": 0}
	)


# --- Windows and paths ------------------------------------------------------------------------


def test_windows_must_end_by_the_decision_time():
	check_windows(WINDOWS, "13:00")
	with pytest.raises(ValueError, match="look-ahead"):
		check_windows({"late": ["08:00", "13:30"]}, "13:00")
	with pytest.raises(ValueError, match="before"):
		check_windows({"backward": ["13:00", "08:00"]}, "13:00")


def test_overnight_path_runs_from_the_previous_1329_bar_to_the_0759_bar():
	bars = minutes(
		(1, "2024-01-02 13:28", 99.0), (1, "2024-01-02 13:29", 100.0), (1, "2024-01-02 13:30", 101.0), (1, "2024-01-02 15:59", 102.0),
		(1, "2024-01-02 17:00", 103.0), (1, "2024-01-03 07:59", 104.0), (1, "2024-01-03 08:00", 105.0),
	)
	refs = references({"trade_date": "2024-01-03", "prev_trade_date": "2024-01-02"})
	path = window_paths(bars, refs, "prev_13:30", "08:00", TZ)
	assert path["close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]
	assert path["ts_utc"].dt.tz_convert(TZ).dt.strftime("%H:%M").tolist() == ["13:29", "13:30", "15:59", "17:00", "07:59"]


def test_reopening_returns_are_jumps_and_short_gaps_are_kept():
	# Previous day 13:29 to 15:59 with one 10-minute and several 30-minute gaps (kept), the 16:00-17:00
	# halt (a jump), then every 30 minutes through the night to 07:59.
	evening = [(1, "2024-01-02 13:29", 100.0), (1, "2024-01-02 13:30", 100.5), (1, "2024-01-02 13:40", 101.0), *ladder(1, "2024-01-02 14:10", "2024-01-02 15:40", 30, 101.1), (1, "2024-01-02 15:59", 101.5)]
	night = [*ladder(1, "2024-01-02 17:00", "2024-01-03 07:30", 30, 103.0), (1, "2024-01-03 07:59", 106.0)]
	bars = minutes(*evening, *night)
	refs = references({"trade_date": "2024-01-03", "prev_trade_date": "2024-01-02"})
	rv = realized_variance(window_paths(bars, refs, "prev_13:30", "08:00", TZ), 60, "on").iloc[0]
	evening_closes, night_closes = [close for _, _, close in evening], [close for _, _, close in night]
	assert rv["rv_on"] == pytest.approx(sum(squared_log_returns(evening_closes)) + sum(squared_log_returns(night_closes)))
	assert rv["n_on"] == len(evening) + len(night) - 2 and rv["bars_on"] == len(evening) + len(night)
	assert rv["jumps_on"] == 1 and rv["jump_var_on"] == pytest.approx(math.log(103.0 / 101.5) ** 2)
	assert rv["max_gap_on"] == 30


def test_weekend_reopen_is_one_jump_and_the_path_follows_daylight_saving():
	# Friday 2025-03-07 (CST) to Monday 2025-03-10 (CDT): the Sunday 17:00 CDT reopen is the only jump.
	friday = [(1, "2025-03-07 13:29", 60.0), *ladder(1, "2025-03-07 13:59", "2025-03-07 15:59", 30, 60.1)]
	sunday = [*ladder(1, "2025-03-09 17:00", "2025-03-10 07:30", 30, 61.0), (1, "2025-03-10 07:59", 63.0)]
	bars = minutes(*friday, *sunday)
	refs = references({"trade_date": "2025-03-10", "prev_trade_date": "2025-03-07"})
	rv = realized_variance(window_paths(bars, refs, "prev_13:30", "08:00", TZ), 60, "on").iloc[0]
	assert rv["jumps_on"] == 1 and rv["n_on"] == len(friday) + len(sunday) - 2
	assert rv["rv_on"] == pytest.approx(sum(squared_log_returns([close for _, _, close in friday])) + sum(squared_log_returns([close for _, _, close in sunday])))
	assert rv["first_bar_on"].tz_convert(TZ).strftime("%m-%d %H:%M") == "03-07 13:29" and rv["last_bar_on"].tz_convert(TZ).strftime("%m-%d %H:%M") == "03-10 07:59"


def test_roll_date_path_uses_the_newly_held_contract_on_the_previous_day():
	old = [(1, "2024-01-02 13:29", 100.0), *ladder(1, "2024-01-02 13:59", "2024-01-02 15:59", 30, 110.0), *ladder(1, "2024-01-02 17:00", "2024-01-03 07:59", 30, 120.0)]
	new = [(2, "2024-01-02 13:29", 50.0), *ladder(2, "2024-01-02 13:59", "2024-01-02 15:59", 30, 50.1), *ladder(2, "2024-01-02 17:00", "2024-01-03 07:30", 30, 50.6), (2, "2024-01-03 07:59", 52.0)]
	bars = minutes(*old, *new)
	refs = references({"trade_date": "2024-01-03", "prev_trade_date": "2024-01-02", "instrument_id": 2, "contract_id": "C2", "p_prev_1330": 50.0, "p_0800": 52.0})
	rv = realized_variance(window_paths(bars, refs, "prev_13:30", "08:00", TZ), 60, "on").iloc[0]
	assert rv["first_close_on"] == 50.0 and rv["last_close_on"] == 52.0
	closes = [close for _, _, close in new]
	assert rv["rv_on"] == pytest.approx(sum(squared_log_returns(closes[:6])) + sum(squared_log_returns(closes[6:])))
	assert rv["jumps_on"] == 1


def test_features_use_no_trade_from_1300_onward_and_match_the_reference_prices():
	night = [(1, "2024-01-02 13:29", 99.5), *ladder(1, "2024-01-02 13:59", "2024-01-02 15:59", 30, 99.6), *ladder(1, "2024-01-02 17:00", "2024-01-03 07:30", 30, 99.0), (1, "2024-01-03 07:59", 100.0)]
	day = [*ladder(1, "2024-01-03 08:29", "2024-01-03 12:29", 30, 100.1), (1, "2024-01-03 12:59", 100.4), (1, "2024-01-03 13:00", 100.9), (1, "2024-01-03 13:29", 101.5)]
	bars = minutes(*night, *day)
	refs = references({"trade_date": "2024-01-03", "prev_trade_date": "2024-01-02", "p_prev_1330": 99.5, "p_0800": 100.0, "p_1300": 100.4})
	base = build_features(refs, bars, quality("2024-01-03"), WINDOWS, TZ, 60, 5)
	later = build_features(refs, bars.assign(close=np.where(bars["ts_utc"] >= pd.Timestamp("2024-01-03 13:00", tz=TZ), 500.0, bars["close"])), quality("2024-01-03"), WINDOWS, TZ, 60, 5)
	night_closes, day_closes = [close for _, _, close in night], [100.0, *[close for _, _, close in day[:-2]]]
	assert base.loc[0, "rv_day"] == pytest.approx(sum(squared_log_returns(day_closes))) and base.loc[0, "n_day"] == len(day_closes) - 1
	assert base.loc[0, "rv_on"] == pytest.approx(sum(squared_log_returns(night_closes[:6])) + sum(squared_log_returns(night_closes[6:]))) and base.loc[0, "jumps_on"] == 1
	assert base.loc[0, "rv_rod"] == pytest.approx(base.loc[0, "rv_on"] + base.loc[0, "rv_day"])
	assert base.loc[0, "endpoints_match_on"] and base.loc[0, "endpoints_match_day"] and base.loc[0, "sample"]
	assert later[["rv_on", "rv_day", "rv_rod"]].equals(base[["rv_on", "rv_day", "rv_rod"]])


# --- Regime labels ----------------------------------------------------------------------------


def test_expanding_threshold_uses_only_earlier_dates():
	measure = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 0.5, 20.0])
	labels = regime_labels(measure, {"method": "expanding", "quantile": 0.5, "min_history": 3})
	assert labels["high"].isna().tolist()[:3] == [True, True, True]
	assert labels.loc[3, "threshold"] == 2.0 and labels.loc[3, "high"] == 1.0
	assert labels.loc[5, "threshold"] == 3.0 and labels.loc[5, "high"] == 0.0
	changed = regime_labels(measure.mask(measure.index > 4, 1000.0), {"method": "expanding", "quantile": 0.5, "min_history": 3})
	assert changed.loc[:4, "high"].equals(labels.loc[:4, "high"])


def test_training_threshold_is_fixed_from_the_first_observations():
	measure = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 0.0])
	labels = regime_labels(measure, {"method": "training", "quantile": 0.75, "train_obs": 4})
	assert labels["threshold"].eq(3.25).all()
	assert labels["high"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


# --- Estimation -------------------------------------------------------------------------------


def test_interaction_design_and_wald_p_values():
	sample = pd.DataFrame({"r_on": [0.01, -0.02, 0.03], "r_day": [0.0, 0.01, -0.01], "high": [0.0, 1.0, 1.0]})
	X, names = interaction_design(sample, {"on": "r_on", "day": "r_day"}, "high")
	assert names == ["on", "day", "high", "on:high", "day:high"]
	assert X[:, 3].tolist() == [0.0, -0.02, 0.03] and X[:, 4].tolist() == [0.0, 0.01, -0.01]
	assert wald_p_value(0.0, 2) == 1.0 and wald_p_value(2 * math.log(4), 2) == pytest.approx(0.25)
	assert wald_p_value(1.96**2, 1) == pytest.approx(0.05, abs=1e-3)


def test_high_regime_slope_is_the_sum_of_slope_and_interaction():
	rng = np.random.default_rng(0)
	n = 600
	high = (np.arange(n) % 2).astype(float)
	on, day = rng.normal(size=n), rng.normal(size=n)
	y = 0.001 + 0.02 * on - 0.01 * day + 0.03 * on * high + rng.normal(scale=0.01, size=n)
	sample = pd.DataFrame({"trade_date": pd.bdate_range("2021-01-01", periods=n), "r_lh": y, "r_on": on, "r_day": day, "high": high})
	terms, slopes, stats = fit_model(sample, "r_lh", {"on": "r_on", "day": "r_day"}, "high", 0, 1.96, 100, None)
	by_term = {row["term"]: row["coef_x100"] for row in terms}
	on_row = next(row for row in slopes if row["predictor"] == "on")
	assert on_row["slope_low_x100"] == pytest.approx(by_term["on"]) and on_row["slope_high_x100"] == pytest.approx(by_term["on"] + by_term["on:high"])
	assert on_row["slope_high_x100"] == pytest.approx(5.0, abs=0.3) and on_row["difference_x100"] == pytest.approx(3.0, abs=0.3)
	assert stats["n_high"] == 300 and stats["n_low"] == 300 and stats["wald_df"] == 2 and stats["wald_p"] < 1e-6


def test_nested_oos_forecasts_use_only_earlier_observations():
	rng = np.random.default_rng(1)
	n = 60
	sample = pd.DataFrame({"trade_date": pd.bdate_range("2021-01-01", periods=n), "r_lh": rng.normal(size=n), "r_on": rng.normal(size=n), "r_day": rng.normal(size=n), "high": (np.arange(n) % 3 == 0).astype(float)})
	stats, forecasts = nested_oos(sample, "r_lh", {"on": "r_on", "day": "r_day"}, "high", 40, 0)
	assert len(forecasts) == 20 and forecasts["trade_date"].iloc[0] == sample["trade_date"].iloc[40]
	later = sample.copy()
	later.loc[41:, ["r_lh", "r_on", "r_day"]] = 100.0
	_, changed = nested_oos(later, "r_lh", {"on": "r_on", "day": "r_day"}, "high", 40, 0)
	assert changed.iloc[0][["base_forecast", "interaction_forecast", "historical_mean"]].equals(forecasts.iloc[0][["base_forecast", "interaction_forecast", "historical_mean"]])
	assert set(stats) == {"oos_r2_vs_base_pct", "clark_west_t_vs_base", "clark_west_p_vs_base"}
