"""Checks for the return windows, sample selection, and look-ahead rules of the return features (#28).

Run from the repository root: python -m pytest
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.build_return_features import build_features, check_returns, column_dictionary, identity_errors, load_config

CONFIG = load_config(Path("config/return_features.yaml"))
RETURNS, TARGET = CONFIG["returns"], CONFIG["target"]
PRICES = ["p_prev_1330", "p_0800", "p_0830", "p_1230", "p_1300", "p_1330"]


def references(*days: dict) -> pd.DataFrame:
	"""Reference-price rows; each day gives trade_date, prev_trade_date and prices, and may override flags."""
	rows = []
	for day in days:
		row = {"contract_id": "CLF2025", "roll_flag": False, "folded_holiday": False, "degraded_date": False, "early_close": False, "low_volume": False, **day}
		row["complete"] = all(pd.notna(row.get(column)) for column in PRICES)
		row["paper_sample"] = row["complete"] and not row["early_close"] and not row["low_volume"]
		rows.append(row)
	frame = pd.DataFrame(rows)
	frame[["trade_date", "prev_trade_date"]] = frame[["trade_date", "prev_trade_date"]].apply(pd.to_datetime)
	return frame


def day(trade_date: str, prev_trade_date: str, prices: list[float], **flags) -> dict:
	return {"trade_date": trade_date, "prev_trade_date": prev_trade_date, **dict(zip(PRICES, prices)), **flags}


def build(frame: pd.DataFrame, start: str = "2025-01-01") -> pd.DataFrame:
	expiry = pd.Series(10, index=frame["trade_date"])
	return build_features(frame, expiry, RETURNS, pd.Timestamp(start), None, True).set_index("trade_date")


# Previous 13:30, 08:00, 08:30, 12:30, 13:00, 13:30.
HAND = [100.0, 101.0, 102.01, 100.9899, 103.0, 103.515]


def test_simple_returns_match_hand_calculations():
	row = build(references(day("2025-03-04", "2025-03-03", HAND))).loc["2025-03-04"]
	expected = {
		"r_on": 101.0 / 100.0 - 1,          # previous 13:30 -> 08:00: 1%
		"r_fh": 102.01 / 101.0 - 1,         # 08:00 -> 08:30: 1%
		"r_onfh": 102.01 / 100.0 - 1,       # previous 13:30 -> 08:30: 2.01%
		"r_m": 100.9899 / 102.01 - 1,       # 08:30 -> 12:30: -1%
		"r_slh": 103.0 / 100.9899 - 1,      # 12:30 -> 13:00
		"r_rod": 103.0 / 100.0 - 1,         # previous 13:30 -> 13:00: 3%
		"r_day": 103.0 / 101.0 - 1,         # 08:00 -> 13:00
		"r_lh": 103.515 / 103.0 - 1,        # 13:00 -> 13:30: 0.5%, the target
	}
	for column, value in expected.items():
		assert row[column] == pytest.approx(value, abs=1e-12), column
	assert (row["r_on"], row["r_fh"], row["r_m"], row["r_rod"], row["r_lh"]) == pytest.approx((0.01, 0.01, -0.01, 0.03, 0.005))


def test_log_returns_add_up_and_simple_returns_compound():
	frame = build(references(day("2025-03-04", "2025-03-03", HAND), day("2025-03-05", "2025-03-04", [103.515, 99.0, 98.0, 99.5, 97.0, 96.5])))
	frame = frame.reset_index()
	assert max(identity_errors(frame, CONFIG["log_identities"]).values()) < 1e-12
	assert np.allclose((1 + frame["r_on"]) * (1 + frame["r_day"]), 1 + frame["r_rod"])
	assert not np.allclose(frame["r_on"] + frame["r_day"], frame["r_rod"])


def test_predictors_do_not_change_when_target_prices_change():
	base = build(references(day("2025-03-04", "2025-03-03", HAND)))
	shocked = build(references(day("2025-03-04", "2025-03-03", [*HAND[:-1], 50.0])))
	predictors = [f"{kind}_{name}" for name in RETURNS if name != TARGET for kind in ("r", "log")]
	assert base[predictors].equals(shocked[predictors])
	assert base["r_lh"].iloc[0] != shocked["r_lh"].iloc[0]


@pytest.mark.parametrize(
	"returns, message",
	[
		({"rod": ["p_prev_1330", "p_1330"], "lh": ["p_1300", "p_1330"]}, "look-ahead"),
		({"rod": ["p_prev_1330", "p_1300"], "lh": ["p_1230", "p_1330"]}, "overlaps"),
		({"rod": ["p_1300", "p_prev_1330"], "lh": ["p_1300", "p_1330"]}, "start before"),
		({True: ["p_prev_1330", "p_0800"], "lh": ["p_1300", "p_1330"]}, "strings"),
	],
)
def test_config_rejects_look_ahead_and_bad_intervals(returns, message):
	with pytest.raises(ValueError, match=message):
		check_returns(returns, "lh", "13:00")


def test_bare_yaml_on_key_is_a_boolean():
	# Why config/return_features.yaml quotes "on": YAML 1.1 reads a bare on as true.
	assert list(yaml.safe_load("on: [p_prev_1330, p_0800]")) == [True]
	assert "on" in RETURNS


def test_excluded_dates_are_logged_and_obs_counts_only_the_sample():
	frame = build(
		references(
			day("2024-12-31", "2024-12-30", HAND),                                   # before the sample start
			day("2025-11-28", "2025-11-26", [*HAND[:4], np.nan, np.nan], early_close=True),
			day("2025-12-01", "2025-11-28", [np.nan, *HAND[1:]]),                    # no previous 13:30 after an early close
			day("2025-12-02", "2025-12-01", HAND, roll_flag=True, degraded_date=True),
			day("2025-12-03", "2025-12-02", HAND, low_volume=True),
			day("2025-12-04", "2025-12-03", HAND),
		)
	)
	assert pd.Timestamp("2024-12-31") not in frame.index
	assert frame["sample"].tolist() == [False, False, True, False, True]
	assert frame["obs"].dropna().tolist() == [1, 2]
	reasons = frame["exclusion_reason"]
	assert reasons["2025-11-28"] == "missing p_1300; missing p_1330; early close"
	assert reasons["2025-12-01"] == "missing p_prev_1330"
	assert reasons["2025-12-03"] == "low volume"
	assert reasons[frame["sample"]].eq("").all()
	# Roll and degraded dates stay in the sample, flagged for robustness checks.
	assert frame.loc["2025-12-02", ["sample", "roll_flag", "degraded_date"]].tolist() == [True, True, True]
	# Returns that can be computed on excluded dates are kept; the rest are missing.
	assert frame.loc["2025-11-28", "r_onfh"] == pytest.approx(0.0201) and pd.isna(frame.loc["2025-11-28", "r_lh"])


def test_weekend_gap_and_column_dictionary():
	frame = build(references(day("2025-03-10", "2025-03-07", HAND)))
	assert (frame["weekday"].iloc[0], frame["gap_days"].iloc[0]) == (0, 3)
	columns = column_dictionary(frame.reset_index(), RETURNS, TARGET).set_index("column")["meaning"]
	assert len(columns) == len(frame.columns) + 1
	assert columns["r_rod"] == "ROD simple return (predictor), previous 13:30 -> 13:00 CT: p_1300 / p_prev_1330 - 1"
	assert columns["r_lh"].startswith("LH simple return (target)")
