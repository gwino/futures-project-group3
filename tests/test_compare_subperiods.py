"""Checks for the subperiod split, the pooled slope-difference test and the chi-square tail (#38).

Run from the repository root: python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.compare_subperiods import chi2_sf, fit_models, lag_length, oos_by_period, period_mask, pooled_difference, regression, variant_mask
from src.replicate_regressions import expanding_forecasts, oos_statistics


@pytest.mark.parametrize("df, critical", [(1, 3.841459), (2, 5.991465), (3, 7.814728), (4, 9.487729), (5, 11.070498)])
def test_chi2_tail_at_the_five_percent_critical_values(df, critical):
	assert chi2_sf(critical, df) == pytest.approx(0.05, abs=1e-6)


def test_chi2_tail_at_zero_and_large_values():
	assert chi2_sf(0.0, 3) == pytest.approx(1.0)
	assert chi2_sf(100.0, 2) == pytest.approx(np.exp(-50))


@pytest.fixture(scope="module")
def two_periods() -> pd.DataFrame:
	rng = np.random.default_rng(38)
	n = 2000
	dates = pd.bdate_range("2018-06-01", periods=n)
	x1, x2 = rng.normal(0, 0.02, n), rng.normal(0, 0.01, n)
	later = dates >= pd.Timestamp("2024-01-01")
	slope = np.where(later, -0.03, 0.01)
	y = 0.0001 + slope * x1 + 0.02 * x2 + rng.normal(0, 0.001, n)
	return pd.DataFrame({"trade_date": dates, "r_lh": y, "x1": x1, "x2": x2})


def test_pooled_slopes_equal_the_separate_period_regressions(two_periods):
	later = period_mask(two_periods, {"start": "2024-01-01", "end": None})
	rows, wald = pooled_difference(two_periods, later, ["x1", "x2"], "r_lh", 3, 1.96, 100)
	assert [row["term"] for row in rows] == ["const", "x1", "x2"]
	for part, column in ((~later, "earlier_x100"), (later, "later_x100")):
		sample = two_periods[part]
		separate = regression(sample["r_lh"].to_numpy(), sample[["x1", "x2"]].to_numpy(), 3)["coefficients"] * 100
		assert [row[column] for row in rows] == pytest.approx(list(separate))
	for row in rows:
		assert row["difference_x100"] == pytest.approx(row["later_x100"] - row["earlier_x100"])
	assert (wald["wald_df"], wald["n"], wald["n_later"]) == (2, 2000, int(later.sum()))
	# The simulated slope on x1 falls from 0.01 to -0.03 and the one on x2 does not change.
	assert rows[1]["difference_x100"] == pytest.approx(-4, abs=0.5) and rows[1]["difference_p"] < 0.01
	assert rows[2]["difference_p"] > 0.01


def test_periods_include_both_end_dates():
	frame = pd.DataFrame({"trade_date": pd.to_datetime(["2023-12-29", "2024-01-02", "2026-10-07"])})
	assert period_mask(frame, {"start": "2021-01-01", "end": "2023-12-31"}).tolist() == [True, False, False]
	assert period_mask(frame, {"start": "2024-01-01", "end": None}).tolist() == [False, True, True]


def test_variants_and_lag_rules():
	frame = pd.DataFrame({"trade_date": pd.to_datetime(["2025-12-31", "2026-01-02", "2026-03-10"]), "r_rod": [0.01, -0.02, -0.15]})
	assert variant_mask(frame, {}).tolist() == [False, False, False]
	assert variant_mask(frame, {"on_or_after": "2026-01-01"}).tolist() == [False, True, True]
	assert variant_mask(frame, {"dates": ["2026-03-10"]}).tolist() == [False, False, True]
	assert (lag_length("common_7", 747, 7), lag_length("auto_n", 747, 7), lag_length("auto_n", 1436, 7)) == (7, 6, 7)


def test_model_wald_is_the_squared_t_for_one_slope_and_r2_matches(two_periods):
	terms, fits = fit_models(two_periods, {"one": ["x1"], "two": ["x1", "x2"]}, "r_lh", 3, 1.96, 100)
	slope = next(row for row in terms if row["model"] == "one" and row["term"] == "x1")
	one = next(row for row in fits if row["model"] == "one")
	assert one["wald_all_slopes"] == pytest.approx(slope["t_stat"] ** 2) and one["wald_df"] == 1
	two = next(row for row in fits if row["model"] == "two")
	y, X = two_periods["r_lh"].to_numpy(), np.column_stack([np.ones(len(two_periods)), two_periods[["x1", "x2"]].to_numpy()])
	residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
	assert two["r2_pct"] == pytest.approx(100 * (1 - residuals @ residuals / ((y - y.mean()) @ (y - y.mean()))))
	assert two["wald_df"] == 2 and two["wald_p"] < 1e-6


def test_oos_split_uses_only_the_forecasts_in_each_period(two_periods):
	periods = {"late": {"start": "2024-01-01", "end": None}}
	rows, forecasts = oos_by_period(two_periods, {"one": ["x1"]}, "r_lh", 500, periods, 3, 100)
	frame = expanding_forecasts(two_periods["r_lh"].to_numpy(), two_periods[["x1"]].to_numpy(), 500)
	dates = two_periods["trade_date"].iloc[frame["position"]].to_numpy()
	expected = oos_statistics(frame[dates >= np.datetime64("2024-01-01")], 3)
	assert rows[0]["oos_forecasts"] == expected["oos_forecasts"]
	assert rows[0]["oos_r2_pct"] == pytest.approx(expected["oos_r2"] * 100) and rows[0]["clark_west_t"] == pytest.approx(expected["clark_west_t"])
	again, _ = oos_by_period(two_periods, {"one": ["x1"]}, "r_lh", 500, periods, 3, 100, forecasts)
	assert again == rows
