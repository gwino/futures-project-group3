"""Checks for the estimators, out-of-sample rules, and paper comparison of the replication (#29).

Run from the repository root: python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.replicate_regressions import (
	exclusion_mask, expanding_forecasts, hac_covariance, newey_west_lags, normal_quantile, ols, oos_statistics, paper_table,
)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(29)
	x = rng.normal(0, 0.02, 400)
	noise = rng.normal(0, 0.005, 400)
	return 0.0002 + 0.03 * x + noise + 0.5 * np.concatenate([[0], noise[:-1]]), x.reshape(-1, 1)


def test_ols_recovers_an_exact_line():
	x = np.linspace(-1, 1, 50).reshape(-1, 1)
	fit = ols(0.1 + 0.5 * x[:, 0], x, 3)
	assert fit["coefficients"] == pytest.approx([0.1, 0.5])
	assert fit["errors"] == pytest.approx([0, 0], abs=1e-12) and fit["r2"] == pytest.approx(1)


def test_newey_west_matches_a_direct_sum(data):
	y, x = data
	X = np.column_stack([np.ones(len(y)), x])
	residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
	n, k = X.shape
	for lags in (0, 3):
		# Written out term by term: sum_t sum_s w(|t - s|) u_t u_s x_t x_s' for |t - s| <= lags.
		meat = np.zeros((k, k))
		for t in range(n):
			for s in range(max(0, t - lags), min(n, t + lags + 1)):
				meat += (1 - abs(t - s) / (lags + 1)) * residuals[t] * residuals[s] * np.outer(X[t], X[s])
		bread = np.linalg.inv(X.T @ X)
		assert hac_covariance(X, residuals, lags) == pytest.approx(bread @ meat @ bread * n / (n - k))


def test_automatic_lags_and_normal_quantile():
	assert (newey_west_lags(100), newey_west_lags(1436), newey_west_lags(8336)) == (4, 7, 10)
	assert normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-6)


def test_expanding_forecasts_use_only_the_past(data):
	y, x = data
	forecasts = expanding_forecasts(y, x, 300)
	assert forecasts["position"].tolist() == list(range(300, 400))
	row = forecasts.iloc[0]
	X = np.column_stack([np.ones(300), x[:300]])
	expected = np.r_[1, x[300]] @ np.linalg.lstsq(X, y[:300], rcond=None)[0]
	assert (row["forecast"], row["historical_mean"], row["actual"]) == pytest.approx((expected, y[:300].mean(), y[300]))
	# Changing the target from t onward leaves the forecast for t unchanged.
	shocked = y.copy()
	shocked[300:] += 1
	assert expanding_forecasts(shocked, x, 300)["forecast"].iloc[0] == pytest.approx(row["forecast"])


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")  # f_t is all zero in the first case
def test_oos_r2_and_clark_west_at_the_benchmarks():
	actual = np.array([0.01, -0.02, 0.015, 0.0, -0.005])
	mean = np.array([0.0, 0.002, -0.001, 0.001, 0.0])
	same = oos_statistics(pd.DataFrame({"actual": actual, "forecast": mean, "historical_mean": mean}), 0)
	assert same["oos_r2"] == pytest.approx(0) and same["clark_west_mean"] == pytest.approx(0)
	perfect = oos_statistics(pd.DataFrame({"actual": actual, "forecast": actual, "historical_mean": mean}), 0)
	assert perfect["oos_r2"] == pytest.approx(1)
	# f_t = (r - mean)^2 - [(r - forecast)^2 - (mean - forecast)^2] = 2 (r - mean)^2 when the forecast is perfect.
	assert perfect["clark_west_mean"] == pytest.approx(np.mean(2 * (actual - mean) ** 2))


def test_paper_standard_errors_are_implied_by_slope_and_t():
	paper = {"sample": "1987-2020", "rod": {"coefficients": {"r_rod": [0.67, 0.25]}, "adj_r2": 0.03, "oos_r2": -1.04}}
	row = paper_table(paper, 1.96).iloc[0]
	assert row["se_x100"] == pytest.approx(2.68)
	assert (row["ci_low_x100"], row["ci_high_x100"]) == pytest.approx((0.67 - 1.96 * 2.68, 0.67 + 1.96 * 2.68))


def test_exclusion_rules():
	sample = pd.DataFrame({"roll_flag": [True, False, False, False], "sessions_to_expiry": [2, 3, 4, pd.NA], "gap_days": [1, 3, 1, 1], "r_rod": [0.01, -0.2, 0.0, 0.02]})
	assert exclusion_mask(sample, {"column": "roll_flag"}).tolist() == [True, False, False, False]
	assert exclusion_mask(sample.astype({"sessions_to_expiry": "Int64"}), {"column": "sessions_to_expiry", "at_most": 3}).tolist() == [True, True, False, False]
	assert exclusion_mask(sample, {"column": "gap_days", "above": 1}).tolist() == [False, True, False, False]
	assert exclusion_mask(sample, {"column": "r_rod", "abs_quantile_above": 0.75}).tolist() == [False, True, False, False]
	dated = sample.assign(trade_date=pd.to_datetime(["2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12"]))
	assert exclusion_mask(dated, {"dates": ["2026-03-10"]}).tolist() == [False, True, False, False]


def test_importing_the_module_keeps_the_matplotlib_backend():
	# The Agg backend is set in main(), so notebooks that import the module keep inline figures.
	import importlib

	import matplotlib

	import src.replicate_regressions as module

	before = matplotlib.get_backend()
	importlib.reload(module)
	assert matplotlib.get_backend() == before
