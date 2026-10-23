"""Checks for the overnight-versus-daytime predictability analysis (#31).

Run from the repository root: python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.analyze_on_day import fit_models, load_config, prepare

CONFIG = {
	"scale": 100,
	"models": {"on": ["on"], "day": ["day"], "on_day": ["on", "day"], "rod": ["rod"]},
	"nested": [["on_day", "on"], ["on_day", "day"]],
	"equality_test": {"model": "on_day", "terms": ["on", "day"]},
}


def sample(n: int = 1500, seed: int = 0, beta_on: float = 0.0, beta_day: float = -0.05, noise: float = 0.003) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	on, day = rng.normal(scale=0.02, size=n), rng.normal(scale=0.015, size=n)
	lh = beta_on * on + beta_day * day + rng.normal(scale=noise, size=n)
	frame = pd.DataFrame({"trade_date": pd.bdate_range("2021-01-04", periods=n), "r_on": on, "r_day": day, "r_rod": (1 + on) * (1 + day) - 1, "r_lh": lh})
	return prepare(frame, "r_lh", {"on": "r_on", "day": "r_day", "rod": "r_rod"})


def test_prepare_maps_the_columns_and_keeps_the_dates():
	frame = sample(5)
	assert list(frame.columns) == ["trade_date", "y", "on", "day", "rod"]
	assert frame["rod"].iloc[0] == pytest.approx((1 + frame["on"].iloc[0]) * (1 + frame["day"].iloc[0]) - 1)


def test_joint_model_recovers_a_daytime_only_effect_and_rejects_equal_slopes():
	terms, models, forecasts = fit_models(sample(), CONFIG, 0, 1.96, 500)
	joint = terms[terms["model"].eq("on_day")].set_index("term")
	assert joint.loc["day", "coef_x100"] == pytest.approx(-5.0, abs=0.8) and joint.loc["on", "coef_x100"] == pytest.approx(0.0, abs=0.8)
	assert joint.loc["day", "p_value"] < 1e-6 and joint.loc["on", "p_value"] > 0.05
	row = models.set_index("model").loc["on_day"]
	assert row["equality_difference_x100"] == pytest.approx(joint.loc["on", "coef_x100"] - joint.loc["day", "coef_x100"])
	assert row["equality_p"] < 1e-6
	# DAY adds information to ON out of sample, ON adds nothing to DAY.
	assert row["oos_r2_vs_on_pct"] > 1 and row["clark_west_p_vs_on"] < 0.01
	assert abs(row["oos_r2_vs_day_pct"]) < 1
	assert row["adj_r2_gain_vs_on_pct"] > 0
	assert len(forecasts) == 4 * 1000 and forecasts["trade_date"].min() == sample()["trade_date"].iloc[500]


def test_standardized_slope_is_the_response_per_standard_deviation():
	frame = sample(400)
	terms, _, _ = fit_models(frame, CONFIG, 0, 1.96, None)
	day = terms[terms["model"].eq("day") & terms["term"].eq("day")].iloc[0]
	expected = day["coef_x100"] / 100 * frame["day"].std(ddof=1) / frame["y"].std(ddof=1)
	assert day["standardized"] == pytest.approx(expected)
	assert day["response_bps_per_sd"] == pytest.approx(day["coef_x100"] / 100 * frame["day"].std(ddof=1) * 1e4)
	assert np.isnan(terms[terms["term"].eq("const")]["standardized"]).all()


def test_log_decomposition_is_exactly_additive():
	frame = sample(50)
	log_on, log_day, log_rod = np.log1p(frame["on"]), np.log1p(frame["day"]), np.log1p(frame["rod"])
	assert np.abs(log_on + log_day - log_rod).max() < 1e-12


def test_config_rejects_unknown_predictors_and_non_nested_pairs(tmp_path):
	import yaml

	base = {
		"return_features": "x", "reference_config": "x", "front_minutes": "x", "roll_comparison": "x", "return_config": "x", "tables_dir": "x", "figures_dir": "x",
		"target": "r_lh", "predictors": {"on": "r_on", "day": "r_day", "rod": "r_rod"}, "log_variant": {"target": "log_lh", "predictors": {"on": "log_on", "day": "log_day", "rod": "log_rod"}},
		"scale": 100, "models": {"on": ["on"], "on_day": ["on", "day"]}, "nested": [["on_day", "on"]], "equality_test": {"model": "on_day", "terms": ["on", "day"]},
		"inference": {}, "oos": {}, "rolling": {}, "robustness": {},
	}
	path = tmp_path / "ok.yaml"
	path.write_text(yaml.safe_dump(base), encoding="utf-8")
	load_config(path)
	bad = dict(base, models={"on": ["on"], "on_day": ["on", "night"]})
	path.write_text(yaml.safe_dump(bad), encoding="utf-8")
	with pytest.raises(ValueError, match="unknown predictors"):
		load_config(path)
	bad = dict(base, nested=[["on", "on_day"]])
	path.write_text(yaml.safe_dump(bad), encoding="utf-8")
	with pytest.raises(ValueError, match="not nested"):
		load_config(path)
