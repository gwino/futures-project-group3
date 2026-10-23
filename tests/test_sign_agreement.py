"""Checks for the overnight/daytime sign-agreement analysis (#33).

Run from the repository root: python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.analyze_sign_agreement import add_terms, category_table, classify, fit_models, grid_table, strategy_positions

CONFIG = {
	"target": "r_lh", "components": {"on": "r_on", "day": "r_day"}, "rod": "r_rod",
	"categories": {"agree": ["ON+/DAY+", "ON-/DAY-"], "disagree": ["ON+/DAY-", "ON-/DAY+"]},
	"family_level": 0.05, "family_tests": 5,
}


def features(on: list[float], day: list[float], lh: list[float]) -> pd.DataFrame:
	on, day, lh = np.asarray(on), np.asarray(day), np.asarray(lh)
	return pd.DataFrame({"trade_date": pd.bdate_range("2021-01-04", periods=len(on)), "r_on": on, "r_day": day, "r_rod": (1 + on) * (1 + day) - 1, "r_lh": lh})


def test_categories_follow_the_signs_and_zero_returns_are_set_aside():
	frame = features([0.01, -0.01, 0.01, -0.01, 0.0, 0.02], [0.02, -0.02, -0.02, 0.02, 0.01, 0.0], [0.0] * 6)
	assert classify(frame, "r_on", "r_day").tolist() == ["ON+/DAY+", "ON-/DAY-", "ON+/DAY-", "ON-/DAY+", "zero", "zero"]
	terms = add_terms(frame, CONFIG)
	assert terms["agree"].tolist() == [1, 1, 0, 0, 0, 0] and terms["disagree"].tolist() == [0, 0, 1, 1, 0, 0]
	assert terms["on_x_day"].iloc[0] == pytest.approx(0.0002) and terms["rod_x_agree"].iloc[2] == 0.0


def test_strategy_return_and_hit_rate_follow_the_sign_of_rod():
	frame = add_terms(features([0.01, -0.01], [0.02, -0.02], [0.003, 0.004]), CONFIG)
	assert frame["eta"].tolist() == pytest.approx([0.003, -0.004])
	assert frame["lh_follows_rod"].tolist() == [1.0, 0.0]


def test_category_table_means_and_the_agree_minus_disagree_difference():
	# Agreement days: strategy return +10 bps each; disagreement days: -10 bps each. Difference = 20 bps.
	on = [0.01, -0.01, 0.01, -0.01] * 10
	day = [0.02, -0.02, -0.02, 0.02] * 10
	lh = [0.001, -0.001, 0.001, -0.001] * 10   # ROD signs: +, -, -, + -> eta: +10, +10, -10, -10 bps
	table = category_table(add_terms(features(on, day, lh), CONFIG), CONFIG, 0, 1.96).set_index("group")
	assert table.loc["agree", "n"] == 20 and table.loc["disagree", "n"] == 20
	assert table.loc["agree", "eta_bps"] == pytest.approx(10.0) and table.loc["disagree", "eta_bps"] == pytest.approx(-10.0)
	assert table.loc["agree - disagree", "eta_bps"] == pytest.approx(20.0)
	assert table.loc["ON+/DAY+", "mean_lh_bps"] == pytest.approx(10.0) and table.loc["ON-/DAY+", "mean_lh_bps"] == pytest.approx(-10.0)
	assert table.loc["agree", "lh_follows_rod"] == 1.0 and table.loc["disagree", "lh_follows_rod"] == 0.0
	assert table.loc["zero", "n"] == 0 and table.loc["agree - disagree", "bonferroni_level"] == pytest.approx(0.01)


def test_grid_cells_cover_every_date_once():
	rng = np.random.default_rng(0)
	frame = add_terms(features(rng.normal(size=300), rng.normal(size=300), rng.normal(size=300) / 100), CONFIG)
	grid = grid_table(frame, 3, 20, 0, 1.96)
	assert len(grid) == 9 and grid["n"].sum() == 300 and not grid["sparse"].any()
	cell = grid[grid["on_bin"].eq("high") & grid["day_bin"].eq("high")].iloc[0]
	selected = frame[(frame["on"] > frame["on"].quantile(2 / 3)) & (frame["day"] > frame["day"].quantile(2 / 3))]
	assert cell["n"] == len(selected) and cell["mean_lh_bps"] == pytest.approx(selected["y"].mean() * 1e4)


def test_interaction_model_recovers_a_planted_agreement_effect():
	rng = np.random.default_rng(1)
	n = 1200
	on, day = rng.normal(scale=0.02, size=n), rng.normal(scale=0.015, size=n)
	rod = (1 + on) * (1 + day) - 1
	agree = (np.sign(on) == np.sign(day)).astype(float)
	lh = 0.0 + 0.01 * rod - 0.06 * rod * agree + rng.normal(scale=0.002, size=n)
	frame = add_terms(pd.DataFrame({"trade_date": pd.bdate_range("2021-01-04", periods=n), "r_on": on, "r_day": day, "r_rod": rod, "r_lh": lh}), CONFIG)
	models = {"rod": ["rod"], "rod_by_agreement": ["rod", "agree", "rod_x_agree"]}
	terms, stats, forecasts = fit_models(frame, models, 0, 1.96, 100, 500, [["rod_by_agreement", "rod"]])
	interaction = terms[terms["term"].eq("rod_x_agree")].iloc[0]
	assert interaction["coef_x100"] == pytest.approx(-6.0, abs=1.5) and interaction["p_value"] < 1e-4
	row = stats.set_index("model").loc["rod_by_agreement"]
	assert row["wald_df"] == 1 and row["wald_p"] < 1e-4 and row["nested_in"] == "rod"
	assert row["oos_r2_vs_nested_pct"] > 0 and row["oos_forecasts"] == 700
	assert forecasts["trade_date"].min() == frame["trade_date"].iloc[500]


def test_strategy_positions_respect_the_sign_filters():
	frame = add_terms(features([0.01, -0.01, 0.01, -0.01, 0.0], [0.02, -0.02, -0.02, 0.02, 0.01], [0.0] * 5), CONFIG)
	assert strategy_positions(frame, {"rule": "sign_of_rod"}).tolist() == [1, -1, -1, 1, 1]
	assert strategy_positions(frame, {"rule": "sign_of_rod", "when": "agree"}).tolist() == [1, -1, 0, 0, 0]
	assert strategy_positions(frame, {"rule": "sign_of_rod", "when": "disagree"}).tolist() == [0, 0, -1, 1, 0]
	assert strategy_positions(frame, {"rule": "sign_of_day"}).tolist() == [1, -1, -1, 1, 1]
	assert strategy_positions(frame, {"rule": "sign_of_on"}).tolist() == [1, -1, 1, -1, 0]


def test_group_means_equal_plain_means_and_zero_days_and_ties_are_set_aside():
	from src.analyze_sign_agreement import group_means, hit_rate, regression_sample

	frame = add_terms(features([0.01, -0.01, 0.01, -0.01, 0.0, 0.02], [0.02, -0.02, -0.02, 0.02, 0.01, 0.03], [0.001, -0.002, 0.0, 0.004, 0.005, -0.001]), CONFIG)
	kept = regression_sample(frame)
	assert kept["category"].tolist() == ["ON+/DAY+", "ON-/DAY-", "ON+/DAY-", "ON-/DAY+", "ON+/DAY+"]
	means = group_means(kept, kept["category"], "y", 0)
	for category, rows in kept.groupby("category"):
		assert means.loc[category, "mean"] == pytest.approx(rows["y"].mean())
	share, se, decided = hit_rate(frame["lh_follows_rod"])
	assert decided == 5 and np.isnan(frame.loc[2, "lh_follows_rod"])   # LH = 0 on the third day: neither hit nor miss
	assert share == pytest.approx(frame["lh_follows_rod"].dropna().mean()) and se == pytest.approx(np.sqrt(share * (1 - share) / 5))
