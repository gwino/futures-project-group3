"""Test whether overnight/daytime sign agreement changes last-half-hour momentum on CL (#33).

Run from the repository root, after the return features (#28) and the minute pipeline (#27):

    python -m src.analyze_sign_agreement --config config/sign_agreement.yaml

At 13:00 CT the overnight return ON (previous 13:30 -> 08:00) and the daytime return DAY (08:00 ->
13:00) are known. Days are classified by their signs (ON+/DAY+, ON-/DAY-, ON+/DAY-, ON-/DAY+); the
first two are agreement, the other two disagreement, and the few days on which a return is exactly
zero are set aside from every group comparison and regression. The last-half-hour return LH, the
direction hit rate and the paper's sign-of-ROD strategy return are compared across the groups, with
Newey-West standard errors taken from one regression of the chronological series on group
indicators, so the lags run over consecutive sample dates. Coherence is also treated as the
two-dimensional vector (ON, DAY): the mean LH on a grid of ON and DAY quantile bins, and regressions
with the product ON x DAY (a nonlinear term, not a directional one) and with agreement
interactions, evaluated in sample and with expanding-window forecasts. Sign-filtered strategies are
priced with the #30 execution and cost settings.

Outputs (tables in reports/tables, figures in reports/figures):
- sign_agreement_categories.csv: per category and pooled agree/disagree: counts, mean LH, hit rates,
  strategy return, with standard errors; sign_agreement_grid.csv: the ON x DAY bin grid.
- sign_agreement_regressions.csv, sign_agreement_oos.csv, sign_agreement_oos_forecasts.csv.
- sign_agreement_strategies.csv and sign_agreement_strategy_by_year.csv: trading comparison.
- sign_agreement_by_year.csv; sign_agreement_robustness*.csv: excluded dates, the pre-expiry roll
  rule, log returns and the zero-day coding, with out-of-sample statistics per variant.
- sign_agreement_*.png and sign_agreement.manifest.json.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml

from src.backtest_lh_strategy import apply_costs, build_ledger, evaluate, fills
from src.compare_subperiods import chi2_sf, regression
from src.prepare_daily_data import sha256
from src.replicate_regressions import exclusion_mask, expanding_forecasts, hac_covariance, newey_west_lags, normal_cdf, normal_quantile, oos_statistics, pre_expiry_features

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#6250d6"
CATEGORY_ORDER = ["ON+/DAY+", "ON-/DAY-", "ON+/DAY-", "ON-/DAY+"]
STRATEGY_LABELS = {"rod": "Sign of ROD (baseline)", "agree_only": "Sign of ROD, agreement days only", "disagree_only": "Sign of ROD, disagreement days only", "day_sign": "Sign of DAY", "on_sign": "Sign of ON"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/sign_agreement.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> tuple[dict, dict]:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {
		"return_features", "reference_prices", "front_minutes", "reference_config", "roll_comparison", "return_config", "strategy_config", "tables_dir", "figures_dir",
		"target", "components", "rod", "log_variant", "scale", "inference", "categories", "grid", "models", "nested", "strategies", "baseline_strategy", "oos",
		"robustness", "family_tests", "family_level",
	}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	for group in (config["components"], config["log_variant"]["components"]):
		names = [name for name in group if not isinstance(name, str)]
		if names:
			raise ValueError(f"Component names must be strings, got {names}; quote YAML keys such as \"on\".")
	if set(config["components"]) != {"on", "day"}:
		raise ValueError("components must name exactly on and day.")
	for larger, smaller in config["nested"]:
		if not set(config["models"][smaller]) < set(config["models"][larger]):
			raise ValueError(f"{smaller} is not nested in {larger}.")
	with Path(config["strategy_config"]).open(encoding="utf-8") as strategy_file:
		strategy = yaml.safe_load(strategy_file)
	return config, strategy


# --- Classification ---------------------------------------------------------------------------


def classify(frame: pd.DataFrame, on: str, day: str) -> pd.Series:
	"""Sign category of each date: ON+/DAY+, ON-/DAY-, ON+/DAY-, ON-/DAY+, or 'zero' when a return is exactly 0."""
	on_sign, day_sign = np.sign(frame[on].to_numpy()), np.sign(frame[day].to_numpy())
	label = np.where(on_sign > 0, "ON+", "ON-") + "/" + np.where(day_sign > 0, "DAY+", "DAY-")
	return pd.Series(np.where((on_sign == 0) | (day_sign == 0), "zero", label), index=frame.index, name="category")


def add_terms(frame: pd.DataFrame, config: dict, labels: dict | None = None) -> pd.DataFrame:
	"""Columns on, day, rod, agree, the interactions, the strategy return eta, the hit indicators and the sign category.

	`agree` and `disagree` are 0 on zero-return days; those days are dropped before any regression
	or group comparison (see regression_sample), so the coding never defines a reference group.
	The hit indicators are missing when LH is exactly zero, so ties are neither hits nor misses.
	"""
	target, components, rod = config["target"], config["components"], config["rod"]
	if labels is not None:
		target, components, rod = labels["target"], labels["components"], labels["rod"]
	out = frame.copy()
	out["on"], out["day"], out["rod"], out["y"] = frame[components["on"]], frame[components["day"]], frame[rod], frame[target]
	out["category"] = classify(out, "on", "day")
	out["agree"] = out["category"].isin(config["categories"]["agree"]).astype(float)
	out["disagree"] = out["category"].isin(config["categories"]["disagree"]).astype(float)
	out["rod_x_agree"], out["on_x_agree"], out["day_x_agree"] = out["rod"] * out["agree"], out["on"] * out["agree"], out["day"] * out["agree"]
	out["on_x_day"] = out["on"] * out["day"]
	out["eta"] = np.sign(out["rod"]) * out["y"]
	tie = out["y"].eq(0)
	out["lh_follows_rod"] = pd.Series(np.where(tie, np.nan, (np.sign(out["y"]) == np.sign(out["rod"])).astype(float)), index=out.index)
	out["lh_follows_day"] = pd.Series(np.where(tie, np.nan, (np.sign(out["y"]) == np.sign(out["day"])).astype(float)), index=out.index)
	return out


def regression_sample(frame: pd.DataFrame) -> pd.DataFrame:
	"""The dates with a sign category: zero-return days are set aside."""
	return frame[frame["category"].ne("zero")].reset_index(drop=True)


def variant_mask(frame: pd.DataFrame, rule: dict) -> pd.Series:
	"""True for the dates a robustness rule drops: the #29 rules plus a list of calendar years."""
	if "years" in rule:
		return frame["trade_date"].dt.year.isin(rule["years"])
	return exclusion_mask(frame, rule)


# --- Group statistics -------------------------------------------------------------------------


def group_means(frame: pd.DataFrame, labels: pd.Series, value: str, lags: int) -> pd.DataFrame:
	"""Mean of `value` per label with Newey-West standard errors, from one regression on the label indicators.

	The regression runs over the chronological sample with one indicator per label and no intercept,
	so the coefficients are the group means and the Bartlett lags run over consecutive sample dates
	rather than over the dates of a single group.
	"""
	names = [name for name in pd.unique(labels) if pd.notna(name)]
	X = np.column_stack([labels.eq(name).to_numpy(dtype=float) for name in names])
	y = frame[value].to_numpy(dtype=float)
	coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
	covariance = hac_covariance(X, y - X @ coefficients, lags)
	return pd.DataFrame({"mean": coefficients, "se": np.sqrt(np.diag(covariance))}, index=names)


def hit_rate(values: pd.Series) -> tuple[float, float, int]:
	"""Share of hits among non-tied days and its binomial standard error."""
	valid = values.dropna()
	if valid.empty:
		return np.nan, np.nan, 0
	share = valid.mean()
	return share, math.sqrt(share * (1 - share) / len(valid)), len(valid)


def group_statistics(frame: pd.DataFrame, labels: pd.Series, lags: int, z: float) -> dict[str, dict]:
	"""Counts, mean LH, hit rates and the sign-of-ROD strategy return for each label, keyed by label."""
	lh, eta = group_means(frame, labels, "y", lags), group_means(frame, labels, "eta", lags)
	rows = {}
	for name in lh.index:
		sub = frame[labels.eq(name)]
		follows, follows_se, decided = hit_rate(sub["lh_follows_rod"])
		follows_day, follows_day_se, _ = hit_rate(sub["lh_follows_day"])
		mean_lh, se_lh, mean_eta, se_eta = lh.loc[name, "mean"], lh.loc[name, "se"], eta.loc[name, "mean"], eta.loc[name, "se"]
		rows[name] = {
			"n": len(sub), "mean_lh_bps": mean_lh * 1e4, "se_lh_bps": se_lh * 1e4, "ci_low_lh_bps": (mean_lh - z * se_lh) * 1e4, "ci_high_lh_bps": (mean_lh + z * se_lh) * 1e4,
			"sd_lh_pct": sub["y"].std() * 100, "mean_rod_pct": sub["rod"].mean() * 100, "mean_on_pct": sub["on"].mean() * 100, "mean_day_pct": sub["day"].mean() * 100,
			"lh_follows_rod": follows, "lh_follows_rod_se": follows_se, "decided_days": decided, "lh_follows_day": follows_day, "lh_follows_day_se": follows_day_se,
			"eta_bps": mean_eta * 1e4, "se_eta_bps": se_eta * 1e4, "ci_low_eta_bps": (mean_eta - z * se_eta) * 1e4, "ci_high_eta_bps": (mean_eta + z * se_eta) * 1e4,
			"t_eta": mean_eta / se_eta if se_eta else np.nan,
		}
	return rows


def category_table(frame: pd.DataFrame, config: dict, lags: int, z: float) -> pd.DataFrame:
	"""One row per sign category, the pooled agree and disagree groups, the zero days, and the agree - disagree difference.

	Group means and their HAC standard errors come from indicator regressions on the chronological
	sample (see group_means). The zero-return days are reported for completeness and take no part in
	the pooled groups or the difference.
	"""
	by_category = group_statistics(frame, frame["category"], lags, z)
	pooled_labels = pd.Series(np.where(frame["agree"].eq(1), "agree", np.where(frame["disagree"].eq(1), "disagree", "zero")), index=frame.index)
	pooled = group_statistics(frame, pooled_labels, lags, z)
	rows = [{"group": category, "kind": "category", **by_category[category]} for category in CATEGORY_ORDER if category in by_category]
	rows += [{"group": name, "kind": "pooled", **pooled[name]} for name in ("agree", "disagree") if name in pooled]
	zero_count = int(frame["category"].eq("zero").sum())
	rows.append({"group": "zero", "kind": "excluded", **(by_category["zero"] if zero_count > 2 else {"n": zero_count})})
	both = regression_sample(frame)
	fit = regression(both["eta"].to_numpy(dtype=float), both[["agree"]].to_numpy(dtype=float), lags)
	diff, se = fit["coefficients"][1], math.sqrt(fit["covariance"][1, 1])
	decided = both[both["lh_follows_rod"].notna()]
	hit_fit = regression(decided["lh_follows_rod"].to_numpy(dtype=float), decided[["agree"]].to_numpy(dtype=float), lags)
	rows.append(
		{
			"group": "agree - disagree", "kind": "difference", "n": len(both), "eta_bps": diff * 1e4, "se_eta_bps": se * 1e4,
			"ci_low_eta_bps": (diff - z * se) * 1e4, "ci_high_eta_bps": (diff + z * se) * 1e4, "t_eta": diff / se, "p_eta": 2 * (1 - normal_cdf(abs(diff / se))),
			"lh_follows_rod": hit_fit["coefficients"][1], "lh_follows_rod_se": math.sqrt(hit_fit["covariance"][1, 1]), "decided_days": len(decided),
			"bonferroni_level": config["family_level"] / config["family_tests"],
		}
	)
	return pd.DataFrame(rows)


def grid_table(frame: pd.DataFrame, bins: int, min_cell: int, lags: int, z: float) -> pd.DataFrame:
	"""Mean LH, its standard error and the count on a grid of ON and DAY quantile bins (the two-dimensional view)."""
	labels = {3: ["low", "mid", "high"]}.get(bins, [f"q{index + 1}" for index in range(bins)])
	on_bin = pd.qcut(frame["on"], bins, labels=labels).astype(str)
	day_bin = pd.qcut(frame["day"], bins, labels=labels).astype(str)
	edges = {"on": pd.qcut(frame["on"], bins, retbins=True)[1] * 100, "day": pd.qcut(frame["day"], bins, retbins=True)[1] * 100}
	cells = on_bin + "|" + day_bin
	stats = group_statistics(frame, cells, lags, z)
	rows = []
	for on_label in labels:
		for day_label in labels:
			cell = stats.get(f"{on_label}|{day_label}", {"n": 0})
			rows.append(
				{
					"on_bin": on_label, "day_bin": day_label, "n": cell["n"], "sparse": cell["n"] < min_cell,
					"on_range_pct": f"[{edges['on'][labels.index(on_label)]:.2f}, {edges['on'][labels.index(on_label) + 1]:.2f}]",
					"day_range_pct": f"[{edges['day'][labels.index(day_label)]:.2f}, {edges['day'][labels.index(day_label) + 1]:.2f}]",
					**{key: cell.get(key, np.nan) for key in ("mean_lh_bps", "se_lh_bps", "ci_low_lh_bps", "ci_high_lh_bps", "eta_bps", "se_eta_bps", "lh_follows_rod")},
				}
			)
	return pd.DataFrame(rows)


# --- Regressions ------------------------------------------------------------------------------


def fit_models(frame: pd.DataFrame, models: dict, lags: int, z: float, scale: float, min_train: int | None, nested: list, variant: str = "baseline") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Term table, model-level table (with Wald tests on interaction terms and OOS statistics) and OOS forecasts.

	The HAC covariance is computed once per model (compare_subperiods.regression) and used for the
	standard errors and the Wald test alike.
	"""
	y = frame["y"].to_numpy(dtype=float)
	term_rows, model_rows, forecasts, fitted = [], [], [], {}
	for model, predictors in models.items():
		X = frame[predictors].to_numpy(dtype=float)
		fit = regression(y, X, lags)
		errors = np.sqrt(np.diag(fit["covariance"]))
		terms = ["const", *predictors]
		stats = {"variant": variant, "model": model, "n": fit["n"], "adj_r2_pct": fit["adj_r2"] * scale, "hac_lags": lags}
		interactions = [index for index, term in enumerate(terms) if "_x_" in term]
		if interactions:
			gamma = fit["coefficients"][interactions]
			stats["wald_stat"] = float(gamma @ np.linalg.solve(fit["covariance"][np.ix_(interactions, interactions)], gamma))
			stats["wald_df"] = len(interactions)
			stats["wald_p"] = chi2_sf(stats["wald_stat"], len(interactions))
		if min_train is not None and len(y) > min_train:
			rows = expanding_forecasts(y, X, min_train)
			oos = oos_statistics(rows, lags)
			stats.update({"oos_forecasts": oos["oos_forecasts"], "first_forecast": frame["trade_date"].iloc[min_train].date(), "oos_r2_pct": oos["oos_r2"] * scale, "clark_west_t": oos["clark_west_t"], "clark_west_p": oos["clark_west_p"]})
			fitted[model] = rows
			forecasts.append(rows.assign(model=model, trade_date=frame["trade_date"].iloc[rows["position"]].to_numpy()).drop(columns="position"))
		for index, term in enumerate(terms):
			coefficient, error = fit["coefficients"][index], errors[index]
			term_rows.append(
				{
					"variant": variant, "model": model, "term": term, "coef_x100": coefficient * scale, "se_x100": error * scale,
					"ci_low_x100": (coefficient - z * error) * scale, "ci_high_x100": (coefficient + z * error) * scale,
					"t_stat": coefficient / error, "p_value": 2 * (1 - normal_cdf(abs(coefficient / error))), "n": fit["n"], "adj_r2_pct": fit["adj_r2"] * scale,
				}
			)
		model_rows.append(stats)
	model_table = pd.DataFrame(model_rows)
	if fitted:
		for larger, smaller in nested:
			if larger in fitted and smaller in fitted:
				paired = fitted[larger].assign(historical_mean=fitted[smaller]["forecast"].to_numpy())
				compare = oos_statistics(paired, lags)
				index = model_table.index[model_table["model"].eq(larger)][0]
				model_table.loc[index, "nested_in"] = smaller
				model_table.loc[index, "oos_r2_vs_nested_pct"] = compare["oos_r2"] * scale
				model_table.loc[index, "clark_west_t_vs_nested"] = compare["clark_west_t"]
				model_table.loc[index, "clark_west_p_vs_nested"] = compare["clark_west_p"]
	return pd.DataFrame(term_rows), model_table, pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()


# --- Trading comparison -----------------------------------------------------------------------


def strategy_positions(frame: pd.DataFrame, spec: dict) -> pd.Series:
	"""Position from signs known at 13:00 CT: the sign of ROD, DAY or ON, optionally only on agreement or disagreement days."""
	source = {"sign_of_rod": "rod", "sign_of_day": "day", "sign_of_on": "on"}[spec["rule"]]
	position = np.sign(frame[source].to_numpy(dtype=float)).astype(int)
	if spec.get("when") == "agree":
		position = np.where(frame["agree"].eq(1), position, 0)
	elif spec.get("when") == "disagree":
		position = np.where(frame["disagree"].eq(1), position, 0)
	return pd.Series(position, index=frame.index, dtype=int)


def trading_tables(frame: pd.DataFrame, references: pd.DataFrame, minutes: pd.DataFrame, config: dict, strategy_cfg: dict, timezone_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
	"""Frictionless, gross and net metrics for each sign-filtered strategy, priced with the #30 baseline execution variant and costs."""
	variant = strategy_cfg["execution_variants"][strategy_cfg["baseline_execution"]]
	execution = {**strategy_cfg["execution"], **{key: value for key, value in variant.items() if key in ("entry_time", "exit_time", "max_delay_minutes", "settlement_proxy_minutes")}}
	costs = strategy_cfg["costs"]
	evaluation = dict(strategy_cfg["evaluation"])
	evaluation["hac_lags"] = newey_west_lags(len(frame)) if evaluation["hac_lags"] == "auto" else int(evaluation["hac_lags"])
	refs = references[references["trade_date"].isin(frame["trade_date"])].sort_values("trade_date").reset_index(drop=True)
	held = refs.set_index("trade_date")["instrument_id"]
	entry = fills(variant["entry"], execution["entry_time"], refs, minutes, held, timezone_name, execution)
	exit = fills(variant["exit"], execution["exit_time"], refs, minutes, held, timezone_name, execution)
	indexed = frame.set_index("trade_date")
	summary_rows, year_rows, ledgers = [], [], {}
	for name, spec in config["strategies"].items():
		position = strategy_positions(indexed, spec).reindex(refs["trade_date"]).fillna(0).astype(int)
		position.index = refs["trade_date"]
		predictor = indexed["rod"].reindex(refs["trade_date"])
		predictor.index = refs["trade_date"]
		ledger = apply_costs(build_ledger(refs, position, predictor, entry, exit, name, execution), costs["commission_per_side_usd"], costs["slippage_ticks_per_side"], execution, variant.get("exit_slippage_ticks_per_side"))
		ledgers[name] = ledger
		for accounting, return_column, pnl_column in (("frictionless", "r_frictionless", None), ("gross", "r_gross", "pnl_gross_usd"), ("net", "r_net", "pnl_net_usd")):
			summary_rows.append({"strategy": name, "accounting": accounting, **evaluate(ledger, return_column, pnl_column, evaluation)})
		for year, group in ledger.groupby(ledger["trade_date"].dt.year):
			year_rows.append({"strategy": name, "year": year, **evaluate(group.reset_index(drop=True), "r_net", "pnl_net_usd", evaluation)})
	return pd.DataFrame(summary_rows), pd.DataFrame(year_rows), ledgers


# --- Stability --------------------------------------------------------------------------------


def by_year_table(frame: pd.DataFrame, lags: int, z: float) -> pd.DataFrame:
	rows = []
	for year, group in frame.groupby(frame["trade_date"].dt.year):
		group = group.reset_index(drop=True)
		labels = pd.Series(np.where(group["agree"].eq(1), "agree", np.where(group["disagree"].eq(1), "disagree", "zero")), index=group.index)
		stats = group_statistics(group, labels, min(lags, len(group) - 4), z)
		rows += [{"year": year, "group": name, **stats[name]} for name in ("agree", "disagree") if name in stats]
		both = regression_sample(group)
		fit = regression(both["eta"].to_numpy(dtype=float), both[["agree"]].to_numpy(dtype=float), min(lags, len(both) - 3))
		se = math.sqrt(fit["covariance"][1, 1])
		rows.append({"year": year, "group": "agree - disagree", "n": len(both), "eta_bps": fit["coefficients"][1] * 1e4, "se_eta_bps": se * 1e4, "t_eta": fit["coefficients"][1] / se})
	return pd.DataFrame(rows)


# --- Figures ----------------------------------------------------------------------------------


def style_axis(axis: plt.Axes) -> None:
	axis.set_facecolor(SURFACE)
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		axis.spines[side].set_color(GRID)
	axis.grid(axis="y", color=GRID, linewidth=0.8)
	axis.tick_params(colors=INK_MUTED, labelsize=9)


def figure_categories(categories: pd.DataFrame, path: Path) -> None:
	"""Mean LH per sign category and the sign-of-ROD strategy return for agreement and disagreement days."""
	figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
	table = categories.set_index("group")
	x = np.arange(len(CATEGORY_ORDER))
	colors = [BLUE, BLUE, ORANGE, ORANGE]
	rows = table.loc[CATEGORY_ORDER]
	for position, (color, (_, row)) in enumerate(zip(colors, rows.iterrows())):
		left.errorbar(position, row["mean_lh_bps"], yerr=[[row["mean_lh_bps"] - row["ci_low_lh_bps"]], [row["ci_high_lh_bps"] - row["mean_lh_bps"]]], fmt="o", color=color, markersize=7, elinewidth=1.8, capsize=4)
		left.annotate(f"n = {int(row['n'])}", (position, row["ci_low_lh_bps"]), xytext=(0, -14), textcoords="offset points", ha="center", color=INK_MUTED, fontsize=8)
	left.axhline(0, color=INK_MUTED, linewidth=0.8)
	left.set_xticks(x, CATEGORY_ORDER)
	left.set_ylabel("Mean last-half-hour return, basis points", color=INK_MUTED, fontsize=9)
	left.set_title("Mean LH by ON/DAY sign category, 95% intervals (blue: agreement, orange: disagreement)", loc="left", color=INK, fontsize=10)
	pooled = table.loc[["agree", "disagree"]]
	for position, (color, (_, row)) in enumerate(zip((BLUE, ORANGE), pooled.iterrows())):
		right.errorbar(position, row["eta_bps"], yerr=[[row["eta_bps"] - row["ci_low_eta_bps"]], [row["ci_high_eta_bps"] - row["eta_bps"]]], fmt="o", color=color, markersize=7, elinewidth=1.8, capsize=4)
	right.axhline(0, color=INK_MUTED, linewidth=0.8)
	right.set_xticks([0, 1], [f"Agreement\nn = {int(pooled.loc['agree', 'n'])}", f"Disagreement\nn = {int(pooled.loc['disagree', 'n'])}"])
	right.set_xlim(-0.6, 1.6)
	right.set_ylabel("Mean sign(ROD) x LH, basis points", color=INK_MUTED, fontsize=9)
	right.set_title("Paper's strategy return by group", loc="left", color=INK, fontsize=10)
	for axis in (left, right):
		style_axis(axis)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_grid(grid: pd.DataFrame, path: Path) -> None:
	"""Heat map of the mean LH on the ON x DAY bin grid: coherence as a two-dimensional vector."""
	labels = list(dict.fromkeys(grid["on_bin"]))
	matrix = grid.pivot(index="on_bin", columns="day_bin", values="mean_lh_bps").reindex(index=labels[::-1], columns=labels)
	counts = grid.pivot(index="on_bin", columns="day_bin", values="n").reindex(index=labels[::-1], columns=labels)
	errors = grid.pivot(index="on_bin", columns="day_bin", values="se_lh_bps").reindex(index=labels[::-1], columns=labels)
	limit = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
	cmap = LinearSegmentedColormap.from_list("diverging", [BLUE, "#f0efec", RED])
	figure, axis = plt.subplots(figsize=(7.6, 6), facecolor=SURFACE)
	image = axis.imshow(matrix.to_numpy(dtype=float), cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), aspect="equal")
	for row_index, on_label in enumerate(matrix.index):
		for column_index, day_label in enumerate(matrix.columns):
			value, count, error = matrix.loc[on_label, day_label], counts.loc[on_label, day_label], errors.loc[on_label, day_label]
			axis.text(column_index, row_index - 0.12, f"{value:+.1f} bps", ha="center", va="center", color=INK, fontsize=10)
			axis.text(column_index, row_index + 0.2, f"SE {error:.1f}, n = {int(count)}", ha="center", va="center", color=INK_MUTED, fontsize=8)
	axis.set_xticks(range(len(matrix.columns)), [f"DAY {label}" for label in matrix.columns])
	axis.set_yticks(range(len(matrix.index)), [f"ON {label}" for label in matrix.index])
	axis.set_xlabel("Daytime return tercile (08:00 to 13:00 CT)", color=INK_MUTED, fontsize=9)
	axis.set_ylabel("Overnight return tercile (previous 13:30 to 08:00 CT)", color=INK_MUTED, fontsize=9)
	axis.set_title("Mean last-half-hour return by overnight and daytime terciles", loc="left", color=INK, fontsize=11)
	axis.tick_params(colors=INK_MUTED, labelsize=9)
	for spine in axis.spines.values():
		spine.set_visible(False)
	bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	bar.set_label("Mean LH, basis points", color=INK_MUTED, fontsize=9)
	bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
	bar.outline.set_visible(False)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_by_year(by_year: pd.DataFrame, path: Path) -> None:
	figure, axis = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
	years = sorted(by_year["year"].unique())
	x = np.arange(len(years))
	for offset, group, color in ((-0.15, "agree", BLUE), (0.15, "disagree", ORANGE)):
		rows = by_year[by_year["group"].eq(group)].set_index("year").reindex(years)
		axis.errorbar(x + offset, rows["eta_bps"], yerr=[rows["eta_bps"] - rows["ci_low_eta_bps"], rows["ci_high_eta_bps"] - rows["eta_bps"]], fmt="o", color=color, markersize=6, capsize=3, linewidth=1.8, label=f"{group.capitalize()} days")
	axis.axhline(0, color=INK_MUTED, linewidth=0.8)
	axis.set_xticks(x, [str(year) for year in years])
	axis.set_ylabel("Mean sign(ROD) x LH, basis points", color=INK_MUTED, fontsize=9)
	axis.set_title("Sign-of-ROD strategy return on agreement and disagreement days, by year (95% intervals)", loc="left", color=INK, fontsize=11)
	axis.legend(frameon=False, fontsize=9, labelcolor=INK)
	style_axis(axis)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_cumulative(ledgers: dict[str, pd.DataFrame], path: Path) -> None:
	figure, axis = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
	colors = {"rod": BLUE, "agree_only": AQUA, "disagree_only": ORANGE, "day_sign": YELLOW, "on_sign": VIOLET}
	for name, ledger in ledgers.items():
		values = ledger["pnl_net_usd"].cumsum()
		axis.plot(ledger["trade_date"], values, color=colors[name], linewidth=2, label=STRATEGY_LABELS[name])
	axis.axhline(0, color=INK_MUTED, linewidth=0.8)
	axis.set_ylabel("Cumulative net P&L, USD per contract", color=INK_MUTED, fontsize=9)
	axis.set_title("Sign-filtered strategies, net of the #30 baseline costs", loc="left", color=INK, fontsize=11, pad=36)
	axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
	style_axis(axis)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	matplotlib.use("Agg")
	args = parse_args()
	config, strategy_cfg = load_config(args.config)
	tables_dir, figures_dir = Path(config["tables_dir"]), Path(config["figures_dir"])
	tables_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)
	z = normal_quantile(1 - (1 - config["inference"]["confidence"]) / 2)
	scale, min_train = config["scale"], config["oos"]["min_train_obs"]

	features = pd.read_parquet(config["return_features"])
	sample = features[features["sample"]].sort_values("trade_date").reset_index(drop=True)
	lags = newey_west_lags(len(sample)) if config["inference"]["hac_lags"] == "auto" else int(config["inference"]["hac_lags"])
	frame = add_terms(sample, config)
	regression_frame = regression_sample(frame)

	categories = category_table(frame, config, lags, z)
	grid = grid_table(regression_frame, config["grid"]["bins"], config["grid"]["min_cell"], lags, z)
	terms, models, forecasts = fit_models(regression_frame, config["models"], lags, z, scale, min_train, config["nested"])
	by_year = by_year_table(frame, lags, z)

	# Robustness: excluded dates, the pre-expiry roll rule, log returns, and the zero-day coding; each with
	# the group comparison and the models, with expanding-window OOS where the variant is prospective.
	robust_groups, robust_terms, robust_models = [], [], []

	def run_variant(name: str, variant_frame: pd.DataFrame, dropped: int, oos: bool = True) -> None:
		robust_groups.append(category_table(variant_frame, config, lags, z).assign(variant=name, dropped=dropped))
		rt, rm, _ = fit_models(regression_sample(variant_frame), config["models"], lags, z, scale, min_train if oos else None, config["nested"], variant=name)
		robust_terms.append(rt)
		robust_models.append(rm.assign(dropped=dropped))

	for name, rule in config["robustness"].items():
		drop = variant_mask(frame, rule)
		run_variant(f"exclude:{name}", frame[~drop].reset_index(drop=True), int(drop.sum()), oos="abs_quantile_above" not in rule)
	alternative = pre_expiry_features({key: config[key] for key in ("reference_config", "front_minutes", "roll_comparison", "return_config")}, features)
	alternative = alternative[alternative["sample"]].sort_values("trade_date").reset_index(drop=True)
	changed = int(alternative.set_index("trade_date")["contract_id"].ne(sample.set_index("trade_date")["contract_id"]).sum())
	run_variant("pre_expiry_roll_3_sessions", add_terms(alternative, config), len(sample) - len(alternative))
	run_variant("log_returns", add_terms(sample, config, labels=config["log_variant"]), 0)
	# The zero-return days coded as agree = 0 (the reference group is then disagreement plus zero days).
	rt, rm, _ = fit_models(frame, config["models"], lags, z, scale, min_train, config["nested"], variant="include_zero_return_days")
	robust_terms.append(rt)
	robust_models.append(rm.assign(dropped=0))
	robustness = pd.concat(robust_groups, ignore_index=True)
	robustness_terms = pd.concat(robust_terms, ignore_index=True)
	robustness_models = pd.concat(robust_models, ignore_index=True)

	# Trading comparison with the #30 execution and costs.
	references = pd.read_parquet(config["reference_prices"])
	minutes = pd.read_parquet(config["front_minutes"], columns=["ts_utc", "instrument_id", "open", "high", "low", "close", "volume"])
	with Path(strategy_cfg["minute_config"]).open(encoding="utf-8") as minute_file:
		timezone_name = yaml.safe_load(minute_file)["session"]["timezone"]
	strategies, strategy_by_year, ledgers = trading_tables(frame, references, minutes, config, strategy_cfg, timezone_name)

	outputs = {
		"categories": tables_dir / "sign_agreement_categories.csv", "grid": tables_dir / "sign_agreement_grid.csv", "regressions": tables_dir / "sign_agreement_regressions.csv",
		"models": tables_dir / "sign_agreement_oos.csv", "oos_forecasts": tables_dir / "sign_agreement_oos_forecasts.csv", "strategies": tables_dir / "sign_agreement_strategies.csv",
		"strategy_by_year": tables_dir / "sign_agreement_strategy_by_year.csv", "by_year": tables_dir / "sign_agreement_by_year.csv",
		"robustness": tables_dir / "sign_agreement_robustness.csv", "robustness_regressions": tables_dir / "sign_agreement_robustness_regressions.csv",
		"robustness_models": tables_dir / "sign_agreement_robustness_models.csv",
		"figure_categories": figures_dir / "sign_agreement_categories.png", "figure_grid": figures_dir / "sign_agreement_grid.png",
		"figure_by_year": figures_dir / "sign_agreement_by_year.png", "figure_cumulative": figures_dir / "sign_agreement_cumulative.png",
	}
	tables = {
		"categories": categories, "grid": grid, "regressions": terms, "models": models, "oos_forecasts": forecasts, "strategies": strategies,
		"strategy_by_year": strategy_by_year, "by_year": by_year, "robustness": robustness, "robustness_regressions": robustness_terms, "robustness_models": robustness_models,
	}
	for name, table in tables.items():
		table.to_csv(outputs[name], index=False, float_format="%.6g")
	figure_categories(categories, outputs["figure_categories"])
	figure_grid(grid, outputs["figure_grid"])
	figure_by_year(by_year, outputs["figure_by_year"])
	figure_cumulative(ledgers, outputs["figure_cumulative"])

	difference = categories[categories["group"].eq("agree - disagree")].iloc[0]
	manifest = {
		"config": str(args.config), "settings": config, "strategy_settings": {key: strategy_cfg[key] for key in ("execution", "costs", "evaluation", "baseline_execution")}, "hac_lags_used": lags,
		"sample": {
			"observations": len(sample), "dates": f"{sample['trade_date'].iloc[0].date()} to {sample['trade_date'].iloc[-1].date()}", "zero_return_days": int(frame["category"].eq("zero").sum()),
			"regression_observations": len(regression_frame), "lh_zero_days": int(frame["y"].eq(0).sum()), "first_oos_forecast": str(regression_frame["trade_date"].iloc[min_train].date()),
		},
		"category_counts": frame["category"].value_counts().to_dict(), "bonferroni_level": config["family_level"] / config["family_tests"], "pre_expiry_roll_contract_changed_dates": changed,
		"inputs": {name: {"file": config[name], "sha256": sha256(Path(config[name]))} for name in ("return_features", "reference_prices", "front_minutes", "roll_comparison")},
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
	}
	(tables_dir / "sign_agreement.manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
		print(f"Sample {manifest['sample']['dates']}, n = {len(sample)}, Newey-West lags = {lags}, zero-return days {manifest['sample']['zero_return_days']} (set aside), LH = 0 on {manifest['sample']['lh_zero_days']} days (no hit, no miss)\n")
		print(categories[["group", "n", "mean_lh_bps", "se_lh_bps", "lh_follows_rod", "eta_bps", "se_eta_bps", "t_eta"]].to_string(index=False))
		print(f"\nAgree - disagree strategy return: {difference['eta_bps']:.2f} bps (SE {difference['se_eta_bps']:.2f}, p = {difference['p_eta']:.3f}, Bonferroni level {manifest['bonferroni_level']:.3f})\n")
		print(terms[terms["term"].str.contains("_x_")][["model", "term", "coef_x100", "se_x100", "t_stat", "p_value"]].to_string(index=False))
		print()
		print(models[["model", "adj_r2_pct", "wald_p", "oos_r2_pct", "clark_west_p", "nested_in", "oos_r2_vs_nested_pct", "clark_west_p_vs_nested"]].to_string(index=False))
		print()
		print(robustness_models[robustness_models["model"].eq("rod_by_agreement")][["variant", "dropped", "n", "oos_r2_pct", "clark_west_p", "oos_r2_vs_nested_pct", "clark_west_p_vs_nested"]].to_string(index=False))
		print()
		print(strategies[strategies["accounting"].eq("net")][["strategy", "traded_days", "mean_bps", "se_hac_bps", "sharpe", "hit_rate", "total_pnl_usd"]].to_string(index=False))
	print(f"\nSaved tables to {tables_dir}/, figures to {figures_dir}/")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
