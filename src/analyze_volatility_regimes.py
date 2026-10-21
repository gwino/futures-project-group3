"""Test whether high-volatility regimes change overnight and daytime predictive power for CL (#32).

Run from the repository root, after the return features (#28) and the volatility features (#32):

    python -m src.analyze_volatility_regimes --config config/volatility_regimes.yaml

For each regime definition (a realized-variance measure known by 13:00 CT), a trade date is high
volatility when its measure exceeds the upper quartile of the measure over the earlier trade dates.
The models are LH ~ ON + DAY (base), LH ~ ON + DAY + High + ON x High + DAY x High (interaction), and
the same pair for the paper's ROD predictor. The interaction slopes say how much the ON and DAY
slopes change in high-volatility regimes; the low- and high-regime slopes are reported with
Newey-West standard errors, a joint Wald test of the interactions, and expanding-window
out-of-sample comparisons against the historical mean and the nested base model.

Outputs (tables in reports/tables, figures in reports/figures):
- volatility_regimes_regressions.csv: every term of every model and regime, with standard errors,
  intervals, t-statistics, adjusted R2, Wald test and OOS statistics.
- volatility_regimes_slopes.csv: low- and high-regime slopes of ON, DAY and ROD per regime definition.
- volatility_regimes_counts.csv: observations and LH dispersion per regime.
- volatility_regimes_oos.csv and volatility_regimes_oos_forecasts.csv: out-of-sample results and forecasts.
- volatility_regimes_robustness.csv: other thresholds, log returns, and excluded dates.
- volatility_regimes_by_year.csv: regime slopes by calendar year for the baseline regime definition.
- volatility_regimes_labels.csv: the measures, thresholds and labels on every sample date.
- volatility_regimes_*.png and volatility_regimes.manifest.json.
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
import numpy as np
import pandas as pd
import yaml

from src.prepare_daily_data import sha256
from src.replicate_regressions import exclusion_mask, expanding_forecasts, hac_covariance, newey_west_lags, normal_cdf, normal_quantile, ols, oos_statistics

# Chart colors (dataviz reference palette): low regime in blue, high regime in orange, overnight aqua, daytime yellow.
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LOW, HIGH, OVERNIGHT, DAYTIME, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
REGIME_LABELS = {"combined_vol": "Combined RV", "day_vol": "Daytime RV", "overnight_vol": "Overnight RV", "trailing_5d_vol": "Trailing 5-day RV"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/volatility_regimes.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> dict:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {
		"return_features", "volatility_features", "tables_dir", "figures_dir", "target", "predictors", "benchmark", "log_variant", "scale", "inference",
		"regimes", "baseline_regime", "threshold", "threshold_sensitivity", "oos", "robustness", "family_level",
	}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	for group in (config["predictors"], config["benchmark"], config["log_variant"]["predictors"], config["log_variant"]["benchmark"]):
		names = [name for name in group if not isinstance(name, str)]
		if names:
			raise ValueError(f"Predictor names must be strings, got {names}; quote YAML keys such as \"on\".")
	if config["baseline_regime"] not in config["regimes"]:
		raise ValueError(f"baseline_regime {config['baseline_regime']} is not among the regimes.")
	for name, spec in {"threshold": config["threshold"], **config["threshold_sensitivity"]}.items():
		if spec["method"] not in ("expanding", "training"):
			raise ValueError(f"{name}: method must be expanding or training.")
	return config


# --- Regime labels ----------------------------------------------------------------------------


def regime_labels(measure: pd.Series, spec: dict) -> pd.DataFrame:
	"""Threshold and high-volatility label on each date, using only the measure's earlier values.

	expanding: the threshold on date t is the quantile of the measure over the dates strictly before t,
	and is missing until min_history earlier values exist. training: one quantile of the first train_obs
	values, applied to every date. The label is True when the measure exceeds the threshold.
	"""
	values = measure.to_numpy(dtype=float)
	if spec["method"] == "expanding":
		threshold = pd.Series(np.nan, index=measure.index)
		for position in range(spec["min_history"], len(values)):
			threshold.iloc[position] = np.nanquantile(values[:position], spec["quantile"])
	else:
		threshold = pd.Series(np.nanquantile(values[: spec["train_obs"]], spec["quantile"]), index=measure.index)
	high = pd.Series(np.where(threshold.notna() & measure.notna(), measure.gt(threshold), np.nan), index=measure.index)
	return pd.DataFrame({"threshold": threshold, "high": high})


# --- Estimation -------------------------------------------------------------------------------


def wald_p_value(statistic: float, df: int) -> float:
	"""Upper tail of a chi-square with 1 or 2 degrees of freedom (closed forms; no scipy)."""
	if df == 1:
		return 2 * (1 - normal_cdf(math.sqrt(statistic)))
	if df == 2:
		return math.exp(-statistic / 2)
	raise ValueError("Only 1 or 2 degrees of freedom are supported.")


def interaction_design(sample: pd.DataFrame, predictors: dict, regime_column: str | None) -> tuple[np.ndarray, list[str]]:
	"""Columns [predictors..., high, predictor x high...] (or just the predictors when regime_column is None)."""
	columns = {name: sample[column].to_numpy(dtype=float) for name, column in predictors.items()}
	if regime_column is not None:
		high = sample[regime_column].to_numpy(dtype=float)
		columns["high"] = high
		for name, values in list(columns.items())[: len(predictors)]:
			columns[f"{name}:high"] = values * high
	return np.column_stack(list(columns.values())), list(columns)


def fit_model(sample: pd.DataFrame, target: str, predictors: dict, regime_column: str | None, lags: int, z: float, scale: float, min_train: int | None) -> tuple[list[dict], list[dict], dict]:
	"""Term rows, regime-slope rows and model statistics for one model.

	With a regime column the model is y ~ predictors + high + predictors x high. The high-regime slope
	of a predictor is its slope plus its interaction, with the standard error from the HAC covariance.
	The Wald statistic tests all interaction terms jointly.
	"""
	y = sample[target].to_numpy(dtype=float)
	X, names = interaction_design(sample, predictors, regime_column)
	fit = ols(y, X, lags)
	design = np.column_stack([np.ones(len(y)), X])
	covariance = hac_covariance(design, y - design @ fit["coefficients"], lags)
	terms = ["const", *names]
	stats = {"n": fit["n"], "adj_r2_pct": fit["adj_r2"] * scale, "hac_lags": lags}
	if regime_column is not None:
		high = sample[regime_column].to_numpy(dtype=float)
		stats.update({"n_high": int(high.sum()), "n_low": int(len(high) - high.sum())})
		interactions = [terms.index(f"{name}:high") for name in predictors]
		gamma = fit["coefficients"][interactions]
		stats["wald_stat"] = float(gamma @ np.linalg.solve(covariance[np.ix_(interactions, interactions)], gamma))
		stats["wald_df"] = len(interactions)
		stats["wald_p"] = wald_p_value(stats["wald_stat"], len(interactions))
	if min_train is not None and len(y) > min_train:
		oos = oos_statistics(expanding_forecasts(y, X, min_train), lags)
		stats.update({"oos_r2_pct": oos["oos_r2"] * scale, "oos_forecasts": oos["oos_forecasts"], "clark_west_t": oos["clark_west_t"], "clark_west_p": oos["clark_west_p"]})
	term_rows = []
	for index, term in enumerate(terms):
		coefficient, error = fit["coefficients"][index], fit["errors"][index]
		term_rows.append(
			{
				"term": term, "coef_x100": coefficient * scale, "se_x100": error * scale, "ci_low_x100": (coefficient - z * error) * scale,
				"ci_high_x100": (coefficient + z * error) * scale, "t_stat": coefficient / error, "p_value": 2 * (1 - normal_cdf(abs(coefficient / error))),
			}
		)
	slope_rows = []
	if regime_column is not None:
		for name in predictors:
			low_index, gamma_index = terms.index(name), terms.index(f"{name}:high")
			low, gamma = fit["coefficients"][low_index], fit["coefficients"][gamma_index]
			low_se = math.sqrt(covariance[low_index, low_index])
			high_se = math.sqrt(covariance[low_index, low_index] + covariance[gamma_index, gamma_index] + 2 * covariance[low_index, gamma_index])
			gamma_se = math.sqrt(covariance[gamma_index, gamma_index])
			slope_rows.append(
				{
					"predictor": name, "slope_low_x100": low * scale, "se_low_x100": low_se * scale, "ci_low_low_x100": (low - z * low_se) * scale, "ci_high_low_x100": (low + z * low_se) * scale,
					"slope_high_x100": (low + gamma) * scale, "se_high_x100": high_se * scale, "ci_low_high_x100": (low + gamma - z * high_se) * scale, "ci_high_high_x100": (low + gamma + z * high_se) * scale,
					"difference_x100": gamma * scale, "se_difference_x100": gamma_se * scale, "t_difference": gamma / gamma_se, "p_difference": 2 * (1 - normal_cdf(abs(gamma / gamma_se))),
				}
			)
	return term_rows, slope_rows, stats


def nested_oos(sample: pd.DataFrame, target: str, predictors: dict, regime_column: str, min_train: int, lags: int) -> tuple[dict, pd.DataFrame]:
	"""Interaction model against the nested base model out of sample: OOS R2 relative to the base forecasts and Clark-West."""
	y = sample[target].to_numpy(dtype=float)
	base = expanding_forecasts(y, interaction_design(sample, predictors, None)[0], min_train)
	full = expanding_forecasts(y, interaction_design(sample, predictors, regime_column)[0], min_train)
	paired = full.rename(columns={"forecast": "forecast"}).assign(historical_mean=base["forecast"].to_numpy())
	stats = oos_statistics(paired, lags)
	forecasts = pd.DataFrame(
		{
			"trade_date": sample["trade_date"].iloc[full["position"]].to_numpy(), "actual": full["actual"], "historical_mean": base["historical_mean"],
			"base_forecast": base["forecast"], "interaction_forecast": full["forecast"],
		}
	)
	return {"oos_r2_vs_base_pct": stats["oos_r2"] * 100, "clark_west_t_vs_base": stats["clark_west_t"], "clark_west_p_vs_base": stats["clark_west_p"]}, forecasts


def estimate_regime(sample: pd.DataFrame, regime: str, measure: str, threshold_spec: dict, config: dict, lags: int, z: float, min_train: int | None, labels: dict | None = None, variant: str = "baseline") -> dict:
	"""All models for one regime definition on one sample; returns the tables as lists of rows."""
	scale, target = config["scale"], config["target"]
	predictors, benchmark = config["predictors"], config["benchmark"]
	if labels is not None:
		target, predictors, benchmark = labels["target"], labels["predictors"], labels["benchmark"]
	frame = sample.copy()
	label = regime_labels(frame[measure], threshold_spec)
	frame["threshold"], frame["high"] = label["threshold"].to_numpy(), label["high"].to_numpy()
	frame = frame[frame["high"].notna()].reset_index(drop=True)
	models = {"base": (predictors, None), "interaction": (predictors, "high"), "rod": (benchmark, None), "rod_interaction": (benchmark, "high")}
	term_rows, slope_rows, stat_rows, forecasts = [], [], [], []
	for model, (columns, regime_column) in models.items():
		terms, slopes, stats = fit_model(frame, target, columns, regime_column, lags, z, scale, min_train)
		if regime_column is not None and min_train is not None and len(frame) > min_train:
			nested, rows = nested_oos(frame, target, columns, regime_column, min_train, lags)
			stats.update(nested)
			forecasts.append(rows.assign(regime=regime, model=model))
		key = {"variant": variant, "regime": regime, "measure": measure, "model": model, "first_date": frame["trade_date"].iloc[0].date(), "last_date": frame["trade_date"].iloc[-1].date()}
		term_rows += [{**key, **row, **stats} for row in terms]
		slope_rows += [{**key, **row, "n_low": stats["n_low"], "n_high": stats["n_high"], "wald_stat": stats["wald_stat"], "wald_p": stats["wald_p"]} for row in slopes]
		stat_rows.append({**key, **stats})
	high = frame["high"].astype(bool)
	counts = {
		"variant": variant, "regime": regime, "measure": measure, "n": len(frame), "n_low": int((~high).sum()), "n_high": int(high.sum()), "first_date": frame["trade_date"].iloc[0].date(),
		"last_threshold_vol_pct": math.sqrt(frame["threshold"].iloc[-1]) * 100, "lh_sd_low_pct": frame.loc[~high, target].std() * 100, "lh_sd_high_pct": frame.loc[high, target].std() * 100,
		"lh_mean_low_bps": frame.loc[~high, target].mean() * 1e4, "lh_mean_high_bps": frame.loc[high, target].mean() * 1e4,
		"measure_vol_low_pct": math.sqrt(frame.loc[~high, measure].mean()) * 100, "measure_vol_high_pct": math.sqrt(frame.loc[high, measure].mean()) * 100,
	}
	labels_table = frame[["trade_date", "threshold", "high"]].assign(regime=regime, measure_value=frame[measure])
	return {"terms": term_rows, "slopes": slope_rows, "stats": stat_rows, "counts": counts, "forecasts": forecasts, "labels": labels_table}


# --- Figures ----------------------------------------------------------------------------------


def style_axis(axis: plt.Axes) -> None:
	axis.set_facecolor(SURFACE)
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		axis.spines[side].set_color(GRID)
	axis.grid(axis="y", color=GRID, linewidth=0.8)
	axis.tick_params(colors=INK_MUTED, labelsize=9)


def figure_measures(features: pd.DataFrame, labels: pd.DataFrame, baseline: str, path: Path) -> None:
	"""Overnight and daytime realized volatility over time, and the baseline measure against its expanding threshold."""
	figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.5), facecolor=SURFACE, sharex=True)
	dates = features["trade_date"]
	top.plot(dates, features["vol_on_pct"], color=OVERNIGHT, linewidth=1.2, label="Overnight (previous 13:30 to 08:00 CT)")
	top.plot(dates, features["vol_day_pct"], color=DAYTIME, linewidth=1.2, label="Daytime (08:00 to 13:00 CT)")
	top.set_ylabel("Realized volatility, % per window", color=INK_MUTED, fontsize=9)
	top.set_title("CL realized volatility from 1-minute returns, 2021-2026", loc="left", color=INK, fontsize=11)
	top.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
	chosen = labels[labels["regime"].eq(baseline)].sort_values("trade_date")
	bottom.plot(dates, features["vol_rod_pct"], color=LOW, linewidth=1.2, label="Combined (previous 13:30 to 13:00 CT)")
	bottom.plot(chosen["trade_date"], np.sqrt(chosen["threshold"]) * 100, color=HIGH, linewidth=2, label="Expanding upper-quartile threshold")
	high_dates = chosen.loc[chosen["high"].astype(bool), "trade_date"]
	bottom.scatter(high_dates, np.sqrt(chosen.loc[chosen["high"].astype(bool), "measure_value"]) * 100, color=HIGH, s=9, label="High-volatility dates", zorder=3)
	bottom.set_ylabel("Realized volatility, % per window", color=INK_MUTED, fontsize=9)
	bottom.set_title(f"{REGIME_LABELS.get(baseline, baseline)} regime: {len(high_dates)} high-volatility dates of {len(chosen)}", loc="left", color=INK, fontsize=11)
	bottom.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
	for axis in (top, bottom):
		style_axis(axis)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_slopes(slopes: pd.DataFrame, path: Path) -> None:
	"""Low- and high-regime slopes of ON, DAY and ROD, with intervals, for each regime definition."""
	table = slopes[slopes["variant"].eq("baseline")]
	predictors = [("on", "ON slope x 100"), ("day", "DAY slope x 100"), ("rod", "ROD slope x 100")]
	figure, axes = plt.subplots(1, 3, figsize=(12, 4.6), facecolor=SURFACE, sharey=False)
	regimes = list(dict.fromkeys(table["regime"]))
	for axis, (predictor, title) in zip(axes, predictors):
		rows = table[table["predictor"].eq(predictor)].set_index("regime").reindex(regimes)
		x = np.arange(len(regimes))
		for offset, level, color in ((-0.15, "low", LOW), (0.15, "high", HIGH)):
			axis.errorbar(
				x + offset, rows[f"slope_{level}_x100"], yerr=[rows[f"slope_{level}_x100"] - rows[f"ci_low_{level}_x100"], rows[f"ci_high_{level}_x100"] - rows[f"slope_{level}_x100"]],
				fmt="o", color=color, markersize=6, capsize=3, linewidth=1.8, label=f"{level.capitalize()}-volatility regime",
			)
		axis.axhline(0, color=INK_MUTED, linewidth=0.8)
		axis.set_xticks(x, [REGIME_LABELS.get(regime, regime) for regime in regimes], rotation=20, ha="right")
		axis.set_title(title, loc="left", color=INK, fontsize=10)
		style_axis(axis)
	axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
	figure.suptitle("Last-half-hour slopes by volatility regime, 95% intervals (Newey-West)", x=0.01, ha="left", color=INK, fontsize=11)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_by_year(by_year: pd.DataFrame, baseline: str, path: Path) -> None:
	figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
	for axis, predictor in zip(axes, ("on", "day")):
		rows = by_year[by_year["predictor"].eq(predictor)].sort_values("year")
		x = np.arange(len(rows))
		for offset, level, color in ((-0.15, "low", LOW), (0.15, "high", HIGH)):
			axis.errorbar(
				x + offset, rows[f"slope_{level}_x100"], yerr=[rows[f"slope_{level}_x100"] - rows[f"ci_low_{level}_x100"], rows[f"ci_high_{level}_x100"] - rows[f"slope_{level}_x100"]],
				fmt="o", color=color, markersize=6, capsize=3, linewidth=1.8, label=f"{level.capitalize()}-volatility regime",
			)
		axis.axhline(0, color=INK_MUTED, linewidth=0.8)
		axis.set_xticks(x, rows["year"].astype(str))
		axis.set_title(f"{predictor.upper()} slope x 100 by year, {REGIME_LABELS.get(baseline, baseline)} regime", loc="left", color=INK, fontsize=10)
		style_axis(axis)
	axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_oos(forecasts: pd.DataFrame, path: Path) -> None:
	"""Cumulative squared-error advantage of the interaction model over the base model, per regime definition."""
	figure, axis = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
	colors = [LOW, HIGH, OVERNIGHT, DAYTIME, VIOLET]
	table = forecasts[forecasts["model"].eq("interaction")]
	for color, (regime, rows) in zip(colors, table.groupby("regime", sort=False)):
		rows = rows.sort_values("trade_date")
		advantage = ((rows["actual"] - rows["base_forecast"]) ** 2 - (rows["actual"] - rows["interaction_forecast"]) ** 2).cumsum() * 1e4
		axis.plot(rows["trade_date"], advantage, color=color, linewidth=2, label=REGIME_LABELS.get(regime, regime))
	axis.axhline(0, color=INK_MUTED, linewidth=0.8)
	axis.set_ylabel("Cumulative SSE(base) - SSE(interaction), x 10,000", color=INK_MUTED, fontsize=9)
	axis.set_title("Out of sample: does adding the volatility interaction improve the ON + DAY forecasts?", loc="left", color=INK, fontsize=11)
	axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
	style_axis(axis)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	matplotlib.use("Agg")
	args = parse_args()
	config = load_config(args.config)
	tables_dir, figures_dir = Path(config["tables_dir"]), Path(config["figures_dir"])
	tables_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)
	z = normal_quantile(1 - (1 - config["inference"]["confidence"]) / 2)
	min_train = config["oos"]["min_train_obs"]

	returns = pd.read_parquet(config["return_features"])
	volatility = pd.read_parquet(config["volatility_features"])
	measures = sorted({spec["measure"] for spec in config["regimes"].values()})
	merged = returns.merge(volatility[["trade_date", "sample", *measures, "vol_on_pct", "vol_day_pct", "vol_rod_pct"]].rename(columns={"sample": "vol_sample"}), on="trade_date", how="inner")
	sample = merged[merged["sample"] & merged["vol_sample"]].sort_values("trade_date").reset_index(drop=True)
	lags = newey_west_lags(len(sample)) if config["inference"]["hac_lags"] == "auto" else int(config["inference"]["hac_lags"])
	tests = len(config["regimes"]) * len(config["predictors"])
	bonferroni = config["family_level"] / tests

	results = {"terms": [], "slopes": [], "stats": [], "counts": [], "forecasts": [], "labels": []}

	def collect(result: dict) -> None:
		for key in ("terms", "slopes", "stats"):
			results[key] += result[key]
		results["counts"].append(result["counts"])
		results["forecasts"] += result["forecasts"]
		results["labels"].append(result["labels"])

	for regime, spec in config["regimes"].items():
		collect(estimate_regime(sample, regime, spec["measure"], config["threshold"], config, lags, z, min_train))
	baseline_labels = pd.concat(results["labels"], ignore_index=True)
	baseline_forecasts = pd.concat(results["forecasts"], ignore_index=True)

	# Robustness: thresholds, log returns, excluded dates. No OOS for these variants.
	for name, threshold_spec in config["threshold_sensitivity"].items():
		for regime, spec in config["regimes"].items():
			collect(estimate_regime(sample, regime, spec["measure"], threshold_spec, config, lags, z, None, variant=f"threshold:{name}"))
	for regime, spec in config["regimes"].items():
		collect(estimate_regime(sample, regime, spec["measure"], config["threshold"], config, lags, z, None, labels=config["log_variant"], variant="log_returns"))
	for name, rule in config["robustness"].items():
		kept = sample[~exclusion_mask(sample, rule)].reset_index(drop=True)
		for regime, spec in config["regimes"].items():
			collect(estimate_regime(kept, regime, spec["measure"], config["threshold"], config, lags, z, None, variant=f"exclude:{name}"))

	# By calendar year, baseline regime, with the labels fixed from the full-sample expanding rule.
	baseline, measure = config["baseline_regime"], config["regimes"][config["baseline_regime"]]["measure"]
	labeled = sample.merge(baseline_labels[baseline_labels["regime"].eq(baseline)][["trade_date", "high"]], on="trade_date", how="inner")
	by_year_rows = []
	for year, rows in labeled.groupby(labeled["trade_date"].dt.year):
		rows = rows.reset_index(drop=True)
		if rows["high"].sum() < 10 or (1 - rows["high"]).sum() < 10:
			continue
		_, slopes, stats = fit_model(rows, config["target"], config["predictors"], "high", lags, z, config["scale"], None)
		by_year_rows += [{"year": year, "regime": baseline, **row, "n_low": stats["n_low"], "n_high": stats["n_high"]} for row in slopes]
	by_year = pd.DataFrame(by_year_rows)

	terms = pd.DataFrame(results["terms"])
	slopes = pd.DataFrame(results["slopes"])
	slopes["bonferroni_level"] = bonferroni
	stats = pd.DataFrame(results["stats"])
	counts = pd.DataFrame(results["counts"])
	oos_table = stats[stats["variant"].eq("baseline")][["regime", "measure", "model", "n", "oos_forecasts", "oos_r2_pct", "clark_west_t", "clark_west_p", "oos_r2_vs_base_pct", "clark_west_t_vs_base", "clark_west_p_vs_base"]]
	outputs = {
		"regressions": tables_dir / "volatility_regimes_regressions.csv", "slopes": tables_dir / "volatility_regimes_slopes.csv", "counts": tables_dir / "volatility_regimes_counts.csv",
		"oos": tables_dir / "volatility_regimes_oos.csv", "oos_forecasts": tables_dir / "volatility_regimes_oos_forecasts.csv", "robustness": tables_dir / "volatility_regimes_robustness.csv",
		"by_year": tables_dir / "volatility_regimes_by_year.csv", "labels": tables_dir / "volatility_regimes_labels.csv",
		"figure_measures": figures_dir / "volatility_regimes_measures.png", "figure_slopes": figures_dir / "volatility_regimes_slopes.png",
		"figure_by_year": figures_dir / "volatility_regimes_by_year.png", "figure_oos": figures_dir / "volatility_regimes_oos.png",
	}
	tables = {
		"regressions": terms[terms["variant"].eq("baseline")].drop(columns="variant"), "slopes": slopes[slopes["variant"].eq("baseline")].drop(columns="variant"),
		"counts": counts, "oos": oos_table, "oos_forecasts": baseline_forecasts, "robustness": slopes[slopes["variant"].ne("baseline")], "by_year": by_year, "labels": baseline_labels,
	}
	for name, table in tables.items():
		table.to_csv(outputs[name], index=False, float_format="%.6g")
	figure_measures(volatility[volatility["sample"]].reset_index(drop=True), baseline_labels, baseline, outputs["figure_measures"])
	figure_slopes(slopes, outputs["figure_slopes"])
	figure_by_year(by_year, baseline, outputs["figure_by_year"])
	figure_oos(baseline_forecasts, outputs["figure_oos"])

	manifest = {
		"config": str(args.config), "settings": config, "hac_lags_used": lags, "bonferroni_level": bonferroni, "interaction_tests": tests,
		"sample": {"merged_observations": len(sample), "dates": f"{sample['trade_date'].iloc[0].date()} to {sample['trade_date'].iloc[-1].date()}", "regression_observations": counts[counts["variant"].eq("baseline")].set_index("regime")["n"].to_dict()},
		"inputs": {name: {"file": config[name], "sha256": sha256(Path(config[name]))} for name in ("return_features", "volatility_features")},
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
	}
	(tables_dir / "volatility_regimes.manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	shown = ["regime", "predictor", "n_low", "n_high", "slope_low_x100", "se_low_x100", "slope_high_x100", "se_high_x100", "difference_x100", "se_difference_x100", "p_difference", "wald_p"]
	with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
		print(f"Sample {manifest['sample']['dates']}, {len(sample)} merged observations, Newey-West lags = {lags}, Bonferroni level {bonferroni:.4f} for {tests} interaction tests\n")
		print(tables["slopes"][shown].to_string(index=False))
		print()
		print(oos_table.to_string(index=False))
	print(f"\nSaved tables to {tables_dir}/, figures to {figures_dir}/")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
