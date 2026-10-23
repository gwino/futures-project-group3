"""Test overnight versus daytime return predictability for the last half hour of CL (#31).

Run from the repository root, after the return features (#28):

    python -m src.analyze_on_day --config config/on_day.yaml

The paper's rest-of-day return ROD (previous 13:30 -> 13:00 CT) compounds an overnight part ON
(previous 13:30 -> 08:00) and a daytime part DAY (08:00 -> 13:00). The models LH ~ ON, LH ~ DAY,
LH ~ ON + DAY and the benchmark LH ~ ROD are estimated with Newey-West standard errors; slopes are
also reported per standard deviation of the predictor so that the two components can be compared
on the same footing. A Wald test checks whether the two slopes are equal, and expanding-window
forecasts show whether either component adds information to the other out of sample. Stability is
checked by calendar year, over a rolling window, across the roll rule, HAC lags and excluded dates.

Outputs (tables in reports/tables, figures in reports/figures):
- on_day_regressions.csv: every term of every model with standard errors, intervals, t-statistics,
  standardized slopes, adjusted R2 and OOS statistics; on_day_models.csv: model-level statistics,
  the equality test and the nested comparisons.
- on_day_oos_forecasts.csv, on_day_oos_sensitivity.csv, on_day_by_year.csv, on_day_rolling.csv,
  on_day_robustness.csv; on_day_*.png and on_day.manifest.json.
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

from src.analyze_volatility_regimes import wald_p_value
from src.prepare_daily_data import sha256
from src.replicate_regressions import exclusion_mask, expanding_forecasts, hac_covariance, newey_west_lags, normal_cdf, normal_quantile, ols, oos_statistics, pre_expiry_features

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OVERNIGHT, DAYTIME, JOINT, ROD = "#1baf7a", "#eda100", "#2a78d6", "#eb6834"
MODEL_LABELS = {"on": "LH ~ ON", "day": "LH ~ DAY", "on_day": "LH ~ ON + DAY", "rod": "LH ~ ROD"}
TERM_LABELS = {"on": "ON", "day": "DAY", "rod": "ROD"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/on_day.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> dict:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {
		"return_features", "reference_config", "front_minutes", "roll_comparison", "return_config", "tables_dir", "figures_dir", "target", "predictors",
		"log_variant", "scale", "models", "nested", "equality_test", "inference", "oos", "rolling", "robustness",
	}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	for group in (config["predictors"], config["log_variant"]["predictors"], config["models"]):
		names = [name for name in group if not isinstance(name, str)]
		if names:
			raise ValueError(f"Names must be strings, got {names}; quote YAML keys such as \"on\".")
	for model, terms in config["models"].items():
		unknown = set(terms) - set(config["predictors"])
		if unknown:
			raise ValueError(f"Model {model} uses unknown predictors {sorted(unknown)}.")
	for larger, smaller in config["nested"]:
		if not set(config["models"][smaller]) < set(config["models"][larger]):
			raise ValueError(f"{smaller} is not nested in {larger}.")
	equality = config["equality_test"]
	if not set(equality["terms"]) <= set(config["models"][equality["model"]]):
		raise ValueError("The equality test names terms that are not in its model.")
	return config


# --- Estimation -------------------------------------------------------------------------------


def prepare(frame: pd.DataFrame, target: str, predictors: dict) -> pd.DataFrame:
	"""The sample with columns y and the predictor names (on, day, rod) in the chosen return convention."""
	out = frame[["trade_date"]].copy()
	out["y"] = frame[target].to_numpy(dtype=float)
	for name, column in predictors.items():
		out[name] = frame[column].to_numpy(dtype=float)
	return out


def fit_models(sample: pd.DataFrame, config: dict, lags: int, z: float, min_train: int | None, variant: str = "baseline") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Term rows (with standardized slopes), model rows (with the equality test and nested comparisons) and OOS forecasts.

	The standardized slope is coefficient x sd(predictor) / sd(target): the LH response, in standard
	deviations of LH, to a one-standard-deviation move of the predictor, so ON and DAY compare on
	the same footing although DAY has a smaller spread.
	"""
	scale = config["scale"]
	y = sample["y"].to_numpy(dtype=float)
	sd_y = y.std(ddof=1)
	term_rows, model_rows, forecasts, fitted = [], [], [], {}
	for model, predictors in config["models"].items():
		X = sample[predictors].to_numpy(dtype=float)
		fit = ols(y, X, lags)
		design = np.column_stack([np.ones(len(y)), X])
		covariance = hac_covariance(design, y - design @ fit["coefficients"], lags)
		terms = ["const", *predictors]
		stats = {"variant": variant, "model": model, "predictors": " + ".join(TERM_LABELS[term] for term in predictors), "n": fit["n"], "r2_pct": fit["r2"] * scale, "adj_r2_pct": fit["adj_r2"] * scale, "hac_lags": lags}
		equality = config["equality_test"]
		if model == equality["model"]:
			first, second = (terms.index(term) for term in equality["terms"])
			difference = fit["coefficients"][first] - fit["coefficients"][second]
			variance = covariance[first, first] + covariance[second, second] - 2 * covariance[first, second]
			stats.update({"equality_difference_x100": difference * scale, "equality_se_x100": math.sqrt(variance) * scale, "equality_wald": difference**2 / variance, "equality_p": wald_p_value(difference**2 / variance, 1)})
		if min_train is not None and len(y) > min_train:
			rows = expanding_forecasts(y, X, min_train)
			oos = oos_statistics(rows, lags)
			stats.update({"oos_forecasts": oos["oos_forecasts"], "oos_r2_pct": oos["oos_r2"] * scale, "clark_west_t": oos["clark_west_t"], "clark_west_p": oos["clark_west_p"]})
			fitted[model] = rows
			forecasts.append(rows.assign(model=model, trade_date=sample["trade_date"].iloc[rows["position"]].to_numpy()).drop(columns="position"))
		for index, term in enumerate(terms):
			coefficient, error = fit["coefficients"][index], fit["errors"][index]
			standardized = coefficient * sample[term].std(ddof=1) / sd_y if term != "const" else np.nan
			term_rows.append(
				{
					"variant": variant, "model": model, "term": term, "coef_x100": coefficient * scale, "se_x100": error * scale,
					"ci_low_x100": (coefficient - z * error) * scale, "ci_high_x100": (coefficient + z * error) * scale,
					"t_stat": coefficient / error, "p_value": 2 * (1 - normal_cdf(abs(coefficient / error))),
					"standardized": standardized, "standardized_se": error * sample[term].std(ddof=1) / sd_y if term != "const" else np.nan,
					"response_bps_per_sd": coefficient * sample[term].std(ddof=1) * 1e4 if term != "const" else np.nan,
					"n": fit["n"], "adj_r2_pct": fit["adj_r2"] * scale,
				}
			)
		model_rows.append(stats)
	models = pd.DataFrame(model_rows).set_index("model")
	for larger, smaller in config["nested"]:
		models.loc[larger, f"adj_r2_gain_vs_{smaller}_pct"] = models.loc[larger, "adj_r2_pct"] - models.loc[smaller, "adj_r2_pct"]
		if larger in fitted and smaller in fitted:
			compare = oos_statistics(fitted[larger].assign(historical_mean=fitted[smaller]["forecast"].to_numpy()), lags)
			models.loc[larger, f"oos_r2_vs_{smaller}_pct"] = compare["oos_r2"] * scale
			models.loc[larger, f"clark_west_p_vs_{smaller}"] = compare["clark_west_p"]
	return pd.DataFrame(term_rows), models.reset_index(), pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()


# --- Figures ----------------------------------------------------------------------------------


def style_axis(axis: plt.Axes) -> None:
	axis.set_facecolor(SURFACE)
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		axis.spines[side].set_color(GRID)
	axis.grid(axis="y", color=GRID, linewidth=0.8)
	axis.tick_params(colors=INK_MUTED, labelsize=9)


def figure_slopes(terms: pd.DataFrame, z: float, path: Path) -> None:
	"""Slopes x 100 and responses per standard deviation, for each component alone, jointly, and ROD."""
	rows = [("on", "on", "ON alone", OVERNIGHT), ("day", "day", "DAY alone", DAYTIME), ("on_day", "on", "ON, joint", JOINT), ("on_day", "day", "DAY, joint", JOINT), ("rod", "rod", "ROD (paper)", ROD)]
	table = terms[terms["variant"].eq("baseline")].set_index(["model", "term"])
	figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
	x = np.arange(len(rows))
	for position, (model, term, label, color) in enumerate(rows):
		row = table.loc[(model, term)]
		left.errorbar(position, row["coef_x100"], yerr=[[row["coef_x100"] - row["ci_low_x100"]], [row["ci_high_x100"] - row["coef_x100"]]], fmt="o", color=color, markersize=7, capsize=4, linewidth=1.8)
		right.errorbar(position, row["response_bps_per_sd"], yerr=[[z * row["standardized_se"] * row["response_bps_per_sd"] / row["standardized"]]] if row["standardized"] else 0, fmt="o", color=color, markersize=7, capsize=4, linewidth=1.8)
	for axis, title, ylabel in ((left, "Slope x 100, 95% intervals (Newey-West)", "Slope x 100"), (right, "LH response to a one-SD move of the predictor", "Basis points of LH per SD")):
		axis.axhline(0, color=INK_MUTED, linewidth=0.8)
		axis.set_xticks(x, [label for _, _, label, _ in rows], rotation=15, ha="right")
		axis.set_title(title, loc="left", color=INK, fontsize=10)
		axis.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
		style_axis(axis)
	figure.suptitle("Overnight and daytime predictors of the last half hour, CL 2021-2026", x=0.01, ha="left", color=INK, fontsize=11)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_over_time(rolling: pd.DataFrame, by_year: pd.DataFrame, window: int, path: Path) -> None:
	"""Rolling joint-model slopes with bands and the calendar-year estimates, one panel per component."""
	figure, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor=SURFACE, sharex=True)
	for axis, (term, color) in zip(axes, (("on", OVERNIGHT), ("day", DAYTIME))):
		rows = rolling[rolling["term"].eq(term)]
		axis.fill_between(rows["trade_date"], rows["ci_low_x100"], rows["ci_high_x100"], color=color, alpha=0.18, linewidth=0, label="95% band, rolling")
		axis.plot(rows["trade_date"], rows["coef_x100"], color=color, linewidth=2, label=f"Rolling {window}-day slope")
		years = by_year[by_year["term"].eq(term) & by_year["model"].eq("on_day")]
		centers = pd.to_datetime(years["year"].astype(str) + "-07-01")
		axis.errorbar(centers, years["coef_x100"], yerr=[years["coef_x100"] - years["ci_low_x100"], years["ci_high_x100"] - years["coef_x100"]], fmt="s", color=INK, markersize=5, capsize=3, linewidth=1.2, label="Calendar year")
		axis.axhline(0, color=INK_MUTED, linewidth=0.8)
		axis.set_title(f"{TERM_LABELS[term]} slope x 100 in LH ~ ON + DAY", loc="left", color=INK, fontsize=10)
		axis.set_ylabel("Slope x 100", color=INK_MUTED, fontsize=9)
		style_axis(axis)
	axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncol=3)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_oos(forecasts: pd.DataFrame, path: Path) -> None:
	figure, axis = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
	colors = {"on": OVERNIGHT, "day": DAYTIME, "on_day": JOINT, "rod": ROD}
	for model, rows in forecasts.groupby("model", sort=False):
		rows = rows.sort_values("trade_date")
		gain = ((rows["actual"] - rows["historical_mean"]) ** 2 - (rows["actual"] - rows["forecast"]) ** 2).cumsum() * 1e4
		axis.plot(rows["trade_date"], gain, color=colors[model], linewidth=2, label=MODEL_LABELS[model])
	axis.axhline(0, color=INK_MUTED, linewidth=0.8)
	axis.set_ylabel("Cumulative SSE(historical mean) - SSE(model), x 10,000", color=INK_MUTED, fontsize=9)
	axis.set_title("Out of sample: expanding-window forecasts against the historical mean", loc="left", color=INK, fontsize=11)
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
	min_train, window = config["oos"]["min_train_obs"], config["rolling"]["window"]

	features = pd.read_parquet(config["return_features"])
	sample_features = features[features["sample"]].sort_values("trade_date").reset_index(drop=True)
	sample = prepare(sample_features, config["target"], config["predictors"])
	lags = newey_west_lags(len(sample)) if config["inference"]["hac_lags"] == "auto" else int(config["inference"]["hac_lags"])

	terms, models, forecasts = fit_models(sample, config, lags, z, min_train)

	# OOS sensitivity to the first forecast date.
	sensitivity = []
	for initial in config["oos"]["sensitivity_min_train"]:
		for model, predictors in config["models"].items():
			rows = expanding_forecasts(sample["y"].to_numpy(), sample[predictors].to_numpy(), initial)
			stats = oos_statistics(rows, lags)
			sensitivity.append({"model": model, "min_train_obs": initial, "first_forecast": sample["trade_date"].iloc[initial].date(), "oos_forecasts": stats["oos_forecasts"], "oos_r2_pct": stats["oos_r2"] * config["scale"], "clark_west_t": stats["clark_west_t"], "clark_west_p": stats["clark_west_p"]})
	sensitivity = pd.DataFrame(sensitivity)

	# Robustness: HAC lags, log returns, the pre-expiry roll rule, excluded dates.
	robust_terms, robust_models = [terms], [models]
	for other in config["inference"]["sensitivity_lags"]:
		if other != lags:
			rt, rm, _ = fit_models(sample, config, other, z, None, variant=f"hac_lags_{other}")
			robust_terms.append(rt)
			robust_models.append(rm.assign(dropped=0))
	log_sample = prepare(sample_features, config["log_variant"]["target"], config["log_variant"]["predictors"])
	rt, rm, _ = fit_models(log_sample, config, lags, z, min_train, variant="log_returns")
	robust_terms.append(rt)
	robust_models.append(rm.assign(dropped=0))
	alternative = pre_expiry_features(config, features)
	alternative = alternative[alternative["sample"]].sort_values("trade_date").reset_index(drop=True)
	changed = int(alternative.set_index("trade_date")["contract_id"].ne(sample_features.set_index("trade_date")["contract_id"]).sum())
	rt, rm, _ = fit_models(prepare(alternative, config["target"], config["predictors"]), config, lags, z, min_train, variant="pre_expiry_roll_3_sessions")
	robust_terms.append(rt)
	robust_models.append(rm.assign(dropped=len(sample) - len(alternative), contract_changed_dates=changed))
	for name, rule in config["robustness"].items():
		drop = exclusion_mask(sample_features, rule)
		kept = prepare(sample_features[~drop].reset_index(drop=True), config["target"], config["predictors"])
		rt, rm, _ = fit_models(kept, config, lags, z, min_train, variant=name)
		robust_terms.append(rt)
		robust_models.append(rm.assign(dropped=int(drop.sum())))
	robustness = pd.concat(robust_terms, ignore_index=True)
	robustness_models = pd.concat(robust_models, ignore_index=True)
	robustness_models["dropped"] = robustness_models["dropped"].fillna(0).astype(int)

	# Over time: calendar years and a rolling window of the joint model.
	by_year = pd.concat([fit_models(rows.reset_index(drop=True), config, lags, z, None, variant=str(year))[0].assign(year=year) for year, rows in sample.groupby(sample["trade_date"].dt.year)], ignore_index=True)
	rolling_rows = []
	joint = config["equality_test"]["model"]
	for end in range(window, len(sample) + 1):
		piece = sample.iloc[end - window:end]
		fit = ols(piece["y"].to_numpy(), piece[config["models"][joint]].to_numpy(), lags)
		for index, term in enumerate(config["models"][joint], start=1):
			coefficient, error = fit["coefficients"][index], fit["errors"][index]
			rolling_rows.append({"trade_date": piece["trade_date"].iloc[-1], "model": joint, "term": term, "coef_x100": coefficient * config["scale"], "se_x100": error * config["scale"], "ci_low_x100": (coefficient - z * error) * config["scale"], "ci_high_x100": (coefficient + z * error) * config["scale"]})
	rolling = pd.DataFrame(rolling_rows)

	outputs = {
		"regressions": tables_dir / "on_day_regressions.csv", "models": tables_dir / "on_day_models.csv", "oos_forecasts": tables_dir / "on_day_oos_forecasts.csv",
		"oos_sensitivity": tables_dir / "on_day_oos_sensitivity.csv", "by_year": tables_dir / "on_day_by_year.csv", "rolling": tables_dir / "on_day_rolling.csv",
		"robustness": tables_dir / "on_day_robustness.csv", "robustness_models": tables_dir / "on_day_robustness_models.csv",
		"figure_slopes": figures_dir / "on_day_slopes.png", "figure_over_time": figures_dir / "on_day_over_time.png", "figure_oos": figures_dir / "on_day_oos.png",
	}
	tables = {
		"regressions": terms, "models": models, "oos_forecasts": forecasts, "oos_sensitivity": sensitivity, "by_year": by_year, "rolling": rolling,
		"robustness": robustness[robustness["variant"].ne("baseline")], "robustness_models": robustness_models,
	}
	for name, table in tables.items():
		table.to_csv(outputs[name], index=False, float_format="%.6g")
	figure_slopes(terms, z, outputs["figure_slopes"])
	figure_over_time(rolling, by_year, window, outputs["figure_over_time"])
	figure_oos(forecasts, outputs["figure_oos"])

	manifest = {
		"config": str(args.config), "settings": config, "hac_lags_used": lags,
		"sample": {"observations": len(sample), "dates": f"{sample['trade_date'].iloc[0].date()} to {sample['trade_date'].iloc[-1].date()}", "first_oos_forecast": str(sample["trade_date"].iloc[min_train].date()), "sd_pct": {name: float(sample[name].std(ddof=1) * 100) for name in ("y", "on", "day", "rod")}, "corr_on_day": float(sample["on"].corr(sample["day"]))},
		"pre_expiry_roll": {"contract_changed_dates": changed, "observations": len(alternative)},
		"inputs": {name: {"file": config[name], "sha256": sha256(Path(config[name]))} for name in ("return_features", "front_minutes", "roll_comparison")},
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
	}
	(tables_dir / "on_day.manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
		print(f"Sample {manifest['sample']['dates']}, n = {len(sample)}, Newey-West lags = {lags}, corr(ON, DAY) = {manifest['sample']['corr_on_day']:.3f}\n")
		print(terms[terms["term"].ne("const")][["model", "term", "coef_x100", "se_x100", "ci_low_x100", "ci_high_x100", "t_stat", "standardized", "response_bps_per_sd", "adj_r2_pct"]].to_string(index=False))
		print()
		print(models.drop(columns=["variant", "hac_lags"]).to_string(index=False))
	print(f"\nSaved tables to {tables_dir}/, figures to {figures_dir}/")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
