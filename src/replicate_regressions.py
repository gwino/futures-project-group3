"""Replicate the CL intraday momentum regressions of Baltussen et al. (2021) for 2021-2026 (#29).

Run from the repository root, after the return features (#28):

    python -m src.replicate_regressions --config config/replication.yaml

The three models regress the last half-hour return LH (13:00-13:30 CT) on returns known by 13:00 CT:
ONFH (Eq. 5), ONFH + M + SLH (Eq. 6) and ROD (Eq. 7). Each estimate is reported with its Newey-West
standard error and confidence interval, with slopes multiplied by 100 as in the paper.

Outputs (tables in reports/tables, figures in reports/figures):
- replication_regressions.csv: full-sample estimates, standard errors, intervals, adjusted R2, OOS R2
  and the Clark-West statistic for each model.
- replication_vs_paper.csv: the same estimates next to Appendix Table B3 (CL, 1987-2020), with the
  paper's standard errors implied by its slopes and t-statistics.
- replication_oos_forecasts.csv: every expanding-window forecast, the historical-mean benchmark and
  the realized target; replication_oos_sensitivity.csv: OOS R2 for other initial windows.
- replication_robustness.csv: estimates under other HAC lags, the pre-expiry roll rule, and without
  roll, degraded, expiry-adjacent, post-holiday and extreme-ROD dates.
- replication_by_year.csv and replication_rolling.csv: estimates by calendar year and over a rolling window.
- replication_*.png: estimates against the paper, over time, and cumulative OOS performance.
- replication.manifest.json: settings with input and output hashes.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.build_return_features import build_features
from src.prepare_daily_data import cme_holidays, sha256, trade_calendar
from src.prepare_minute_data import held_contracts, reference_prices

# Chart colors: our sample in blue and the paper in orange throughout; the out-of-sample chart, which
# shows only our models, uses blue, aqua and violet (both sets pass a colorblind-separation check).
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, PAPER = "#2a78d6", "#eb6834"
MODEL_COLORS = ["#2a78d6", "#1baf7a", "#4a3aa7"]
LABELS = {"onfh": "ONFH", "onfh_m_slh": "ONFH + M + SLH", "rod": "ROD", "r_onfh": "ONFH", "r_m": "M", "r_slh": "SLH", "r_rod": "ROD"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/replication.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> dict:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {"features", "tables_dir", "figures_dir", "target", "models", "scale", "inference", "oos", "rolling", "robustness", "paper"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	return config


# --- Estimation -------------------------------------------------------------------------------


def newey_west_lags(n: int) -> int:
	"""The usual automatic lag length, floor(4 (n / 100)^(2/9))."""
	return int(math.floor(4 * (n / 100) ** (2 / 9)))


def normal_cdf(value: float) -> float:
	return 0.5 * math.erfc(-value / math.sqrt(2))


def normal_quantile(probability: float) -> float:
	"""Inverse of normal_cdf by bisection (accurate to about 1e-12, enough for interval widths)."""
	low, high = -10.0, 10.0
	for _ in range(100):
		middle = (low + high) / 2
		low, high = (middle, high) if normal_cdf(middle) < probability else (low, middle)
	return (low + high) / 2


def hac_covariance(X: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
	"""Newey-West covariance of OLS coefficients: Bartlett weights 1 - l / (lags + 1), times n / (n - k).

	Observations are taken in order, so a date excluded from the sample simply joins its neighbours.
	"""
	n, k = X.shape
	scores = X * residuals[:, None]
	meat = scores.T @ scores
	for lag in range(1, lags + 1):
		weight = 1 - lag / (lags + 1)
		cross = scores[lag:].T @ scores[:-lag]
		meat += weight * (cross + cross.T)
	bread = np.linalg.inv(X.T @ X)
	return bread @ meat @ bread * n / (n - k)


def ols(y: np.ndarray, X: np.ndarray, lags: int) -> dict:
	"""OLS with an intercept (added here) and Newey-West standard errors."""
	design = np.column_stack([np.ones(len(y)), X])
	coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
	residuals = y - design @ coefficients
	n, k = design.shape
	r2 = 1 - residuals @ residuals / ((y - y.mean()) @ (y - y.mean()))
	errors = np.sqrt(np.diag(hac_covariance(design, residuals, lags)))
	return {"coefficients": coefficients, "errors": errors, "n": n, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k), "lags": lags}


def expanding_forecasts(y: np.ndarray, X: np.ndarray, min_train: int) -> pd.DataFrame:
	"""One-step forecasts of y[t] from an OLS fit on observations 0..t-1, for t >= min_train.

	The benchmark is the historical mean of y[0..t-1]. Nothing from t onward enters the forecast for t.
	"""
	design = np.column_stack([np.ones(len(y)), X])
	rows = []
	for t in range(min_train, len(y)):
		coefficients, *_ = np.linalg.lstsq(design[:t], y[:t], rcond=None)
		rows.append({"position": t, "forecast": float(design[t] @ coefficients), "historical_mean": float(y[:t].mean()), "actual": float(y[t])})
	return pd.DataFrame(rows)


def oos_statistics(forecasts: pd.DataFrame, lags: int) -> dict:
	"""Out-of-sample R2 against the historical mean (Eq. 8) and the Clark-West (2007) test (Eq. 9).

	The Clark-West statistic is the mean of f_t with a Newey-West standard error; a positive mean
	means the model's adjusted squared error is below the historical mean's. The p-value is one-sided.
	"""
	actual, forecast, mean = forecasts["actual"].to_numpy(), forecasts["forecast"].to_numpy(), forecasts["historical_mean"].to_numpy()
	model_error, mean_error = (actual - forecast) ** 2, (actual - mean) ** 2
	f = mean_error - (model_error - (mean - forecast) ** 2)
	fit = ols(f, np.empty((len(f), 0)), lags)
	t_stat = fit["coefficients"][0] / fit["errors"][0]
	return {
		"oos_r2": 1 - model_error.sum() / mean_error.sum(), "oos_forecasts": len(forecasts),
		"clark_west_mean": fit["coefficients"][0], "clark_west_se": fit["errors"][0], "clark_west_t": t_stat, "clark_west_p": 1 - normal_cdf(t_stat),
	}


def estimate_models(sample: pd.DataFrame, models: dict, target: str, lags: int, z: float, scale: float, min_train: int | None) -> pd.DataFrame:
	"""One row per model and slope: estimate, standard error, interval and t-statistic (slopes x scale)."""
	rows = []
	y = sample[target].to_numpy()
	for model, predictors in models.items():
		X = sample[predictors].to_numpy()
		fit = ols(y, X, lags)
		extra = {}
		if min_train is not None and len(y) > min_train:
			extra = oos_statistics(expanding_forecasts(y, X, min_train), lags)
			extra["oos_r2_pct"] = extra.pop("oos_r2") * scale
		for index, term in enumerate(["const", *predictors]):
			coefficient, error = fit["coefficients"][index], fit["errors"][index]
			rows.append(
				{
					"model": model, "term": term, "coef_x100": coefficient * scale, "se_x100": error * scale,
					"ci_low_x100": (coefficient - z * error) * scale, "ci_high_x100": (coefficient + z * error) * scale,
					"t_stat": coefficient / error, "p_value": 2 * (1 - normal_cdf(abs(coefficient / error))),
					"n": fit["n"], "first_date": sample["trade_date"].iloc[0].date(), "last_date": sample["trade_date"].iloc[-1].date(),
					"hac_lags": lags, "r2_pct": fit["r2"] * scale, "adj_r2_pct": fit["adj_r2"] * scale, **extra,
				}
			)
	return pd.DataFrame(rows)


# --- Sample variants --------------------------------------------------------------------------


def exclusion_mask(sample: pd.DataFrame, rule: dict) -> pd.Series:
	"""True for the dates a robustness rule drops."""
	column = sample[rule["column"]]
	if "at_most" in rule:
		return column.le(rule["at_most"]).fillna(False)
	if "above" in rule:
		return column.gt(rule["above"]).fillna(False)
	if "abs_quantile_above" in rule:
		return column.abs().gt(column.abs().quantile(rule["abs_quantile_above"]))
	return column.astype(bool)


def pre_expiry_features(config: dict, features: pd.DataFrame) -> pd.DataFrame:
	"""The return features rebuilt under the existing roll rule, 3 trade dates before expiry (#17/#18).

	Between that roll and the paper's later one the held contract is the next listed month, whose
	minutes are in the front-contract file, so the reference prices can be taken again from the same
	bars without rerunning the minute pipeline. Session flags stay those of the paper rule.
	"""
	with Path(config["reference_config"]).open(encoding="utf-8") as reference_file:
		minute = yaml.safe_load(reference_file)
	with Path(minute["daily_config"]).open(encoding="utf-8") as daily_file:
		daily = yaml.safe_load(daily_file)
	with Path(config["return_config"]).open(encoding="utf-8") as return_file:
		returns = yaml.safe_load(return_file)
	front = pd.read_parquet(config["front_minutes"], columns=["ts_utc", "instrument_id", "contract_id", "close"])
	instruments = front.drop_duplicates("contract_id").set_index("contract_id")["instrument_id"]
	references = pd.read_parquet(minute["output_dir"] + "/cl_minute_reference_prices.parquet")
	rolls = pd.read_csv(config["roll_comparison"], parse_dates=["roll_date_pre_expiry"]).dropna(subset=["roll_date_pre_expiry"])
	rolls = pd.DataFrame(
		{
			"roll_date": rolls["roll_date_pre_expiry"], "from_instrument_id": rolls["from_contract"].map(instruments),
			"to_instrument_id": rolls["to_contract"].map(instruments),
		}
	).sort_values("roll_date")
	dates = pd.DatetimeIndex(references["trade_date"])
	held = held_contracts(dates, rolls).astype(front["instrument_id"].dtype)
	start, end = dates.min() - pd.Timedelta(days=30), dates.max() + pd.Timedelta(days=400)
	calendar = trade_calendar(start, end, cme_holidays(start, end, daily["session"]["juneteenth_from"]))
	reference_cfg = minute["reference"]
	prices = reference_prices(
		front, held, calendar, minute["session"]["timezone"], reference_cfg["previous_close"], reference_cfg["points"], reference_cfg["max_stale_minutes"]
	).set_index("trade_date")
	rebuilt = references.set_index("trade_date")
	price_columns = [column for column in prices.columns if column.startswith(("p_", "stale_"))]
	rebuilt[price_columns] = prices[price_columns]
	rebuilt["contract_id"] = held.map(dict(zip(instruments.to_numpy(), instruments.index)))
	rebuilt["roll_flag"] = rebuilt.index.isin(rolls["roll_date"])
	rebuilt["complete"] = prices["complete"]
	rebuilt["paper_sample"] = rebuilt["complete"] & ~rebuilt["early_close"] & ~rebuilt["low_volume"]
	expiry = features.set_index("trade_date")["sessions_to_expiry"]
	start_date, end_date = pd.Timestamp(returns["sample"]["start"]), returns["sample"].get("end")
	return build_features(rebuilt.reset_index(), expiry, returns["returns"], start_date, pd.Timestamp(end_date) if end_date else None, False)


# --- Paper comparison -------------------------------------------------------------------------


def paper_table(paper: dict, z: float) -> pd.DataFrame:
	"""Appendix Table B3 for CL, with the standard error implied by each slope and t-statistic."""
	rows = []
	for model, entry in paper.items():
		if not isinstance(entry, dict):
			continue
		for term, (coefficient, t_stat) in entry["coefficients"].items():
			error = abs(coefficient / t_stat)
			rows.append(
				{
					"model": model, "term": term, "coef_x100": coefficient, "se_x100": error, "ci_low_x100": coefficient - z * error,
					"ci_high_x100": coefficient + z * error, "t_stat": t_stat, "adj_r2_pct": entry["adj_r2"], "oos_r2_pct": entry["oos_r2"],
				}
			)
	return pd.DataFrame(rows)


def compare_with_paper(ours: pd.DataFrame, paper: pd.DataFrame, paper_sample: str, paper_n: int, our_sample: str) -> pd.DataFrame:
	columns = ["coef_x100", "se_x100", "ci_low_x100", "ci_high_x100", "t_stat", "adj_r2_pct", "oos_r2_pct"]
	slopes = ours[ours["term"].ne("const")]
	table = paper[["model", "term", *columns]].merge(slopes[["model", "term", *columns, "n"]], on=["model", "term"], suffixes=("_paper", "_ours"))
	table.insert(2, "paper_sample", f"{paper_sample} (n = {paper_n})")
	table.insert(3, "our_sample", our_sample)
	# Difference of the two slopes, with the standard error of independent samples.
	table["difference_x100"] = table["coef_x100_ours"] - table["coef_x100_paper"]
	table["difference_se_x100"] = np.hypot(table["se_x100_ours"], table["se_x100_paper"])
	return table


# --- Figures ----------------------------------------------------------------------------------


def style_axis(axis: plt.Axes) -> None:
	axis.set_facecolor(SURFACE)
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		axis.spines[side].set_color(GRID)
	axis.tick_params(colors=INK_MUTED, labelsize=9)
	axis.grid(True, color=GRID, linewidth=0.6)
	axis.set_axisbelow(True)


def figure_vs_paper(comparison: pd.DataFrame, path: Path) -> None:
	"""Slope estimates with 95% intervals: the paper's 1987-2020 sample against 2021-2026, one panel per slope."""
	panels = comparison[["model", "term"]].drop_duplicates().to_records(index=False)
	figure, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.8), facecolor=SURFACE)
	for axis, (model, term) in zip(axes, panels):
		row = comparison[(comparison["model"] == model) & (comparison["term"] == term)].iloc[0]
		style_axis(axis)
		axis.grid(False, axis="y")
		axis.axvline(0, color=INK_MUTED, linewidth=0.8)
		for position, (suffix, color, label) in enumerate([("paper", PAPER, "1987–2020 (paper)"), ("ours", OURS, "2021–2026")]):
			axis.plot([row[f"ci_low_x100_{suffix}"], row[f"ci_high_x100_{suffix}"]], [position, position], color=color, linewidth=2, solid_capstyle="round")
			axis.plot(row[f"coef_x100_{suffix}"], position, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
		axis.set_yticks([0, 1], ["1987–2020\n(paper)", "2021–2026"] if axis is axes[0] else ["", ""])
		axis.set_ylim(-0.6, 1.6)
		axis.set_title(f"{LABELS[term]} in {LABELS[model]}", fontsize=10, color=INK, loc="left")
		axis.set_xlabel("slope × 100", fontsize=9, color=INK_MUTED)
	figure.suptitle("CL last-half-hour slopes: estimates and 95% Newey-West intervals", fontsize=11, color=INK, x=0.01, ha="left")
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_over_time(rolling: pd.DataFrame, by_year: pd.DataFrame, paper: pd.DataFrame, window: int, path: Path) -> None:
	"""Rolling-window slopes of the two single-predictor models with 95% bands, and the paper's estimate."""
	terms = [("onfh", "r_onfh"), ("rod", "r_rod")]
	figure, axes = plt.subplots(len(terms), 1, figsize=(9, 5.6), sharex=True, facecolor=SURFACE)
	for axis, (model, term) in zip(axes, terms):
		style_axis(axis)
		path_rows = rolling[(rolling["model"] == model) & (rolling["term"] == term)]
		axis.fill_between(path_rows["last_date"], path_rows["ci_low_x100"], path_rows["ci_high_x100"], color=OURS, alpha=0.15, linewidth=0)
		axis.plot(path_rows["last_date"], path_rows["coef_x100"], color=OURS, linewidth=2, label=f"{window}-day rolling estimate")
		years = by_year[(by_year["model"] == model) & (by_year["term"] == term)]
		middles = pd.to_datetime(years["first_date"]) + (pd.to_datetime(years["last_date"]) - pd.to_datetime(years["first_date"])) / 2
		axis.errorbar(middles, years["coef_x100"], yerr=[years["coef_x100"] - years["ci_low_x100"], years["ci_high_x100"] - years["coef_x100"]],
			fmt="o", color=INK_MUTED, markersize=5, linewidth=1, capsize=0, label="calendar year")
		reference = paper[(paper["model"] == model) & (paper["term"] == term)].iloc[0]
		axis.axhline(reference["coef_x100"], color=PAPER, linewidth=2, linestyle=(0, (4, 2)), label="paper, 1987–2020")
		axis.axhline(0, color=INK_MUTED, linewidth=0.8)
		axis.set_title(f"{LABELS[term]} slope × 100 (LH ~ {LABELS[model]})", fontsize=10, color=INK, loc="left")
	axes[-1].set_xlabel("last trade date of the rolling window; calendar-year estimates at mid-year", fontsize=9, color=INK_MUTED)
	figure.suptitle("CL intraday momentum over time, with 95% Newey-West bands", fontsize=11, color=INK, x=0.01, ha="left")
	handles, labels = axes[0].get_legend_handles_labels()
	figure.tight_layout(rect=(0, 0, 1, 0.93))
	figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), fontsize=8, frameon=False, labelcolor=INK_MUTED, ncol=3)
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def figure_oos(forecasts: pd.DataFrame, path: Path) -> None:
	"""Cumulative squared-error advantage of each model over the historical mean; rising means the model is better."""
	figure, axis = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
	style_axis(axis)
	axis.axhline(0, color=INK_MUTED, linewidth=0.8)
	for model, color in zip(forecasts["model"].unique(), MODEL_COLORS):
		rows = forecasts[forecasts["model"] == model]
		gain = ((rows["actual"] - rows["historical_mean"]) ** 2 - (rows["actual"] - rows["forecast"]) ** 2).cumsum() * 1e4
		axis.plot(rows["trade_date"], gain, color=color, linewidth=2, label=LABELS[model])
		axis.annotate(LABELS[model], (rows["trade_date"].iloc[-1], gain.iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
	axis.set_ylabel("cumulative SSE gain (%²)", fontsize=9, color=INK_MUTED)
	axis.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK_MUTED)
	axis.set_title("Out-of-sample: historical-mean squared error minus model squared error, summed", fontsize=11, color=INK, loc="left")
	axis.margins(x=0.08)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	args = parse_args()
	config = load_config(args.config)
	tables_dir, figures_dir = Path(config["tables_dir"]), Path(config["figures_dir"])
	tables_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)
	models, target, scale = config["models"], config["target"], config["scale"]
	inference, oos = config["inference"], config["oos"]
	z = normal_quantile(1 - (1 - inference["confidence"]) / 2)

	features = pd.read_parquet(config["features"])
	sample = features[features["sample"]].sort_values("trade_date").reset_index(drop=True)
	lags = newey_west_lags(len(sample)) if inference["hac_lags"] == "auto" else int(inference["hac_lags"])
	min_train = oos["min_train_obs"]
	our_sample = f"{sample['trade_date'].iloc[0].date()} to {sample['trade_date'].iloc[-1].date()}"

	# Full sample, with expanding-window forecasts.
	baseline = estimate_models(sample, models, target, lags, z, scale, min_train)
	forecasts = []
	for model, predictors in models.items():
		rows = expanding_forecasts(sample[target].to_numpy(), sample[predictors].to_numpy(), min_train)
		rows.insert(0, "trade_date", sample["trade_date"].iloc[rows["position"]].to_numpy())
		forecasts.append(rows.drop(columns="position").assign(model=model))
	forecasts = pd.concat(forecasts, ignore_index=True)

	paper = paper_table(config["paper"], z)
	comparison = compare_with_paper(baseline, paper, config["paper"]["sample"], config["paper"]["observations"], our_sample)

	sensitivity = []
	for initial in oos["sensitivity_min_train"]:
		for model, predictors in models.items():
			rows = expanding_forecasts(sample[target].to_numpy(), sample[predictors].to_numpy(), initial)
			stats = oos_statistics(rows, lags)
			sensitivity.append(
				{
					"model": model, "min_train_obs": initial, "first_forecast": sample["trade_date"].iloc[initial].date(),
					"oos_forecasts": stats["oos_forecasts"], "oos_r2_pct": stats["oos_r2"] * scale,
					"clark_west_t": stats["clark_west_t"], "clark_west_p": stats["clark_west_p"],
				}
			)
	sensitivity = pd.DataFrame(sensitivity)

	# Robustness: HAC lags, the pre-expiry roll rule, and excluded dates.
	variants = [baseline.assign(variant="baseline", dropped=0)]
	for other in inference["sensitivity_lags"]:
		if other != lags:
			variants.append(estimate_models(sample, models, target, other, z, scale, None).assign(variant=f"hac_lags_{other}", dropped=0))
	alternative = pre_expiry_features(config, features)
	alternative_sample = alternative[alternative["sample"]].sort_values("trade_date").reset_index(drop=True)
	changed = alternative_sample.set_index("trade_date")["contract_id"].ne(sample.set_index("trade_date")["contract_id"]).sum()
	variants.append(
		estimate_models(alternative_sample, models, target, lags, z, scale, min_train).assign(
			variant="pre_expiry_roll_3_sessions", dropped=len(sample) - len(alternative_sample), contract_changed_dates=int(changed)
		)
	)
	for name, rule in config["robustness"].items():
		drop = exclusion_mask(sample, rule)
		kept = sample[~drop].reset_index(drop=True)
		variants.append(estimate_models(kept, models, target, lags, z, scale, min_train).assign(variant=name, dropped=int(drop.sum())))
	robustness = pd.concat(variants, ignore_index=True)
	robustness = robustness[["variant", "dropped", *[column for column in robustness.columns if column not in ("variant", "dropped")]]]

	# Over time: calendar years and a rolling window.
	by_year = pd.concat(
		[estimate_models(rows, models, target, lags, z, scale, None).assign(year=year) for year, rows in sample.groupby(sample["trade_date"].dt.year)],
		ignore_index=True,
	)
	window = config["rolling"]["window"]
	single = {model: predictors for model, predictors in models.items() if len(predictors) == 1}
	rolling = pd.concat(
		[estimate_models(sample.iloc[end - window:end], single, target, lags, z, scale, None) for end in range(window, len(sample) + 1)],
		ignore_index=True,
	)
	rolling = rolling[rolling["term"].ne("const")].drop(columns=["r2_pct"])

	outputs = {
		"regressions": tables_dir / "replication_regressions.csv",
		"vs_paper": tables_dir / "replication_vs_paper.csv",
		"oos_forecasts": tables_dir / "replication_oos_forecasts.csv",
		"oos_sensitivity": tables_dir / "replication_oos_sensitivity.csv",
		"robustness": tables_dir / "replication_robustness.csv",
		"by_year": tables_dir / "replication_by_year.csv",
		"rolling": tables_dir / "replication_rolling.csv",
		"figure_vs_paper": figures_dir / "replication_vs_paper.png",
		"figure_over_time": figures_dir / "replication_slopes_over_time.png",
		"figure_oos": figures_dir / "replication_oos_cumulative.png",
	}
	tables = {"regressions": baseline, "vs_paper": comparison, "oos_forecasts": forecasts, "oos_sensitivity": sensitivity, "robustness": robustness, "by_year": by_year, "rolling": rolling}
	for name, table in tables.items():
		table.to_csv(outputs[name], index=False, float_format="%.6g")
	figure_vs_paper(comparison, outputs["figure_vs_paper"])
	figure_over_time(rolling, by_year, paper, window, outputs["figure_over_time"])
	figure_oos(forecasts, outputs["figure_oos"])

	inputs = [Path(config[key]) for key in ("features", "front_minutes", "roll_comparison")]
	manifest = {
		"config": str(args.config),
		"settings": config,
		"hac_lags_used": lags,
		"sample": {"observations": len(sample), "dates": our_sample, "first_oos_forecast": str(sample["trade_date"].iloc[min_train].date())},
		"inputs": [{"file": str(path), "sha256": sha256(path)} for path in inputs],
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"python": platform.python_version(),
		"pandas": pd.__version__,
		"numpy": np.__version__,
	}
	manifest_path = tables_dir / "replication.manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	shown = ["model", "term", "coef_x100", "se_x100", "ci_low_x100", "ci_high_x100", "t_stat", "adj_r2_pct", "oos_r2_pct", "clark_west_t"]
	with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
		print(f"Sample {our_sample}, n = {len(sample)}, Newey-West lags = {lags}, first OOS forecast {manifest['sample']['first_oos_forecast']}\n")
		print(baseline[baseline["term"].ne("const")][shown].to_string(index=False))
		print("\nPaper (Table B3, CL 1987-2020):")
		print(paper.to_string(index=False))
		print(f"\nPre-expiry roll rule: {int(changed)} trade dates hold a different contract.")
	print(f"\nSaved tables to {tables_dir}/, figures to {figures_dir}/ and manifest to {manifest_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
