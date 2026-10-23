"""Compare the CL intraday momentum regressions of Baltussen et al. (2021) across 2021-2023 and 2024-2026 (#38).

Run from the repository root, after the return features (#28):

    python -m src.compare_subperiods --config config/subperiods.yaml

The three models of #29 (LH on ONFH; on ONFH, M and SLH; on ROD) are estimated on each subperiod with
Newey-West standard errors. The slope difference between the periods is tested in one pooled regression,
LH = a + d D + b x + g x D + e, where D = 1 in 2024-2026: g is the change in the slope, with a HAC
standard error, and a Wald test covers all of a model's slopes jointly. A significant slope in one period
and not in the other is therefore never read as a difference.

Outputs (tables in reports/tables, the figure in reports/figures):
- replication_subperiods.csv: every coefficient (intercept and slopes) for the full sample and each
  subperiod, with its standard error, t-statistic, p-value and interval, for every variant (baseline
  and 2026 exclusions) and lag rule (common 7 lags, and each regression's own automatic lag).
- replication_subperiod_fit.csv: one row per model and period: R2, adjusted R2, the HAC Wald test that
  all slopes are zero, residual and target standard deviations, and the out-of-sample R2 and
  Clark-West test of the #29 expanding-window forecasts that fall in that period. The forecasts come
  from one expanding window over the whole sample, split by forecast date; no period is trained on
  its own. `oos_interpretation` marks the OOS results of variants whose excluded dates were chosen
  after the fact (ex_post_sensitivity) as influence checks, not prospective forecast performance.
- replication_subperiod_differences.csv: the pooled tests of the 2024-2026 minus 2021-2023 change. When
  both lag rules give the same lag length for the pooled sample (7 for every variant here), the test
  is run once and labeled common_7=auto_n.
- replication_subperiods_bogoev.csv: Bogoev (2025) Table IA1's joint ONFH + ROD specification on the
  overlapping 2021-2023 window, next to the values reported there (context only, not a replication).
- replication_subperiods.png and replication_subperiods.manifest.json.
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
from src.replicate_regressions import (
	INK, INK_MUTED, LABELS, OURS, PAPER, SURFACE, exclusion_mask, expanding_forecasts, hac_covariance, newey_west_lags, normal_cdf, normal_quantile,
	oos_statistics, paper_table, style_axis,
)

# Subperiod colors extend the #29 roles (paper orange, full sample blue) with aqua and violet; the four
# pass the colorblind-separation check, and every row is labeled on the axis.
PERIOD_COLORS = {"2021_2023": "#1baf7a", "2024_2026": "#4a3aa7"}
PERIOD_LABELS = {"full": "2021–2026", "2021_2023": "2021–2023", "2024_2026": "2024–2026"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/subperiods.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> tuple[dict, dict]:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {"features", "replication_config", "tables_dir", "figures_dir", "periods", "lags", "variants", "bogoev"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	if len(config["periods"]) != 2:
		raise ValueError("The comparison needs exactly two periods.")
	with Path(config["replication_config"]).open(encoding="utf-8") as replication_file:
		replication = yaml.safe_load(replication_file)
	return config, replication


# --- Statistics -------------------------------------------------------------------------------


def chi2_sf(statistic: float, df: int) -> float:
	"""Upper tail of a chi-square distribution with integer degrees of freedom (closed forms; no scipy)."""
	if df < 1:
		raise ValueError("Degrees of freedom must be at least 1.")
	half = statistic / 2
	if df % 2 == 0:
		return math.exp(-half) * sum(half**i / math.factorial(i) for i in range(df // 2))
	root = math.sqrt(statistic)
	tail, term = 2 * (1 - normal_cdf(root)), 0.0
	density = math.exp(-half) / math.sqrt(2 * math.pi)
	for i in range(1, (df - 1) // 2 + 1):
		term += statistic ** (i - 0.5) / math.prod(range(1, 2 * i, 2))
	return tail + 2 * density * term


def regression(y: np.ndarray, X: np.ndarray, lags: int) -> dict:
	"""OLS with an intercept and the full Newey-West covariance (as in #29: Bartlett weights, n / (n - k))."""
	design = np.column_stack([np.ones(len(y)), X])
	coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
	residuals = y - design @ coefficients
	n, k = design.shape
	r2 = 1 - residuals @ residuals / ((y - y.mean()) @ (y - y.mean()))
	return {"coefficients": coefficients, "covariance": hac_covariance(design, residuals, lags), "n": n, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k)}


def lag_length(rule: str, n: int, common: int) -> int:
	return common if rule == "common_7" else newey_west_lags(n)


def period_mask(sample: pd.DataFrame, period: dict) -> pd.Series:
	mask = sample["trade_date"].ge(pd.Timestamp(period["start"]))
	if period.get("end"):
		mask &= sample["trade_date"].le(pd.Timestamp(period["end"]))
	return mask


def variant_mask(sample: pd.DataFrame, rule: dict) -> pd.Series:
	"""True for the dates a variant drops: the #29 exclusion rules, plus every date on or after a day.

	The `oos` key only labels how the variant's out-of-sample results may be read; it selects nothing.
	"""
	rule = {key: value for key, value in rule.items() if key != "oos"}
	if not rule:
		return pd.Series(False, index=sample.index)
	if "on_or_after" in rule:
		return sample["trade_date"].ge(pd.Timestamp(rule["on_or_after"]))
	return exclusion_mask(sample, rule)


def fit_models(sample: pd.DataFrame, models: dict, target: str, lags: int, z: float, scale: float) -> tuple[list[dict], list[dict]]:
	"""Coefficient rows and model-fit rows for each model on one sample.

	Coefficient rows: estimate x scale, Newey-West standard error, t, two-sided p and interval, for the
	intercept and every slope. Fit rows: R2 and adjusted R2 (x scale), the HAC Wald test that every slope
	is zero (the model's overall significance), and the residual and target standard deviations in percent.
	"""
	terms, fits = [], []
	y = sample[target].to_numpy(dtype=float)
	dates = {"n": len(y), "first_date": sample["trade_date"].iloc[0].date(), "last_date": sample["trade_date"].iloc[-1].date(), "hac_lags": lags}
	for model, predictors in models.items():
		fit = regression(y, sample[predictors].to_numpy(dtype=float), lags)
		b, V = fit["coefficients"], fit["covariance"]
		errors = np.sqrt(np.diag(V))
		for index, term in enumerate(["const", *predictors]):
			coefficient, error = b[index], errors[index]
			terms.append(
				{
					"model": model, "term": term, **dates, "coef_x100": coefficient * scale, "se_x100": error * scale, "t_stat": coefficient / error,
					"p_value": 2 * (1 - normal_cdf(abs(coefficient / error))), "ci_low_x100": (coefficient - z * error) * scale,
					"ci_high_x100": (coefficient + z * error) * scale, "r2_pct": fit["r2"] * scale, "adj_r2_pct": fit["adj_r2"] * scale,
				}
			)
		slopes = list(range(1, len(predictors) + 1))
		wald = float(b[slopes] @ np.linalg.solve(V[np.ix_(slopes, slopes)], b[slopes]))
		residuals = y - np.column_stack([np.ones(len(y)), sample[predictors].to_numpy(dtype=float)]) @ b
		fits.append(
			{
				"model": model, **dates, "r2_pct": fit["r2"] * scale, "adj_r2_pct": fit["adj_r2"] * scale,
				"wald_all_slopes": wald, "wald_df": len(slopes), "wald_p": chi2_sf(wald, len(slopes)),
				"residual_sd_pct": residuals.std(ddof=len(predictors) + 1) * 100, "lh_sd_pct": y.std(ddof=1) * 100, "lh_mean_bps": y.mean() * 1e4,
			}
		)
	return terms, fits


def oos_by_period(sample: pd.DataFrame, models: dict, target: str, min_train: int, periods: dict, lags: int, scale: float, forecasts: dict | None = None) -> tuple[list[dict], dict]:
	"""Out-of-sample R2 and Clark-West test of the #29 expanding-window forecasts, split by forecast date.

	The forecasts are fitted once on the whole sample in time order (training from its first date, the
	first forecast after min_train observations), as in #29; each period's statistics use only the
	forecasts that fall in it. Pass `forecasts` to reuse them for another lag length.
	"""
	forecasts = forecasts or {}
	rows = []
	for model, predictors in models.items():
		if model not in forecasts:
			frame = expanding_forecasts(sample[target].to_numpy(dtype=float), sample[predictors].to_numpy(dtype=float), min_train)
			forecasts[model] = frame.assign(trade_date=sample["trade_date"].iloc[frame["position"]].to_numpy())
		frame = forecasts[model]
		for period, spec in periods.items():
			part = frame[period_mask(frame, spec)] if spec else frame
			if len(part) < 2:
				continue
			stats = oos_statistics(part, lags)
			rows.append(
				{
					"model": model, "period": period, "oos_forecasts": stats["oos_forecasts"], "oos_first_date": part["trade_date"].iloc[0].date(),
					"oos_last_date": part["trade_date"].iloc[-1].date(), "oos_r2_pct": stats["oos_r2"] * scale,
					"clark_west_t": stats["clark_west_t"], "clark_west_p": stats["clark_west_p"],
				}
			)
	return rows, forecasts


def pooled_difference(sample: pd.DataFrame, later: pd.Series, predictors: list[str], target: str, lags: int, z: float, scale: float) -> tuple[list[dict], dict]:
	"""Slope change between the periods from LH = a + d D + b x + g x D, with D = 1 in the later period.

	Rows give each predictor's earlier slope b, later slope b + g and the change g, all from one HAC
	covariance of the pooled regression (dates in time order across the boundary), plus the same for
	the intercept (term "const", change d). The Wald test asks whether every slope change g is zero.
	"""
	y = sample[target].to_numpy(dtype=float)
	x = sample[predictors].to_numpy(dtype=float)
	d = later.to_numpy(dtype=float)
	fit = regression(y, np.column_stack([x, d, x * d[:, None]]), lags)
	b, V = fit["coefficients"], fit["covariance"]
	k = len(predictors)
	slopes, changes = list(range(1, k + 1)), list(range(k + 2, 2 * k + 2))
	rows = []
	for term, slope, change in zip(["const", *predictors], [0, *slopes], [k + 1, *changes]):
		later_se = math.sqrt(V[slope, slope] + V[change, change] + 2 * V[slope, change])
		change_se = math.sqrt(V[change, change])
		rows.append(
			{
				"term": term, "earlier_x100": b[slope] * scale, "earlier_se_x100": math.sqrt(V[slope, slope]) * scale,
				"later_x100": (b[slope] + b[change]) * scale, "later_se_x100": later_se * scale,
				"difference_x100": b[change] * scale, "difference_se_x100": change_se * scale,
				"difference_ci_low_x100": (b[change] - z * change_se) * scale, "difference_ci_high_x100": (b[change] + z * change_se) * scale,
				"difference_t": b[change] / change_se, "difference_p": 2 * (1 - normal_cdf(abs(b[change] / change_se))),
			}
		)
	gamma = b[changes]
	wald = float(gamma @ np.linalg.solve(V[np.ix_(changes, changes)], gamma))
	return rows, {"n": fit["n"], "n_later": int(d.sum()), "hac_lags": lags, "wald_stat": wald, "wald_df": k, "wald_p": chi2_sf(wald, k)}


# --- Figure -----------------------------------------------------------------------------------


def figure_subperiods(estimates: pd.DataFrame, paper: pd.DataFrame, path: Path) -> None:
	"""Each slope with its 95% interval: the paper, the full 2021-2026 sample, and the two subperiods.

	The 2024-2026 row is repeated without 10 March 2026 (open marker), the day that dominates the full
	sample (#29). Common 7-lag standard errors throughout.
	"""
	panels = estimates.loc[estimates["term"].ne("const"), ["model", "term"]].drop_duplicates().to_records(index=False)
	figure, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 3.4), facecolor=SURFACE)
	rows = [
		("paper", "1987–2020 (paper)", PAPER, "o"),
		("full", "2021–2026", OURS, "o"),
		("2021_2023", "2021–2023", PERIOD_COLORS["2021_2023"], "o"),
		("2024_2026", "2024–2026", PERIOD_COLORS["2024_2026"], "o"),
		("2024_2026_ex", "2024–2026 without\n10 March 2026", PERIOD_COLORS["2024_2026"], "open"),
	]
	common = estimates[estimates["lag_rule"].eq("common_7")]
	for axis, (model, term) in zip(axes, panels):
		style_axis(axis)
		axis.grid(False, axis="y")
		axis.axvline(0, color=INK_MUTED, linewidth=0.8)
		for position, (key, label, color, marker) in enumerate(rows):
			if key == "paper":
				match = paper[(paper["model"] == model) & (paper["term"] == term)]
			else:
				period, variant = ("2024_2026", "exclude_2026_03_10") if key == "2024_2026_ex" else (key, "baseline")
				match = common[(common["period"] == period) & (common["variant"] == variant) & (common["model"] == model) & (common["term"] == term)]
			if match.empty:
				continue
			row = match.iloc[0]
			y = len(rows) - 1 - position
			axis.plot([row["ci_low_x100"], row["ci_high_x100"]], [y, y], color=color, linewidth=2, solid_capstyle="round")
			face = SURFACE if marker == "open" else color
			axis.plot(row["coef_x100"], y, "o", color=color, markerfacecolor=face, markersize=7, markeredgecolor=color if marker == "open" else SURFACE, markeredgewidth=1.5)
		axis.set_yticks(range(len(rows)), [label for _, label, _, _ in reversed(rows)] if axis is axes[0] else [""] * len(rows))
		axis.set_ylim(-0.6, len(rows) - 0.4)
		axis.set_title(f"{LABELS[term]} in {LABELS[model]}", fontsize=10, color=INK, loc="left")
		axis.set_xlabel("slope × 100", fontsize=9, color=INK_MUTED)
	figure.suptitle("CL last-half-hour slopes by subperiod: estimates and 95% Newey-West intervals (7 lags)", fontsize=11, color=INK, x=0.01, ha="left")
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	# Set here rather than at import, so notebooks that import this module keep inline figures.
	matplotlib.use("Agg")
	args = parse_args()
	config, replication = load_config(args.config)
	tables_dir, figures_dir = Path(config["tables_dir"]), Path(config["figures_dir"])
	tables_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)
	models, target, scale = replication["models"], replication["target"], replication["scale"]
	z = normal_quantile(1 - (1 - replication["inference"]["confidence"]) / 2)
	common = int(config["lags"]["common"])
	(earlier_name, earlier), (later_name, later) = config["periods"].items()

	features = pd.read_parquet(config["features"])
	full = features[features["sample"]].sort_values("trade_date").reset_index(drop=True)

	min_train = replication["oos"]["min_train_obs"]
	estimates, fits, differences, walds = [], [], [], []
	for variant, rule in config["variants"].items():
		drop = variant_mask(full, rule)
		sample = full[~drop].reset_index(drop=True)
		parts = {"full": sample, earlier_name: sample[period_mask(sample, earlier)], later_name: sample[period_mask(sample, later)]}
		forecasts = None
		oos_label = rule.get("oos", "prospective")
		pooled = sample[period_mask(sample, earlier) | period_mask(sample, later)].reset_index(drop=True)
		pooled_lags = {lag_rule: lag_length(lag_rule, len(pooled), common) for lag_rule in ("common_7", "auto_n")}
		for lag_rule in ("common_7", "auto_n"):
			fit_rows = []
			for period, rows in parts.items():
				if rows.empty:
					continue
				lags = lag_length(lag_rule, len(rows), common)
				key = {"variant": variant, "dropped": int(drop.sum()), "lag_rule": lag_rule, "period": period}
				terms, model_fits = fit_models(rows, models, target, lags, z, scale)
				estimates += [{**key, **row} for row in terms]
				fit_rows += [{**key, **row} for row in model_fits]
			# Out of sample: the forecasts in each period, with the period's own Clark-West lag length.
			oos = []
			for period, spec in {"full": None, earlier_name: earlier, later_name: later}.items():
				count = len(parts[period]) if period in parts else len(sample)
				rows, forecasts = oos_by_period(sample, models, target, min_train, {period: spec}, lag_length(lag_rule, count, common), scale, forecasts)
				oos += rows
			oos = pd.DataFrame(oos).assign(
				oos_interpretation=oos_label, oos_training=f"one expanding window from {sample['trade_date'].iloc[0].date()}, split by forecast date",
			)
			fits += pd.DataFrame(fit_rows).merge(oos, on=["model", "period"], how="left").to_dict("records")
			indicator = period_mask(pooled, later)
			if indicator.all() or not indicator.any():
				continue
			# With equal lag lengths the two labels give the same pooled test, so it is run once.
			if lag_rule == "auto_n" and pooled_lags["auto_n"] == pooled_lags["common_7"]:
				continue
			label = "common_7=auto_n" if pooled_lags["auto_n"] == pooled_lags["common_7"] else lag_rule
			for model, predictors in models.items():
				rows, wald = pooled_difference(pooled, indicator, predictors, target, pooled_lags[lag_rule], z, scale)
				key = {"variant": variant, "lag_rule": label, "model": model, "earlier": earlier_name, "later": later_name}
				differences += [{**key, **row, "n": wald["n"], "n_later": wald["n_later"], "hac_lags": wald["hac_lags"], "wald_p": wald["wald_p"]} for row in rows]
				walds.append({**key, **wald})
	estimates = pd.DataFrame(estimates)
	fits = pd.DataFrame(fits)
	differences = pd.DataFrame(differences)

	# Bogoev (2025) Table IA1: LH on ONFH and ROD jointly, on the overlapping window of our sample.
	bogoev = config["bogoev"]
	window = full[period_mask(full, bogoev["window"])].reset_index(drop=True)
	bogoev_rows = []
	for lag_rule in ("common_7", "auto_n"):
		lags = lag_length(lag_rule, len(window), common)
		rows, _ = fit_models(window, {"onfh_rod_joint": bogoev["predictors"]}, target, lags, z, scale)
		bogoev_rows += [{"source": "this project, same specification", "lag_rule": lag_rule, **row} for row in rows if row["term"] != "const"]
	for term in bogoev["predictors"]:
		coefficient, parenthesis = bogoev["reported"][term]
		bogoev_rows.append(
			{
				"source": bogoev["source"], "lag_rule": "as reported", "model": "onfh_rod_joint", "term": term, "coef_as_reported": coefficient,
				"parenthesis_as_reported": parenthesis, "adj_r2_as_reported": bogoev["reported"]["adj_r2"],
			}
		)
	bogoev_table = pd.DataFrame(bogoev_rows)

	paper = paper_table(replication["paper"], z)
	outputs = {
		"estimates": tables_dir / "replication_subperiods.csv",
		"fit": tables_dir / "replication_subperiod_fit.csv",
		"differences": tables_dir / "replication_subperiod_differences.csv",
		"bogoev": tables_dir / "replication_subperiods_bogoev.csv",
		"figure": figures_dir / "replication_subperiods.png",
	}
	estimates.to_csv(outputs["estimates"], index=False, float_format="%.6g")
	fits.to_csv(outputs["fit"], index=False, float_format="%.6g")
	differences.to_csv(outputs["differences"], index=False, float_format="%.6g")
	bogoev_table.to_csv(outputs["bogoev"], index=False, float_format="%.6g")
	figure_subperiods(estimates[estimates["variant"].isin(["baseline", "exclude_2026_03_10"])], paper, outputs["figure"])

	counts = estimates[estimates["variant"].eq("baseline") & estimates["lag_rule"].eq("common_7")].drop_duplicates("period").set_index("period")
	manifest = {
		"config": Path(args.config).as_posix(),
		"settings": config,
		"replication_settings": {key: replication[key] for key in ("target", "models", "scale", "inference")},
		"periods": {period: {"n": int(row["n"]), "dates": f"{row['first_date']} to {row['last_date']}"} for period, row in counts.iterrows()},
		"wald_tests": walds,
		"inputs": [{"file": Path(path).as_posix(), "sha256": sha256(Path(path))} for path in (config["features"], config["replication_config"])],
		"outputs": {name: {"file": path.as_posix(), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"python": platform.python_version(),
		"pandas": pd.__version__,
		"numpy": np.__version__,
	}
	manifest_path = tables_dir / "replication_subperiods.manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	shown = ["period", "model", "term", "n", "coef_x100", "se_x100", "t_stat", "p_value", "ci_low_x100", "ci_high_x100"]
	fit_shown = ["period", "model", "n", "r2_pct", "adj_r2_pct", "wald_all_slopes", "wald_p", "oos_forecasts", "oos_r2_pct", "clark_west_t", "clark_west_p"]
	tested = ["model", "term", "earlier_x100", "later_x100", "difference_x100", "difference_se_x100", "difference_p", "wald_p"]
	baseline = estimates["variant"].eq("baseline") & estimates["lag_rule"].eq("common_7") & estimates["term"].ne("const")
	with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
		print(estimates.loc[baseline, shown].to_string(index=False))
		print("\nModel fit, overall significance and out-of-sample results (7 lags):")
		print(fits.loc[fits["variant"].eq("baseline") & fits["lag_rule"].eq("common_7"), fit_shown].to_string(index=False))
		print("\nSlope change, 2024-2026 minus 2021-2023 (pooled regression, 7 lags):")
		for variant in config["variants"]:
			rows = differences[differences["variant"].eq(variant) & differences["lag_rule"].str.startswith("common_7")]
			print(f"\n{variant}:")
			print(rows[tested].to_string(index=False))
	print(f"\nSaved tables to {tables_dir}/, the figure to {outputs['figure']} and manifest to {manifest_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
