"""Build overnight and daytime realized variance for CL from the 1-minute bars (#32).

Run from the repository root, after the minute pipeline (#27):

    python -m src.build_volatility_features --config config/volatility_features.yaml

For each trade date t the overnight window runs from the previous 13:30 CT to 08:00 CT and the daytime
window from 08:00 to 13:00 CT, both on the contract held on t. Realized variance is the sum of squared
1-minute log returns inside the window; the first return after a scheduled halt (the daily 16:00-17:00
CT halt, weekends, holidays) is a reopening jump and is recorded separately rather than summed. Every
window ends by 13:00 CT, so no feature uses a trade from the 13:00-13:30 CT target interval.

Outputs (in the configured output directory):
- cl_volatility_features.parquet: one row per trade date with rv_on, rv_day, rv_rod (their sum), the
  trailing mean, the number of returns and bars behind each, the reopening jumps, the coverage of each
  path, the sample flags, and the reference prices the paths start and end at.
- cl_volatility_features_columns.csv: the column dictionary.
- cl_volatility_features_summary.csv and cl_volatility_features.manifest.json: checks, settings, hashes.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.prepare_daily_data import sha256
from src.prepare_minute_data import at_clock, clock_label

MINUTE = pd.Timedelta(minutes=1)
FLAG_COLUMNS = ["roll_flag", "folded_holiday", "degraded_date", "early_close", "low_volume", "complete", "paper_sample"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/volatility_features.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> tuple[dict, dict]:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {"reference_prices", "front_minutes", "session_quality", "minute_config", "output_dir", "sample", "windows", "decision_time", "returns", "trailing"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	with Path(config["minute_config"]).open(encoding="utf-8") as minute_file:
		minute = yaml.safe_load(minute_file)
	check_windows(config["windows"], config["decision_time"])
	return config, minute


# --- Windows ----------------------------------------------------------------------------------


def clock_minutes(clock: str) -> int:
	"""Minutes from midnight on the trade date; a "prev_" clock is on the previous trade date."""
	previous = clock.startswith("prev_")
	hours, minutes = clock.removeprefix("prev_").split(":")
	return int(hours) * 60 + int(minutes) - (24 * 60 if previous else 0)


def check_windows(windows: dict, decision_time: str) -> None:
	"""Every window runs forward in time and ends by the decision time."""
	names = [name for name in windows if not isinstance(name, str)]
	if names:
		raise ValueError(f"Window names must be strings, got {names}; quote YAML keys such as \"on\".")
	for name, (start, end) in windows.items():
		if clock_minutes(start) >= clock_minutes(end):
			raise ValueError(f"Window {name} must start before it ends: {start} -> {end}.")
		if clock_minutes(end) > clock_minutes(decision_time):
			raise ValueError(f"Window {name} ends at {end}, after the {decision_time} decision time (look-ahead).")


def window_instants(references: pd.DataFrame, clock: str, timezone_name: str) -> pd.DatetimeIndex:
	"""UTC instant of a window boundary on each trade date (or on the previous trade date for prev_ clocks)."""
	dates = references["prev_trade_date"] if clock.startswith("prev_") else references["trade_date"]
	return pd.DatetimeIndex(at_clock(pd.DatetimeIndex(dates), clock.removeprefix("prev_"), timezone_name))


def reference_column(clock: str) -> str:
	return "p_" + ("prev_" if clock.startswith("prev_") else "") + clock_label(clock.removeprefix("prev_"))


# --- Realized variance ------------------------------------------------------------------------


def window_paths(minutes: pd.DataFrame, references: pd.DataFrame, start: str, end: str, timezone_name: str) -> pd.DataFrame:
	"""The held contract's bars from the one starting a minute before `start` to the one starting a minute before `end`.

	Bars are selected by contract and time, not by their trade date, so a roll date's overnight path
	starts on the previous trade date in the newly held contract. Windows of one kind never overlap
	across trade dates, so each bar belongs to at most one of them.
	"""
	bounds = pd.DataFrame(
		{
			"trade_date": references["trade_date"].to_numpy(), "instrument_id": references["instrument_id"].to_numpy(),
			"first_bar": window_instants(references, start, timezone_name) - MINUTE, "last_bar": window_instants(references, end, timezone_name) - MINUTE,
		}
	)
	ordered = bounds.sort_values(["instrument_id", "first_bar"])
	if ordered["first_bar"].le(ordered.groupby("instrument_id")["last_bar"].shift()).any():
		raise ValueError(f"Windows {start} -> {end} overlap across trade dates.")
	bars = minutes[["ts_utc", "instrument_id", "close"]].sort_values("ts_utc")
	keyed = pd.merge_asof(bars, bounds.sort_values("first_bar"), left_on="ts_utc", right_on="first_bar", by="instrument_id", direction="backward")
	keyed = keyed[keyed["trade_date"].notna() & keyed["ts_utc"].le(keyed["last_bar"])]
	return keyed.sort_values(["trade_date", "ts_utc"]).reset_index(drop=True)


def realized_variance(paths: pd.DataFrame, halt_gap_minutes: int, name: str) -> pd.DataFrame:
	"""Per trade date: sum of squared 1-minute log returns, with returns across a halt counted as jumps.

	`gap` is the minutes between consecutive bars. A return with gap >= halt_gap_minutes spans a halt and
	goes into jumps_/jump_var_ instead of rv_. Columns are suffixed with the window name.
	"""
	frame = paths.copy()
	group = frame.groupby("trade_date", sort=True)
	frame["log_return"] = group["close"].transform(lambda close: np.log(close).diff())
	frame["gap"] = group["ts_utc"].diff() / MINUTE
	frame["jump"] = frame["gap"].ge(halt_gap_minutes)
	kept = frame[frame["log_return"].notna() & ~frame["jump"]]
	jumped = frame[frame["jump"]]
	per_date = pd.DataFrame(
		{
			f"rv_{name}": kept.groupby("trade_date")["log_return"].apply(lambda r: float((r**2).sum())),
			f"n_{name}": kept.groupby("trade_date").size(),
			f"max_gap_{name}": kept.groupby("trade_date")["gap"].max(),
			f"jumps_{name}": jumped.groupby("trade_date").size(),
			f"jump_var_{name}": jumped.groupby("trade_date")["log_return"].apply(lambda r: float((r**2).sum())),
			f"bars_{name}": group.size(),
			f"first_close_{name}": group["close"].first(),
			f"last_close_{name}": group["close"].last(),
			f"first_bar_{name}": group["ts_utc"].first(),
			f"last_bar_{name}": group["ts_utc"].last(),
		}
	)
	per_date[[f"jumps_{name}", f"jump_var_{name}"]] = per_date[[f"jumps_{name}", f"jump_var_{name}"]].fillna(0)
	per_date[f"n_{name}"] = per_date[f"n_{name}"].fillna(0).astype(int)
	per_date[f"jumps_{name}"] = per_date[f"jumps_{name}"].astype(int)
	return per_date


def build_features(
	references: pd.DataFrame, minutes: pd.DataFrame, quality: pd.DataFrame, windows: dict, timezone_name: str, halt_gap_minutes: int, trailing_days: int,
) -> pd.DataFrame:
	"""One row per trade date with the realized variance of each window, their sum, and the trailing mean."""
	frame = references.sort_values("trade_date").reset_index(drop=True)
	features = frame[["trade_date", "prev_trade_date", "contract_id"]].copy()
	for name, (start, end) in windows.items():
		paths = window_paths(minutes, frame, start, end, timezone_name)
		per_date = realized_variance(paths, halt_gap_minutes, name)
		features = features.merge(per_date, left_on="trade_date", right_index=True, how="left")
		features[f"start_price_{name}"] = frame[reference_column(start)]
		features[f"end_price_{name}"] = frame[reference_column(end)]
		features[f"endpoints_match_{name}"] = features[f"first_close_{name}"].eq(features[f"start_price_{name}"]) & features[f"last_close_{name}"].eq(features[f"end_price_{name}"])
		features[f"n_{name}"] = features[f"n_{name}"].fillna(0).astype(int)
		features[f"jumps_{name}"] = features[f"jumps_{name}"].fillna(0).astype(int)
		features[f"bars_{name}"] = features[f"bars_{name}"].fillna(0).astype(int)
	names = list(windows)
	features["rv_rod"] = features[[f"rv_{name}" for name in names]].sum(axis=1, min_count=len(names))
	features["rv_rod_with_jumps"] = features["rv_rod"] + features[[f"jump_var_{name}" for name in names]].sum(axis=1)
	features[f"rv_rod_{trailing_days}d"] = features["rv_rod"].rolling(trailing_days).mean()
	for name in [*names, "rod"]:
		features[f"vol_{name}_pct"] = np.sqrt(features[f"rv_{name}"]) * 100
	coverage = quality.set_index("trade_date")[["overnight_minutes", "overnight_missing_minutes", "overnight_longest_gap_minutes", "window_minutes", "window_missing_minutes"]]
	features = features.merge(coverage, left_on="trade_date", right_index=True, how="left")
	features[FLAG_COLUMNS] = frame[FLAG_COLUMNS]
	features["sample"] = features["paper_sample"] & features["rv_rod"].notna()
	return features


# --- Documentation ----------------------------------------------------------------------------


def column_dictionary(features: pd.DataFrame, windows: dict, halt_gap_minutes: int, trailing_days: int) -> pd.DataFrame:
	meaning = {
		"trade_date": "CME trade date t", "prev_trade_date": "Previous trade date", "contract_id": "Contract held on t; every path uses it",
		"rv_rod": "Overnight plus daytime realized variance (previous 13:30 -> 13:00 CT), reopening jumps excluded",
		"rv_rod_with_jumps": "rv_rod plus the squared reopening jumps of both windows",
		f"rv_rod_{trailing_days}d": f"Mean of rv_rod over the last {trailing_days} trade dates including t (known at 13:00 CT on t)",
		"vol_rod_pct": "sqrt(rv_rod) x 100: realized volatility of the combined window in percent",
		"overnight_minutes": "Held-contract minutes traded between the previous 13:30 and 08:00 CT (session quality, #27)",
		"overnight_missing_minutes": "Minutes in that interval in which another outright traded but the held contract did not",
		"overnight_longest_gap_minutes": "Longest such run overnight", "window_minutes": "Held-contract minutes traded 08:00-13:30 CT",
		"window_missing_minutes": "Minutes 08:00-13:30 CT in which another outright traded but the held contract did not",
		"sample": "paper_sample and both windows have a path: the regression sample", "roll_flag": "t is the first session in a new contract",
		"folded_holiday": "A weekend or holiday session was folded into t", "degraded_date": "Databento flags data quality on t",
		"early_close": "The market closed 30+ minutes before 16:00 CT on t", "low_volume": "Held-contract session volume under 100 contracts",
		"complete": "All reference prices exist on t", "paper_sample": "complete, not an early close, and not low volume (paper filters)",
	}
	for name, (start, end) in windows.items():
		label = f"{start.replace('prev_', 'previous ')} -> {end} CT"
		meaning.update(
			{
				f"rv_{name}": f"Realized variance {label}: sum of squared 1-minute log returns of the held contract, returns across a halt of {halt_gap_minutes}+ minutes excluded",
				f"n_{name}": f"Number of 1-minute returns in rv_{name}", f"max_gap_{name}": f"Longest gap in minutes between consecutive bars among the returns in rv_{name}",
				f"jumps_{name}": f"Returns across a halt of {halt_gap_minutes}+ minutes {label} (the daily 16:00-17:00 halt, weekends, holidays), excluded from rv_{name}",
				f"jump_var_{name}": f"Sum of the squared excluded reopening returns {label}", f"bars_{name}": f"Bars in the path {label}, including the starting bar",
				f"first_close_{name}": f"Close of the first bar of the path (the price at {start.replace('prev_', 'previous ')})", f"last_close_{name}": f"Close of the last bar of the path (the price at {end})",
				f"first_bar_{name}": "Start of the first bar of the path, UTC", f"last_bar_{name}": "Start of the last bar of the path, UTC",
				f"start_price_{name}": f"Reference price at {start.replace('prev_', 'previous ')} (#27)", f"end_price_{name}": f"Reference price at {end} (#27)",
				f"endpoints_match_{name}": "The path's first and last closes equal the reference prices", f"vol_{name}_pct": f"sqrt(rv_{name}) x 100: realized volatility {label} in percent",
			}
		)
	return pd.DataFrame({"column": features.columns, "dtype": [str(dtype) for dtype in features.dtypes], "meaning": [meaning.get(column, "") for column in features.columns]})


def summary_checks(features: pd.DataFrame, windows: dict) -> list[dict]:
	sample = features[features["sample"]]
	rows = [
		{"check": "trade_dates", "value": len(features), "note": f"{features['trade_date'].iloc[0].date()} to {features['trade_date'].iloc[-1].date()}"},
		{"check": "sample_trade_dates", "value": int(features["sample"].sum()), "note": "paper_sample with both paths"},
	]
	for name in windows:
		missing = features[f"rv_{name}"].isna()
		rows += [
			{"check": f"paths_{name}", "value": f"{int((~missing).sum())}/{len(features)}", "note": "missing: " + ", ".join(str(date.date()) for date in features.loc[missing, "trade_date"])},
			{"check": f"endpoints_match_{name}", "value": f"{int(sample[f'endpoints_match_{name}'].sum())}/{len(sample)}", "note": "path starts and ends at the reference prices (sample dates)"},
			{"check": f"returns_{name}", "value": f"median {sample[f'n_{name}'].median():.0f}, min {sample[f'n_{name}'].min()}, max {sample[f'n_{name}'].max()}", "note": "1-minute returns per path in the sample"},
			{"check": f"jumps_{name}", "value": sample[f"jumps_{name}"].value_counts().sort_index().to_dict(), "note": "reopening returns excluded per path: count of paths"},
			{"check": f"jump_share_{name}", "value": f"{sample[f'jump_var_{name}'].sum() / (sample[f'rv_{name}'].sum() + sample[f'jump_var_{name}'].sum()):.3f}", "note": "share of the window's total squared returns that sits in the excluded reopening jumps"},
			{"check": f"max_gap_{name}", "value": sample[f"max_gap_{name}"].value_counts().sort_index().tail(5).to_dict(), "note": "longest kept gap (minutes) per path: count of paths, largest five"},
			{"check": f"vol_{name}_pct", "value": f"median {sample[f'vol_{name}_pct'].median():.3f}, p90 {sample[f'vol_{name}_pct'].quantile(0.9):.3f}, max {sample[f'vol_{name}_pct'].max():.3f}", "note": "realized volatility per window, percent"},
		]
	rows.append({"check": "corr_rv_on_rv_day", "value": f"{sample['rv_on'].corr(sample['rv_day']):.3f}", "note": "correlation of overnight and daytime realized variance across sample dates"})
	return rows


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	args = parse_args()
	config, minute_cfg = load_config(args.config)
	timezone_name = minute_cfg["session"]["timezone"]
	output_dir = Path(config["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)

	references = pd.read_parquet(config["reference_prices"])
	in_period = references["trade_date"].ge(pd.Timestamp(config["sample"]["start"]))
	if config["sample"].get("end"):
		in_period &= references["trade_date"].le(pd.Timestamp(config["sample"]["end"]))
	references = references[in_period].sort_values("trade_date").reset_index(drop=True)
	minutes = pd.read_parquet(config["front_minutes"], columns=["ts_utc", "instrument_id", "close"])
	quality = pd.read_csv(config["session_quality"], parse_dates=["trade_date"])

	features = build_features(references, minutes, quality, config["windows"], timezone_name, config["returns"]["halt_gap_minutes"], config["trailing"]["days"])
	dictionary = column_dictionary(features, config["windows"], config["returns"]["halt_gap_minutes"], config["trailing"]["days"])
	summary = pd.DataFrame(summary_checks(features, config["windows"]))

	outputs = {
		"features": output_dir / "cl_volatility_features.parquet", "columns": output_dir / "cl_volatility_features_columns.csv", "summary": output_dir / "cl_volatility_features_summary.csv",
	}
	features.to_parquet(outputs["features"], index=False)
	dictionary.to_csv(outputs["columns"], index=False)
	summary.to_csv(outputs["summary"], index=False)
	manifest = {
		"config": str(args.config), "settings": config, "minute_settings": {"session": minute_cfg["session"], "reference": minute_cfg["reference"]},
		"inputs": {name: {"file": config[name], "sha256": sha256(Path(config[name]))} for name in ("reference_prices", "front_minutes", "session_quality")},
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
	}
	(output_dir / "cl_volatility_features.manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	with pd.option_context("display.width", 200, "display.max_colwidth", 120):
		print(summary.to_string(index=False))
	print(f"\nSaved {outputs['features']} ({len(features)} rows, {int(features['sample'].sum())} in the sample).")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
