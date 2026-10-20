"""Build the CL return variables of Baltussen et al. (2021) from the minute reference prices (#28).

Run from the repository root, after the minute pipeline (#27):

    python -m src.build_return_features --config config/return_features.yaml

Each return on trade date t is a simple return between two reference prices of the contract held on t,
so a roll never enters a return. Predictors end by the decision time (13:00 CT) and only the target
(13:00-13:30 CT) uses later prices; the config is rejected otherwise.

Outputs (in the configured output directory):
- cl_return_features.parquet: one row per trade date in the sample period, with the reference prices,
  simple returns `r_*`, log returns `log_*`, roll and session flags, `sample` (the regression sample),
  `obs` (its observation number) and `exclusion_reason` for the dates left out.
- cl_return_features_columns.csv: the column dictionary, with each return's formula from the config.
- cl_return_features_summary.csv and cl_return_features.manifest.json: checks, settings, and hashes.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.prepare_daily_data import sha256

PRICE_PATTERN = re.compile(r"^p_(prev_)?(\d{2})(\d{2})$")
FLAG_COLUMNS = ["roll_flag", "folded_holiday", "degraded_date", "early_close", "low_volume", "complete", "paper_sample"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/return_features.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> dict:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {"reference_prices", "session_quality", "output_dir", "sample", "returns", "target", "decision_time", "oos"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	check_returns(config["returns"], config["target"], config["decision_time"])
	return config


# --- Return definitions -----------------------------------------------------------------------


def price_time(column: str) -> int:
	"""Minutes from midnight on the trade date of a reference price column; `p_prev_*` is a day earlier."""
	match = PRICE_PATTERN.match(column)
	if not match:
		raise ValueError(f"{column} is not a reference price column (p_HHMM or p_prev_HHMM).")
	previous, hours, minutes = match.groups()
	return int(hours) * 60 + int(minutes) - (24 * 60 if previous else 0)


def check_returns(returns: dict, target: str, decision_time: str) -> None:
	"""Every interval runs forward in time, and only the target ends after the decision time."""
	names = [name for name in returns if not isinstance(name, str)]
	if names:
		raise ValueError(f"Return names must be strings, got {names}; quote YAML keys such as \"on\".")
	if target not in returns:
		raise ValueError(f"Target {target} is not among the returns.")
	hours, minutes = decision_time.split(":")
	decision = int(hours) * 60 + int(minutes)
	for name, (start, end) in returns.items():
		if price_time(start) >= price_time(end):
			raise ValueError(f"Return {name} must start before it ends: {start} -> {end}.")
		if name != target and price_time(end) > decision:
			raise ValueError(f"Predictor {name} ends at {end}, after the {decision_time} decision time (look-ahead).")
	if price_time(returns[target][0]) < decision:
		raise ValueError(f"Target {target} starts before the {decision_time} decision time and overlaps the predictors.")


def describe_price(column: str) -> str:
	match = PRICE_PATTERN.match(column)
	previous, hours, minutes = match.groups()
	return f"{'previous ' if previous else ''}{hours}:{minutes}"


# --- Features ---------------------------------------------------------------------------------


def build_features(
	references: pd.DataFrame, sessions_to_expiry: pd.Series, returns: dict, start: pd.Timestamp, end: pd.Timestamp | None,
	log_returns: bool,
) -> pd.DataFrame:
	"""Returns for each trade date in [start, end], the regression sample and the reason for each exclusion.

	A date is in the sample when it passes the paper filters (`paper_sample`: all reference prices
	exist, no early close, enough volume). Returns are computed for every date whose two prices exist,
	so excluded dates keep whatever returns they have.
	"""
	in_period = references["trade_date"].ge(start) & (references["trade_date"].le(end) if end is not None else True)
	frame = references.loc[in_period].sort_values("trade_date").reset_index(drop=True)
	prices = sorted({column for interval in returns.values() for column in interval}, key=price_time)
	features = frame[["trade_date", "prev_trade_date", "contract_id"]].copy()
	features["weekday"] = features["trade_date"].dt.dayofweek
	features["gap_days"] = (features["trade_date"] - features["prev_trade_date"]).dt.days
	features[prices] = frame[prices]
	for name, (first, last) in returns.items():
		features[f"r_{name}"] = frame[last] / frame[first] - 1
	if log_returns:
		for name, (first, last) in returns.items():
			features[f"log_{name}"] = np.log(frame[last] / frame[first])
	features["sessions_to_expiry"] = features["trade_date"].map(sessions_to_expiry).astype("Int64")
	features[FLAG_COLUMNS] = frame[FLAG_COLUMNS]
	features["sample"] = features["paper_sample"]
	features["obs"] = pd.Series(np.arange(1, features["sample"].sum() + 1), index=features.index[features["sample"]]).reindex(features.index).astype("Int64")
	features["exclusion_reason"] = exclusion_reasons(frame, prices)
	return features


def exclusion_reasons(frame: pd.DataFrame, prices: list[str]) -> pd.Series:
	"""Why each date is out of the sample ('' when it is in): missing prices, early close, low volume."""
	parts = [frame[column].isna().map({True: f"missing {column}", False: ""}) for column in prices]
	parts += [frame["early_close"].map({True: "early close", False: ""}), frame["low_volume"].map({True: "low volume", False: ""})]
	reasons = pd.concat(parts, axis=1).apply(lambda row: "; ".join(reason for reason in row if reason), axis=1)
	unexplained = ~frame["paper_sample"] & reasons.eq("")
	return reasons.mask(unexplained, "not in paper_sample")


def identity_errors(features: pd.DataFrame, identities: list) -> dict[str, float]:
	"""Largest absolute error of each log-return identity on the sample."""
	sample = features[features["sample"]]
	errors = {}
	for total, parts in identities:
		error = sample[f"log_{total}"] - sample[[f"log_{part}" for part in parts]].sum(axis=1)
		errors[f"log {total} = " + " + ".join(f"log {part}" for part in parts)] = float(error.abs().max())
	return errors


def column_dictionary(features: pd.DataFrame, returns: dict, target: str) -> pd.DataFrame:
	"""One row per output column: its type and meaning. Return formulas come from the config."""
	meaning = {
		"trade_date": "CME trade date t (the session from 17:00 CT on the previous trade date to 16:00 CT on t)",
		"prev_trade_date": "Previous trade date (Friday for a Monday; skips holidays)",
		"contract_id": "Contract held on t, e.g. CLK2022; every price and return on t uses it",
		"weekday": "Day of week of t (0 = Monday)",
		"gap_days": "Calendar days from prev_trade_date to t (3 after a weekend); overnight returns span the gap",
		"sessions_to_expiry": "Trade dates from t to the held contract's last trade date",
		"roll_flag": "t is the first session in a new contract (paper volume-crossover rule)",
		"folded_holiday": "A weekend or holiday session was folded into t",
		"degraded_date": "Databento flags data quality on t; kept, for robustness checks",
		"early_close": "The market closed 30+ minutes before 16:00 CT on t",
		"low_volume": "Held-contract session volume under the paper's 100-contract minimum",
		"complete": "All reference prices exist on t",
		"paper_sample": "complete, not an early close, and not low volume (paper filters)",
		"sample": "In the regression sample: paper_sample and within the configured period",
		"obs": "Observation number within the sample (1, 2, ...); empty outside it",
		"exclusion_reason": "Why t is out of the sample; empty when it is in",
	}
	for column in features.columns:
		if PRICE_PATTERN.match(column):
			meaning[column] = f"Held contract's price at {describe_price(column)} CT: close of the last 1-minute bar starting before that time"
	for name, (first, last) in returns.items():
		role = "target" if name == target else "predictor"
		interval = f"{describe_price(first)} -> {describe_price(last)} CT"
		meaning[f"r_{name}"] = f"{name.upper()} simple return ({role}), {interval}: {last} / {first} - 1"
		meaning[f"log_{name}"] = f"{name.upper()} log return ({role}), {interval}: log({last} / {first})"
	return pd.DataFrame(
		{"column": features.columns, "dtype": [str(dtype) for dtype in features.dtypes], "meaning": [meaning[column] for column in features.columns]}
	)


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	args = parse_args()
	config = load_config(args.config)
	output_dir = Path(config["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)
	returns, target = config["returns"], config["target"]
	start = pd.Timestamp(config["sample"]["start"])
	end = pd.Timestamp(config["sample"]["end"]) if config["sample"].get("end") else None

	summary: list[dict] = []

	def report(check: str, value, note: str = "") -> None:
		summary.append({"check": check, "value": value, "note": note})

	references = pd.read_parquet(config["reference_prices"])
	quality = pd.read_csv(config["session_quality"], parse_dates=["trade_date"], usecols=["trade_date", "contract_id", "sessions_to_expiry"])
	held = references.set_index("trade_date")["contract_id"]
	mismatch = quality["contract_id"].ne(quality["trade_date"].map(held))
	if mismatch.any():
		raise ValueError(f"Reference prices and session quality disagree on the held contract for {int(mismatch.sum())} trade dates.")
	features = build_features(references, quality.set_index("trade_date")["sessions_to_expiry"], returns, start, end, config.get("log_returns", False))

	sample = features[features["sample"]]
	report("period_trade_dates", len(features), f"{features['trade_date'].min().date()} to {features['trade_date'].max().date()}")
	report("sample_trade_dates", len(sample), f"{sample['trade_date'].min().date()} to {sample['trade_date'].max().date()}")
	excluded = features[~features["sample"]]
	report("excluded_trade_dates", len(excluded), "; ".join(f"{row.trade_date.date()} ({row.exclusion_reason})" for row in excluded.itertuples()))
	return_columns = [f"r_{name}" for name in returns]
	missing = int(sample[return_columns].isna().sum().sum())
	if missing:
		raise ValueError(f"{missing} missing returns in the sample.")
	report("sample_missing_returns", missing)
	report("sample_roll_dates", int(sample["roll_flag"].sum()), "kept: every return is within the contract held on its trade date")
	report("sample_degraded_dates", int(sample["degraded_date"].sum()), "kept, flagged for robustness checks")
	report("sample_weekend_or_holiday_gaps", int(sample["gap_days"].gt(1).sum()), ", ".join(f"{k} days: {v}" for k, v in sample["gap_days"].value_counts().sort_index().items()))
	report("predictors_end_by", config["decision_time"], f"target {target}: {returns[target][0]} -> {returns[target][1]}; checked when the config is loaded")
	if config.get("log_returns"):
		for identity, error in identity_errors(features, config.get("log_identities", [])).items():
			report("max_identity_error", f"{error:.1e}", identity)
	min_train = config["oos"]["min_train_obs"]
	if len(sample) > min_train:
		report(
			"first_oos_forecast", str(sample["trade_date"].iloc[min_train].date()),
			f"after a {min_train}-observation initial window ending {sample['trade_date'].iloc[min_train - 1].date()}; {len(sample) - min_train} forecasts",
		)
	for name in returns:
		values = sample[f"r_{name}"] * 100
		report(f"r_{name}_percent", f"mean {values.mean():.4f}, sd {values.std():.4f}", f"min {values.min():.2f}, max {values.max():.2f}")

	outputs = {
		"features": output_dir / "cl_return_features.parquet",
		"columns": output_dir / "cl_return_features_columns.csv",
		"summary": output_dir / "cl_return_features_summary.csv",
	}
	features.to_parquet(outputs["features"], index=False)
	column_dictionary(features, returns, target).to_csv(outputs["columns"], index=False)
	summary_frame = pd.DataFrame(summary)
	summary_frame.to_csv(outputs["summary"], index=False)

	inputs = [Path(config["reference_prices"]), Path(config["session_quality"])]
	manifest = {
		"config": str(args.config),
		"settings": config,
		"inputs": [{"file": str(path), "sha256": sha256(path)} for path in inputs],
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"python": platform.python_version(),
		"pandas": pd.__version__,
	}
	manifest_path = output_dir / "cl_return_features.manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	with pd.option_context("display.max_colwidth", 110, "display.width", 200):
		print(summary_frame.assign(note=summary_frame["note"].str.slice(0, 110)).to_string(index=False))
	print(f"\nSaved outputs to {output_dir}/ and manifest to {manifest_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
