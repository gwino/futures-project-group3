"""Validate raw Databento daily CL bars and build a rolled continuous series.

Outputs (in the configured output directory):
- cl_daily_contracts.parquet: validated session bars for every outright contract.
- cl_daily_continuous.parquet: the held contract per trade date, with raw (tradable) prices,
  back-adjusted signal prices, and roll markers.
- cl_daily_roll_calendar.csv: one row per roll.
- cl_daily_contract_quality.csv and cl_daily_quality_summary.csv: data-quality reports.
- cl_daily_roll_crosscheck.csv: roll dates compared with Databento continuous symbology.
- cl_daily_prepare.manifest.json: inputs, settings, and output hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import yaml
from pandas.tseries.holiday import (
	AbstractHolidayCalendar,
	GoodFriday,
	Holiday,
	USLaborDay,
	USMartinLutherKingJr,
	USMemorialDay,
	USPresidentsDay,
	USThanksgivingDay,
	nearest_workday,
	sunday_to_monday,
)

MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
PRICE_COLUMNS = ["open", "high", "low", "close"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/prepare_daily.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> dict:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {"input", "output_dir", "session", "contracts", "roll", "quality"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	if config["roll"]["rule"] not in {"days_before_expiry", "volume_crossover"}:
		raise ValueError("roll.rule must be days_before_expiry or volume_crossover.")
	if config["roll"]["adjustment"] not in {"additive", "ratio"}:
		raise ValueError("roll.adjustment must be additive or ratio.")
	return config


def sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as raw_file:
		for chunk in iter(lambda: raw_file.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


# --- Trading calendar -------------------------------------------------------------------------


def cme_holidays(start: pd.Timestamp, end: pd.Timestamp, juneteenth_from: str) -> pd.DatetimeIndex:
	"""CME holidays without a CL settlement (full closures and early-halt sessions)."""

	class CMEEnergyHolidays(AbstractHolidayCalendar):
		rules = [
			# A Saturday New Year's Day is not observed on the preceding Friday.
			Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
			USMartinLutherKingJr,
			USPresidentsDay,
			GoodFriday,
			USMemorialDay,
			Holiday("Juneteenth", month=6, day=19, start_date=juneteenth_from, observance=nearest_workday),
			Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
			USLaborDay,
			USThanksgivingDay,
			Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
		]

	return CMEEnergyHolidays().holidays(start, end)


def trade_calendar(start: pd.Timestamp, end: pd.Timestamp, holidays: pd.DatetimeIndex) -> pd.DatetimeIndex:
	return pd.bdate_range(start, end).difference(holidays)


def last_trade_date(year: int, month: int, calendar: pd.DatetimeIndex) -> pd.Timestamp:
	"""CME CL rule: 3 business days before the 25th of the prior month (4 if the 25th is not one)."""
	anchor = pd.Timestamp(year, month, 1) - pd.offsets.MonthBegin(1) + pd.Timedelta(days=24)
	prior = calendar[calendar < anchor]
	return prior[-3] if anchor in calendar else prior[-4]


# --- Loading and session assignment -----------------------------------------------------------


def load_raw(path: Path) -> pd.DataFrame:
	store = db.DBNStore.from_file(path)
	raw = store.to_df().reset_index()
	raw = raw.rename(columns={"ts_event": "ts_utc"})
	raw["file_order"] = np.arange(len(raw))
	return raw[["file_order", "ts_utc", "instrument_id", "symbol", *PRICE_COLUMNS, "volume"]]


def assign_trade_dates(bars: pd.DataFrame, calendar: pd.DatetimeIndex, fold: bool) -> pd.DataFrame:
	bars = bars.copy()
	bars["utc_date"] = bars["ts_utc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
	position = calendar.searchsorted(bars["utc_date"].to_numpy(), side="left")
	if (position >= len(calendar)).any():
		raise ValueError("Bars fall after the end of the trade calendar.")
	bars["trade_date"] = calendar[position]
	bars["folded"] = bars["trade_date"].ne(bars["utc_date"])
	if not fold:
		bars = bars[~bars["folded"]]
	return bars


def aggregate_sessions(bars: pd.DataFrame) -> pd.DataFrame:
	bars = bars.sort_values(["instrument_id", "ts_utc"])
	grouped = bars.groupby(["instrument_id", "trade_date"], sort=True)
	sessions = grouped.agg(
		symbol=("symbol", "first"),
		open=("open", "first"),
		high=("high", "max"),
		low=("low", "min"),
		close=("close", "last"),
		volume=("volume", "sum"),
		source_bars=("ts_utc", "size"),
		first_utc_date=("utc_date", "min"),
	)
	return sessions.reset_index()


# --- Contract identity ------------------------------------------------------------------------


def contract_table(sessions: pd.DataFrame, calendar: pd.DatetimeIndex, data_end: pd.Timestamp) -> pd.DataFrame:
	"""Resolve each instrument_id to a full contract month; raw symbols reuse one-digit years."""
	spans = sessions.groupby("instrument_id").agg(
		symbol=("symbol", "first"),
		symbols=("symbol", "nunique"),
		first_date=("trade_date", "min"),
		last_date=("trade_date", "max"),
		sessions=("trade_date", "size"),
	)
	if (spans["symbols"] > 1).any():
		raise ValueError("An instrument_id maps to more than one raw symbol.")

	rows = []
	for instrument_id, span in spans.iterrows():
		match = re.fullmatch(r"CL([FGHJKMNQUVXZ])([0-9]{1,2})", span["symbol"])
		month = MONTH_CODES[match.group(1)]
		digits = match.group(2)
		if len(digits) == 2:
			year = 2000 + int(digits)
		else:
			# Smallest year with the right final digit whose expiry is on or after the last session.
			year = next(
				candidate
				for candidate in range(span["first_date"].year - 1, span["first_date"].year + 12)
				if candidate % 10 == int(digits)
				and last_trade_date(candidate, month, calendar) >= span["last_date"]
			)
		ltd = last_trade_date(year, month, calendar)
		rows.append(
			{
				"instrument_id": instrument_id,
				"symbol": span["symbol"],
				"contract_id": f"CL{match.group(1)}{year}",
				"contract_month": pd.Timestamp(year, month, 1),
				"last_trade_date": ltd,
				"first_date": span["first_date"],
				"last_date": span["last_date"],
				"sessions": span["sessions"],
				"expired_in_sample": ltd <= data_end,
			}
		)
	contracts = pd.DataFrame(rows).sort_values(["contract_month", "first_date"]).reset_index(drop=True)
	if contracts["contract_id"].duplicated().any():
		duplicated = contracts.loc[contracts["contract_id"].duplicated(keep=False), "contract_id"].unique()
		raise ValueError(f"Contracts with more than one instrument_id: {', '.join(duplicated)}")
	contracts["last_session_is_ltd"] = contracts["last_date"].eq(contracts["last_trade_date"])
	return contracts


# --- Validation -------------------------------------------------------------------------------


def flag_outliers(sessions: pd.DataFrame, window: int, threshold: float) -> pd.Series:
	ordered = sessions.sort_values(["instrument_id", "trade_date"])
	change = ordered.groupby("instrument_id")["close"].diff()
	scale = (
		change.abs()
		.groupby(ordered["instrument_id"])
		.transform(lambda values: values.rolling(window, min_periods=20).median().shift(1))
	)
	flags = change.abs().gt(threshold * scale) & scale.gt(0)
	return flags.reindex(sessions.index).fillna(False)


def contract_quality(sessions: pd.DataFrame, contracts: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
	rows = []
	for contract in contracts.itertuples():
		dates = sessions.loc[sessions["instrument_id"].eq(contract.instrument_id), "trade_date"]
		expected = calendar[(calendar >= contract.first_date) & (calendar <= contract.last_date)]
		missing = expected.difference(pd.DatetimeIndex(dates))
		# Longest run of consecutive missing trade dates.
		runs = pd.Series(expected.isin(missing)).astype(int)
		longest = int(runs.groupby((runs == 0).cumsum()).sum().max()) if len(runs) else 0
		rows.append(
			{
				"contract_id": contract.contract_id,
				"instrument_id": contract.instrument_id,
				"sessions": len(dates),
				"expected_sessions": len(expected),
				"missing_sessions": len(missing),
				"longest_gap_sessions": longest,
			}
		)
	return pd.DataFrame(rows)


# --- Rolls and continuous series --------------------------------------------------------------


def build_rolls(
	sessions: pd.DataFrame,
	contracts: pd.DataFrame,
	calendar: pd.DatetimeIndex,
	rule: str,
	days_before_expiry: int,
) -> pd.DataFrame:
	"""Roll from each contract to the next listed month; roll_date is the first session held in the new contract."""
	data_start, data_end = sessions["trade_date"].min(), sessions["trade_date"].max()
	volume = sessions.pivot(index="trade_date", columns="instrument_id", values="volume")
	close = sessions.pivot(index="trade_date", columns="instrument_id", values="close")
	open_ = sessions.pivot(index="trade_date", columns="instrument_id", values="open")
	chain = contracts.sort_values("contract_month").reset_index(drop=True)

	rows = []
	held_since = data_start
	for current, upcoming in zip(chain.itertuples(), chain.iloc[1:].itertuples()):
		deadline = calendar[calendar.get_loc(current.last_trade_date) - days_before_expiry]
		if deadline <= data_start:
			continue
		roll_date = deadline
		if rule == "volume_crossover":
			window = calendar[(calendar >= held_since) & (calendar < deadline)]
			window = window[window.isin(volume.index)]
			ahead = volume.loc[window, upcoming.instrument_id].gt(volume.loc[window, current.instrument_id])
			if ahead.any():
				# Decide on the crossover session's volume; act from the next session.
				decision = ahead.idxmax()
				roll_date = calendar[calendar.get_loc(decision) + 1]
		if roll_date > data_end:
			break

		# Measure the price gap on the last session before the roll where both contracts traded.
		both = close.loc[close.index < roll_date, [current.instrument_id, upcoming.instrument_id]].dropna()
		gap_date = both.index.max()
		from_close, to_close = both.loc[gap_date]
		rows.append(
			{
				"roll_date": roll_date,
				"from_contract": current.contract_id,
				"to_contract": upcoming.contract_id,
				"from_instrument_id": current.instrument_id,
				"to_instrument_id": upcoming.instrument_id,
				"from_last_trade_date": current.last_trade_date,
				"sessions_before_expiry": int(calendar.get_loc(current.last_trade_date) - calendar.get_loc(roll_date)),
				"gap_date": gap_date,
				"from_close": from_close,
				"to_close": to_close,
				"gap": to_close - from_close,
				"ratio": to_close / from_close if from_close > 0 and to_close > 0 else np.nan,
				"from_open_on_roll": open_.at[roll_date, current.instrument_id] if roll_date in open_.index else np.nan,
				"to_open_on_roll": open_.at[roll_date, upcoming.instrument_id] if roll_date in open_.index else np.nan,
			}
		)
		held_since = roll_date
	return pd.DataFrame(rows)


def build_continuous(
	sessions: pd.DataFrame,
	contracts: pd.DataFrame,
	rolls: pd.DataFrame,
	adjustment: str,
	session_close: str,
	timezone_name: str,
) -> pd.DataFrame:
	dates = pd.DatetimeIndex(sorted(sessions["trade_date"].unique()))
	first_held = rolls["from_instrument_id"].iloc[0]
	held = pd.Series(first_held, index=dates)
	for roll in rolls.itertuples():
		held[held.index >= roll.roll_date] = roll.to_instrument_id

	keyed = sessions.set_index(["trade_date", "instrument_id"])
	index = pd.MultiIndex.from_arrays([held.index, held.to_numpy()], names=["trade_date", "instrument_id"])
	series = keyed.reindex(index).reset_index()
	# Dates where the held contract has no session (market-wide outages) are dropped.
	series = series.dropna(subset=["close"]).reset_index(drop=True)
	series["instrument_id"] = series["instrument_id"].astype(sessions["instrument_id"].dtype)

	identity = contracts.set_index("instrument_id")
	series["contract_id"] = series["instrument_id"].map(identity["contract_id"])
	series["roll_flag"] = series["trade_date"].isin(rolls["roll_date"])

	# Back-adjust: each roll shifts (or scales) all earlier prices onto the newer contract's level.
	later_gap = series["trade_date"].map(lambda date: rolls.loc[rolls["roll_date"] > date, "gap"].sum())
	later_ratio = series["trade_date"].map(lambda date: rolls.loc[rolls["roll_date"] > date, "ratio"].prod())
	for column in PRICE_COLUMNS:
		if adjustment == "additive":
			series[f"signal_{column}"] = series[column] + later_gap
		else:
			series[f"signal_{column}"] = series[column] * later_ratio
	series["back_adjustment"] = later_gap if adjustment == "additive" else later_ratio

	close_time = pd.Timedelta(f"{session_close}:00")
	series["timestamp"] = (series["trade_date"] + close_time).dt.tz_localize(timezone_name)
	return series


def add_sessions_to_expiry(frame: pd.DataFrame, contracts: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
	ltd = frame["instrument_id"].map(contracts.set_index("instrument_id")["last_trade_date"])
	return pd.Series(calendar.searchsorted(ltd.to_numpy()) - calendar.searchsorted(frame["trade_date"].to_numpy()), index=frame.index)


# --- Cross-check ------------------------------------------------------------------------------


def databento_switches(path: Path, calendar: pd.DatetimeIndex) -> pd.DataFrame:
	"""Trade dates on which Databento's continuous symbols switch away from each instrument."""
	mappings = db.DBNStore.from_file(path).metadata.mappings
	rows = []
	for symbol in ("CL.v.0", "CL.c.0"):
		intervals = mappings.get(symbol, [])
		for previous, following in zip(intervals, intervals[1:]):
			switch_utc = pd.Timestamp(following["start_date"])
			rows.append(
				{
					"series": symbol,
					"from_instrument_id": int(previous["symbol"]),
					"switch_date": calendar[calendar.searchsorted(switch_utc)],
				}
			)
	return pd.DataFrame(rows)


def roll_crosscheck(rolls: pd.DataFrame, contracts: pd.DataFrame, switches: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
	position = pd.Series(np.arange(len(calendar)), index=calendar)
	check = rolls[["roll_date", "from_contract", "from_instrument_id", "from_last_trade_date"]].copy()
	for symbol, label in (("CL.v.0", "databento_volume"), ("CL.c.0", "databento_calendar")):
		dates = switches[switches["series"].eq(symbol)].set_index("from_instrument_id")["switch_date"]
		check[f"{label}_switch"] = check["from_instrument_id"].map(dates)
		check[f"{label}_minus_ours"] = check[f"{label}_switch"].map(position) - check["roll_date"].map(position)
	# CL.c.0 switches on the session after expiry, which validates the computed last trade dates.
	after_ltd = check["from_last_trade_date"].map(lambda date: calendar[calendar.get_loc(date) + 1])
	check["calendar_switch_is_day_after_ltd"] = check["databento_calendar_switch"].eq(after_ltd)
	return check


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	args = parse_args()
	config = load_config(args.config)
	session_cfg, roll_cfg, quality_cfg = config["session"], config["roll"], config["quality"]
	input_path = Path(config["input"])
	output_dir = Path(config["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)

	raw = load_raw(input_path)
	summary: list[dict] = []

	def report(check: str, value, note: str = "") -> None:
		summary.append({"check": check, "value": value, "note": note})

	report("raw_rows", len(raw))
	report("raw_ts_monotonic_in_file", bool(raw["ts_utc"].is_monotonic_increasing))
	report("raw_missing_values", int(raw[PRICE_COLUMNS + ["volume", "symbol"]].isna().sum().sum()))

	# Outright contracts only; spreads and strategies are excluded.
	is_outright = raw["symbol"].str.fullmatch(config["contracts"]["symbol_pattern"])
	report("non_outright_rows_excluded", int((~is_outright).sum()), "calendar spreads and other strategies")
	bars = raw[is_outright].copy()
	report("outright_rows", len(bars))

	duplicate = bars.duplicated(subset=["instrument_id", "ts_utc"], keep="first")
	report("duplicate_instrument_timestamp_rows_removed", int(duplicate.sum()))
	bars = bars[~duplicate]
	per_instrument_order = bars.groupby("instrument_id")["ts_utc"].apply(lambda ts: ts.is_monotonic_increasing)
	report("ts_monotonic_within_each_instrument", bool(per_instrument_order.all()))

	start = bars["ts_utc"].min().tz_localize(None).normalize()
	end = bars["ts_utc"].max().tz_localize(None).normalize()
	# Span two years either side so expiries of early and far-dated contracts can be computed.
	calendar_start, calendar_end = pd.Timestamp(start.year - 2, 1, 1), pd.Timestamp(end.year + 12, 12, 31)
	holidays = cme_holidays(calendar_start, calendar_end, session_cfg["juneteenth_from"])
	calendar = trade_calendar(calendar_start, calendar_end, holidays)

	bars = assign_trade_dates(bars, calendar, session_cfg["fold_non_trading_days"])
	utc_day = bars["utc_date"]
	report("sunday_bars_folded", int((utc_day.dt.dayofweek == 6).sum()), "Sunday-evening open, part of Monday's session")
	report("holiday_bars_folded", int((bars["folded"] & (utc_day.dt.dayofweek < 5)).sum()), "early-halt holiday sessions settle on the next trade date")

	sessions = aggregate_sessions(bars)
	data_start, data_end = sessions["trade_date"].min(), sessions["trade_date"].max()
	sample_calendar = calendar[(calendar >= data_start) & (calendar <= data_end)]
	report("session_rows", len(sessions))
	report("sample_trade_dates", len(sample_calendar), f"{data_start.date()} to {data_end.date()}")

	missing_market = sample_calendar.difference(pd.DatetimeIndex(sessions["trade_date"].unique()))
	degraded = pd.DatetimeIndex(pd.to_datetime(config.get("degraded_dates", [])))
	report(
		"trade_dates_without_any_outright",
		len(missing_market),
		", ".join(str(date.date()) for date in missing_market) + " (all within Databento degraded dates)"
		if missing_market.isin(degraded).all()
		else ", ".join(str(date.date()) for date in missing_market),
	)

	# Sanity checks.
	ohlc_ok = (
		sessions["low"].le(sessions[["open", "close"]].min(axis=1))
		& sessions["high"].ge(sessions[["open", "close"]].max(axis=1))
	)
	sessions["ohlc_ok"] = ohlc_ok
	report("ohlc_inconsistent_sessions", int((~ohlc_ok).sum()))
	report("negative_volume_sessions", int(sessions["volume"].lt(0).sum()))
	report("zero_volume_sessions", int(sessions["volume"].eq(0).sum()))
	nonpositive = sessions[sessions[PRICE_COLUMNS].le(0).any(axis=1)]
	report(
		"nonpositive_price_sessions",
		len(nonpositive),
		"; ".join(f"{row.symbol} {row.trade_date.date()} low={row.low} close={row.close}" for row in nonpositive.itertuples()),
	)
	degraded_trade_dates = calendar[calendar.searchsorted(degraded.to_numpy())]
	sessions["degraded_date"] = sessions["trade_date"].isin(degraded_trade_dates)
	report("sessions_on_degraded_dates", int(sessions["degraded_date"].sum()), "flagged, not removed")

	contracts = contract_table(sessions, calendar, data_end)
	expired = contracts[contracts["expired_in_sample"] & contracts["first_date"].gt(data_start)]
	report("contracts", len(contracts))
	report(
		"expired_contracts_last_session_on_computed_ltd",
		f"{int(expired['last_session_is_ltd'].sum())}/{len(expired)}",
		", ".join(expired.loc[~expired["last_session_is_ltd"], "contract_id"]),
	)
	sessions = sessions.merge(
		contracts[["instrument_id", "contract_id", "contract_month", "last_trade_date"]], on="instrument_id", how="left"
	)
	sessions["sessions_to_expiry"] = add_sessions_to_expiry(sessions, contracts, calendar)
	sessions["outlier_flag"] = flag_outliers(sessions, quality_cfg["outlier_window"], quality_cfg["outlier_threshold"])
	report("outlier_sessions_all_contracts", int(sessions["outlier_flag"].sum()), "flagged, not removed")
	close_time = pd.Timedelta(f"{session_cfg['close_time']}:00")
	sessions["timestamp"] = (sessions["trade_date"] + close_time).dt.tz_localize(session_cfg["timezone"])
	report(
		"timestamp_monotonic_within_each_contract",
		bool(sessions.groupby("instrument_id")["timestamp"].apply(lambda ts: ts.is_monotonic_increasing).all()),
		session_cfg["timezone"],
	)

	quality = contract_quality(sessions, contracts, calendar)

	rolls = build_rolls(sessions, contracts, calendar, roll_cfg["rule"], roll_cfg["days_before_expiry"])
	continuous = build_continuous(
		sessions, contracts, rolls, roll_cfg["adjustment"], session_cfg["close_time"], session_cfg["timezone"]
	)
	report("rolls", len(rolls), f"rule={roll_cfg['rule']}, days_before_expiry={roll_cfg['days_before_expiry']}")
	report("continuous_sessions", len(continuous))
	report("continuous_missing_trade_dates", len(sample_calendar) - len(continuous))
	# An outlier in the held contract is corroborated when the next listed month moved the same way.
	change = sessions.sort_values("trade_date").groupby("instrument_id")["close"].diff()
	change_by_key = pd.Series(change.to_numpy(), index=pd.MultiIndex.from_frame(sessions.loc[change.index, ["trade_date", "instrument_id"]]))
	next_listed = dict(zip(contracts.sort_values("contract_month")["instrument_id"], contracts.sort_values("contract_month")["instrument_id"].shift(-1)))
	notes = []
	for row in continuous[continuous["outlier_flag"]].itertuples():
		own = change_by_key.get((row.trade_date, row.instrument_id))
		other = change_by_key.get((row.trade_date, next_listed.get(row.instrument_id)))
		corroborated = other is not None and np.sign(own) == np.sign(other) and abs(other) >= 0.5 * abs(own)
		notes.append(f"{row.contract_id} {row.trade_date.date()} {own:+.2f} ({'next month ' + format(other, '+.2f') if corroborated else 'NOT corroborated'})")
	report("continuous_outlier_sessions", int(continuous["outlier_flag"].sum()), "; ".join(notes))
	report("continuous_degraded_sessions", int(continuous["degraded_date"].sum()), "flagged, not removed")

	crosscheck = None
	check_path = config.get("continuous_check")
	if check_path and Path(check_path).exists():
		crosscheck = roll_crosscheck(rolls, contracts, databento_switches(Path(check_path), calendar), calendar)
		mismatch = crosscheck[crosscheck["databento_calendar_switch"].notna() & ~crosscheck["calendar_switch_is_day_after_ltd"]]
		report(
			"databento_calendar_switch_matches_ltd",
			f"{int(crosscheck['calendar_switch_is_day_after_ltd'].sum())}/{int(crosscheck['databento_calendar_switch'].notna().sum())}",
			"CL.c.0 switches on the session after our computed last trade date; exceptions: "
			+ (", ".join(
				f"{row.from_contract} (switch {row.databento_calendar_switch.date()}, LTD {row.from_last_trade_date.date()} {row.from_last_trade_date.day_name()})"
				for row in mismatch.itertuples()
			) or "none"),
		)
		difference = crosscheck["databento_volume_minus_ours"].dropna()
		report(
			"databento_volume_roll_minus_ours_sessions",
			f"median={difference.median():.0f}, min={difference.min():.0f}, max={difference.max():.0f}",
			"positive means CL.v.0 rolls later than our rule",
		)

	# Write outputs.
	contract_columns = [
		"timestamp", "trade_date", "contract_id", "instrument_id", "symbol", "contract_month", "last_trade_date",
		"sessions_to_expiry", *PRICE_COLUMNS, "volume", "source_bars", "ohlc_ok", "outlier_flag", "degraded_date",
	]
	continuous_columns = [
		"timestamp", "trade_date", "contract_id", "instrument_id", "symbol", *PRICE_COLUMNS, "volume",
		*(f"signal_{column}" for column in PRICE_COLUMNS), "back_adjustment", "roll_flag", "sessions_to_expiry",
		"source_bars", "outlier_flag", "degraded_date",
	]
	outputs = {
		"contracts": output_dir / "cl_daily_contracts.parquet",
		"continuous": output_dir / "cl_daily_continuous.parquet",
		"roll_calendar": output_dir / "cl_daily_roll_calendar.csv",
		"contract_quality": output_dir / "cl_daily_contract_quality.csv",
		"quality_summary": output_dir / "cl_daily_quality_summary.csv",
	}
	sessions.sort_values(["contract_month", "trade_date"])[contract_columns].to_parquet(outputs["contracts"], index=False)
	continuous[continuous_columns].to_parquet(outputs["continuous"], index=False)
	rolls.to_csv(outputs["roll_calendar"], index=False)
	quality.merge(contracts[["contract_id", "first_date", "last_date", "last_trade_date"]], on="contract_id").to_csv(
		outputs["contract_quality"], index=False
	)
	summary_frame = pd.DataFrame(summary)
	summary_frame.to_csv(outputs["quality_summary"], index=False)
	if crosscheck is not None:
		outputs["roll_crosscheck"] = output_dir / "cl_daily_roll_crosscheck.csv"
		crosscheck.to_csv(outputs["roll_crosscheck"], index=False)

	manifest = {
		"config": str(args.config),
		"settings": config,
		"input": {"file": str(input_path), "sha256": sha256(input_path)},
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"python": platform.python_version(),
		"pandas": pd.__version__,
	}
	manifest_path = output_dir / "cl_daily_prepare.manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	with pd.option_context("display.max_colwidth", 90, "display.width", 160):
		print(summary_frame.to_string(index=False))
	print(f"\nSaved outputs to {output_dir}/ and manifest to {manifest_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
