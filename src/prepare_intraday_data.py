"""Validate raw Databento 1-minute CL bars and build 15-minute and CME session bars.

Run from the repository root, after the daily preparation step (which writes the daily roll
calendar used to cross-check the intraday rolls):

    python -m src.prepare_daily_data --config config/prepare_daily.yaml
    python -m src.prepare_intraday_data --config config/prepare_intraday.yaml

Outputs (in the configured output directory):
- cl_intraday_15m_contracts.parquet: validated 15-minute bars for every outright contract.
- cl_intraday_15m_continuous.parquet: the held contract's 15-minute bars, with raw (tradable)
  prices, back-adjusted signal prices, roll markers, and the trading-window flags from #4.
- cl_session_daily_contracts.parquet and cl_session_daily_continuous.parquet: CME session bars
  (17:00 to 16:00 CT) built from the same 1-minute data, for the common-window daily horizon.
- cl_intraday_roll_calendar.csv: one row per roll, with session and 15-minute price gaps.
- cl_intraday_session_coverage.csv: expected, observed, and missing bars (and the longest gap) for
  every contract on every trade date; cl_intraday_contract_coverage.csv: the same, totalled per contract.
- cl_intraday_window_coverage.csv and cl_intraday_quality_summary.csv: data-quality reports.
- cl_intraday_prepare.manifest.json: inputs, settings, and output hashes.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import yaml

from src.prepare_daily_data import (
	PRICE_COLUMNS,
	add_sessions_to_expiry,
	build_continuous,
	build_rolls,
	cme_holidays,
	contract_table,
	sha256,
	trade_calendar,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/prepare_intraday.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> tuple[dict, dict]:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {"input", "daily_config", "output_dir", "session", "bars", "quality"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	with Path(config["daily_config"]).open(encoding="utf-8") as daily_file:
		daily = yaml.safe_load(daily_file)
	if daily["session"]["timezone"] != config["session"]["timezone"]:
		raise ValueError("The intraday and daily configs must use the same session timezone.")
	return config, daily


def minute_of_day(clock: str) -> int:
	hours, minutes = (int(part) for part in clock.split(":"))
	return hours * 60 + minutes


# --- Loading and session assignment -----------------------------------------------------------


def load_raw(path: Path, symbol_pattern: str) -> tuple[pd.DataFrame, int, pd.Timestamp]:
	"""Load 1-minute bars and keep outright contracts; returns the bars, the raw row count, and the query end.

	Symbols are mapped from the file's instrument mappings instead of row by row, which is
	much faster for the parent query (most rows are spreads).
	"""
	store = db.DBNStore.from_file(path)
	raw = store.to_df(map_symbols=False).reset_index()
	pattern = re.compile(symbol_pattern)
	symbols: dict[int, str] = {}
	for symbol, intervals in store.metadata.mappings.items():
		if pattern.fullmatch(symbol):
			for interval in intervals:
				instrument_id = int(interval["symbol"])
				if symbols.setdefault(instrument_id, symbol) != symbol:
					raise ValueError(f"instrument_id {instrument_id} maps to more than one outright symbol.")
	bars = raw[raw["instrument_id"].isin(symbols)].copy()
	bars["symbol"] = bars["instrument_id"].map(symbols)
	bars = bars.rename(columns={"ts_event": "ts_utc"})
	columns = ["ts_utc", "instrument_id", "symbol", *PRICE_COLUMNS, "volume"]
	return bars[columns].reset_index(drop=True), len(raw), pd.Timestamp(store.end)


def assign_trade_dates(
	bars: pd.DataFrame, calendar: pd.DatetimeIndex, timezone_name: str, open_time: str, close_time: str
) -> pd.DataFrame:
	"""Assign each minute to the CME trade date whose session it belongs to.

	A session runs from open_time on the prior calendar day to close_time. Minutes from a
	weekend or holiday session (for example the Sunday-evening open) fold into the next trade
	date, as in the daily pipeline. Minutes in the daily halt are marked `after_close`.
	"""
	bars = bars.copy()
	wall = bars["ts_utc"].dt.tz_convert(timezone_name).dt.tz_localize(None)
	minute = wall.dt.hour * 60 + wall.dt.minute
	bars["after_close"] = minute.ge(minute_of_day(close_time)) & minute.lt(minute_of_day(open_time))
	shift = pd.Timedelta(days=1) - pd.Timedelta(minutes=minute_of_day(open_time))
	session_day = (wall + shift).dt.normalize()
	position = calendar.searchsorted(session_day.to_numpy(), side="left")
	if (position >= len(calendar)).any():
		raise ValueError("Bars fall after the end of the trade calendar.")
	bars["trade_date"] = calendar[position]
	bars["folded"] = bars["trade_date"].ne(session_day)
	return bars


def aggregate_bars(minutes: pd.DataFrame, frequency: str) -> pd.DataFrame:
	"""Aggregate 1-minute bars to `frequency` bars labeled by their end time.

	Each 1-minute ts_event marks the start of its minute, so the bins are left-closed and the
	label is the right edge. Bins are cut on the UTC clock; Chicago's UTC offset is a whole
	number of hours, so they align with the local quarter hours and the 17:00 and 16:00 session
	boundaries, and the repeated hour at the end of daylight saving time is not merged.
	"""
	minutes = minutes.sort_values(["instrument_id", "ts_utc"])
	step = pd.Timedelta(frequency)
	bar_end = minutes["ts_utc"].dt.floor(frequency) + step
	grouped = minutes.assign(bar_end=bar_end).groupby(["instrument_id", "bar_end"], sort=True)
	bars = grouped.agg(
		symbol=("symbol", "first"),
		trade_date=("trade_date", "first"),
		open=("open", "first"),
		high=("high", "max"),
		low=("low", "min"),
		close=("close", "last"),
		volume=("volume", "sum"),
		minutes=("ts_utc", "size"),
		trade_dates=("trade_date", "nunique"),
	)
	if (bars["trade_dates"] > 1).any():
		raise ValueError(f"A {frequency} bar spans two trade dates; check the session boundaries.")
	return bars.drop(columns="trade_dates").reset_index()


def aggregate_sessions(minutes: pd.DataFrame) -> pd.DataFrame:
	minutes = minutes.sort_values(["instrument_id", "ts_utc"])
	sessions = minutes.groupby(["instrument_id", "trade_date"], sort=True).agg(
		symbol=("symbol", "first"),
		open=("open", "first"),
		high=("high", "max"),
		low=("low", "min"),
		close=("close", "last"),
		volume=("volume", "sum"),
		source_minutes=("ts_utc", "size"),
	)
	return sessions.reset_index()


def flag_trade_window(bars: pd.DataFrame, timezone_name: str, start: str, end: str, frequency: str) -> pd.Series:
	"""True for bars that start at or after `start` and end at or before `end` on the trade date itself.

	Bars on the calendar day before the trade date (the evening session) and bars from folded
	holiday sessions are outside the window.
	"""
	end_wall = bars["bar_end"].dt.tz_convert(timezone_name).dt.tz_localize(None)
	start_wall = end_wall - pd.Timedelta(frequency)
	start_minute = start_wall.dt.hour * 60 + start_wall.dt.minute
	end_minute = end_wall.dt.hour * 60 + end_wall.dt.minute
	return (
		start_wall.dt.normalize().eq(bars["trade_date"])
		& start_minute.ge(minute_of_day(start))
		& end_minute.le(minute_of_day(end))
		& end_minute.gt(start_minute)
	)


# --- Validation -------------------------------------------------------------------------------


def flag_outliers(bars: pd.DataFrame, window: int, threshold: float) -> pd.Series:
	ordered = bars.sort_values(["instrument_id", "bar_end"])
	change = ordered.groupby("instrument_id")["close"].diff()
	scale = (
		change.abs()
		.groupby(ordered["instrument_id"])
		.transform(lambda values: values.rolling(window, min_periods=window // 4).median().shift(1))
	)
	flags = change.abs().gt(threshold * scale) & scale.gt(0)
	return flags.reindex(bars.index).fillna(False)


def session_coverage(bars: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
	"""Expected, observed, and missing bars for every contract on every trade date of its life in the data.

	A bar is expected when any outright contract traded in it on that trade date, so early closes and
	holiday sessions need no separate calendar. A contract's life runs from its first to its last trade
	date in the data; trade dates in that range with no bars at all count as fully missing.
	"""
	market = bars[["trade_date", "bar_end", "in_trade_window"]].drop_duplicates(["trade_date", "bar_end"])
	life = contracts[["instrument_id", "contract_id", "first_date", "last_date"]]
	grid = market.merge(life, how="cross")
	grid = grid[grid["trade_date"].between(grid["first_date"], grid["last_date"])]
	observed = bars[["instrument_id", "bar_end"]].assign(observed=True)
	grid = grid.merge(observed, on=["instrument_id", "bar_end"], how="left")
	grid["observed"] = grid["observed"].notna()
	grid = grid.sort_values(["instrument_id", "bar_end"]).reset_index(drop=True)

	key = ["instrument_id", "trade_date"]
	# Longest run of consecutive missing bars within each contract-session.
	run = grid.groupby(key)["observed"].cumsum()
	missing = grid[~grid["observed"]]
	longest = missing.groupby([*key, run[missing.index]]).size().groupby(level=[0, 1]).max().rename("longest_gap_bars")

	grid["window_observed"] = grid["observed"] & grid["in_trade_window"]
	coverage = grid.groupby(key).agg(
		contract_id=("contract_id", "first"),
		expected_bars=("bar_end", "size"),
		observed_bars=("observed", "sum"),
		expected_window_bars=("in_trade_window", "sum"),
		observed_window_bars=("window_observed", "sum"),
	)
	coverage = coverage.join(longest).fillna({"longest_gap_bars": 0}).reset_index()
	coverage["missing_bars"] = coverage["expected_bars"] - coverage["observed_bars"]
	coverage["missing_window_bars"] = coverage["expected_window_bars"] - coverage["observed_window_bars"]
	columns = [
		"contract_id", "instrument_id", "trade_date", "expected_bars", "observed_bars", "missing_bars", "longest_gap_bars",
		"expected_window_bars", "observed_window_bars", "missing_window_bars",
	]
	return coverage[columns].astype({"longest_gap_bars": int})


def contract_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
	"""Per-contract totals of the session coverage report, overall and for the sessions in which the contract is held."""
	held = coverage[coverage["held"]]
	totals = coverage.groupby("contract_id").agg(
		instrument_id=("instrument_id", "first"),
		first_date=("trade_date", "min"),
		last_date=("trade_date", "max"),
		sessions=("trade_date", "size"),
		sessions_without_bars=("observed_bars", lambda observed: int(observed.eq(0).sum())),
		expected_bars=("expected_bars", "sum"),
		observed_bars=("observed_bars", "sum"),
		missing_bars=("missing_bars", "sum"),
		longest_gap_bars=("longest_gap_bars", "max"),
	)
	totals["missing_share"] = totals["missing_bars"] / totals["expected_bars"]
	held_totals = held.groupby("contract_id").agg(
		held_sessions=("trade_date", "size"),
		held_missing_bars=("missing_bars", "sum"),
		held_missing_window_bars=("missing_window_bars", "sum"),
		held_longest_gap_bars=("longest_gap_bars", "max"),
	)
	totals = totals.join(held_totals).fillna({"held_sessions": 0, "held_missing_bars": 0, "held_missing_window_bars": 0, "held_longest_gap_bars": 0})
	held_columns = ["held_sessions", "held_missing_bars", "held_missing_window_bars", "held_longest_gap_bars"]
	totals[held_columns] = totals[held_columns].astype(int)
	return totals.sort_values(["last_date", "contract_id"]).reset_index()


# --- Rolls and continuous series --------------------------------------------------------------


def intraday_gaps(bars: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
	"""Measure each roll's price gap at the last 15-minute bar before the roll session where both contracts traded."""
	close = bars.pivot(index="bar_end", columns="instrument_id", values="close")
	trade_date = bars.drop_duplicates("bar_end").set_index("bar_end")["trade_date"]
	rows = []
	for roll in rolls.itertuples():
		pair = close.loc[trade_date.reindex(close.index).lt(roll.roll_date), [roll.from_instrument_id, roll.to_instrument_id]]
		both = pair.dropna()
		gap_bar = both.index.max()
		from_close, to_close = both.loc[gap_bar]
		rows.append(
			{
				"roll_date": roll.roll_date,
				"intraday_gap_bar_end": gap_bar,
				"intraday_from_close": from_close,
				"intraday_to_close": to_close,
				"intraday_gap": to_close - from_close,
			}
		)
	return rolls.merge(pd.DataFrame(rows), on="roll_date")


def build_intraday_continuous(bars: pd.DataFrame, contracts: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
	"""Held contract's bars per trade date, with additively back-adjusted signal prices.

	The held contract changes only at a session boundary (the roll date's 17:00 CT open), so both
	horizons hold the same contract on every trade date.
	"""
	dates = pd.DatetimeIndex(sorted(bars["trade_date"].unique()))
	held = pd.Series(rolls["from_instrument_id"].iloc[0], index=dates)
	for roll in rolls.itertuples():
		held[held.index >= roll.roll_date] = roll.to_instrument_id
	held = held.rename("instrument_id").rename_axis("trade_date").reset_index()
	series = bars.merge(held, on=["trade_date", "instrument_id"], how="inner").sort_values("bar_end").reset_index(drop=True)

	identity = contracts.set_index("instrument_id")
	series["contract_id"] = series["instrument_id"].map(identity["contract_id"])
	series["roll_flag"] = series["trade_date"].isin(rolls["roll_date"])
	roll_dates = rolls["roll_date"].to_numpy()
	gaps = rolls["intraday_gap"].to_numpy()
	later_gap = series["trade_date"].map(lambda date: gaps[roll_dates > date].sum())
	for column in PRICE_COLUMNS:
		series[f"signal_{column}"] = series[column] + later_gap
	series["back_adjustment"] = later_gap

	in_window = series[series["in_trade_window"]]
	last_in_window = in_window.groupby("trade_date")["bar_end"].transform("max")
	series["is_cutoff_bar"] = False
	series.loc[in_window.index, "is_cutoff_bar"] = in_window["bar_end"].eq(last_in_window)
	return series


def plot_roll_examples(series: pd.DataFrame, bars: pd.DataFrame, rolls: pd.DataFrame, timezone_name: str, path: Path) -> None:
	import warnings

	with warnings.catch_warnings():
		# matplotlib 3.9 calls pyparsing APIs that newer pyparsing releases mark as deprecated.
		warnings.simplefilter("ignore", DeprecationWarning)
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

	examples = rolls.iloc[np.linspace(0, len(rolls) - 1, 3).round().astype(int)]
	fig, axes = plt.subplots(1, len(examples), figsize=(15, 4.2))
	for ax, roll in zip(np.atleast_1d(axes), examples.itertuples()):
		dates = pd.DatetimeIndex(sorted(series["trade_date"].unique()))
		centre = dates.get_loc(roll.roll_date)
		window = dates[max(centre - 2, 0) : centre + 2]
		for instrument_id, label in ((roll.from_instrument_id, roll.from_contract), (roll.to_instrument_id, roll.to_contract)):
			raw = bars[bars["instrument_id"].eq(instrument_id) & bars["trade_date"].isin(window)]
			ax.plot(raw["bar_end"].dt.tz_convert(timezone_name), raw["close"], lw=0.8, label=f"{label} raw close")
		held = series[series["trade_date"].isin(window)]
		# Remove the adjustment from later rolls so the line sits on the new contract's price level.
		later = held.loc[held["trade_date"].ge(roll.roll_date), "back_adjustment"].iloc[0]
		ax.plot(
			held["bar_end"].dt.tz_convert(timezone_name), held["signal_close"] - later, lw=1.4, color="black",
			label="signal_close (adjusted)",
		)
		ax.axvline(pd.Timestamp(roll.roll_date).tz_localize(timezone_name) - pd.Timedelta(hours=7), color="grey", ls="--", lw=0.8)
		ax.set_title(f"{roll.from_contract} → {roll.to_contract}, roll {roll.roll_date.date()}")
		ax.tick_params(axis="x", labelrotation=30, labelsize=8)
		ax.legend(fontsize=7)
	axes[0].set_ylabel("USD/bbl")
	fig.tight_layout()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=130)
	plt.close(fig)


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	args = parse_args()
	config, daily = load_config(args.config)
	session_cfg, bar_cfg, quality_cfg = config["session"], config["bars"], config["quality"]
	roll_cfg = daily["roll"]
	timezone_name, frequency = session_cfg["timezone"], bar_cfg["frequency"]
	input_path = Path(config["input"])
	output_dir = Path(config["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)

	summary: list[dict] = []

	def report(check: str, value, note: str = "") -> None:
		summary.append({"check": check, "value": value, "note": note})

	minutes, raw_rows, query_end = load_raw(input_path, daily["contracts"]["symbol_pattern"])
	report("raw_rows", raw_rows)
	report("non_outright_rows_excluded", raw_rows - len(minutes), "calendar spreads and other strategies")
	report("outright_rows", len(minutes))
	report("outright_missing_values", int(minutes[PRICE_COLUMNS + ["volume", "symbol"]].isna().sum().sum()))

	duplicate = minutes.duplicated(subset=["instrument_id", "ts_utc"], keep="first")
	report("duplicate_instrument_timestamp_rows_removed", int(duplicate.sum()))
	minutes = minutes[~duplicate]
	report(
		"ts_monotonic_within_each_instrument",
		bool(minutes.groupby("instrument_id")["ts_utc"].apply(lambda ts: ts.is_monotonic_increasing).all()),
	)

	start = minutes["ts_utc"].min().tz_localize(None).normalize()
	end = minutes["ts_utc"].max().tz_localize(None).normalize()
	calendar_start, calendar_end = pd.Timestamp(start.year - 2, 1, 1), pd.Timestamp(end.year + 12, 12, 31)
	holidays = cme_holidays(calendar_start, calendar_end, daily["session"]["juneteenth_from"])
	calendar = trade_calendar(calendar_start, calendar_end, holidays)

	minutes = assign_trade_dates(minutes, calendar, timezone_name, session_cfg["open_time"], session_cfg["close_time"])
	after_close = minutes[minutes["after_close"]]
	report(
		"after_close_minutes_dropped",
		len(after_close),
		"; ".join(f"{row.symbol} {row.ts_utc.tz_convert(timezone_name)} volume={row.volume}" for row in after_close.itertuples()),
	)
	minutes = minutes[~minutes["after_close"]]
	report("folded_minutes", int(minutes["folded"].sum()), "holiday sessions folded into the next trade date")
	# The query ends at midnight UTC, partway through the evening session of the next trade date.
	session_close = (minutes["trade_date"] + pd.Timedelta(f"{session_cfg['close_time']}:00")).dt.tz_localize(timezone_name)
	incomplete = session_close.gt(query_end)
	report(
		"incomplete_trade_date_minutes_dropped",
		int(incomplete.sum()),
		", ".join(str(date.date()) for date in minutes.loc[incomplete, "trade_date"].unique()) + f" (query ends {query_end})",
	)
	minutes = minutes[~incomplete]

	# Session bars (common-window daily horizon) and contract identity.
	sessions = aggregate_sessions(minutes)
	data_start, data_end = sessions["trade_date"].min(), sessions["trade_date"].max()
	sample_calendar = calendar[(calendar >= data_start) & (calendar <= data_end)]
	report("sample_trade_dates", len(sample_calendar), f"{data_start.date()} to {data_end.date()}")
	missing_market = sample_calendar.difference(pd.DatetimeIndex(sessions["trade_date"].unique()))
	report("trade_dates_without_any_outright", len(missing_market), ", ".join(str(date.date()) for date in missing_market))

	contracts = contract_table(sessions, calendar, data_end)
	expired = contracts[contracts["expired_in_sample"] & contracts["first_date"].gt(data_start)]
	report("contracts", len(contracts))
	report(
		"expired_contracts_last_session_on_computed_ltd",
		f"{int(expired['last_session_is_ltd'].sum())}/{len(expired)}",
		", ".join(expired.loc[~expired["last_session_is_ltd"], "contract_id"]),
	)
	identity_columns = ["instrument_id", "contract_id", "contract_month", "last_trade_date"]
	sessions = sessions.merge(contracts[identity_columns], on="instrument_id", how="left")
	sessions["sessions_to_expiry"] = add_sessions_to_expiry(sessions, contracts, calendar)
	sessions["ohlc_ok"] = sessions["low"].le(sessions[["open", "close"]].min(axis=1)) & sessions["high"].ge(
		sessions[["open", "close"]].max(axis=1)
	)
	sessions["timestamp"] = (sessions["trade_date"] + pd.Timedelta(f"{session_cfg['close_time']}:00")).dt.tz_localize(timezone_name)

	# 15-minute bars.
	bars = aggregate_bars(minutes, frequency)
	bars = bars.merge(contracts[identity_columns], on="instrument_id", how="left")
	bars["sessions_to_expiry"] = add_sessions_to_expiry(bars, contracts, calendar)
	bars["timestamp"] = bars["bar_end"].dt.tz_convert(timezone_name)
	bars["in_trade_window"] = flag_trade_window(
		bars, timezone_name, session_cfg["trade_window_start"], session_cfg["trade_window_end"], frequency
	)
	bars["ohlc_ok"] = bars["low"].le(bars[["open", "close"]].min(axis=1)) & bars["high"].ge(bars[["open", "close"]].max(axis=1))
	bars["outlier_flag"] = flag_outliers(bars, quality_cfg["outlier_window"], quality_cfg["outlier_threshold"])
	degraded = pd.DatetimeIndex(pd.to_datetime(daily.get("degraded_dates", [])))
	bar_start_utc_date = (bars["bar_end"] - pd.Timedelta(frequency)).dt.tz_localize(None).dt.normalize()
	bars["degraded_date"] = bar_start_utc_date.isin(degraded)

	report(f"bars_{frequency}", len(bars), f"from {len(minutes)} 1-minute rows")
	report("ohlc_inconsistent_bars", int((~bars["ohlc_ok"]).sum()))
	report("ohlc_inconsistent_sessions", int((~sessions["ohlc_ok"]).sum()))
	report("negative_volume_bars", int(bars["volume"].lt(0).sum()))
	report("zero_volume_bars", int(bars["volume"].eq(0).sum()))
	report("nonpositive_price_bars", int(bars[PRICE_COLUMNS].le(0).any(axis=1).sum()))
	report("bars_on_degraded_dates", int(bars["degraded_date"].sum()), "flagged, not removed")
	report("outlier_bars_all_contracts", int(bars["outlier_flag"].sum()), "flagged, not removed")

	# Rolls: the same calendar rule as the daily pipeline, applied to the session bars.
	rolls = build_rolls(sessions, contracts, calendar, roll_cfg["rule"], roll_cfg["days_before_expiry"])
	rolls = intraday_gaps(bars, rolls)
	report("rolls", len(rolls), f"rule={roll_cfg['rule']}, days_before_expiry={roll_cfg['days_before_expiry']}")
	report(
		"roll_gap_intraday_minus_session",
		f"median={(rolls['intraday_gap'] - rolls['gap']).abs().median():.3f}, max={(rolls['intraday_gap'] - rolls['gap']).abs().max():.3f}",
		"absolute difference between the 15-minute and session-close gaps, USD/bbl",
	)
	daily_roll_path = config.get("daily_roll_calendar")
	if daily_roll_path and Path(daily_roll_path).exists():
		daily_rolls = pd.read_csv(daily_roll_path, parse_dates=["roll_date"])
		key = ["roll_date", "from_contract", "to_contract"]
		overlap = daily_rolls[daily_rolls["roll_date"].between(rolls["roll_date"].min(), rolls["roll_date"].max())]
		matched = rolls[key].merge(overlap[key], on=key, how="inner")
		report(
			"rolls_match_daily_roll_calendar",
			f"{len(matched)}/{len(rolls)}",
			f"{len(overlap)} daily rolls in the same period",
		)
	else:
		report(
			"rolls_match_daily_roll_calendar",
			"skipped",
			f"{daily_roll_path} not found; run python -m src.prepare_daily_data --config config/prepare_daily.yaml first",
		)

	# Continuous series for both horizons.
	session_continuous = build_continuous(
		sessions, contracts, rolls, roll_cfg["adjustment"], session_cfg["close_time"], timezone_name
	)
	session_continuous["sessions_to_expiry"] = add_sessions_to_expiry(session_continuous, contracts, calendar)
	continuous = build_intraday_continuous(bars, contracts, rolls)
	report("session_continuous_trade_dates", len(session_continuous))
	report("intraday_continuous_bars", len(continuous))
	report(
		"intraday_and_session_hold_same_contract",
		bool(
			continuous.groupby("trade_date")["contract_id"].first().reindex(session_continuous["trade_date"]).to_numpy().tolist()
			== session_continuous["contract_id"].tolist()
		),
	)

	window_bars = int(
		(pd.Timestamp(f"2000-01-01 {session_cfg['trade_window_end']}") - pd.Timestamp(f"2000-01-01 {session_cfg['trade_window_start']}"))
		/ pd.Timedelta(frequency)
	)
	coverage = (
		continuous[continuous["in_trade_window"]]
		.groupby("trade_date")
		.agg(window_bars=("bar_end", "size"), window_volume=("volume", "sum"), contract_id=("contract_id", "first"))
		.reindex(session_continuous["trade_date"], fill_value=0)
		.reset_index()
	)
	coverage["expected_window_bars"] = window_bars
	short = coverage[coverage["window_bars"].lt(window_bars)]
	report(
		"trade_dates_with_full_window",
		f"{int(coverage['window_bars'].eq(window_bars).sum())}/{len(coverage)}",
		f"{window_bars} bars from {session_cfg['trade_window_start']} to {session_cfg['trade_window_end']} CT; short: "
		+ ", ".join(f"{row.trade_date.date()} ({row.window_bars})" for row in short.itertuples()),
	)
	report("missing_window_bars", int((window_bars - coverage["window_bars"]).sum()), f"against a fixed {window_bars}-bar window")

	# Missing bars and gaps per contract and per session.
	held_by_date = continuous.drop_duplicates("trade_date").set_index("trade_date")["instrument_id"]
	bar_coverage = session_coverage(bars, contracts)
	bar_coverage["held"] = bar_coverage["instrument_id"].eq(bar_coverage["trade_date"].map(held_by_date))
	per_contract = contract_coverage(bar_coverage)
	held_coverage = bar_coverage[bar_coverage["held"]]
	report(
		"contract_session_bars_observed",
		f"{int(bar_coverage['observed_bars'].sum())}/{int(bar_coverage['expected_bars'].sum())}",
		f"all {len(per_contract)} contracts over {len(bar_coverage)} contract-sessions; deferred months trade sparsely, "
		"see cl_intraday_session_coverage.csv and cl_intraday_contract_coverage.csv",
	)
	report(
		"held_contract_missing_bars",
		int(held_coverage["missing_bars"].sum()),
		f"of {int(held_coverage['expected_bars'].sum())} expected bars in {len(held_coverage)} held sessions; "
		f"longest gap {int(held_coverage['longest_gap_bars'].max())} bars",
	)
	report(
		"held_contract_missing_window_bars",
		int(held_coverage["missing_window_bars"].sum()),
		f"of {int(held_coverage['expected_window_bars'].sum())} window bars in which the market traded",
	)
	report(
		"window_share_of_held_volume",
		f"{continuous.loc[continuous['in_trade_window'], 'volume'].sum() / continuous['volume'].sum():.1%}",
		"share of the held contract's volume traded inside the trading window",
	)

	# Outliers in the held series are corroborated when the next listed month moved the same way.
	change = bars.sort_values(["instrument_id", "bar_end"]).groupby("instrument_id")["close"].diff()
	change_by_key = pd.Series(change.to_numpy(), index=pd.MultiIndex.from_frame(bars.loc[change.index, ["bar_end", "instrument_id"]]))
	chain = contracts.sort_values("contract_month")["instrument_id"]
	next_listed = dict(zip(chain, chain.shift(-1)))
	uncorroborated = []
	flagged = continuous[continuous["outlier_flag"]]
	for row in flagged.itertuples():
		own = change_by_key.get((row.bar_end, row.instrument_id))
		other = change_by_key.get((row.bar_end, next_listed.get(row.instrument_id)))
		corroborated = other is not None and not np.isnan(other) and np.sign(own) == np.sign(other) and abs(other) >= 0.5 * abs(own)
		if not corroborated:
			uncorroborated.append(f"{row.contract_id} {row.timestamp} {own:+.2f}")
	report(
		"continuous_outlier_bars",
		len(flagged),
		f"{len(flagged) - len(uncorroborated)} corroborated by the next listed month; not corroborated: "
		+ ("; ".join(uncorroborated) or "none"),
	)

	split_dates = pd.DatetimeIndex(session_continuous["trade_date"])
	development = int(np.floor(config.get("development_share", 0.7) * len(split_dates)))
	report(
		"common_window_split",
		f"{split_dates[development - 1].date()} / {split_dates[development].date()}",
		f"last development / first holdout trade date ({development} of {len(split_dates)} trade dates)",
	)

	# Write outputs.
	bar_columns = [
		"timestamp", "trade_date", "contract_id", "instrument_id", "symbol", "contract_month", "last_trade_date",
		"sessions_to_expiry", *PRICE_COLUMNS, "volume", "minutes", "in_trade_window", "ohlc_ok", "outlier_flag", "degraded_date",
	]
	continuous_columns = [
		"timestamp", "trade_date", "contract_id", "instrument_id", "symbol", *PRICE_COLUMNS, "volume",
		*(f"signal_{column}" for column in PRICE_COLUMNS), "back_adjustment", "roll_flag", "in_trade_window", "is_cutoff_bar",
		"sessions_to_expiry", "minutes", "outlier_flag", "degraded_date",
	]
	session_columns = [
		"timestamp", "trade_date", "contract_id", "instrument_id", "symbol", "contract_month", "last_trade_date",
		"sessions_to_expiry", *PRICE_COLUMNS, "volume", "source_minutes", "ohlc_ok",
	]
	session_continuous_columns = [
		"timestamp", "trade_date", "contract_id", "instrument_id", "symbol", *PRICE_COLUMNS, "volume",
		*(f"signal_{column}" for column in PRICE_COLUMNS), "back_adjustment", "roll_flag", "sessions_to_expiry", "source_minutes",
	]
	outputs = {
		"contracts_15m": output_dir / "cl_intraday_15m_contracts.parquet",
		"continuous_15m": output_dir / "cl_intraday_15m_continuous.parquet",
		"session_contracts": output_dir / "cl_session_daily_contracts.parquet",
		"session_continuous": output_dir / "cl_session_daily_continuous.parquet",
		"roll_calendar": output_dir / "cl_intraday_roll_calendar.csv",
		"window_coverage": output_dir / "cl_intraday_window_coverage.csv",
		"session_coverage": output_dir / "cl_intraday_session_coverage.csv",
		"contract_coverage": output_dir / "cl_intraday_contract_coverage.csv",
		"quality_summary": output_dir / "cl_intraday_quality_summary.csv",
	}
	bars.sort_values(["contract_month", "bar_end"])[bar_columns].to_parquet(outputs["contracts_15m"], index=False)
	continuous[continuous_columns].to_parquet(outputs["continuous_15m"], index=False)
	sessions.sort_values(["contract_month", "trade_date"])[session_columns].to_parquet(outputs["session_contracts"], index=False)
	session_continuous[session_continuous_columns].to_parquet(outputs["session_continuous"], index=False)
	rolls.to_csv(outputs["roll_calendar"], index=False)
	coverage.to_csv(outputs["window_coverage"], index=False)
	bar_coverage.sort_values(["trade_date", "contract_id"]).to_csv(outputs["session_coverage"], index=False)
	per_contract.to_csv(outputs["contract_coverage"], index=False)
	summary_frame = pd.DataFrame(summary)
	summary_frame.to_csv(outputs["quality_summary"], index=False)

	figure_dir = config.get("figure_dir")
	if figure_dir:
		figure_path = Path(figure_dir) / "intraday_roll_examples.png"
		plot_roll_examples(continuous, bars, rolls, timezone_name, figure_path)
		outputs["roll_figure"] = figure_path

	manifest = {
		"config": str(args.config),
		"settings": config,
		"shared_settings": {"daily_config": config["daily_config"], "contracts": daily["contracts"], "roll": roll_cfg},
		"input": {"file": str(input_path), "sha256": sha256(input_path)},
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"python": platform.python_version(),
		"pandas": pd.__version__,
	}
	manifest_path = output_dir / "cl_intraday_prepare.manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	# Long notes are kept in full in the CSV; the console shows a shortened copy.
	with pd.option_context("display.max_colwidth", 110, "display.width", 200):
		print(summary_frame.assign(note=summary_frame["note"].str.slice(0, 110)).to_string(index=False))
	print(f"\nSaved outputs to {output_dir}/ and manifest to {manifest_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
