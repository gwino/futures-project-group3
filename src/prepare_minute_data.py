"""Clean full-session 1-minute CL bars and extract the reference prices of Baltussen et al. (2021).

Run from the repository root. The daily step is optional; it writes the daily roll calendar used to
cross-check the comparison roll rule:

    python -m src.prepare_daily_data --config config/prepare_daily.yaml
    python -m src.prepare_minute_data --config config/prepare_minute.yaml

Every return on trade date t is meant to be computed within the contract held on t, including the
previous 13:30 CT reference close, so roll gaps never enter overnight returns or realized volatility.

Outputs (in the configured output directory unless noted):
- cl_minute_front_contracts.parquet: validated 1-minute bars of the held and the next listed contract on
  every trade date (raw prices, contract IDs, quality flags), for return and realized-volatility work.
- cl_minute_reference_prices.parquet: one row per trade date with the held contract's prices at the
  previous 13:30 and today's 08:00, 08:30, 12:30, 13:00 and 13:30 CT, their staleness, and sample flags.
- cl_minute_session_quality.csv: per trade date coverage of the overnight (previous 13:30 to 08:00) and
  trading-window (08:00 to 13:30) minute paths of the held contract, early closes, volume and flags.
- cl_minute_roll_calendar.csv: rolls under the paper's volume-crossover rule; cl_minute_roll_comparison.csv:
  those rolls next to the existing 3-sessions-before-expiry rule (#17/#18).
- cl_minute_quality_summary.csv and cl_minute_prepare.manifest.json: checks, settings, and hashes.
- reports/tables/cl_intraday_coverage.csv: the monthly coverage inventory for #25.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import yaml

from src.prepare_daily_data import PRICE_COLUMNS, add_sessions_to_expiry, build_rolls, cme_holidays, contract_table, sha256, trade_calendar
from src.prepare_intraday_data import aggregate_sessions, assign_trade_dates, flag_outliers, outright_symbols

MINUTE = pd.Timedelta(minutes=1)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/prepare_minute.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> tuple[dict, dict]:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {"inputs", "daily_config", "output_dir", "session", "reference", "roll", "filters", "quality"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	with Path(config["daily_config"]).open(encoding="utf-8") as daily_file:
		daily = yaml.safe_load(daily_file)
	if daily["session"]["timezone"] != config["session"]["timezone"]:
		raise ValueError("The minute and daily configs must use the same session timezone.")
	return config, daily


def clock_label(clock: str) -> str:
	return clock.replace(":", "")


def at_clock(dates, clock: str, timezone_name: str) -> pd.Series:
	"""UTC instants of a Chicago wall-clock time on each date (daylight saving handled by the zone)."""
	local = pd.DatetimeIndex(dates) + pd.Timedelta(f"{clock}:00")
	return pd.Series(local.tz_localize(timezone_name).tz_convert("UTC"))


# --- Loading ----------------------------------------------------------------------------------


def load_minutes(paths: list[Path], symbol_pattern: str, chunk_rows: int = 2_000_000) -> tuple[pd.DataFrame, list[dict]]:
	"""Load outright 1-minute bars from several contiguous DBN files, reading each file in chunks.

	Most rows of a parent query are spreads, so filtering each chunk keeps memory low.
	"""
	frames, files, symbols = [], [], {}
	for source, path in enumerate(paths):
		store = db.DBNStore.from_file(path)
		file_symbols = outright_symbols(store, symbol_pattern)
		for instrument_id, symbol in file_symbols.items():
			if symbols.setdefault(instrument_id, symbol) != symbol:
				raise ValueError(f"instrument_id {instrument_id} maps to different symbols in different files.")
		raw_rows = outright_rows = 0
		for chunk in store.to_df(map_symbols=False, count=chunk_rows):
			raw_rows += len(chunk)
			chunk = chunk.reset_index()
			chunk = chunk[chunk["instrument_id"].isin(file_symbols)]
			outright_rows += len(chunk)
			frames.append(chunk[["ts_event", "instrument_id", *PRICE_COLUMNS, "volume"]].assign(source=source))
		files.append(
			{
				"source": source, "file": str(path), "start": pd.Timestamp(store.start), "end": pd.Timestamp(store.end),
				"raw_rows": raw_rows, "outright_rows": outright_rows,
			}
		)
	minutes = pd.concat(frames, ignore_index=True).rename(columns={"ts_event": "ts_utc"})
	minutes["symbol"] = minutes["instrument_id"].map(symbols).astype("category")
	return minutes, files


def check_contiguous(files: list[dict]) -> None:
	for earlier, later in zip(files, files[1:]):
		if earlier["end"] != later["start"]:
			raise ValueError(f"{earlier['file']} ends {earlier['end']} but {later['file']} starts {later['start']}.")


def complete_sessions(
	minutes: pd.DataFrame, calendar: pd.DatetimeIndex, timezone_name: str, open_time: str, close_time: str,
	data_start: pd.Timestamp, data_end: pd.Timestamp,
) -> pd.Series:
	"""True for minutes whose trade date lies wholly inside the queried range.

	A trade date's session opens at open_time on the previous trade date (weekend and holiday sessions
	fold forward) and closes at close_time, so queries that start or end at midnight UTC cut the first
	and last sessions.
	"""
	dates = pd.DatetimeIndex(minutes["trade_date"].unique())
	previous = calendar[calendar.searchsorted(dates) - 1]
	opens = at_clock(previous, open_time, timezone_name).set_axis(dates)
	closes = at_clock(dates, close_time, timezone_name).set_axis(dates)
	whole = opens.ge(data_start) & closes.le(data_end)
	return minutes["trade_date"].map(whole)


# --- Held contract ----------------------------------------------------------------------------


def held_contracts(dates: pd.DatetimeIndex, rolls: pd.DataFrame) -> pd.Series:
	"""instrument_id held on each trade date; a roll takes effect at the 17:00 CT open of its roll date."""
	held = pd.Series(rolls["from_instrument_id"].iloc[0], index=dates, name="instrument_id")
	for roll in rolls.itertuples():
		held[held.index >= roll.roll_date] = roll.to_instrument_id
	return held.rename_axis("trade_date")


def roll_comparison(primary: pd.DataFrame, comparison: pd.DataFrame, sessions: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
	"""Each roll under the paper rule and the pre-expiry rule, with the first session in which the next contract out-traded the held one.

	`paper_roll_by_backstop` is true when the volume crossover never happened early enough and the
	paper rule rolled only because of its days-before-expiry backstop.
	"""
	key = ["from_contract", "to_contract", "from_last_trade_date"]
	columns = [*key, "roll_date", "sessions_before_expiry", "gap"]
	table = primary[columns].merge(comparison[columns], on=key, how="outer", suffixes=("_paper", "_pre_expiry"))
	both = table["roll_date_paper"].notna() & table["roll_date_pre_expiry"].notna()
	table["sessions_paper_minus_pre_expiry"] = np.nan
	table.loc[both, "sessions_paper_minus_pre_expiry"] = calendar.searchsorted(
		table.loc[both, "roll_date_paper"].to_numpy()
	) - calendar.searchsorted(table.loc[both, "roll_date_pre_expiry"].to_numpy())

	volume = sessions.pivot(index="trade_date", columns="instrument_id", values="volume")
	crossover, held_since = {}, volume.index.min()
	for roll in primary.itertuples():
		window = volume.loc[(volume.index >= held_since) & (volume.index <= roll.from_last_trade_date), [roll.from_instrument_id, roll.to_instrument_id]]
		ahead = window[roll.to_instrument_id].gt(window[roll.from_instrument_id])
		crossover[roll.from_contract] = ahead.idxmax() if ahead.any() else pd.NaT
		held_since = roll.roll_date
	table["first_volume_crossover"] = table["from_contract"].map(crossover)
	known = table["first_volume_crossover"].notna()
	table["crossover_sessions_before_expiry"] = np.nan
	table.loc[known, "crossover_sessions_before_expiry"] = calendar.searchsorted(
		table.loc[known, "from_last_trade_date"].to_numpy()
	) - calendar.searchsorted(table.loc[known, "first_volume_crossover"].to_numpy())
	acted = table["first_volume_crossover"].map(lambda date: calendar[calendar.searchsorted(date) + 1] if pd.notna(date) else pd.NaT)
	table["paper_roll_by_backstop"] = table["roll_date_paper"].notna() & ~acted.eq(table["roll_date_paper"])
	return table.sort_values("from_last_trade_date").reset_index(drop=True)


# --- Reference prices -------------------------------------------------------------------------


def reference_prices(
	minutes: pd.DataFrame, held: pd.Series, calendar: pd.DatetimeIndex, timezone_name: str,
	previous_close: str, points: list[str], max_stale_minutes: int,
) -> pd.DataFrame:
	"""The held contract's price at each reference time, using only trades before that time.

	The price at T is the close of the held contract's last 1-minute bar starting before T, provided it
	starts no more than max_stale_minutes before T - 1 minute; otherwise it is missing. The previous
	close comes from the contract held on t, not the one held on t - 1, so a roll never puts a price
	gap into the overnight return. `stale_*` is the number of minutes between the bar used and the bar
	starting at T - 1 minute (0 when that minute traded).
	"""
	dates = pd.DatetimeIndex(held.index)
	previous = calendar[calendar.searchsorted(dates) - 1]
	frame = pd.DataFrame({"trade_date": dates, "instrument_id": held.to_numpy(), "prev_trade_date": previous})
	right = minutes[["ts_utc", "instrument_id", "close"]].sort_values("ts_utc")
	targets = [(f"prev_{clock_label(previous_close)}", previous, previous_close)]
	targets += [(clock_label(clock), dates, clock) for clock in points]
	for label, days, clock in targets:
		left = pd.DataFrame({"trade_date": dates, "instrument_id": held.to_numpy(), "lookup": at_clock(days, clock, timezone_name) - MINUTE})
		found = pd.merge_asof(
			left.sort_values("lookup"), right, left_on="lookup", right_on="ts_utc", by="instrument_id",
			direction="backward", tolerance=pd.Timedelta(minutes=max_stale_minutes),
		).set_index("trade_date")
		frame[f"p_{label}"] = frame["trade_date"].map(found["close"])
		frame[f"stale_{label}"] = frame["trade_date"].map((found["lookup"] - found["ts_utc"]) / MINUTE)
	price_columns = [f"p_{label}" for label, _, _ in targets]
	frame["complete"] = frame[price_columns].notna().all(axis=1)
	return frame


def period_coverage(minutes: pd.DataFrame, market_minutes: pd.DatetimeIndex, periods: pd.DataFrame, name: str) -> pd.DataFrame:
	"""Held-contract minutes inside [start, end) for each trade date, against the minutes in which any outright traded.

	`periods` has trade_date, instrument_id (the contract held on that trade date), start and end (UTC).
	`minutes` must include that contract's minutes before it became held, so a roll date's overnight path
	starts at the previous 13:30 in the new contract. Gaps are counted in market minutes (minutes in which
	any outright traded), so scheduled halts, weekends, and holiday closures are not gaps. The longest gap
	includes the period's edges.
	"""
	keyed = pd.merge_asof(
		minutes[["ts_utc", "instrument_id"]].sort_values("ts_utc"), periods.sort_values("start"), left_on="ts_utc", right_on="start",
		by="instrument_id", direction="backward",
	)
	keyed = keyed[keyed["ts_utc"].lt(keyed["end"])].sort_values(["trade_date", "ts_utc"])
	keyed["position"] = market_minutes.searchsorted(keyed["ts_utc"])
	keyed["gap"] = keyed.groupby("trade_date")["position"].diff() - 1
	per_date = keyed.groupby("trade_date").agg(
		observed=("position", "size"), first=("position", "min"), last=("position", "max"), internal_gap=("gap", "max")
	)
	stats = periods.set_index("trade_date")[["start", "end"]].join(per_date)
	start, end = market_minutes.searchsorted(stats["start"]), market_minutes.searchsorted(stats["end"])
	market = pd.Series(end - start, index=stats.index)
	edge = np.maximum(stats["first"] - start, end - stats["last"] - 1)
	longest = np.maximum(stats["internal_gap"].fillna(0), edge).where(stats["observed"].notna(), market)
	observed = stats["observed"].fillna(0).astype(int)
	return pd.DataFrame(
		{
			f"{name}_minutes": observed,
			f"{name}_market_minutes": market,
			f"{name}_missing_minutes": market - observed,
			f"{name}_longest_gap_minutes": longest.astype(int),
		}
	)


def monthly_coverage(quality: pd.DataFrame, rolls: pd.DataFrame, files: list[dict]) -> pd.DataFrame:
	"""Coverage inventory by calendar month of the trade date (#25)."""
	frame = quality.assign(month=quality["trade_date"].dt.to_period("M").astype(str))
	frame["file"] = frame["source"].map({entry["source"]: Path(entry["file"]).name for entry in files})
	frame["overnight_path"] = frame["overnight_minutes"].gt(0)
	frame["full_window"] = frame["window_missing_minutes"].eq(0) & ~frame["early_close"]
	monthly = frame.groupby("month").agg(
		source_file=("file", lambda names: ", ".join(sorted(set(names)))),
		trade_dates=("trade_date", "size"),
		held_contracts=("contract_id", lambda ids: " ".join(dict.fromkeys(ids))),
		complete_reference=("complete", "sum"),
		paper_sample=("paper_sample", "sum"),
		early_close=("early_close", "sum"),
		low_volume=("low_volume", "sum"),
		folded_holiday=("folded_holiday", "sum"),
		degraded=("degraded_date", "sum"),
		sessions_with_overnight_path=("overnight_path", "sum"),
		median_overnight_minutes=("overnight_minutes", "median"),
		overnight_missing_minutes=("overnight_missing_minutes", "sum"),
		max_overnight_gap_minutes=("overnight_longest_gap_minutes", "max"),
		sessions_with_full_window=("full_window", "sum"),
		window_missing_minutes=("window_missing_minutes", "sum"),
		max_window_gap_minutes=("window_longest_gap_minutes", "max"),
		held_volume=("session_volume", "sum"),
	)
	roll_months = rolls["roll_date"].dt.to_period("M").astype(str).value_counts()
	monthly.insert(3, "rolls", roll_months.reindex(monthly.index, fill_value=0))
	return monthly.reset_index()


# --- Main -------------------------------------------------------------------------------------


def main() -> int:
	args = parse_args()
	config, daily = load_config(args.config)
	session_cfg, reference_cfg, roll_cfg = config["session"], config["reference"], config["roll"]
	timezone_name = session_cfg["timezone"]
	output_dir = Path(config["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)
	input_paths = [Path(path) for path in config["inputs"]]

	summary: list[dict] = []

	def report(check: str, value, note: str = "") -> None:
		summary.append({"check": check, "value": value, "note": note})

	# Load, de-duplicate, and remove non-positive prices.
	minutes, files = load_minutes(input_paths, daily["contracts"]["symbol_pattern"])
	check_contiguous(files)
	data_start, data_end = files[0]["start"], files[-1]["end"]
	for entry in files:
		report(f"raw_rows[{Path(entry['file']).name}]", entry["raw_rows"], f"{entry['start']} to {entry['end']}; {entry['outright_rows']} outright rows")
	report("outright_rows", len(minutes), "calendar spreads and other strategies excluded")
	report("outright_missing_values", int(minutes[PRICE_COLUMNS + ["volume"]].isna().sum().sum()))
	minutes = minutes.sort_values(["instrument_id", "ts_utc"], kind="stable").reset_index(drop=True)
	duplicate = minutes.duplicated(subset=["instrument_id", "ts_utc"], keep="first")
	report("duplicate_instrument_minute_rows_removed", int(duplicate.sum()), "includes any overlap between input files")
	minutes = minutes[~duplicate]
	nonpositive = minutes[PRICE_COLUMNS].le(0).any(axis=1)
	report(
		"nonpositive_price_minutes_removed", int(nonpositive.sum()),
		"paper filter (i); " + "; ".join(f"{row.symbol} {row.ts_utc} low={row.low}" for row in minutes[nonpositive].head(10).itertuples()),
	)
	minutes = minutes[~nonpositive]

	# Trade dates and session boundaries.
	calendar_start = pd.Timestamp(data_start.year - 2, 1, 1)
	calendar_end = pd.Timestamp(data_end.year + 12, 12, 31)
	holidays = cme_holidays(calendar_start, calendar_end, daily["session"]["juneteenth_from"])
	calendar = trade_calendar(calendar_start, calendar_end, holidays)
	minutes = assign_trade_dates(minutes, calendar, timezone_name, session_cfg["open_time"], session_cfg["close_time"])
	after_close = minutes["after_close"]
	report(
		"after_close_minutes_dropped", int(after_close.sum()),
		"; ".join(f"{row.symbol} {row.ts_utc.tz_convert(timezone_name)} volume={row.volume}" for row in minutes[after_close].head(20).itertuples()),
	)
	minutes = minutes[~after_close]
	whole = complete_sessions(minutes, calendar, timezone_name, session_cfg["open_time"], session_cfg["close_time"], data_start, data_end)
	report(
		"partial_session_minutes_dropped", int((~whole).sum()),
		"trade dates cut by the query range: " + ", ".join(str(date.date()) for date in minutes.loc[~whole, "trade_date"].unique()),
	)
	minutes = minutes[whole].reset_index(drop=True)
	report("folded_minutes", int(minutes["folded"].sum()), "weekend-holiday sessions folded into the next trade date")
	minutes["ohlc_ok"] = minutes["low"].le(minutes[["open", "close"]].min(axis=1)) & minutes["high"].ge(minutes[["open", "close"]].max(axis=1))
	report("ohlc_inconsistent_minutes", int((~minutes["ohlc_ok"]).sum()), "flagged, not removed")
	report("negative_volume_minutes", int(minutes["volume"].lt(0).sum()))
	report("zero_volume_minutes", int(minutes["volume"].eq(0).sum()))

	sessions = aggregate_sessions(minutes)
	first_date, last_date = sessions["trade_date"].min(), sessions["trade_date"].max()
	sample_calendar = calendar[(calendar >= first_date) & (calendar <= last_date)]
	report("sample_trade_dates", len(sample_calendar), f"{first_date.date()} to {last_date.date()}")
	missing_market = sample_calendar.difference(pd.DatetimeIndex(sessions["trade_date"].unique()))
	report("trade_dates_without_any_outright", len(missing_market), ", ".join(str(date.date()) for date in missing_market))

	contracts = contract_table(sessions, calendar, last_date)
	expired = contracts[contracts["expired_in_sample"] & contracts["first_date"].gt(first_date)]
	report("contracts", len(contracts))
	report(
		"expired_contracts_last_session_on_computed_ltd", f"{int(expired['last_session_is_ltd'].sum())}/{len(expired)}",
		", ".join(expired.loc[~expired["last_session_is_ltd"], "contract_id"]),
	)
	identity = contracts.set_index("instrument_id")

	# Rolls: the paper's volume crossover, and the existing pre-expiry rule for comparison.
	rolls = build_rolls(sessions, contracts, calendar, roll_cfg["rule"], roll_cfg["days_before_expiry"])
	comparison = build_rolls(sessions, contracts, calendar, roll_cfg["comparison_rule"], roll_cfg["comparison_days_before_expiry"])
	compared = roll_comparison(rolls, comparison, sessions, calendar)
	report("rolls", len(rolls), f"rule={roll_cfg['rule']}, backstop {roll_cfg['days_before_expiry']} trade date(s) before expiry")
	crossovers = compared["crossover_sessions_before_expiry"].dropna().astype(int).value_counts().sort_index()
	report(
		"first_volume_crossover_sessions_before_expiry", ", ".join(f"{k}: {v}" for k, v in crossovers.items()),
		f"paper rolls set by the backstop: {int(compared['paper_roll_by_backstop'].sum())}",
	)
	offsets = rolls["sessions_before_expiry"].value_counts().sort_index()
	report("roll_sessions_before_expiry", ", ".join(f"{k}: {v}" for k, v in offsets.items()), "trade dates between the roll date and the expiring contract's last trade date")
	difference = compared["sessions_paper_minus_pre_expiry"].dropna().astype(int).value_counts().sort_index()
	report(
		"roll_paper_minus_pre_expiry_sessions", ", ".join(f"{k:+d}: {v}" for k, v in difference.items()),
		f"positive means the paper rule rolls later than the {roll_cfg['comparison_days_before_expiry']}-day pre-expiry rule",
	)
	daily_roll_path = config.get("daily_roll_calendar")
	if daily_roll_path and Path(daily_roll_path).exists():
		daily_rolls = pd.read_csv(daily_roll_path, parse_dates=["roll_date"])
		key = ["roll_date", "from_contract", "to_contract"]
		overlap = daily_rolls[daily_rolls["roll_date"].between(comparison["roll_date"].min(), comparison["roll_date"].max())]
		matched = comparison[key].merge(overlap[key], on=key, how="inner")
		report("pre_expiry_rolls_match_daily_roll_calendar", f"{len(matched)}/{len(comparison)}", f"{len(overlap)} daily rolls in the same period")
	else:
		report("pre_expiry_rolls_match_daily_roll_calendar", "skipped", f"{daily_roll_path} not found")

	# The held and next listed contract on every trade date.
	dates = pd.DatetimeIndex(sorted(sessions["trade_date"].unique()))
	held = held_contracts(dates, rolls).astype(minutes["instrument_id"].dtype)
	chain = contracts.sort_values("contract_month")["instrument_id"]
	next_listed = dict(zip(chain, chain.shift(-1)))
	front_keys = pd.concat(
		[
			held.reset_index().assign(held=True),
			held.map(next_listed).dropna().astype(held.dtype).reset_index().assign(held=False),
		]
	)
	front = minutes.merge(front_keys, on=["trade_date", "instrument_id"], how="inner")
	front["contract_id"] = front["instrument_id"].map(identity["contract_id"])
	front["timestamp"] = front["ts_utc"].dt.tz_convert(timezone_name)
	degraded = pd.DatetimeIndex(pd.to_datetime(daily.get("degraded_dates", [])))
	front["degraded_date"] = front["ts_utc"].dt.tz_localize(None).dt.normalize().isin(degraded)
	front["outlier_flag"] = flag_outliers(
		front.rename(columns={"ts_utc": "bar_end"}), config["quality"]["outlier_window"], config["quality"]["outlier_threshold"]
	)

	# Held-contract outliers are corroborated when the next listed month moved the same way in the same minute.
	change = front.sort_values(["instrument_id", "ts_utc"]).groupby("instrument_id")["close"].diff()
	change_by_key = pd.Series(change.to_numpy(), index=pd.MultiIndex.from_frame(front.loc[change.index, ["ts_utc", "instrument_id"]]))
	flagged = front[front["outlier_flag"] & front["held"]]
	other_keys = pd.MultiIndex.from_arrays([flagged["ts_utc"], flagged["instrument_id"].map(next_listed)])
	own = change_by_key.reindex(pd.MultiIndex.from_frame(flagged[["ts_utc", "instrument_id"]])).to_numpy()
	other = change_by_key[~change_by_key.index.duplicated()].reindex(other_keys).to_numpy()
	corroborated = (np.sign(own) == np.sign(other)) & (np.abs(other) >= 0.5 * np.abs(own))
	front["outlier_corroborated"] = False
	front.loc[flagged.index, "outlier_corroborated"] = corroborated
	uncorroborated = flagged[~corroborated]
	report(
		"held_outlier_minutes", len(flagged),
		f"{int(corroborated.sum())} corroborated by the next listed month; not corroborated: "
		+ ("; ".join(f"{row.contract_id} {row.timestamp} {row.close:.2f}" for row in uncorroborated.head(20).itertuples()) or "none"),
	)

	# Reference prices and per-session quality.
	held_minutes = front[front["held"]]
	references = reference_prices(
		front, held, calendar, timezone_name, reference_cfg["previous_close"], reference_cfg["points"], reference_cfg["max_stale_minutes"]
	)
	labels = [f"prev_{clock_label(reference_cfg['previous_close'])}", *(clock_label(clock) for clock in reference_cfg["points"])]
	market_minutes = pd.DatetimeIndex(minutes["ts_utc"].unique()).sort_values()
	# Overnight: previous reference close to today's open; window: today's open to today's reference close.
	previous_close = at_clock(references["prev_trade_date"], reference_cfg["previous_close"], timezone_name)
	window_open = at_clock(references["trade_date"], reference_cfg["points"][0], timezone_name)
	window_close = at_clock(references["trade_date"], reference_cfg["points"][-1], timezone_name)
	overnight = pd.DataFrame(
		{"trade_date": references["trade_date"], "instrument_id": references["instrument_id"], "start": previous_close, "end": window_open}
	)
	window = overnight.assign(start=window_open, end=window_close)
	quality = references[["trade_date", "prev_trade_date", "instrument_id"]].copy()
	quality["contract_id"] = quality["instrument_id"].map(identity["contract_id"])
	quality["roll_flag"] = quality["trade_date"].isin(rolls["roll_date"])
	quality["sessions_to_expiry"] = add_sessions_to_expiry(quality, contracts, calendar)
	session_volume = sessions.set_index(["trade_date", "instrument_id"])["volume"]
	quality["session_volume"] = session_volume.reindex(pd.MultiIndex.from_frame(quality[["trade_date", "instrument_id"]])).fillna(0).astype(int).to_numpy()
	per_date = held_minutes.groupby("trade_date")
	quality["session_minutes"] = quality["trade_date"].map(per_date.size()).fillna(0).astype(int)
	quality["source"] = quality["trade_date"].map(per_date["source"].agg(lambda values: values.mode().iloc[0]))
	quality["folded_holiday"] = quality["trade_date"].map(minutes.groupby("trade_date")["folded"].any()).fillna(False)
	quality["degraded_date"] = quality["trade_date"].map(per_date["degraded_date"].any()).fillna(False)
	quality["outlier_minutes"] = quality["trade_date"].map(per_date["outlier_flag"].sum()).fillna(0).astype(int)
	quality["ohlc_inconsistent_minutes"] = quality["trade_date"].map((~held_minutes["ohlc_ok"]).groupby(held_minutes["trade_date"]).sum()).fillna(0).astype(int)
	market_last = minutes.groupby("trade_date")["ts_utc"].max() + MINUTE
	quality["market_close"] = quality["trade_date"].map(market_last).dt.tz_convert(timezone_name)
	scheduled_close = at_clock(quality["trade_date"], session_cfg["close_time"], timezone_name)
	quality["early_close"] = (scheduled_close - quality["market_close"].dt.tz_convert("UTC")).ge(pd.Timedelta(minutes=session_cfg["early_close_minutes"])).to_numpy()
	quality = quality.join(period_coverage(front, market_minutes, overnight, "overnight"), on="trade_date")
	quality = quality.join(period_coverage(front, market_minutes, window, "window"), on="trade_date")
	quality["complete"] = references["complete"].to_numpy()
	quality["low_volume"] = quality["session_volume"].lt(config["filters"]["min_session_volume"])
	quality["paper_sample"] = quality["complete"] & ~quality["early_close"] & ~quality["low_volume"]
	flags = ["roll_flag", "early_close", "low_volume", "folded_holiday", "degraded_date", "complete", "paper_sample"]
	shared = ["contract_id", "session_volume", *(flag for flag in flags if flag != "complete")]
	references = references.merge(quality[["trade_date", *shared]], on="trade_date")

	report("held_sessions", len(quality), f"{held_minutes['ts_utc'].size} held-contract minutes")
	for label in labels:
		stale = references[f"stale_{label}"]
		report(
			f"reference_{label}", f"{int(references[f'p_{label}'].notna().sum())}/{len(references)}",
			f"stale (>0 min): {int(stale.gt(0).sum())}, missing: {int(references[f'p_{label}'].isna().sum())}: "
			+ ", ".join(str(date.date()) for date in references.loc[references[f"p_{label}"].isna(), "trade_date"].head(15)),
		)
	report("complete_reference_sessions", f"{int(quality['complete'].sum())}/{len(quality)}")
	report(
		"early_close_sessions", int(quality["early_close"].sum()),
		", ".join(f"{row.trade_date.date()} ({row.market_close.strftime('%H:%M')})" for row in quality[quality["early_close"]].itertuples()),
	)
	report("low_volume_sessions", int(quality["low_volume"].sum()), f"held contract volume < {config['filters']['min_session_volume']}")
	report("paper_sample_sessions", f"{int(quality['paper_sample'].sum())}/{len(quality)}", "complete, not an early close, volume >= threshold")
	report(
		"window_missing_minutes", int(quality["window_missing_minutes"].sum()),
		f"held-contract minutes without a trade while another outright traded, 08:00-13:30; longest gap {int(quality['window_longest_gap_minutes'].max())} min",
	)
	report(
		"overnight_missing_minutes", int(quality["overnight_missing_minutes"].sum()),
		f"previous 13:30 to 08:00; sessions without any overnight minute: {int(quality['overnight_minutes'].eq(0).sum())}",
	)
	report("degraded_sessions", int(quality["degraded_date"].sum()), "flagged, not removed")

	# Write outputs.
	coverage = monthly_coverage(quality, rolls, files)
	front_columns = [
		"timestamp", "ts_utc", "trade_date", "contract_id", "instrument_id", "symbol", *PRICE_COLUMNS, "volume", "held",
		"folded", "ohlc_ok", "outlier_flag", "outlier_corroborated", "degraded_date", "source",
	]
	reference_columns = [
		"trade_date", "prev_trade_date", "contract_id", "instrument_id",
		*(column for label in labels for column in (f"p_{label}", f"stale_{label}")), "session_volume", *flags,
	]
	outputs = {
		"front_minutes": output_dir / "cl_minute_front_contracts.parquet",
		"reference_prices": output_dir / "cl_minute_reference_prices.parquet",
		"session_quality": output_dir / "cl_minute_session_quality.csv",
		"roll_calendar": output_dir / "cl_minute_roll_calendar.csv",
		"roll_comparison": output_dir / "cl_minute_roll_comparison.csv",
		"quality_summary": output_dir / "cl_minute_quality_summary.csv",
		"coverage_table": Path(config.get("coverage_table", output_dir / "cl_intraday_coverage.csv")),
	}
	front.sort_values(["ts_utc", "held"], ascending=[True, False])[front_columns].to_parquet(outputs["front_minutes"], index=False)
	references[reference_columns].to_parquet(outputs["reference_prices"], index=False)
	quality.to_csv(outputs["session_quality"], index=False)
	rolls.to_csv(outputs["roll_calendar"], index=False)
	compared.to_csv(outputs["roll_comparison"], index=False)
	outputs["coverage_table"].parent.mkdir(parents=True, exist_ok=True)
	coverage.to_csv(outputs["coverage_table"], index=False)
	summary_frame = pd.DataFrame(summary)
	summary_frame.to_csv(outputs["quality_summary"], index=False)

	manifest = {
		"config": str(args.config),
		"settings": config,
		"shared_settings": {"daily_config": config["daily_config"], "contracts": daily["contracts"], "degraded_dates": daily.get("degraded_dates", [])},
		"inputs": [{**{k: str(v) for k, v in entry.items()}, "sha256": sha256(Path(entry["file"]))} for entry in files],
		"outputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"python": platform.python_version(),
		"pandas": pd.__version__,
	}
	manifest_path = output_dir / "cl_minute_prepare.manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

	with pd.option_context("display.max_colwidth", 110, "display.width", 200):
		print(summary_frame.assign(note=summary_frame["note"].str.slice(0, 110)).to_string(index=False))
	print(f"\nSaved outputs to {output_dir}/, {outputs['coverage_table']}, and manifest to {manifest_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
