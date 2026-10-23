"""Backtest the final-half-hour directional strategy of Baltussen et al. (2021) on CL, 2021-2026 (#30).

Run from the repository root, after the minute pipeline (#27):

    python -m src.backtest_lh_strategy --config config/lh_strategy.yaml

The paper's timing strategy (Eq. 12) is long over the last half hour when the rest-of-day return is
positive and short otherwise. Here the signal is read at 13:00 CT from the pipeline's reference
prices, the position is opened at the first held-contract trade at or after 13:00 CT and closed at
the first trade at or after 13:30 CT, and commissions and a spread/slippage proxy are charged per
side. The paper's frictionless returns, the feasible gross P&L and the net P&L are reported apart.

Outputs (in the configured tables and figures directories):
- lh_strategy_trades.csv: the trade ledger of the baseline signal and execution, one row per sample
  trade date, with signal, fills, underlying return, gross and net P&L and the reason for any skip.
- lh_strategy_summary.csv: every signal under frictionless, feasible gross and net accounting.
- lh_strategy_cost_sensitivity.csv, lh_strategy_by_year.csv, lh_strategy_roll_rules.csv,
  lh_strategy_robustness.csv, lh_strategy_worst_days.csv, lh_strategy_rolling.csv.
- lh_strategy.manifest.json: settings, checks and output hashes.
- lh_strategy_cumulative.png, lh_strategy_by_year.png, lh_strategy_rolling.png,
  lh_strategy_cost_sensitivity.png.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.prepare_daily_data import sha256
from src.prepare_minute_data import at_clock, clock_label, held_contracts, reference_prices
from src.replicate_regressions import newey_west_lags

MINUTE = pd.Timedelta(minutes=1)
LEDGER_COLUMNS = [
	"trade_date", "contract_id", "signal", "predictor", "position", "entry_time", "entry_price", "entry_delay",
	"exit_time", "exit_price", "exit_delay", "p_1300", "p_1330", "r_lh", "r_frictionless", "r_gross", "pnl_gross_usd",
	"sides", "commission_usd", "slippage_usd", "cost_usd", "pnl_net_usd", "r_net", "traded", "skip_reason",
]
# Categorical palette and text tokens (dataviz skill reference palette, light mode).
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--config", default=Path("config/lh_strategy.yaml"), type=Path)
	return parser.parse_args()


def load_config(path: Path) -> tuple[dict, dict]:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	required = {
		"reference_prices", "front_minutes", "session_quality", "roll_comparison", "minute_config", "tables_dir", "figures_dir",
		"sample", "signals", "baseline_signal", "decision_time", "execution", "execution_variants", "baseline_execution",
		"costs", "roll_rules", "blackouts", "evaluation",
	}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	with Path(config["minute_config"]).open(encoding="utf-8") as minute_file:
		minute = yaml.safe_load(minute_file)
	check_signals(config["signals"], config["decision_time"])
	if config["baseline_signal"] not in config["signals"]:
		raise ValueError(f"baseline_signal {config['baseline_signal']} is not among the signals.")
	if config["baseline_execution"] not in config["execution_variants"]:
		raise ValueError(f"baseline_execution {config['baseline_execution']} is not among the execution variants.")
	execution = config["execution"]
	if clock_minutes(execution["entry_time"]) < clock_minutes(config["decision_time"]):
		raise ValueError("The entry time cannot be before the decision time.")
	if clock_minutes(execution["exit_time"]) <= clock_minutes(execution["entry_time"]):
		raise ValueError("The exit time must be after the entry time.")
	return config, minute


# --- Signals ----------------------------------------------------------------------------------


def clock_minutes(clock: str) -> int:
	hours, minutes = clock.split(":")
	return int(hours) * 60 + int(minutes)


def price_minutes(column: str) -> int:
	"""Minutes from midnight on the trade date of a reference price column; `p_prev_*` is a day earlier."""
	if not column.startswith("p_"):
		raise ValueError(f"{column} is not a reference price column (p_HHMM or p_prev_HHMM).")
	label = column[2:]
	previous = label.startswith("prev_")
	clock = label[5:] if previous else label
	return int(clock[:2]) * 60 + int(clock[2:]) - (24 * 60 if previous else 0)


def signal_intervals(spec: dict) -> list[list[str]]:
	if "long_if_positive" in spec:
		return [spec["long_if_positive"]]
	if "long_if_all_positive" in spec:
		return spec["long_if_all_positive"]
	if "constant" in spec:
		return []
	raise ValueError(f"Unknown signal specification {spec}.")


def check_signals(signals: dict, decision_time: str) -> None:
	"""Every signal price is observed at or before the decision time, and intervals run forward."""
	decision = clock_minutes(decision_time)
	for name, spec in signals.items():
		for start, end in signal_intervals(spec):
			if price_minutes(start) >= price_minutes(end):
				raise ValueError(f"Signal {name}: {start} must come before {end}.")
			if price_minutes(end) > decision:
				raise ValueError(f"Signal {name} uses {end}, after the {decision_time} decision time (look-ahead).")
		if "constant" in spec and spec["constant"] not in (-1, 0, 1):
			raise ValueError(f"Signal {name}: a constant position must be -1, 0 or 1.")


def positions(prices: pd.DataFrame, spec: dict) -> tuple[pd.Series, pd.Series]:
	"""Position (+1, -1 or 0) on each trade date and the predictor value the position is based on.

	A return of exactly zero gives no position. For long_if_all_positive the predictor is the first return
	listed; the position is +1 only when every return is positive and -1 only when every one is negative.
	"""
	if "constant" in spec:
		return pd.Series(spec["constant"], index=prices.index, dtype=int), pd.Series(np.nan, index=prices.index)
	returns = [prices[end] / prices[start] - 1 for start, end in signal_intervals(spec)]
	frame = pd.concat(returns, axis=1)
	position = pd.Series(0, index=prices.index, dtype=int)
	position[frame.gt(0).all(axis=1)] = 1
	position[frame.lt(0).all(axis=1)] = -1
	position[frame.isna().any(axis=1)] = 0
	return position, returns[0]


# --- Fills ------------------------------------------------------------------------------------


def next_bar_fills(minutes: pd.DataFrame, held: pd.Series, clock: str, timezone_name: str, max_delay_minutes: int) -> pd.DataFrame:
	"""Open of the first held-contract bar starting at or after the Chicago time on each trade date.

	`held` maps trade_date to instrument_id. The bar must start within max_delay_minutes of the time;
	otherwise price and time are missing. `delay` is minutes from the time to the bar's start (0 when the
	market traded in the minute starting at the time). Only bars at or after the time are considered, so
	no fill uses an earlier trade.
	"""
	dates = pd.DatetimeIndex(held.index)
	left = pd.DataFrame({"trade_date": dates, "instrument_id": held.to_numpy(), "lookup": at_clock(dates, clock, timezone_name).to_numpy()})
	right = minutes[["ts_utc", "instrument_id", "open"]].sort_values("ts_utc")
	found = pd.merge_asof(
		left.sort_values("lookup"), right, left_on="lookup", right_on="ts_utc", by="instrument_id",
		direction="forward", tolerance=pd.Timedelta(minutes=max_delay_minutes),
	).set_index("trade_date").sort_index()
	return pd.DataFrame(
		{
			"time": found["ts_utc"].dt.tz_convert(timezone_name),
			"price": found["open"],
			"delay": (found["ts_utc"] - found["lookup"]) / MINUTE,
		}
	)


def reference_fills(references: pd.DataFrame, clock: str, timezone_name: str) -> pd.DataFrame:
	"""The pipeline's price at the time: the close of the bar before it (the paper's convention, not executable)."""
	label = clock_label(clock)
	if f"p_{label}" not in references:
		raise ValueError(f"No reference price at {clock}; use next_bar_open fills for that time.")
	frame = references.set_index("trade_date")
	stale = frame[f"stale_{label}"]
	instants = pd.DatetimeIndex(at_clock(frame.index, clock, timezone_name)) - pd.to_timedelta(1 + stale.fillna(0).to_numpy(), unit="min")
	time = pd.Series(instants.tz_convert(timezone_name), index=frame.index).where(frame[f"p_{label}"].notna())
	return pd.DataFrame({"time": time, "price": frame[f"p_{label}"], "delay": -1 - stale}, index=frame.index)


def settlement_proxy_fills(minutes: pd.DataFrame, held: pd.Series, clock: str, timezone_name: str, window_minutes: int) -> pd.DataFrame:
	"""Volume-weighted average price of the held contract's bars in the window before the time: a settlement proxy.

	CL's daily settlement is the volume-weighted average of outright trades from 13:28:00 to 13:30:00 CT,
	and a trade-at-settlement (TAS) order clears at that price plus its traded differential. The proxy
	weights each bar's OHLC average by its volume over the bars starting in [time - window, time). The
	fill time is the time itself (the settlement is fixed then) and the delay 0; the price is missing
	when no bar traded in the window.
	"""
	dates = pd.DatetimeIndex(held.index)
	end = pd.DatetimeIndex(at_clock(dates, clock, timezone_name))
	bounds = pd.DataFrame({"trade_date": dates, "instrument_id": held.to_numpy(), "start": end - pd.Timedelta(minutes=window_minutes), "end": end})
	bars = minutes[["ts_utc", "instrument_id", "open", "high", "low", "close", "volume"]].sort_values("ts_utc")
	keyed = pd.merge_asof(bars, bounds.sort_values("start"), left_on="ts_utc", right_on="start", by="instrument_id", direction="backward")
	keyed = keyed[keyed["trade_date"].notna() & keyed["ts_utc"].lt(keyed["end"])]
	keyed["weighted"] = keyed[["open", "high", "low", "close"]].mean(axis=1) * keyed["volume"]
	totals = keyed.groupby("trade_date")[["weighted", "volume"]].sum()
	price = (totals["weighted"] / totals["volume"]).reindex(dates)
	return pd.DataFrame({"time": pd.Series(end.tz_convert(timezone_name), index=dates).where(price.notna()), "price": price, "delay": pd.Series(0.0, index=dates).where(price.notna())})


def fills(kind: str, clock: str, references: pd.DataFrame, minutes: pd.DataFrame, held: pd.Series, timezone_name: str, execution: dict) -> pd.DataFrame:
	if kind == "reference":
		return reference_fills(references, clock, timezone_name)
	if kind == "next_bar_open":
		return next_bar_fills(minutes, held, clock, timezone_name, execution["max_delay_minutes"])
	if kind == "settlement_proxy":
		return settlement_proxy_fills(minutes, held, clock, timezone_name, execution["settlement_proxy_minutes"])
	raise ValueError(f"Unknown fill kind {kind}.")


# --- Ledger and costs -------------------------------------------------------------------------


def build_ledger(references: pd.DataFrame, position: pd.Series, predictor: pd.Series, entry: pd.DataFrame, exit: pd.DataFrame, signal: str, execution: dict) -> pd.DataFrame:
	"""One row per trade date: signal, fills, underlying last-half-hour return and gross P&L before costs.

	The underlying return r_lh and the paper's frictionless strategy return use the reference prices.
	r_gross and pnl_gross_usd use the fills. A day without both fills, or with no position, is not traded.
	"""
	frame = references.set_index("trade_date")
	ledger = pd.DataFrame(index=frame.index)
	ledger["contract_id"] = frame["contract_id"]
	ledger["signal"] = signal
	ledger["predictor"] = predictor
	ledger["position"] = position
	for name, fill in (("entry", entry), ("exit", exit)):
		ledger[f"{name}_time"] = fill["time"].reindex(ledger.index)
		ledger[f"{name}_price"] = fill["price"].reindex(ledger.index)
		ledger[f"{name}_delay"] = fill["delay"].reindex(ledger.index)
	ledger["p_1300"], ledger["p_1330"] = frame["p_1300"], frame["p_1330"]
	ledger["r_lh"] = frame["p_1330"] / frame["p_1300"] - 1
	ledger["r_frictionless"] = ledger["position"] * ledger["r_lh"]
	filled = ledger["entry_price"].notna() & ledger["exit_price"].notna()
	ledger["traded"] = filled & ledger["position"].ne(0)
	ledger["skip_reason"] = ""
	ledger.loc[~filled & ledger["entry_price"].isna(), "skip_reason"] = f"no entry fill within {execution['max_delay_minutes']} minutes"
	ledger.loc[~filled & ledger["exit_price"].isna(), "skip_reason"] = f"no exit fill within {execution['max_delay_minutes']} minutes"
	ledger.loc[filled & ledger["position"].eq(0), "skip_reason"] = "no position (signal zero or missing)"
	traded_position = ledger["position"].where(ledger["traded"], 0)
	ledger["r_gross"] = traded_position * (ledger["exit_price"] / ledger["entry_price"] - 1).fillna(0)
	ledger["pnl_gross_usd"] = traded_position * (ledger["exit_price"] - ledger["entry_price"]).fillna(0) * execution["multiplier"] * execution["contracts"]
	ledger["sides"] = 2 * ledger["traded"].astype(int)
	return ledger.reset_index()


def apply_costs(ledger: pd.DataFrame, commission_per_side_usd: float, slippage_ticks_per_side: float, execution: dict, exit_slippage_ticks_per_side: float | None = None) -> pd.DataFrame:
	"""Commission and slippage per side; net P&L and the net return on the entry notional of the position.

	The exit side can carry its own slippage (for example 0 for a trade-at-settlement exit); by default
	both sides use slippage_ticks_per_side.
	"""
	ledger = ledger.copy()
	exit_ticks = slippage_ticks_per_side if exit_slippage_ticks_per_side is None else exit_slippage_ticks_per_side
	ledger["commission_usd"] = ledger["sides"] * commission_per_side_usd
	ledger["slippage_usd"] = ledger["traded"].astype(int) * (slippage_ticks_per_side + exit_ticks) * execution["tick_size"] * execution["multiplier"] * execution["contracts"]
	ledger["cost_usd"] = ledger["commission_usd"] + ledger["slippage_usd"]
	ledger["pnl_net_usd"] = ledger["pnl_gross_usd"] - ledger["cost_usd"]
	notional = ledger["entry_price"] * execution["multiplier"] * execution["contracts"]
	ledger["r_net"] = (ledger["pnl_net_usd"] / notional).where(ledger["traded"], 0.0)
	return ledger[LEDGER_COLUMNS]


# --- Evaluation -------------------------------------------------------------------------------


def hac_standard_error(values: np.ndarray, lags: int) -> float:
	"""Newey-West standard error of the mean with Bartlett weights."""
	centered = values - values.mean()
	n = len(centered)
	variance = centered @ centered / n
	for lag in range(1, min(lags, n - 1) + 1):
		variance += 2 * (1 - lag / (lags + 1)) * (centered[lag:] @ centered[:-lag]) / n
	return float(np.sqrt(variance / n))


def max_drawdown(values: np.ndarray) -> tuple[float, int]:
	"""Largest peak-to-trough fall of a cumulative-sum path and the day it bottoms."""
	path = np.concatenate([[0.0], np.cumsum(values)])
	drawdown = path - np.maximum.accumulate(path)
	return float(-drawdown.min()), int(drawdown.argmin()) - 1


def evaluate(ledger: pd.DataFrame, return_column: str, pnl_column: str | None, evaluation: dict) -> dict:
	"""Estimates with standard errors for a daily strategy return series, plus P&L, hit rate, drawdown and turnover.

	Daily returns are the last-half-hour return on the position's notional; days without a position
	count as zero. Annualization treats each trade date as one observation (mean x 252, std x sqrt 252).
	The Sharpe ratio's standard error follows Lo (2002) for independent returns.
	"""
	days = evaluation["trading_days_per_year"]
	returns = ledger[return_column].to_numpy(dtype=float)
	n, traded = len(returns), int(ledger["traded"].sum())
	mean, std = returns.mean(), returns.std(ddof=1)
	se = std / np.sqrt(n)
	se_hac = hac_standard_error(returns, evaluation["hac_lags"])
	z = NormalDist().inv_cdf(0.5 + evaluation["confidence"] / 2)
	sharpe_daily = mean / std if std > 0 else np.nan
	sharpe_se_daily = np.sqrt((1 + sharpe_daily**2 / 2) / n)
	active = ledger[ledger["traded"]]
	wins = int(active[return_column].gt(0).sum())
	hit = wins / traded if traded else np.nan
	years = n / days
	result = {
		"days": n, "traded_days": traded, "long_days": int(active["position"].eq(1).sum()), "short_days": int(active["position"].eq(-1).sum()),
		"mean_bps": mean * 1e4, "se_bps": se * 1e4, "se_hac_bps": se_hac * 1e4,
		"ci_low_bps": (mean - z * se_hac) * 1e4, "ci_high_bps": (mean + z * se_hac) * 1e4, "t_stat_hac": mean / se_hac if se_hac > 0 else np.nan,
		"std_bps": std * 1e4, "ann_return_pct": mean * days * 100, "ann_vol_pct": std * np.sqrt(days) * 100,
		"sharpe": sharpe_daily * np.sqrt(days), "sharpe_se": sharpe_se_daily * np.sqrt(days),
		"hit_rate": hit, "hit_rate_se": np.sqrt(hit * (1 - hit) / traded) if traded else np.nan,
		"autocorr_lag1": float(pd.Series(returns).autocorr(1)) if n > 2 else np.nan,
		"round_trips_per_year": traded / years, "contracts_traded_per_year": active["sides"].sum() / years,
	}
	drawdown_pct, trough = max_drawdown(returns)
	result["max_drawdown_pct"] = drawdown_pct * 100
	result["max_drawdown_trough"] = ledger["trade_date"].iloc[trough].date() if trough >= 0 else pd.NaT
	if pnl_column is not None:
		pnl = ledger[pnl_column].to_numpy(dtype=float)
		drawdown_usd, _ = max_drawdown(pnl)
		result.update(
			{
				"total_pnl_usd": pnl.sum(), "mean_pnl_per_day_usd": pnl.mean(), "mean_pnl_per_trade_usd": pnl.sum() / traded if traded else np.nan,
				"total_cost_usd": ledger["cost_usd"].sum() if "cost_usd" in ledger else 0.0,
				"max_drawdown_usd": drawdown_usd, "worst_day_usd": pnl.min(), "worst_day": ledger["trade_date"].iloc[pnl.argmin()].date(),
				"adverse_days": int((pnl < -evaluation["adverse_loss_usd"]).sum()),
			}
		)
	return result


def breakeven_cost(ledger: pd.DataFrame, execution: dict) -> dict:
	"""Round-trip cost per contract at which the mean net P&L is zero, in USD and ticks."""
	traded = ledger[ledger["traded"]]
	usd = traded["pnl_gross_usd"].mean() if len(traded) else np.nan
	return {"breakeven_cost_per_round_trip_usd": usd, "breakeven_cost_per_round_trip_ticks": usd / (execution["tick_size"] * execution["multiplier"] * execution["contracts"])}


def accounting_rows(ledger: pd.DataFrame, evaluation: dict, execution: dict, labels: dict) -> list[dict]:
	"""Frictionless, feasible gross and net metrics for one ledger."""
	rows = []
	for accounting, return_column, pnl_column in (("frictionless", "r_frictionless", None), ("gross", "r_gross", "pnl_gross_usd"), ("net", "r_net", "pnl_net_usd")):
		row = {**labels, "accounting": accounting, **evaluate(ledger, return_column, pnl_column, evaluation)}
		if accounting == "gross":
			row.update(breakeven_cost(ledger, execution))
		rows.append(row)
	return rows


def by_year(ledger: pd.DataFrame, evaluation: dict) -> pd.DataFrame:
	rows = []
	for year, group in ledger.groupby(ledger["trade_date"].dt.year):
		for accounting, return_column, pnl_column in (("gross", "r_gross", "pnl_gross_usd"), ("net", "r_net", "pnl_net_usd")):
			rows.append({"year": year, "accounting": accounting, **evaluate(group.reset_index(drop=True), return_column, pnl_column, evaluation)})
	return pd.DataFrame(rows)


def rolling_estimates(ledger: pd.DataFrame, evaluation: dict) -> pd.DataFrame:
	"""Rolling mean net and gross return with a plain confidence band, in basis points."""
	window = evaluation["rolling_window"]
	z = NormalDist().inv_cdf(0.5 + evaluation["confidence"] / 2)
	frame = pd.DataFrame({"trade_date": ledger["trade_date"]})
	for accounting, column in (("gross", "r_gross"), ("net", "r_net")):
		rolling = ledger[column].rolling(window)
		mean, se = rolling.mean() * 1e4, rolling.std(ddof=1) / np.sqrt(window) * 1e4
		frame[f"{accounting}_mean_bps"], frame[f"{accounting}_ci_low_bps"], frame[f"{accounting}_ci_high_bps"] = mean, mean - z * se, mean + z * se
		frame[f"{accounting}_cum_pnl_usd"] = ledger[f"pnl_{accounting}_usd"].cumsum()
	frame["frictionless_cum_return_pct"] = ledger["r_frictionless"].cumsum() * 100
	return frame


def worst_days(ledger: pd.DataFrame, count: int) -> pd.DataFrame:
	columns = ["trade_date", "contract_id", "position", "predictor", "entry_price", "exit_price", "r_lh", "r_gross", "pnl_gross_usd", "pnl_net_usd"]
	return ledger.nsmallest(count, "pnl_net_usd")[columns].reset_index(drop=True)


# --- Roll rules and blackouts -----------------------------------------------------------------


def alternative_references(references: pd.DataFrame, minutes: pd.DataFrame, roll_comparison: pd.DataFrame, roll_date_column: str, minute_cfg: dict) -> pd.DataFrame:
	"""Reference prices for the contract held under another roll calendar, rebuilt from the minute bars.

	Keeps the sample flags of the paper-rule file (early close, volume, degraded dates) and sets
	`roll_flag` from the alternative calendar. Dates whose rebuilt prices are missing leave the sample.
	"""
	instrument = minutes.drop_duplicates("contract_id").set_index("contract_id")["instrument_id"]
	rolls = pd.DataFrame(
		{
			"roll_date": pd.to_datetime(roll_comparison[roll_date_column]),
			"from_instrument_id": roll_comparison["from_contract"].map(instrument),
			"to_instrument_id": roll_comparison["to_contract"].map(instrument),
		}
	).dropna().sort_values("roll_date")
	dates = pd.DatetimeIndex(references["trade_date"])
	calendar = pd.DatetimeIndex(sorted(set(dates) | set(references["prev_trade_date"].dropna())))
	held = held_contracts(dates, rolls)
	reference_cfg, timezone_name = minute_cfg["reference"], minute_cfg["session"]["timezone"]
	rebuilt = reference_prices(minutes, held, calendar, timezone_name, reference_cfg["previous_close"], reference_cfg["points"], reference_cfg["max_stale_minutes"])
	rebuilt["contract_id"] = rebuilt["instrument_id"].map(instrument.reset_index().set_index("instrument_id")["contract_id"])
	flags = references.set_index("trade_date")[["session_volume", "early_close", "low_volume", "folded_holiday", "degraded_date"]]
	rebuilt = rebuilt.merge(flags, left_on="trade_date", right_index=True, how="left")
	rebuilt["roll_flag"] = rebuilt["trade_date"].isin(rolls["roll_date"])
	rebuilt["paper_sample"] = rebuilt["complete"] & ~rebuilt["early_close"] & ~rebuilt["low_volume"]
	return rebuilt


def blackout_mask(references: pd.DataFrame, spec: dict) -> pd.Series:
	"""True on the trade dates the variant skips."""
	if not spec:
		return pd.Series(False, index=references.index)
	if "dates" in spec:
		return references["trade_date"].isin(pd.to_datetime(spec["dates"]))
	column = references[spec["column"]]
	if "at_most" in spec:
		return column.le(spec["at_most"]).fillna(False)
	if "above" in spec:
		return column.gt(spec["above"]).fillna(False)
	return column.fillna(False).astype(bool)


def skip_days(ledger: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
	"""The ledger with no position on the masked dates (costs are re-applied by the caller)."""
	ledger = ledger.copy()
	skipped = ledger["trade_date"].isin(ledger.loc[mask.to_numpy(), "trade_date"])
	ledger.loc[skipped, ["position", "r_frictionless", "r_gross", "pnl_gross_usd", "sides"]] = 0
	ledger.loc[skipped, "traded"] = False
	ledger.loc[skipped, "skip_reason"] = reason
	return ledger


# --- Figures ----------------------------------------------------------------------------------


def style_axes(axis: plt.Axes, ylabel: str, title: str) -> None:
	axis.set_facecolor(SURFACE)
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		axis.spines[side].set_color(GRID)
	axis.grid(axis="y", color=GRID, linewidth=0.8)
	axis.tick_params(colors=MUTED, labelsize=9)
	axis.set_ylabel(ylabel, color=MUTED, fontsize=9)
	axis.set_title(title, loc="left", color=TEXT, fontsize=11, pad=10)


def label_line_ends(axis: plt.Axes, x, ends: list[tuple[float, str, str]], min_gap: float) -> None:
	"""Direct labels at the right edge, pushed apart so that close line ends do not overlap."""
	ordered = sorted(ends, key=lambda end: end[0])
	placed = []
	for value, text, color in ordered:
		position = value if not placed else max(value, placed[-1] + min_gap)
		placed.append(position)
		axis.annotate(text, (x, value), xytext=(6, 0), textcoords="offset points", color=color, fontsize=9, va="center")
		if position != value:
			axis.texts[-1].xyann = (6, (position - value) / min_gap * 11)


def plot_cumulative(ledgers: dict[str, pd.DataFrame], path: Path, cost_label: str) -> None:
	"""Cumulative P&L per contract: paper's frictionless returns, feasible gross, net, and Always Long."""
	figure, axis = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
	baseline, always_long = ledgers["baseline"], ledgers["always_long"]
	dates = baseline["trade_date"]
	notional = baseline["p_1300"] * 1000
	series = [
		("Paper, frictionless (sign of ROD)", (baseline["r_frictionless"] * notional).cumsum(), SERIES["blue"], "-"),
		("Feasible, gross of costs", baseline["pnl_gross_usd"].cumsum(), SERIES["orange"], "-"),
		(f"Feasible, net ({cost_label})", baseline["pnl_net_usd"].cumsum(), SERIES["aqua"], "-"),
		("Always Long, gross", always_long["pnl_gross_usd"].cumsum(), SERIES["yellow"], "--"),
	]
	for name, values, color, style in series:
		axis.plot(dates, values, color=color, linewidth=2, linestyle=style, label=name)
	span = max(values.max() for _, values, _, _ in series) - min(values.min() for _, values, _, _ in series)
	label_line_ends(axis, dates.iloc[-1], [(values.iloc[-1], f"{values.iloc[-1]:,.0f}", color) for _, values, color, _ in series], 0.04 * span)
	axis.axhline(0, color=MUTED, linewidth=0.8)
	style_axes(axis, "Cumulative P&L, USD per contract", "Last-half-hour strategy on CL, 2021-2026: cumulative P&L per contract")
	axis.set_title(axis.get_title(loc="left"), loc="left", color=TEXT, fontsize=11, pad=36)
	axis.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=TEXT)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def plot_by_year(table: pd.DataFrame, path: Path) -> None:
	"""Mean daily return by calendar year with a confidence interval, gross and net."""
	figure, axis = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
	years = sorted(table["year"].unique())
	offsets = {"gross": -0.15, "net": 0.15}
	colors = {"gross": SERIES["orange"], "net": SERIES["aqua"]}
	for accounting, group in table.groupby("accounting"):
		group = group.set_index("year").reindex(years)
		x = np.arange(len(years)) + offsets[accounting]
		axis.errorbar(
			x, group["mean_bps"], yerr=[group["mean_bps"] - group["ci_low_bps"], group["ci_high_bps"] - group["mean_bps"]],
			fmt="o", color=colors[accounting], markersize=7, capsize=4, linewidth=2, label=f"{accounting.capitalize()} mean with 95% interval",
		)
	axis.axhline(0, color=MUTED, linewidth=0.8)
	axis.set_xticks(np.arange(len(years)), [str(year) for year in years])
	style_axes(axis, "Mean daily return, basis points of notional", "Sign-of-ROD strategy: mean last-half-hour return by year")
	axis.legend(frameon=False, fontsize=9, labelcolor=TEXT)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def plot_rolling(table: pd.DataFrame, window: int, path: Path) -> None:
	figure, axis = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
	dates = table["trade_date"]
	axis.fill_between(dates, table["net_ci_low_bps"], table["net_ci_high_bps"], color=SERIES["aqua"], alpha=0.18, linewidth=0, label="95% interval, net")
	axis.plot(dates, table["gross_mean_bps"], color=SERIES["orange"], linewidth=2, label="Gross")
	axis.plot(dates, table["net_mean_bps"], color=SERIES["aqua"], linewidth=2, label="Net")
	axis.axhline(0, color=MUTED, linewidth=0.8)
	style_axes(axis, "Mean daily return, basis points of notional", f"Sign-of-ROD strategy: rolling {window}-day mean return")
	axis.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


def plot_cost_sensitivity(table: pd.DataFrame, baseline_commission: float, path: Path) -> None:
	figure, axis = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
	colors = [SERIES["blue"], SERIES["orange"], SERIES["aqua"]]
	for color, (commission, group) in zip(colors, table.groupby("commission_per_side_usd")):
		group = group.sort_values("slippage_ticks_per_side")
		with_band = commission == baseline_commission
		axis.errorbar(
			group["slippage_ticks_per_side"], group["mean_bps"], yerr=group["se_hac_bps"] * 1.96 if with_band else None, color=color, linewidth=2,
			marker="o", markersize=6, capsize=3, label=f"Commission USD {commission:.2f} per side" + (" (95% interval shown)" if with_band else ""),
		)
	axis.axhline(0, color=MUTED, linewidth=0.8)
	style_axes(axis, "Mean net daily return, basis points of notional", "Sign-of-ROD strategy: net return against spread and slippage per side")
	axis.set_xlabel("Spread and slippage, ticks per side (1 tick = USD 10 per contract)", color=MUTED, fontsize=9)
	axis.legend(frameon=False, fontsize=9, labelcolor=TEXT)
	figure.tight_layout()
	figure.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(figure)


# --- Main -------------------------------------------------------------------------------------


def sample_references(references: pd.DataFrame, sample: dict) -> pd.DataFrame:
	in_period = references["trade_date"].ge(pd.Timestamp(sample["start"]))
	if sample.get("end"):
		in_period &= references["trade_date"].le(pd.Timestamp(sample["end"]))
	return references[in_period & references["paper_sample"]].sort_values("trade_date").reset_index(drop=True)


def run_signal(references: pd.DataFrame, minutes: pd.DataFrame, signal_name: str, spec: dict, variant: dict, config: dict, timezone_name: str) -> pd.DataFrame:
	"""Ledger with baseline costs for one signal, one execution variant and one set of reference prices."""
	execution = {**config["execution"], **{key: value for key, value in variant.items() if key in ("entry_time", "exit_time", "max_delay_minutes", "settlement_proxy_minutes")}}
	held = references.set_index("trade_date")["instrument_id"]
	position, predictor = positions(references.set_index("trade_date"), spec)
	entry = fills(variant["entry"], execution["entry_time"], references, minutes, held, timezone_name, execution)
	exit = fills(variant["exit"], execution["exit_time"], references, minutes, held, timezone_name, execution)
	ledger = build_ledger(references, position, predictor, entry, exit, signal_name, execution)
	costs = config["costs"]
	return apply_costs(ledger, costs["commission_per_side_usd"], costs["slippage_ticks_per_side"], execution, variant.get("exit_slippage_ticks_per_side"))


def check_against_features(references: pd.DataFrame, path: Path) -> dict:
	"""The signal return equals r_rod from the #28 feature file on every shared date, when that file exists."""
	if not path.exists():
		return {"return_features": "not present; signal computed from the reference prices"}
	features = pd.read_parquet(path).set_index("trade_date")
	rod = (references["p_1300"] / references["p_prev_1330"] - 1).set_axis(references["trade_date"])
	shared = rod.index.intersection(features.index)
	error = (rod.loc[shared] - features.loc[shared, "r_rod"]).abs().max()
	return {"return_features": str(path), "shared_dates": int(len(shared)), "max_abs_rod_difference": float(error)}


def main() -> int:
	args = parse_args()
	config, minute_cfg = load_config(args.config)
	timezone_name = minute_cfg["session"]["timezone"]
	execution, costs, evaluation = config["execution"], config["costs"], dict(config["evaluation"])
	tables_dir, figures_dir = Path(config["tables_dir"]), Path(config["figures_dir"])
	tables_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)

	all_references = pd.read_parquet(config["reference_prices"])
	minutes = pd.read_parquet(config["front_minutes"], columns=["ts_utc", "trade_date", "contract_id", "instrument_id", "open", "high", "low", "close", "volume", "held"])
	quality = pd.read_csv(config["session_quality"], parse_dates=["trade_date"]).set_index("trade_date")
	roll_comparison = pd.read_csv(config["roll_comparison"])
	references = sample_references(all_references, config["sample"])
	references["sessions_to_expiry"] = references["trade_date"].map(quality["sessions_to_expiry"])
	references["gap_days"] = (references["trade_date"] - references["prev_trade_date"]).dt.days
	# Newey-West lags: the automatic rule floor(4 (n / 100)^(2/9)) of #29 and #35, or a fixed number.
	evaluation["hac_lags"] = newey_west_lags(len(references)) if evaluation["hac_lags"] == "auto" else int(evaluation["hac_lags"])
	checks = {"sample_days": int(len(references)), "first_date": str(references["trade_date"].iloc[0].date()), "last_date": str(references["trade_date"].iloc[-1].date()), "hac_lags_used": evaluation["hac_lags"]}
	checks.update(check_against_features(references, Path(config["return_features"])))

	baseline_name, baseline_variant = config["baseline_signal"], config["execution_variants"][config["baseline_execution"]]
	ledgers = {name: run_signal(references, minutes, name, spec, baseline_variant, config, timezone_name) for name, spec in config["signals"].items()}
	baseline = ledgers[baseline_name]
	checks["baseline_traded_days"] = int(baseline["traded"].sum())
	checks["baseline_skipped"] = baseline.loc[~baseline["traded"], "skip_reason"].value_counts().to_dict()
	checks["entry_delay_minutes"] = baseline["entry_delay"].value_counts().sort_index().to_dict()
	checks["exit_delay_minutes"] = baseline["exit_delay"].value_counts().sort_index().to_dict()
	checks["entry_after_decision"] = bool((baseline["entry_time"].dt.strftime("%H:%M") >= execution["entry_time"]).all())
	checks["exit_after_entry"] = bool((baseline["exit_time"] > baseline["entry_time"]).all())

	# Summary: every signal, three accountings.
	summary = pd.DataFrame([row for name, ledger in ledgers.items() for row in accounting_rows(ledger, evaluation, execution, {"signal": name, "execution": config["baseline_execution"]})])

	# Cost sensitivity for the baseline signal.
	cost_rows = []
	for commission in costs["sensitivity_commission_usd"]:
		for ticks in costs["sensitivity_ticks_per_side"]:
			priced = apply_costs(baseline, commission, ticks, execution)
			cost_rows.append({"commission_per_side_usd": commission, "slippage_ticks_per_side": ticks, "round_trip_cost_usd": 2 * (commission + ticks * execution["tick_size"] * execution["multiplier"] * execution["contracts"]), **evaluate(priced, "r_net", "pnl_net_usd", evaluation)})
	cost_table = pd.DataFrame(cost_rows)

	# Execution-timing variants for the baseline signal.
	robustness = []
	for variant_name, variant in config["execution_variants"].items():
		ledger = run_signal(references, minutes, baseline_name, config["signals"][baseline_name], variant, config, timezone_name)
		robustness += accounting_rows(ledger, evaluation, execution, {"variant": f"execution:{variant_name}", "days_skipped": int((~ledger["traded"]).sum())})
	# Blackout variants.
	for blackout_name, spec in config["blackouts"].items():
		mask = blackout_mask(references, spec)
		ledger = apply_costs(skip_days(baseline, mask, f"blackout {blackout_name}"), costs["commission_per_side_usd"], costs["slippage_ticks_per_side"], execution)
		robustness += accounting_rows(ledger, evaluation, execution, {"variant": f"blackout:{blackout_name}", "days_skipped": int(mask.sum())})
	robustness_table = pd.DataFrame(robustness)

	# Roll rules.
	roll_rows = []
	for rule_name, spec in config["roll_rules"].items():
		if spec:
			rule_references = alternative_references(all_references, minutes, roll_comparison, spec["roll_date_column"], minute_cfg)
			rule_references = sample_references(rule_references, config["sample"])
			checks[f"roll_rule_{rule_name}_days"] = int(len(rule_references))
			checks[f"roll_rule_{rule_name}_different_contract_days"] = int((rule_references.set_index("trade_date")["contract_id"] != references.set_index("trade_date")["contract_id"].reindex(rule_references["trade_date"])).sum())
		else:
			rule_references = references
		for name in (baseline_name, "always_long"):
			ledger = run_signal(rule_references, minutes, name, config["signals"][name], baseline_variant, config, timezone_name)
			roll_rows += accounting_rows(ledger, evaluation, execution, {"roll_rule": rule_name, "signal": name, "rolls": int(rule_references["roll_flag"].sum())})
	roll_table = pd.DataFrame(roll_rows)

	year_table = by_year(baseline, evaluation)
	rolling_table = rolling_estimates(baseline, evaluation)
	worst_table = worst_days(baseline, evaluation["worst_days"])

	outputs = {
		"lh_strategy_trades.csv": baseline, "lh_strategy_summary.csv": summary, "lh_strategy_cost_sensitivity.csv": cost_table,
		"lh_strategy_by_year.csv": year_table, "lh_strategy_roll_rules.csv": roll_table, "lh_strategy_robustness.csv": robustness_table,
		"lh_strategy_worst_days.csv": worst_table, "lh_strategy_rolling.csv": rolling_table,
	}
	for name, table in outputs.items():
		table.to_csv(tables_dir / name, index=False)
	cost_label = f"{costs['slippage_ticks_per_side']:g} tick + USD {costs['commission_per_side_usd']:.2f} per side"
	plot_cumulative({"baseline": baseline, "always_long": ledgers["always_long"]}, figures_dir / "lh_strategy_cumulative.png", cost_label)
	plot_by_year(year_table, figures_dir / "lh_strategy_by_year.png")
	plot_rolling(rolling_table, evaluation["rolling_window"], figures_dir / "lh_strategy_rolling.png")
	plot_cost_sensitivity(cost_table, costs["commission_per_side_usd"], figures_dir / "lh_strategy_cost_sensitivity.png")

	manifest = {
		"config": str(args.config), "settings": config, "minute_settings": {"session": minute_cfg["session"], "reference": minute_cfg["reference"]},
		"checks": checks,
		"inputs": {name: sha256(Path(config[name])) for name in ("reference_prices", "front_minutes", "session_quality", "roll_comparison")},
		"outputs": {name: sha256(tables_dir / name) for name in outputs} | {name: sha256(figures_dir / name) for name in ("lh_strategy_cumulative.png", "lh_strategy_by_year.png", "lh_strategy_rolling.png", "lh_strategy_cost_sensitivity.png")},
		"run": {"time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
	}
	with (tables_dir / "lh_strategy.manifest.json").open("w", encoding="utf-8") as manifest_file:
		json.dump(manifest, manifest_file, indent=2, default=str)

	baseline_rows = summary[summary["signal"].eq(baseline_name)].set_index("accounting")
	print(f"Sample: {checks['sample_days']} trade dates, {checks['first_date']} to {checks['last_date']}; traded {checks['baseline_traded_days']}.")
	for accounting in ("frictionless", "gross", "net"):
		row = baseline_rows.loc[accounting]
		print(f"{accounting:>12}: mean {row['mean_bps']:6.2f} bps (HAC se {row['se_hac_bps']:.2f}), Sharpe {row['sharpe']:5.2f} (se {row['sharpe_se']:.2f}), hit rate {row['hit_rate']:.3f}" + (f", total P&L USD {row['total_pnl_usd']:,.0f}" if accounting != "frictionless" else ""))
	print(f"Breakeven round-trip cost: USD {baseline_rows.loc['gross', 'breakeven_cost_per_round_trip_usd']:.2f} per contract.")
	print(f"Tables in {tables_dir}, figures in {figures_dir}.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
