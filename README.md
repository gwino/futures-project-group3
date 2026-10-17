# WTI Crude Oil Intraday Momentum: Replication and New Hypotheses

Group 3 research project for FINM 37000: Futures and Related Derivatives.

> **Current scope (revised):** This project focuses **only** on replicating Baltussen, Da, Lammers & Martens (2021), *Hedging Demand and Market Intraday Momentum*, for post-2020 NYMEX WTI crude oil futures (CL), and testing new hypotheses about overnight versus daytime returns and volatility. The previous **moving-average strategy** and **daily-versus-15-minute strategy comparison** have been discontinued.

## Research questions

**Replication.** Does the paper's intraday momentum relationship hold for CL during **2021–2026**? Specifically, does the return from the previous session's reference close until 30 minutes before today's close predict the final half-hour return?

**New hypotheses.**
1. Do **overnight** and **daytime** returns differ in their ability to predict the final 30-minute CL return?
2. Does that predictive relationship change during **high-volatility periods** (measured using only information known by the prediction time)?
3. Does **agreement versus disagreement** between overnight and daytime return directions affect closing-period momentum?

A statistically significant coefficient is not sufficient evidence of a useful trading strategy. We will examine out-of-sample performance and the sensitivity of any directional backtest to execution costs.

### Why CL?

CL is one of the energy futures in the original paper. Its individual historical rest-of-day results were **not statistically significant** (Appendix Table B3: t-statistic 0.25; adjusted R² 0.03%; out-of-sample R² −1.04%). We will examine whether this remains true in the period after 2020.

**Reference:** Baltussen, G., Da, Z., Lammers, S., and Martens, M. (2021), *Hedging Demand and Market Intraday Momentum*, Journal of Financial Economics, 142, 377–403.

## Data available and required

| Dataset | Current coverage | Intended use |
| --- | --- | --- |
| CL daily history | **2010–2026** | Existing resource for contract/volume metadata, roll cross-checks, and broader context; **not** a daily-strategy comparison |
| CL intraday history | **2025–2026** | Audit actual frequency, trading-hour coverage, bar timestamps, contract IDs, and quality before reuse |
| Missing minute history | **Primarily 2021–2024** | Procure as needed, after assessing price and existing gaps |
| Target analysis | **2021–2026** | One-minute CL OHLCV, ideally across the entire available electronic trading session |

A full electronic-session minute history allows the project to calculate **overnight realized volatility** as well as daytime realized volatility. To compute overnight *returns* alone, only the prior close and current opening reference prices are necessary. Level 2 order-book data is **not required**.

**Data handling:** retain individual CL contract identifiers, exchange session calendars, historical volume, and contract expiration/roll information. Preserve original market-data files locally, record source provenance, and do not commit API keys or restricted raw data.

## Paper-consistent session and return definitions

Use **America/Chicago** (Central Time, with daylight saving changes handled automatically). The paper selects **09:00–14:30 Eastern = 08:00–13:30 Central** as the CL reference trading window, although CL's electronic trading runs for much longer.

| Variable | Time range in Central Time | Meaning |
| --- | --- | --- |
| ON | Previous reference close **13:30** → today's **08:00** | Overnight return |
| FH | **08:00–08:30** | First half-hour return |
| ONFH | Previous reference close **13:30** → today's **08:30** | Paper's opening-period predictor (includes overnight) |
| M | **08:30–12:30** | Middle-of-day return |
| SLH | **12:30–13:00** | Second-to-last half-hour return |
| ROD | Previous reference close **13:30** → today's **13:00** | Paper's rest-of-day predictor (includes overnight) |
| DAY | **08:00–13:00** | New daytime-only predictor |
| LH | **13:00–13:30** | Prediction target: final half-hour return |

The paper's main regressions use **simple returns**. We will retain those for direct replication, and may use clearly labeled log-return versions for an exactly additive ON + DAY decomposition. Reference prices at 13:30 CT must not be confused with the **16:00 CT end of the electronic session**.

### Replication: primary models

1. `LH ~ ONFH`
2. `LH ~ ONFH + M + SLH`
3. `LH ~ ROD`

Report regression coefficients in units matching the paper, intercepts, robust t-statistics, adjusted R², sample sizes, and **expanding-window out-of-sample R²** versus the expanding historical mean. The paper uses a minimum initial estimation window of approximately **500 observations**.

### New hypotheses: planned tests

- **Return decomposition:** `LH ~ ON`, `LH ~ DAY`, and `LH ~ ON + DAY`; compare incremental predictive power, standardized effects, coefficient equality, and out-of-sample accuracy to the original ROD model.
- **Volatility regimes:** construct overnight/daytime realized variance from one-minute returns, carefully handling exchange breaks and reopening jumps. Test whether ON and DAY slopes differ in high-volatility regimes using interaction terms, with thresholds determined from training data only.
- **Direction agreement:** compare the closing return and directional hit rate when overnight and daytime returns agree versus disagree. Evaluate the four sign combinations as well as the pooled agreement/disagreement test.

Features used at the **13:00 CT** trading decision may contain **no information from 13:00–13:30 CT**. Use chronological out-of-sample evaluation; consider data-snooping risk when testing multiple hypotheses.

## Project phases and GitHub issues

**Project tracker:** [#24 — CL intraday momentum replication and new hypotheses](https://github.com/gwino/futures-project-group3/issues/24).

| Phase | GitHub issue | Deliverable |
| --- | --- | --- |
| **1. Data procurement & cleaning** | [#25 Audit existing data](https://github.com/gwino/futures-project-group3/issues/25) | Coverage inventory for 2010–2026 daily and 2025–2026 intraday |
| | [#26 Procure missing one-minute data](https://github.com/gwino/futures-project-group3/issues/26) | Cost-checked 2021–2024 acquisition and source manifests |
| | [#27 Clean full-session data and handle rolls](https://github.com/gwino/futures-project-group3/issues/27) | Validated minute dataset, roll calendar and quality tests |
| **2. Replicate the paper** | [#28 Construct return variables](https://github.com/gwino/futures-project-group3/issues/28) | ONFH, M, SLH, ROD, LH and regression-ready data |
| | [#29 Run post-2020 regressions](https://github.com/gwino/futures-project-group3/issues/29) | Historical-vs-new results and OOS forecasts |
| | [#30 Backtest closing-period strategy](https://github.com/gwino/futures-project-group3/issues/30) | Trade ledger, transaction-cost sensitivity, and performance metrics |
| **3. Test new hypotheses** | [#31 Overnight versus daytime](https://github.com/gwino/futures-project-group3/issues/31) | Single- and joint-predictor tests |
| | [#32 Volatility regime interactions](https://github.com/gwino/futures-project-group3/issues/32) | Pre-close volatility features and interaction models |
| | [#33 Signal agreement & final report](https://github.com/gwino/futures-project-group3/issues/33) | Agreement/disagreement tests and reproducible final findings |

**Dependency order:** #25 → #26 → #27 → #28 → #29, then #30 and #31–#33 as the validated dataset becomes available. Contract-roll logic and pre-existing download utilities may be reused from earlier work.

The repository also contains **older issues from the discontinued moving-average/daily-versus-15-minute proposal**. They are historical planning items and are **not part of this project's current scope**; they can be triaged/closed separately without discarding reusable code.

## Deliverables

- Documented 2021–2026 CL research dataset and reproducible cleaning/roll procedures, including complete and incomplete session handling.
- Faithful reproduction of the paper's three core CL regression specifications, with historical reference metrics and out-of-sample evaluation.
- A closing-half-hour directional backtest with explicit timing, contract sizing, fill assumptions, commissions and slippage.
- Overnight/daytime return comparison, high-volatility interaction tests, and sign-agreement analysis.
- Clear figures and tables covering t-statistics, adjusted and OOS R², hit rates, and gross/net performance where applicable.
- Final report and presentation explaining research questions, data, methods, findings, robustness checks, and limitations.

**Success criterion:** Another team member can reproduce the results from authorized data, documented configurations and runnable code. Null findings (including continued non-significance for CL) are valid research outcomes.

## Team

| Team member | Existing coordination role |
| --- | --- |
| Gabriella Wiem | Tech Lead — data acquisition workflow, research code coordination, integration and reproducibility |
| Jiayi Chen | Communication Lead — research documentation, analysis tables/figures, report and presentation |
| Ethan Huang | Design Lead — session/roll definitions, data preparation methodology, validation and review |

Specific issue owners and reviewers are to be confirmed by the team for the **revised** scope.

## Repository organization and reproducibility

The existing `config/`, `data/`, `src/`, `analysis/`, `tests/` and `reports/` folders may be reused. New return-feature, regression, volatility and evaluation modules should be kept **separate from prior moving-average code**, and named/configured according to this study rather than the retired strategy.

All sample periods, data filters, contract roll rules, session/reference times, return conventions, regression specifications, OOS rules and trading-cost assumptions must be recorded in configuration files and the final methodology note. Existing `src/download_data.py` and related routines should be inspected before reuse. Commands for new research modules will be documented once implemented and tested.

### Suggested four-week sequence

| Week | Focus |
| --- | --- |
| 1 | Audit existing data; confirm vendor coverage, acquisition costs, reference session and contract-selection method |
| 2 | Complete data acquisition, cleaning and return feature construction; start core regressions |
| 3 | Finish replication and feasible directional backtest; test overnight/daytime and volatility hypotheses |
| 4 | Robustness/OOS checks, agreement tests, results review, final report, and presentation |

The research is historical and **does not execute live trades**.
