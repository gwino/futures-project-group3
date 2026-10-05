# Energy Futures Strategy Analysis

Group 3 project for FINM 37000: Futures and Related Derivatives.

## Team Members and Planning Roles

| Team member | Role | Initial responsibility |
| --- | --- | --- |
| Gabriella Wiem | Tech Leader | Maintain the repository, set up the Python environment and data acquisition workflow, and coordinate backtesting implementation and merges. |
| Jiayi Chen | Communication Leader | Maintain the README, document the methodology, organize performance comparisons, and coordinate the final report and presentation. |
| Ethan Huang | Design Leader | Define the analysis workflow and data-cleaning and rollover rules, and review strategy assumptions and backtest validation. |

These responsibilities are proposed for team review. All members will contribute to implementation, review one another's work, and present their contributions in Week 4. Changes to scope or responsibilities will be reflected in this README.

## Desired Project Outcome

**Research question:** How does trading frequency affect the performance of a trend-following strategy in energy futures after transaction costs, and how does that performance compare with a passive futures benchmark?

We will build a reproducible Python analysis using historical data from Databento. The proposed initial market is WTI crude oil futures. We will compare a moving-average trend-following strategy at intraday and daily horizons to investigate how trading frequency, holding periods, costs, and market conditions affect performance.

The strategy will take a long position when a fast moving average exceeds a slow moving average and a short position when it falls below it. Signals will use only information available at the time, with execution at the next eligible price. The intraday version will close positions before the defined session ends; the daily version may hold positions across sessions.

The final deliverables will include:

- A documented dataset and reproducible cleaning pipeline, including contract selection, missing-data handling, and rollover rules.
- Configurable strategy and backtesting code for both trading horizons.
- Comparison tables covering gross and net returns, volatility, Sharpe ratio, maximum drawdown, turnover, and trading costs, with calculation conventions stated.
- Cumulative-performance and drawdown charts, plus a limited parameter and cost sensitivity analysis.
- A final report explaining the research question, data, methodology, findings, and limitations.
- A Week 4 presentation summarizing the analysis and each team member's contribution.

We will compare both strategies with a long-only position in the same rolling futures contracts, using a common starting-capital basis and consistent rollover and cost assumptions. The report will explain differences in overnight exposure and holding periods so the comparison can be interpreted fairly.

Success means that another team member can obtain the documented inputs, reproduce the backtests and key tables, and trace the report's conclusions to saved results. A profitable strategy is not required: evidence of weak or cost-sensitive performance is also a useful research outcome.

### Scope and Decisions to Confirm

This README describes a proposed implementation. By the end of Week 1, the team will confirm the following settings and record them in the configuration files and data documentation:

- **Data:** Databento dataset, individual contract identifiers, exact sample dates, access requirements, and download budget. Both horizons should use a common historical period where coverage permits.
- **Trading horizons:** Proposed 15-minute intraday bars and daily bars. Define the session timezone and closing cutoff, and document whether bars are supplied directly or aggregated from finer data.
- **Strategy:** Fast and slow moving-average windows, one-contract position sizing as the proposed baseline, and execution at the next eligible bar's opening price. Define how intraday positions are closed at the session cutoff.
- **Contract handling:** A reproducible roll rule that retains individual contract identifiers and distinguishes signal prices from the tradable prices used to calculate P&L.
- **Evaluation:** Proposed chronological split of the first 70% of sessions for development and the final 30% for holdout evaluation, with a shared date cutoff. Parameters will be selected on development data and frozen before holdout evaluation.
- **Accounting:** Starting capital, contract multiplier, commissions per side, slippage, roll costs, and conventions for returns, annualization, and Sharpe ratio calculations.

The project will focus on one energy market and one baseline strategy. If data access limits the intraday study, the team will agree on a feasible alternative in Week 1 and update this README. Additional contracts or strategies are optional extensions after the core analysis is complete. The project will analyze historical data without executing live trades.

## Proposed System Design

The repository is currently at the planning stage. The following structure defines where the data, reusable code, analysis, and final deliverables will be stored; the listed implementation files are planned.

```text
futures-project-group3/
├── README.md                  # Project goal, setup, workflow, and timeline
├── requirements.txt           # Pinned Python dependencies
├── .env.example               # Environment variable names; no real credentials
├── .gitignore                 # Exclude credentials, local environments, and data
├── config/
│   ├── intraday.yaml           # Data, session, strategy, and cost settings
│   └── daily.yaml              # Daily strategy and backtest settings
├── data/
│   ├── raw/                   # Original Databento files, preserved unchanged
│   ├── processed/             # Cleaned bars and contract/roll metadata
│   └── README.md              # Provenance, schema, coverage, and access instructions
├── src/
│   ├── __init__.py
│   ├── download_data.py        # Fetch configured historical data
│   ├── prepare_data.py         # Validate, clean, and construct analysis datasets
│   ├── strategies.py           # Generate signals using past observations only
│   ├── backtest.py             # Positions, fills, contract rolls, and net P&L
│   └── evaluate.py             # Metrics, comparisons, and charts
├── analysis/
│   ├── 01_data_exploration.ipynb
│   └── 02_strategy_comparison.ipynb
├── tests/                     # Checks for timing, P&L, costs, and rollover behavior
└── reports/
    ├── figures/               # Performance and drawdown charts
    ├── tables/                # Metrics and sensitivity results
    ├── runs/                  # Per-run settings, trades, positions, and returns
    ├── final_report.md        # Findings and limitations
    └── presentation.pdf       # Final slides for Week 4
```

The workflow will be **Databento → raw data → validated data → strategy signals → backtest → evaluation → report and presentation**. Reusable logic will live in `src/`; notebooks in `analysis/` will explain and visualize the results. Configuration files will record the assumptions for each run so a teammate can reproduce it without editing source code.

Processed data will preserve timestamps, contract identifiers, price and volume fields, and roll markers. Each backtest will save its configuration and data reference together with timestamped trades, positions, gross P&L, costs, and net P&L. Both horizons will share the same accounting and evaluation code to keep the comparison consistent.

Raw and processed market data will stay local unless redistribution is permitted. Credentials must never be committed. The data README will explain how authorized users can obtain the same inputs, including dataset identifiers, dates, timezone, units, and any limitations.

## How to Run the Project

**Planned interface:** the scripts, configuration files, and dependency file below are not implemented yet. These instructions define the workflow we intend to deliver and will be verified before final submission.

### 1. Set Up the Environment

Use Python 3.11 as the proposed baseline. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/gwino/futures-project-group3.git
cd futures-project-group3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. The dependency file will record tested versions of the Databento client, data analysis and plotting libraries, notebook support, configuration parsing, and testing tools.

### 2. Obtain and Prepare Data

Set your own Databento API key in the shell environment:

```bash
export DATABENTO_API_KEY="YOUR_API_KEY"
```

Review the selected contracts, dates, resolution, and estimated data cost before downloading. The planned `data/README.md` will document the required inputs, file format, and access instructions. If those inputs are already available in `data/raw/`, skip downloading and start with preparation.

```bash
python -m src.download_data --config config/intraday.yaml
python -m src.download_data --config config/daily.yaml
python -m src.prepare_data --config config/intraday.yaml
python -m src.prepare_data --config config/daily.yaml
```

Data preparation will check timestamps, duplicates, missing observations, and contract coverage. It will apply the documented session and rollover rules, preserve roll metadata, and distinguish signal prices from tradable contract prices used for P&L.

### 3. Run the Analysis

```bash
python -m src.backtest --config config/intraday.yaml
python -m src.backtest --config config/daily.yaml
python -m src.evaluate --configs config/intraday.yaml config/daily.yaml
python -m pytest
jupyter lab analysis/
```

Each backtest will save its settings and outputs in a separate directory under `reports/runs/`. Evaluation will produce comparison tables in `reports/tables/` and cumulative-performance and drawdown charts in `reports/figures/`. The notebooks will read these outputs and support the narrative in `reports/final_report.md`.

Validation will check signal timing, representative long and short trades, session exits, contract rolls, and cost accounting against simple hand-calculated examples. Before submission, another team member will follow these instructions in a clean environment, verify the key tables, and document any fixes. The final README will identify the tested environment and expected output filenames.

## Project Timeline

The project runs for four weeks, with Week 4 reserved for the presentation and final deliverables. Calendar dates will be added once the course schedule is confirmed. Core implementation and analysis should be complete by the end of Week 3 so the final week can focus on reproduction, communication, and submission.

| Week | Work plan | Deliverable / completion criteria |
| --- | --- | --- |
| 1 — Agree on scope and establish data | All members review the research question and assumptions. Gabriella sets up the environment and obtains the data; Ethan defines processing and roll rules; Jiayi finalizes the plan and documentation. | Team agreement recorded on the README PR; environment installs successfully; exact sample, data coverage, strategy settings, session rules, and evaluation split are documented. |
| 2 — Prepare data and implement the baseline | Ethan develops and reviews data preparation; Gabriella implements signals and backtesting; Jiayi builds the exploration notebook and documents methodology. Add costs and the passive benchmark. | Both horizons run end to end on validated data; sample trades reconcile with hand calculations; timing, session exits, costs, and roll behavior are checked. |
| 3 — Evaluate and draft findings | Jiayi coordinates comparison tables and charts. Gabriella and Ethan verify accounting and sensitivity results. Select and freeze parameters using development data, evaluate the holdout, and draft the report and presentation outline. | Core analysis complete; development and holdout results reported separately; metrics and charts saved; draft report explains assumptions, findings, and limitations. |
| 4 — Present and deliver | Gabriella reproduces the documented workflow; all members resolve final problems, review the report and slides, rehearse their speaking sections, and present and submit the project. | Final report and presentation delivered; run instructions verified; numerical claims trace to saved outputs; each member's contributions and reviews are visible in the repository. |

To keep the schedule feasible, sensitivity checks will use a small predefined set of parameter and cost settings. Data-access problems and scope changes should be resolved in Week 1 and reflected in this README. Each member will review the README PR and record agreement or specific feedback; the team will confirm a shared communication channel and check progress weekly.

## Task Tracking

Tasks are tracked as [GitHub Issues](https://github.com/gwino/futures-project-group3/issues) and grouped into [milestones](https://github.com/gwino/futures-project-group3/milestones). Each issue lists a proposed owner, reviewer, dependencies, and "Done when" criteria. Comment on an issue to claim or swap it, and reference the issue in related PRs (for example, `Closes #5`).

- **Week 1 – Scope & Data:** [#2 Team agreement](https://github.com/gwino/futures-project-group3/issues/2), [#3 Environment setup](https://github.com/gwino/futures-project-group3/issues/3), [#4 Scope decisions](https://github.com/gwino/futures-project-group3/issues/4), [#5 Data download](https://github.com/gwino/futures-project-group3/issues/5)
- **Week 2 – Data Exploration & Planning:** [#6 Data cleaning](https://github.com/gwino/futures-project-group3/issues/6), [#7 Contract rolls](https://github.com/gwino/futures-project-group3/issues/7), [#8 Data exploration](https://github.com/gwino/futures-project-group3/issues/8), [#9 Plan strategy and evaluation tasks](https://github.com/gwino/futures-project-group3/issues/9)

Strategy, backtesting, evaluation, and reporting tasks will be opened from #9 once the team has reviewed the data findings in #8, so the later work is designed around the actual data.
