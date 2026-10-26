# Does WTI Crude Oil Have Intraday Momentum? A Post-2020 Replication

Group 3, FINM 37000 Futures and Related Derivatives. We replicate Baltussen, Da, Lammers and Martens (2021, *Journal of Financial Economics*) on NYMEX WTI crude oil futures (CL) from January 2021 to October 2026, using 1,436 trading days of one-minute Databento data. We then test three new hypotheses about overnight and daytime returns.

## What we test

The paper's claim is **intraday momentum**: how the market moves from yesterday's close until 30 minutes before today's close predicts what it does in the final 30 minutes. We estimate this by regressing the last-half-hour return on the rest-of-day return. All times are Chicago time, and the reference close is 13:30.

- **LH (last half hour):** return from 13:00 to 13:30. This is the quantity we predict.
- **ROD (rest of day):** return from the previous day's 13:30 close to 13:00. This is the paper's predictor.
- **ON (overnight):** return from the previous 13:30 to 08:00. It is the first part of ROD.
- **DAY (daytime):** return from 08:00 to 13:00. It is the second part of ROD, and (1 + ON)(1 + DAY) = 1 + ROD.
- **ONFH, M, SLH:** the paper's other predictors. ONFH is overnight plus first half hour (previous 13:30 to 08:30), M is 08:30–12:30, SLH is 12:30–13:00.
- **Slope:** basis points of LH per 1% of the predictor (reported × 100). Positive means momentum and negative means reversal.

**Sample:** 1,436 trading days from 4 January 2021 to 7 October 2026 (936 out-of-sample forecasts, starting 4 January 2023). Results are estimates with standard errors.

## Headline findings

1. **There is no intraday momentum in post-2020 CL.** The ROD slope is −1.46 (95% interval [−3.35, 0.43]), where the paper found +0.67 for CL (t = 0.25, also insignificant). Its strong results are in other markets.
2. **The signal cannot be traded.** Going long when ROD is positive and short otherwise earns −1.2 basis points a day (standard error 1.3) before costs. With USD 15 per round trip it earns −3.1 and has a Sharpe ratio of −1.1. Gross is already negative, so no cost level helps.
3. **Overnight and daytime returns do not predict differently.** The ON slope is −1.26 (1.01) and the DAY slope is −1.84 (1.43). Their difference is 0.58 (1.43), which is small next to its error.
4. **High-volatility regimes change nothing measurable.** The slope differences are −1.0 (1.8) for ON and −1.8 (2.9) for DAY, and regime models forecast worse out of sample.
5. **Our one positive finding is a reversal when overnight and daytime moves agree.** On those days the paper's strategy earns −4.5 (1.7) basis points, against +1.9 (1.7) on disagreement days. This effect comes almost entirely from 2022 and 2026. Its out-of-sample support rests on one day.

![Slopes against the paper](reports/figures/replication_vs_paper.png)

| Model | Paper slope (CL, 1987–2020) | Our slope (CL, 2021–2026) | Out-of-sample R², paper / ours |
|---|---|---|---|
| LH ~ ONFH | 2.00 | −1.30 | −0.48% / −0.42% |
| LH ~ ONFH + M + SLH (ONFH slope) | 2.40 | −1.32 | −1.77% / −1.49% |
| LH ~ ROD | 0.67 | −1.46 | −1.04% / 0.37% |

Slopes are multiplied by 100. Every post-2020 interval includes zero.

![Cumulative P&L of the closing-period strategy](reports/figures/lh_strategy_cumulative.png)

## One day drives much of what looks like signal

On 10 March 2026, during the Strait of Hormuz closure, ROD was −15.4% and the last half hour then rose 4.6%. That single day explains more than half of the negative ROD slope. It also explains all of ROD's positive out-of-sample R² (0.37%, which falls to −0.22% without it), the worst strategy day, and the out-of-sample gain of the agreement model (0.82% falls to −0.47%). The data is genuine and stays in every baseline, and each write-up also reports results without it.

## Method in brief

- **Data:** Databento `GLBX.MDP3` one-minute bars for the full Globex session, with the paper's volume-crossover roll rule (69 rolls). The paper's filters leave 1,436 dates.
- **Timing:** the paper's window, 08:00–13:30 Chicago time. The code rejects any predictor that uses a price after the 13:00 decision time.
- **Inference:** Newey-West standard errors and intervals, expanding-window out-of-sample R² with Clark-West tests, and a Bonferroni level where several hypotheses are tested.
- **Volatility:** realized variance is the sum of squared one-minute log returns, computed separately for the overnight and daytime windows, with returns across scheduled halts excluded. A date is high volatility when its variance exceeds the upper quartile of earlier dates, so no future data is used.
- **Trading:** one contract, next-trade fills, USD 2.50 commission and half a tick of slippage per side.
- **Robustness:** both roll rules, 0 to 20 lags, log returns, and dropping notable dates.

## Limitations and future work

The sample is one market, six years and two crisis episodes. With no quotes, the spread is a proxy. The agreement result is a lead, not an established effect. Next steps are to measure agreement as a continuous two-dimensional score and not a yes/no label, to test it on 2019–2020 minute data, to extend it to other energy futures, and to measure spreads from quote data.

## Reproducing the results

The full write-up with every table is in [reports/final_report.md](reports/final_report.md). Per-topic results are in `reports/*_results.md`, and the notebooks `analysis/04` to `analysis/08` display the saved tables. Data provenance is in [data/README.md](data/README.md).

```bash
git clone https://github.com/gwino/futures-project-group3.git && cd futures-project-group3
python -m venv .venv && source .venv/bin/activate     # .venv\Scripts\activate on Windows
python -m pip install -r requirements.txt
git lfs pull                                          # processed data is stored in Git LFS
python -m pytest
```

No API key is needed, because the processed data is committed. Each step reads the previous step's committed outputs, so you can run any step on its own. Each script writes a `*.manifest.json` with its settings and file hashes.

| Step | Command (settings in `config/`) | Writes |
|---|---|---|
| Minute dataset and rolls | `python -m src.prepare_minute_data --config config/prepare_minute.yaml` | `data/processed/cl_minute_*` |
| Return features | `python -m src.build_return_features --config config/return_features.yaml` | `data/processed/cl_return_features*` |
| Replication regressions | `python -m src.replicate_regressions --config config/replication.yaml` | `reports/tables/replication_*`, `reports/figures/replication_*` |
| Subperiods | `python -m src.compare_subperiods --config config/subperiods.yaml` | `reports/tables/replication_subperiod*` |
| Strategy backtest | `python -m src.backtest_lh_strategy --config config/lh_strategy.yaml` | `reports/tables/lh_strategy_*`, `reports/figures/lh_strategy_*` |
| Overnight vs daytime | `python -m src.analyze_on_day --config config/on_day.yaml` | `reports/tables/on_day_*` |
| Volatility features | `python -m src.build_volatility_features --config config/volatility_features.yaml` | `data/processed/cl_volatility_features*` |
| Volatility regimes | `python -m src.analyze_volatility_regimes --config config/volatility_regimes.yaml` | `reports/tables/volatility_regimes_*` |
| Sign agreement | `python -m src.analyze_sign_agreement --config config/sign_agreement.yaml` | `reports/tables/sign_agreement_*` |

The first step needs the raw minute `.dbn` files. The 2021–2024 file is licensed and is not committed, only its manifest. You can skip that step and start from the committed processed data.

`PLANNING.md` holds the original project plan. The repository also keeps the retired moving-average code and daily and 15-minute preparation scripts (`src/backtest.py`, `src/strategies.py` and others). Nothing in the results above depends on them.

## Future work: a continuous two-dimensional agreement score

Our agreement result uses a yes/no label: a day is "agree" if the overnight (ON) and daytime (DAY) returns have the same sign. That label ignores size. A day with ON +0.01% and DAY +0.02% counts the same as one with ON +2% and DAY +1.8%, although the first is close to noise.

A continuous score would describe each day by how much ON and DAY agree and how much they disagree, as two numbers instead of one label. For example:

- **Agreement amount:** min(|ON|, |DAY|) when the signs match, and zero otherwise.
- **Disagreement amount:** min(|ON|, |DAY|) when the signs differ, and zero otherwise.
- **Geometric version:** treat each day as the point (ON, DAY). Distance along the diagonal (ON + DAY, which is ROD) measures agreement, and distance across it (ON − DAY) measures disagreement.

We would then regress the last half hour (LH) on both numbers, with standard errors, and plot the estimates over time. This also answers whether the reversal on agreement days comes from large, clear moves or from small ones.

What we have already done in this direction (`reports/sign_agreement_results.md`): the four sign categories, a 3 × 3 tercile grid of ON and DAY (`reports/figures/sign_agreement_grid.png`), and a regression with the product ON × DAY. The product is positive on agreement days in either direction, so it measures a sign-symmetric dependence and does not by itself show a reversal. We have not built the two-number score above. It was suggested in the instructor's feedback.

## Team

Gabriella Wiem (Tech Lead: data acquisition, cleaning and rolls), Jiayi Chen (Communication Lead: return features, replication regressions, subperiods), Ethan Huang (Design Lead: strategy backtest, overnight versus daytime, volatility and sign-agreement analysis).

## Reference

Baltussen, G., Da, Z., Lammers, S., and Martens, M. (2021). Hedging demand and market intraday momentum. *Journal of Financial Economics*, 142(1), 377–403.
