# Macro Regime Strategy: Full Technical Explanation

This document explains how the `macro_regime` code works end-to-end, what the strategy is doing, the math behind it, and exactly how returns are computed.

## 1) Strategy Idea in Plain English

The strategy is a quarterly long-only allocation process built in two stages:

1. Predict which sectors are likely to outperform in the next quarter.
2. Inside those predicted sectors, select high-momentum stocks and hold them equally weighted for one quarter.

It then compares this portfolio to SPY on the same quarterly timeline.

The strategy is not market-neutral and not shorting. It is a directional long equity strategy that tries to improve risk-adjusted returns versus SPY by rotating toward stronger sectors and stronger stocks within those sectors.

## 2) Data Inputs and Where They Are Used

Main input files are in `macro_regime/data/`:

- `stock_prices.csv`: monthly adjusted close panel. Must include `Date` and ideally `SPY`.
- `sp500_constituents.csv`: ticker -> sector mapping, optional point-in-time columns (`DateAdded`, `DateRemoved`).
- `constituents_history_template.csv`: optional event table (`Date`, `Ticker`, `Action`) with `added`/`removed`.
- `delisting_returns.csv`: optional/required in strict mode, with (`Date`, `Ticker`, `DelistReturn`) for terminal delisting events.
- `quality_report.csv`, `dropped_rows.csv`: external audit artifacts, not used in core model logic.

Core code path:

1. `backtest.py` loads inputs, runs data checks, builds features/labels, runs walk-forward backtest, then diagnostics/reliability/statistical reports.
2. `features.py` builds monthly and quarterly feature matrices and aligned targets.
3. `sector_model.py` trains one ridge model per sector.
4. `stock_model.py` selects stocks inside predicted sectors using momentum.
5. `report.py` and additional report modules compute performance and validation outputs.

## 3) Universe and Point-in-Time Handling

Universe construction is handled in `universe.py`.

- Symbols are normalized (uppercase, `.` replaced with `-`).
- If `DateAdded`/`DateRemoved` exist, constituents are filtered `as of` each decision date.
- If history events exist, `added`/`removed` actions are applied through that decision date.

This is intended to reduce survivorship bias. Reliability still depends on how complete the event history is.

## 4) Return Construction from Prices

In `prices.py`:

- Monthly returns:
  \[
  r_{m,t} = \frac{P_t}{P_{t-1}} - 1
  \]
- Quarterly returns:
  prices are resampled to quarter-end, then:
  \[
  r_{q,t} = \frac{P^{(Q)}_t}{P^{(Q)}_{t-1}} - 1
  \]

If local `SPY` exists in `stock_prices.csv`, SPY benchmark uses that column. Otherwise the code can fall back to synthetic benchmark logic.

## 5) Feature Engineering (Sector Level)

Implemented in `features.py` from monthly sector return series \(r_{s,t}\):

- 3-month momentum:
  \[
  \text{mom3}_{s,t} = \frac{1}{3}\sum_{i=0}^{2} r_{s,t-i}
  \]
- 6-month momentum:
  \[
  \text{mom6}_{s,t} = \frac{1}{6}\sum_{i=0}^{5} r_{s,t-i}
  \]
- 12-month momentum:
  \[
  \text{mom12}_{s,t} = \frac{1}{12}\sum_{i=0}^{11} r_{s,t-i}
  \]
- 6-month volatility:
  \[
  \text{vol6}_{s,t} = \text{StdDev}(r_{s,t-5},...,r_{s,t})
  \]
- Relative strength vs SPY (3-month):
  \[
  \text{rel3}_{s,t} = \text{mom3}_{s,t} - \text{mom3}^{SPY}_t
  \]

Features are created monthly and then sampled at quarter-end.

## 6) Label Definition and Timing Alignment

The label for each sector is next-quarter return:

\[
y_{s,T} = r^{(Q)}_{s,T+1}
\]

Code alignment in `make_sector_training_set`:

- `X_q`: features at quarter-end \(T\).
- `Y_q_next`: quarter returns shifted by -1 so each row at \(T\) maps to \(T+1\) return.

This is explicitly designed to avoid lookahead:

- Train uses data only up to current decision quarter.
- Test/trade return is the following quarter.

## 7) Sector Prediction Model (Ridge Regression)

In `sector_model.py`, each sector is modeled independently with ridge regression.

For each sector \(s\):

\[
\hat{\beta}_s = \arg\min_{\beta}\left\{\sum_{t}(y_{s,t} - X_{s,t}\beta)^2 + \alpha \|\beta\|_2^2\right\}
\]

Key details:

- Features are standardized with `StandardScaler`.
- Default ridge penalty is `alpha=10.0`.
- Minimum sample requirement per sector model: 20 aligned observations.

Predictions are made for one quarter using the latest available feature row.

## 8) Sector Selection Rule

For each rebalance date:

1. Predict next-quarter return for each sector with trained sector models.
2. Rank sectors by predicted return.
3. Choose top `K` sectors (`top_k_sectors`, default 3).

The selected sector IDs are then mapped to sector names for stock selection.

## 9) Stock Selection Inside Predicted Sectors

Implemented in `stock_model.py`:

For each predicted sector:

1. Start with constituents in that sector.
2. Keep symbols available in the price panel.
3. Enforce minimum history (`min_history_months_for_stock_selection`, default 24).
4. Use recent window (up to last 18 months).
5. Drop stocks with missing values in that recent window.
6. Optionally apply liquidity filter using average dollar volume if provided.
7. Compute 12-1 momentum:
   \[
   \text{mom12-1} = \frac{1 + R_{12m}}{1 + R_{1m}} - 1
   \]
8. Rank by momentum and keep top `N` per sector (`top_n_stocks_per_sector`, default 10).

The portfolio is the union of picks across selected sectors.

## 10) Portfolio Return Computation

Quarterly realized portfolio return uses equal weight across chosen stocks:

\[
r^{port}_{q,t} = \frac{1}{n_t}\sum_{i=1}^{n_t} r_{i,q,t}
\]

If a stock-level cap is configured, each stock return is clipped before averaging:

\[
r_{i,q,t}^{clipped} = \min(\max(r_{i,q,t}, -c), c)
\]

Then transaction cost is subtracted each rebalance:

\[
r^{net}_{q,t} = r^{port}_{q,t} - \frac{\text{txn\_cost\_bps}}{10000}
\]

Default cost is 10 bps per quarter.

SPY quarter return is:
\[
r^{SPY}_{q,t}
\]
from the same quarter index.

## 11) Equity Curve and Compounding

Compounding is geometric, not additive.

Portfolio equity:
\[
EQ^{port}_t = \prod_{\tau \le t} (1 + r^{net}_{q,\tau})
\]

SPY equity:
\[
EQ^{SPY}_t = \prod_{\tau \le t} (1 + r^{SPY}_{q,\tau})
\]

Alpha equity shown in outputs:
\[
EQ^{alpha}_t = EQ^{port}_t - EQ^{SPY}_t
\]

Yes, this means each new quarter is applied on the current compounded capital (for example 1000 -> 1100 -> next return on 1100).

## 12) Performance Metrics Math

In `report.py` and related reports:

- Mean quarterly return:
  \[
  \bar{r}_q = \frac{1}{N}\sum_t r_t
  \]
- Annualized return (simple transformation used in code):
  \[
  r_{ann} = (1 + \bar{r}_q)^4 - 1
  \]
- Annualized volatility:
  \[
  \sigma_{ann} = \sigma_q \sqrt{4}
  \]
- Sharpe (risk-free assumed 0):
  \[
  \text{Sharpe} = \frac{r_{ann}}{\sigma_{ann}}
  \]
- Max drawdown from equity curve:
  \[
  DD_t = \frac{EQ_t - \max_{\tau \le t} EQ_\tau}{\max_{\tau \le t} EQ_\tau}
  \]
  \[
  \text{MaxDD} = \min_t DD_t
  \]

Other reports include Sortino, hit rate vs SPY, alpha by period/window, and reliability scoring.

## 13) Validation and Robustness Layers

The pipeline now includes multiple checks:

1. `data_checks.py`: schema, date quality, outliers, active-window coverage, history consistency.
2. `validation.py`: leakage/sanity diagnostics:
   - stale features test
   - leaked features test
   - shuffled labels sanity test
   - baseline comparisons
3. `reliability.py`: aggregates warnings/diagnostics to a reliability score.
4. `train_test_report.py`: strict split (train <= split date, test > split date).
5. `rolling_walkforward_report.py`: repeated rolling train/test windows and decay tracking.
6. `bootstrap_report.py`: bootstrap confidence intervals and probability-style significance checks.

## 14) Output Files You Should Read First

In `macro_regime/reports/`:

- `backtest_results.csv`: quarter-by-quarter returns and equity.
- `quarter_log.txt`: detailed quarter decisions and picks.
- `diagnostics_report.txt`: leakage and sanity diagnostics.
- `reliability_report.txt`: high-level trust score.
- `train_test_split_report.txt`: strict out-of-sample split behavior.
- `rolling_walkforward_report.txt`: stability across rolling windows.
- `bootstrap_significance_report.txt`: confidence intervals for Sharpe and alpha.

## 15) Important Practical Notes

- A strong backtest can still fail live if universe data is incomplete, delisting returns are missing, or parameter tuning was excessive.
- Statistical confidence is stronger when multiple independent tests agree (split, rolling windows, bootstrap, leakage tests).
- The strategy should be treated as research-grade until data history and bias controls are fully institutionalized.

## 16) Step 7 and Step 8 Additions

- Step 7 (delisting integration): the code can now inject delisting returns into stock quarterly return series before portfolio construction. This is recorded in `reports/delisting_integration_results.csv`.
- Step 8 (strict governance): default settings now enforce reliability, universe realism, frozen config consistency, and delisting coverage gates. In strict mode, backtest can fail intentionally when controls are not satisfied.
