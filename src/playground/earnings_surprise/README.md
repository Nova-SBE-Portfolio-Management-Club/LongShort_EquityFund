# Earnings Surprise Strategy

Research into post-earnings announcement drift: long firms with strong earnings beats and short firms with strong misses.

## Experiment and observations

The strategy combines standardized unexpected earnings with surprise z-scores, then compares capped and uncapped event portfolios with an equal-weight benchmark. The cap leaves unused exposure in cash when few names qualify; the short overlay requires enough active longs.

In the [bundled-data script reproduction](../../../docs/results/earnings_surprise.md), position caps reduce annualized volatility from 13.61% to 9.47% and the worst drawdown from 16.21% to 8.10%, while CAGR falls from 26.60% to 23.58%. This is a full-sample sizing comparison, not an independent validation of the strategy. The report also identifies unmapped tickers and active positions without usable returns.

The [notebook](earnings_surprise.ipynb) keeps the original exploratory parameter variant and has regenerated plots and tables. Its start date, thresholds, and cash-rate assumption differ from the script report. `stresstest.py` contains subperiod, market-regime, parameter, and tail-risk comparisons.

## Implementation

- `best_earnsurp.py` implements standardized unexpected earnings (SUE) and historical surprise z-scores, configurable entry and holding periods, and position caps.
- Shorts are allowed only alongside active longs; both books have per-name weight caps.
- `stresstest.py` runs subperiod, market-regime, parameter-sensitivity, and tail-risk diagnostics.
- `earnings_surprise.ipynb` contains an exploratory variant with its own parameter settings; its results need not match the script.
- `backtest.py` re-exports the shared implementation in `src/utils/backtest.py` for compatibility with older notebooks. The script and stress diagnostics use its CAGR, volatility, Sharpe, and drawdown calculations.

## Data

The existing inputs are located alongside the scripts:

- `earningssurprise_S&Pall.xlsx`, sheet `Data`, with columns `Security`, `AnnouncementDate`, `EPS_Actual`, and `EPS_Estimate`.
- `S&P_dailyprices_clean.csv`, with a date index and ticker price columns.

Default script paths are resolved from the strategy folder. Change the paths and parameters in `CONFIG` for a different input or experiment. Source provenance and the assumptions behind these datasets still need documentation.

## Run

Install `requirements-research.txt` as described in the [root README](../../../README.md), including `openpyxl` for the Excel input. From the repository root:

```bash
python src/playground/earnings_surprise/best_earnsurp.py
python src/playground/earnings_surprise/stresstest.py
```

For the notebook:

```bash
python -m jupyter lab src/playground/earnings_surprise/earnings_surprise.ipynb
```

Run its cells in order. These are research workflows; the stress diagnostics do not replace automated correctness tests.

## Reproduced results

See the [bundled-data report](../../../docs/results/earnings_surprise.md) for the default script configuration, benchmark, return series, source hashes, and data audit. The report documents unmapped tickers, missing quotes, excluded costs, and unresolved execution and universe assumptions.
