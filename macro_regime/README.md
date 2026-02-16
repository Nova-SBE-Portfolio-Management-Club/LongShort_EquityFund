# Macro Regime Folder Guide

This folder is organized so you can run the full workflow with local data.

## Folder Layout
- `data/`: all input and data-audit CSVs
- `reports/`: generated backtest and diagnostics outputs
- `notebooks/`: analysis notebooks
- `*.py` in root: strategy pipeline code
- `STRATEGY_EXPLANATION.md`: full technical explanation of model logic and math

## Core Code
- `backtest.py`: main pipeline (data checks, backtest, diagnostics, reliability)
- `config.py`: thresholds and runtime settings
- `prices.py`, `features.py`, `sector_model.py`, `stock_model.py`, `universe.py`, `validation.py`, `reliability.py`: strategy logic
- `period_backtests.py`: optional period-slice summaries

## Input Data
- `data/stock_prices.csv`: monthly prices (must include `Date` and `SPY`)
- `data/sp500_constituents.csv`: current constituents with sector info
- `data/constituents_history_template.csv`: add/remove event history
- `data/delisting_returns.csv`: delisting return events (`Date,Ticker,DelistReturn`)
- `data/quality_report.csv`, `data/dropped_rows.csv`: data-cleaning audit files

## Generated Outputs
- `reports/backtest_results.csv`
- `reports/quarter_log.txt`
- `reports/data_checks_results.csv`, `reports/data_checks_report.txt`
- `reports/delisting_integration_results.csv`, `reports/delisting_integration_report.txt`
- `reports/diagnostics_results.csv`, `reports/diagnostics_report.txt`
- `reports/reliability_report.txt`
- `reports/universe_realism_results.csv`, `reports/universe_realism_report.txt`
- `reports/parameter_freeze_results.csv`, `reports/parameter_freeze_report.txt`
- `reports/train_test_split_results.csv`, `reports/train_test_split_report.txt`
- `reports/rolling_walkforward_results.csv`, `reports/rolling_walkforward_report.txt`
- `reports/bootstrap_significance_results.csv`, `reports/bootstrap_significance_report.txt`
- `reports/period_backtests_results.csv`, `reports/period_backtests_report.txt`

## Run
From repo root:

```bash
./ls_venv/bin/python macro_regime/backtest.py
```

Notebook:
- Open `macro_regime/notebooks/stress_test.ipynb`
- Run cells top-to-bottom

Strict mode notes:
- Backtest now enforces reliability, universe realism, frozen-parameter consistency, and delisting data coverage by default.
- Provide `data/delisting_returns.csv` with enough mapped events to pass the delisting gate.
