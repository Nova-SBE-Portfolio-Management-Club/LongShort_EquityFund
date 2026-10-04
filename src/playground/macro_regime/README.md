# Macro Regime Folder Guide

This folder contains a quarterly long-only sector-rotation strategy, with stock selection based on momentum. It supports a local-data workflow and generates data-quality, bias, and robustness diagnostics. See [STRATEGY_EXPLANATION.md](STRATEGY_EXPLANATION.md) for the model logic and assumptions.

## Existing experiment

The [stress notebook](notebooks/stress_test.ipynb) retains the original saved analysis: 82 quarterly observations, comparisons with SPY, regime slices, shifted-feature tests, and drawdowns. Its outputs have not been regenerated; the original report files and configuration baseline are not bundled.

**The saved outputs predate a look-ahead fix:** training included the label at the decision quarter, which contains the next quarter's return. Training now excludes that label. Stock-history and liquidity filters also now return no picks when no stocks qualify. Those saved performance numbers do not evaluate the corrected implementation.

The corrected pipeline completed a local 82-quarter run on the bundled inputs with the default strategy settings and gates enabled. A new configuration baseline was created for that run. Coverage was 85.79% against an 88% target; delisting returns remain absent, and shuffled-label performance still triggers a warning. The built-in reliability score was 76/100 against its 65-point threshold; passing that gate does not resolve those research limitations. The original notebook outputs above have not been replaced with this run.

The saved diagnostics contain a coverage warning and show that stale features do not degrade performance enough. They also compare the main strategy with random-sector, shuffled-label, and intentionally leaked-feature cases. These are useful checks to investigate further, rather than treating the cumulative-return chart alone as the result.

The notebook retains its original CAGR/volatility Sharpe proxy and its own drawdown calculations. Its metric values differ in definition from the updated shared backtester.

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
python -m pip install -r requirements-research.txt
python -m src.playground.macro_regime.main
```

The default configuration in `config.py` enforces reliability, universe realism, and frozen-parameter consistency. A fresh checkout has no `data/frozen_config.json`, so the pipeline stops at the parameter-freeze gate until a baseline is supplied.

To start a new local experiment, review the parameters first, then explicitly create a baseline from the repository root:

```bash
python - <<'PY'
from src.playground.macro_regime.config import Config
from src.playground.macro_regime.parameter_freeze import run_parameter_freeze_check
from src.playground.macro_regime.paths import package_root

cfg = Config()
_, report, matched = run_parameter_freeze_check(
    cfg,
    cfg.resolved_snapshot_path(package_root()),
    auto_write_if_missing=True,
)
print(report)
if not matched:
    raise SystemExit("Existing baseline differs; review the configuration before running.")
PY
```

This creates a local baseline if one is missing and leaves an existing snapshot unchanged. It does not establish that the configuration was frozen before earlier research. Other data and reliability gates may still stop the pipeline.

Delisting return integration is enabled, but `enforce_delisting_data_gate` is `False` by default. The supplied `data/delisting_returns.csv` contains only its header. Supply suitable events and enable the gate when assessing delisting coverage.

For the notebook:

```bash
python -m jupyter lab src/playground/macro_regime/notebooks/stress_test.ipynb
```

Run cells top-to-bottom. If no backtest output exists, the notebook calls the same pipeline and is subject to the same prerequisites and gates. Generated data-audit files and reports remain local; record the configuration and input provenance alongside any published result.
