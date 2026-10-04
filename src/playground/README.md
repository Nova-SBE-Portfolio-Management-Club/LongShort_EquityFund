# Playground

Strategy research lives here, alongside the shared-backtester example. See the [project overview](../../README.md#start-here) for the main entry points and setup.

## Workflows

| Experiment | Question or approach | Evidence and requirements |
| --- | --- | --- |
| [Earnings surprise](earnings_surprise/README.md) | Post-announcement drift with event signals and capped positions | Bundled inputs, reproduced script report, notebook, and stress diagnostics |
| [Macro regime](macro_regime/README.md) | Long-only sector rotation followed by stock momentum selection | Pipeline and historical stress-test outputs; a new run needs a configuration baseline |
| [Cross-sectional momentum](momentum_strategy/momentum_strategy.ipynb) | Three-month rankings with monthly top/bottom-quintile portfolios | Historical input/signal tables and an archived result; rerunning requires Yahoo Finance |
| [Momentum and reversal](momo_rvrs/README.md) | Buffered rankings and overnight reversal | Live-data scripts and period comparisons |
| [Basic ranking](basic_ranking/README.md) | Momentum divided by volatility | Exploratory notebook; requires Yahoo Finance |
| [Ranking exploration](RM.ipynb) | Inspect score components and edge cases | Seeded synthetic-price experiment with original saved plots |
| [Statistical arbitrage](SA.ipynb) | Mean reversion of a selected pair's spread | Original saved tables/plots; a new run uses Yahoo Finance |
| [Pair selection](../pair_generator/tester.ipynb) | Interface for correlation and cointegration screening | Widget prototype; database queries and scoring are unfinished |

## Reading and running the experiments

Notebook introductions explain their inputs and whether outputs are historical or regenerated. The strategies retain their own parameters and backtest calculations; their metrics are not automatically comparable.

[`backtest_demo.py`](backtest_demo.py) is a small synthetic software example. Run `python src/playground/backtest_demo.py` from the repository root with `requirements-core.txt` installed. It needs no market-data connection or database.

Scripts named `test.py`, `close_test.py`, or `stresstest.py` run research diagnostics. The deterministic correctness tests are in [`tests/`](../../tests/); [CONTRIBUTING.md](../../CONTRIBUTING.md) lists the commands.
