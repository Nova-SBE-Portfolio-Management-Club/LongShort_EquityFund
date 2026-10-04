# Long-Short Equity Fund

An equity-research project developed within the Nova SBE Portfolio Management Club. It combines a PostgreSQL application for storing and updating market data with experiments in ranking, pairs, momentum, earnings surprise, and sector rotation.

**Jack Noel's contribution:** database models and migrations, pair and price-data functions, and Bloomberg import/export and yield-curve workflows. **João Fonseca's contribution:** the initial application structure, database connector, migrations, and correlation/cointegration utilities. Strategy research was developed by several club members; see [contributors](docs/CONTRIBUTORS.md).

The data application and research code were developed together, but the strategy experiments also have their own inputs and workflows. Some run from bundled data, while others require PostgreSQL, Bloomberg access, or live downloads.

## Start here

| Workflow | What to look at | Status |
| --- | --- | --- |
| [Data infrastructure](src/database/README.md) | SQLAlchemy models, Alembic migrations, data updates, and Bloomberg CSV exchange | Implemented application code; requires a configured database and has some unfinished commands |
| [Earnings-surprise research](src/playground/earnings_surprise/README.md) | Event signals, position caps, and comparison with an equal-weight benchmark | Script results reproduced locally from bundled inputs; data and execution assumptions remain open |
| [Macro-regime research](src/playground/macro_regime/README.md) | Long-only sector rotation, stock selection, and stress diagnostics | Corrected pipeline rerun on bundled inputs; the notebook retains older historical analysis |

The [research index](src/playground/README.md) links the other experiments. [Known limitations](docs/ROADMAP.md) describe the remaining gaps.

## Quick start

For a small offline example, use Python 3.13 and run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python src/playground/backtest_demo.py
```

The demo uses seeded synthetic returns, prints statistics, and writes `backtest_results.png`. It illustrates the shared backtester rather than a trading result. At 1x gross exposure, the allocation is 50% long and 50% short; see [backtesting conventions](docs/BACKTESTING.md).

For notebooks and the wider research environment:

```bash
python -m pip install -r requirements-research.txt
python -m jupyter lab
```

Individual guides describe the required data and working directory. The [database guide](src/database/README.md) covers PostgreSQL setup and Bloomberg workflows. `requirements.lock.txt` retains the original full environment snapshot; the smaller demo and test environments use their own pinned requirements.

## Repository layout

```text
docs/                   Results, calculation conventions, and known limitations
src/main.py             Interactive database-management application
src/database/           SQLAlchemy models, connector, and Alembic migrations
src/bloomberg/          Bloomberg request templates
src/utils/              Shared ranking, statistics, and backtesting helpers
src/pair_generator/     Pair-selection interface prototype
src/playground/         Strategy research, notebooks, and the synthetic demo
```

## Results and experiments

The [earnings-surprise report](docs/results/earnings_surprise.md) includes a performance chart, capped and uncapped returns, benchmark definitions, and a data audit. Its results reproduce the existing implementation; costs, historical-universe coverage, and execution timing still need research. The [results guide](docs/results/README.md) gives the reproduction commands.

Exploratory notebooks retain their own hypotheses and parameter choices. Saved historical outputs are identified where they have not been rerun against the current code.

## Checks

Tests in [`tests/`](tests/) check shared performance calculations, portfolio exposure, earnings-strategy behavior, and stock-selection filters. With database dependencies installed, they also check ORM queries and deletion boundaries using SQLite. [GitHub Actions](.github/workflows/ci.yml) is configured to run the core checks and offline report reproduction; live PostgreSQL, Bloomberg, and downloads are outside that coverage.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the check commands and a few notes on changing research code.
