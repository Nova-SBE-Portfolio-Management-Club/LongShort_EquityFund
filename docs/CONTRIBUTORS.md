# Contributors

Developed collaboratively within the Nova SBE Portfolio Management Club. The areas below reflect the existing code and Git history; shared research and coordination are not fully captured by commits.

## Core data infrastructure

- **Jack Noel:** asset/equity, futures, macro, and miscellaneous data models and migrations; pair/price-data functions; Bloomberg handlers and imports. Examples: [pair and price functions](https://github.com/Nova-SBE-Portfolio-Management-Club/LongShort_EquityFund/commit/656fbd5daa1cf9da2828b55fb224d78368dac246) and [yield-curve retrieval](https://github.com/Nova-SBE-Portfolio-Management-Club/LongShort_EquityFund/commit/6d7a4846caf47e60fc113caece988d6d5a864e42).
- **João Fonseca (`jmseca`):** initial structure, database connector, Alembic migrations, application commands, and correlation/cointegration utilities. Example: [SQLAlchemy/Alembic integration](https://github.com/Nova-SBE-Portfolio-Management-Club/LongShort_EquityFund/commit/7977ec2a172c97db5f82d9c1ce848ab7489db277).

## Strategy research and additional contributions

| Contributor | Work |
| --- | --- |
| Lucas Albanese (`lucasalbanesee`) | Shared backtesting and metrics; ranking research; macro-regime pipeline and diagnostics |
| Alessandro De Riccardis | Momentum-over-volatility ranking, basic-ranking and momentum notebooks, and the backtest demo |
| Noah | Earnings-surprise strategy, data inputs, notebook, and stress diagnostics |
| Guilherme Cruz | Momentum/reversal signals, loaders, and period comparisons |
| `Vilaca23` | Pair-selection widgets and draft screening logic |
| `leorove` | Country/price models, connector methods, and last-update fields |
| `dnferreira13` | Get/update commands and futures/miscellaneous-data workflows |
| `gunchevsimeon` | Statistical-arbitrage exploratory notebook |

Names follow the original README or Git author identity; handles are retained where a full name is not established.
