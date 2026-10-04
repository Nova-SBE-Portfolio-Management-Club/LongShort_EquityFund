# Known limitations

These are the gaps that matter when using or interpreting the existing work.

## Research results

- **Earnings surprise:** the bundled-data reproduction excludes costs. Six signal tickers do not map to the price file, and 821 active ticker-period cells have no usable return. Data provenance, historical constituents, and execution timing need review before treating the result as evidence of a tradable strategy.
- **Macro regime:** the corrected pipeline completed an 82-quarter local run; the notebook's historical outputs predate the fixes. Coverage remains below its target, delisting returns are absent, and shuffled-label performance remains high. A new local baseline does not establish that earlier research used parameters frozen in advance.
- **Pair selection:** the notebook is a widget prototype; database screening and scoring are unfinished.
- **Other experiments:** some use live downloads or a database rather than a fixed dataset. Historical notebook outputs may differ from a new run. The shared backtest's exposure and metric changes are documented in [BACKTESTING.md](BACKTESTING.md).

## Application and environment

- Some database commands and asset-deletion methods are unfinished. Fresh PostgreSQL setup was checked locally, and ORM regression tests use SQLite; live provider workflows are outside the automated tests.
- Historical migrations include a view removal specific to the original database; the chain passes on an empty PostgreSQL 14 database, but upgrades of an existing deployment need separate validation.
- The full dependency snapshot is inherited from the original project. Additional research dependencies are unpinned, and working-directory/import conventions still differ between workflows.
- Data source dates, adjusted-price conventions, and historical-universe coverage need fuller documentation. The repository has no project license selected.

The next useful research work is to resolve data and execution assumptions and test changes on unseen periods. Further packaging work should follow the needs of an actual workflow.
