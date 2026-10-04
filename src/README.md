# src

The source tree contains both shared application code and strategy research:

- [`main.py`](main.py): interactive PostgreSQL data-management application.
- [`database/`](database/README.md): database models, connector, migrations, and setup instructions.
- [`bloomberg/`](bloomberg/): Bloomberg request templates for environments with Bloomberg access.
- [`utils/`](utils/): shared ranking, correlation, cointegration, and backtesting functions.
- [`pair_generator/`](pair_generator/): pair-selection interface prototype.
- [`playground/`](playground/README.md): strategy experiments and the synthetic backtest demo.

Start with the [root README](../README.md) for environment setup and the research overview. Imports currently vary between workflows; the individual guides specify the command and working directory to use.
