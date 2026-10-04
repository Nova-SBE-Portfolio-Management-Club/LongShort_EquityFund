# Results and experiments

These reports can be regenerated from the repository without market-data downloads or a database:

- [Synthetic backtester demonstration](synthetic_demo.md): seeded data for checking the software's exposure and performance conventions.
- [Bundled earnings-surprise reproduction](earnings_surprise.md): the existing strategy configuration evaluated on its bundled inputs, with a data and sizing audit.

The earnings report compares capped and uncapped positions with an equal-weight benchmark and records the missing-data assumptions. Saved returns and configuration accompany both reports.

```bash
python -m pip install -r requirements-results.txt
python scripts/reproduce_results.py
python scripts/reproduce_results.py --check
```

The check recomputes returns and metrics and verifies the saved artifacts. Input/code hashes and package versions are recorded in the accompanying manifests.

## Earlier experiments

- [Historical momentum result](momentum_historical.md): the original chart and metrics, using the earlier full-spread exposure convention. It has not been regenerated with the current backtester.
- [Macro-regime stress notebook](../../src/playground/macro_regime/notebooks/stress_test.ipynb): original saved performance, data warnings, shifted-feature tests, regime comparisons, and drawdown analysis. These outputs are historical and are not checked by the reproduction command above.

See [BACKTESTING.md](../BACKTESTING.md) for metric definitions and changes that affect earlier outputs. Reproducing an implementation does not resolve its data provenance or establish out-of-sample performance.
