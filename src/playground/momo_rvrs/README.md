# Momentum and reversal

Research scripts for buffered long-short rankings and overnight reversal. Shared signal and portfolio helpers live in `signals.py`; `data_loader.py` downloads and caches market prices.

## Run

Install the additional dependencies in `requirements-research.txt` using the [root setup instructions](../../../README.md). From the repository root:

```bash
python -m src.playground.momo_rvrs.test
python -m src.playground.momo_rvrs.close_test
```

- `test.py` compares a fixed long-short configuration across market periods and transaction-cost levels.
- `close_test.py` compares an overnight strategy with an earnings-month exposure adjustment.

The scripts require internet access for constituent and price downloads. Local CSV and Parquet caches are stored in this folder's ignored `data/` directory. Edit the parameters in each runner before starting a new experiment.

Despite their filenames, these scripts run strategy diagnostics rather than automated unit tests. Data-source provenance, historical-universe assumptions, and repeatable sample results remain follow-up work.
