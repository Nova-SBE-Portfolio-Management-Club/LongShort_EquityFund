# Contributing

Keep changes focused and credit [contributors](docs/CONTRIBUTORS.md). For research changes:

- Explain the experiment and any changes to signals, weights, execution timing, or costs.
- Record the data, parameters, command, and limitations alongside a result. Keep useful notebook outputs; identify historical results and clear failed or misleading outputs.
- Keep credentials in a local `.env`, and leave downloads, caches, and checkpoint copies out of Git.

## Local checks

```bash
python -m pip install -r requirements-dev.txt -r requirements-results.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python scripts/check_repository.py
python scripts/reproduce_results.py --check
```

Tests cover shared backtesting, selected earnings cases, stock-selection filters, and saved reports. ORM tests use SQLite and are skipped when the database dependencies are absent; install `requirements.txt` alongside the check dependencies to run them. Ruff covers the backtester, earnings script, demo, scripts, and tests; the earnings script retains its original layout and is excluded from automatic formatting. The syntax check covers the full source tree. These checks do not exercise live PostgreSQL, Bloomberg, or downloads.

If a change affects a published report, regenerate it with `python scripts/reproduce_results.py` and review the returns and data audit. Source hashes flag changes to the relevant implementation, including comments and formatting. See [known limitations](docs/ROADMAP.md) for outstanding research and application gaps.
