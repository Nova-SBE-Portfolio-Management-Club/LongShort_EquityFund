# src/database

PostgreSQL data storage for companies, assets, prices, countries, pairs, and macroeconomic data. The application supports the club's market-data collection and provides stored histories for research.

## What was built

| Component | Implementation detail |
| --- | --- |
| [Models](models.py) | Shared asset identifiers, company/futures models, country metadata, and macro/miscellaneous series |
| [Price storage](db_engine.py) | Open/close observations keyed by `(date, ticker)`; inserts skip existing keys; queries retrieve histories and date coverage |
| [Migrations](migrations/versions/) | Recorded schema changes for asset/company relationships, foreign keys, price keys, and last-update fields |
| [Update handlers](../utils/handlers/update.py) | Retrieve market data for stored assets and use last-update dates to select the next period |
| [Bloomberg handler](../utils/handlers/bloomberg.py) | Export company identifiers and yield-curve request metadata to CSV; import handling remains partly unfinished |

Jack Noel contributed the models, migrations, pair/price functions, and Bloomberg workflows alongside João Fonseca and other club members. See [contributors](../../docs/CONTRIBUTORS.md) for credit.

## Example: inspecting stored prices

Company records connect asset identifiers to country metadata. Price rows store daily open and close values; the `(date, ticker)` key permits another update to skip observations already stored.

With populated tables, a read-only SQL inspection of the stored format is:

```sql
SELECT ticker, date, open_price, close_price
FROM prices_data
ORDER BY ticker, date
LIMIT 5;
```

The connector methods `get_company_pricedata` and `get_company_pricedata_period` retrieve an asset's history and its stored date range. The [pair-selection notebook](../pair_generator/tester.ipynb) is an interface prototype; its database queries and scoring are unfinished.

## Setup

Activate the environment described in the [root README](../../README.md), then install the broader database dependencies:

```bash
python -m pip install -r requirements.txt
```

Install and start PostgreSQL, then create an empty database for this project and a user with access to it.

From the repository root, create a local configuration:

```bash
cp .env.example .env
```

Edit `.env` with the actual database name, user, password, host, and port. Both the connector and Alembic load these variables through `python-dotenv`.

Run migrations from the database directory so that the current Alembic configuration and model imports resolve correctly:

```bash
cd src/database
python -m alembic upgrade head
cd ../..
```

Run the interactive application from `src/`, where its current import and CSV paths are resolved:

```bash
cd src
python main.py
```

## Bloomberg and other inputs

- `imports/countries_template.csv` documents the CSV columns used by the population command. An optional `yc_code` column supplies Bloomberg yield-curve identifiers; it is not present in the bundled country file.
- `imports/countries.csv` contains the existing country input.
- Company and price updates can use external data services and need internet access.
- Bloomberg templates in [`../bloomberg/template/`](../bloomberg/template/) use `blpapi` or `xbbg`. They run separately from the database menu and require a suitable Bloomberg environment.
- The menu's `BLOOMBERG` → `Export` → `All` or `Country` options construct identifiers from stored company tickers and country Bloomberg codes, and write CSVs in `src/bloomberg/exports/`.
- [`get_yc_data.py`](../bloomberg/template/get_yc_data.py) illustrates yield-curve requests: a manually prepared ticker/start-date file feeds tenor discovery and historical-price requests. The template and menu CSV formats are not a single automated pipeline.

## Current gaps

The migration chain and application startup were verified on an empty PostgreSQL 14 database. Checks covered model columns, the price key, duplicate-price inserts, pair/country queries, and deletion boundaries. [`tests/test_database.py`](../../tests/test_database.py) checks ORM queries and deletion behaviour using SQLite when the database dependencies are installed.

Pair, price, and industry menu queries and asset-deletion methods are unfinished. Bloomberg exports contain request metadata; importing provider responses into the database still needs implementation. Live market-data and Bloomberg requests require separate validation.

These are application follow-ups; the models, migration history, and research work remain available to inspect. See [known limitations](../../docs/ROADMAP.md) for the wider research gaps.
