from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
DATA_DIR = STRATEGY_DIR / "data"

SP500_CSV_PATH = DATA_DIR / "sp500.csv"
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PRICES_PARQUET_PATH = DATA_DIR / "sp500_adjclose.parquet"
VOLUME_PARQUET_PATH = DATA_DIR / "sp500_volume.parquet"
