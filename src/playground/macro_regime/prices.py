import pandas as pd
import yfinance as yf
from pathlib import Path


def configure_yfinance_cache(base_dir: str | Path | None = None) -> None:
    """
    Force yfinance cache files into a writable project-local folder.
    This avoids failures when default user cache directories are read-only.
    """
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    cache_dir = root / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
        yf.cache.set_cache_location(str(cache_dir))
    elif hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

def download_monthly_adjclose(tickers, start, end) -> pd.DataFrame:
    configure_yfinance_cache()
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=True,
        interval="1d"
    )
    # Auto-adjust True => Adj Close is effectively Close
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].rename(columns={"Close": tickers if isinstance(tickers,str) else "Close"})

    px.index = pd.to_datetime(px.index)
    # monthly end
    px_m = px.resample("ME").last()
    return px_m.dropna(how="all")


def load_monthly_prices_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a wide monthly price matrix from CSV:
      - required first column: Date
      - remaining columns: tickers
    """
    p = Path(path)
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        raise ValueError(f"CSV at {p} must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.columns = [str(c).upper().replace(".", "-") for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.resample("ME").last().dropna(how="all")

def monthly_returns_from_prices(px_m: pd.DataFrame) -> pd.DataFrame:
    return px_m.pct_change(fill_method=None).dropna(how="all")

def quarter_returns_from_monthly(px_m: pd.DataFrame) -> pd.DataFrame:
    # Convert monthly prices -> quarter end prices -> pct change
    px_q = px_m.resample("QE").last()
    # Drop trailing partial quarter (e.g., Jan/Feb data mapped to Mar quarter-end).
    if not px_m.empty and not px_q.empty and px_m.index.max() < px_q.index.max():
        px_q = px_q.iloc[:-1]
    return px_q.pct_change(fill_method=None).dropna(how="all")
