# src/playground/MomentumReversal/data_loader.py
from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src.playground.MomentumReversal.config import (
    SP500_CSV_PATH,
    WIKI_SP500_URL,
    PRICES_PARQUET_PATH,
    VOLUME_PARQUET_PATH,
)


def _clean_ticker(sym: str) -> str:
    # Wikipedia uses "." (BRK.B), yfinance often uses "-" (BRK-B)
    sym = str(sym).strip()
    sym = sym.replace(".", "-")
    return sym


def generate_sp500_csv(force: bool = False) -> pd.DataFrame:
    """
    Create (or load) a local CSV containing current S&P 500 tickers + GICS sector.

    Output columns:
      - ticker
      - sector
    """
    if SP500_CSV_PATH.exists() and not force:
        return pd.read_csv(SP500_CSV_PATH)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text), match="Symbol")
    if not tables:
        raise RuntimeError("No tables found on Wikipedia page (unexpected).")

    sp500_table = tables[0].copy()

    if "Symbol" not in sp500_table.columns:
        raise RuntimeError(f"Expected 'Symbol' column, got: {list(sp500_table.columns)}")
    if "GICS Sector" not in sp500_table.columns:
        raise RuntimeError(f"Expected 'GICS Sector' column, got: {list(sp500_table.columns)}")

    sp500_table["Symbol"] = sp500_table["Symbol"].map(_clean_ticker)
    sp500_table["GICS Sector"] = sp500_table["GICS Sector"].astype(str).str.strip()

    df = sp500_table[["Symbol", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "GICS Sector": "sector"}
    )

    # Remove any known-bad tickers (keep your previous behavior)
    bad = {"XYZ"}
    df = df[~df["ticker"].isin(bad)].copy()

    df = df.sort_values("ticker").reset_index(drop=True)

    SP500_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SP500_CSV_PATH, index=False)
    return df


def load_sp500_tickers() -> list[str]:
    if not SP500_CSV_PATH.exists():
        generate_sp500_csv(force=True)

    df = pd.read_csv(SP500_CSV_PATH)
    if "ticker" not in df.columns:
        raise RuntimeError(f"SP500 CSV missing 'ticker' column: {SP500_CSV_PATH}")

    return df["ticker"].astype(str).tolist()


def load_sp500_sector_map() -> pd.Series:
    """
    Returns:
      pd.Series with index=ticker and value=sector
    """
    if not SP500_CSV_PATH.exists():
        generate_sp500_csv(force=True)

    df = pd.read_csv(SP500_CSV_PATH)
    if "ticker" not in df.columns or "sector" not in df.columns:
        raise RuntimeError(
            f"SP500 CSV must contain columns ['ticker','sector'], got: {list(df.columns)}"
        )

    sector_map = pd.Series(df["sector"].values, index=df["ticker"].values, dtype="object")
    sector_map = sector_map[~sector_map.index.duplicated(keep="first")]
    return sector_map


def download_adjclose_panel(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download Adj Close panel only (backward-compatible with your original code).
    """
    if end is None:
        end = date.today().isoformat()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=True,
        threads=True,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned empty data. Network/rate limit?")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise RuntimeError(
                f"Expected 'Adj Close' in yfinance columns, got: {data.columns.levels[0]}"
            )
        adj = data["Adj Close"].copy()
    else:
        # single ticker case
        adj = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    adj.index = pd.to_datetime(adj.index)
    adj = adj.sort_index()
    return adj


def download_adjclose_and_volume_panel(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download both Adj Close and Volume panels.
    Returns:
      (adj_close_df, volume_df)
    """
    if end is None:
        end = date.today().isoformat()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=True,
        threads=True,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned empty data. Network/rate limit?")

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(data.columns.get_level_values(0))
        if "Adj Close" not in lvl0:
            raise RuntimeError(f"Expected 'Adj Close' in yfinance columns, got: {sorted(lvl0)}")
        if "Volume" not in lvl0:
            raise RuntimeError(f"Expected 'Volume' in yfinance columns, got: {sorted(lvl0)}")

        adj = data["Adj Close"].copy()
        vol = data["Volume"].copy()
    else:
        # single ticker case
        adj = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        vol = data[["Volume"]].rename(columns={"Volume": tickers[0]})

    adj.index = pd.to_datetime(adj.index)
    vol.index = pd.to_datetime(vol.index)

    adj = adj.sort_index()
    vol = vol.sort_index()

    return adj, vol


def _to_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    """
    Prefer pyarrow if installed (more robust), fallback to default otherwise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, engine="pyarrow")
    except Exception:
        df.to_parquet(path)


def get_sp500_prices(
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame of Adj Close prices with DatetimeIndex.
    Cached to parquet at PRICES_PARQUET_PATH.

    IMPORTANT:
    - Even when using cached parquet, we still slice to the requested [start, end] range.
    - This prevents the "changing start/end does nothing" bug when force=False.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    # ---------- Load from cache ----------
    if PRICES_PARQUET_PATH.exists() and not force:
        df = pd.read_parquet(PRICES_PARQUET_PATH, engine="pyarrow")
        if "Date" not in df.columns:
            raise RuntimeError(
                f"Cached parquet missing 'Date' column: {PRICES_PARQUET_PATH} "
                f"(columns={list(df.columns)})"
            )
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        if end_dt is None:
            return df.loc[start_dt:]
        return df.loc[start_dt:end_dt]

    # ---------- Download fresh ----------
    tickers = load_sp500_tickers()
    prices = download_adjclose_panel(tickers, start=start, end=end)

    out = prices.copy()
    out.index = pd.to_datetime(out.index)
    out = out.reset_index().rename(columns={"index": "Date"})

    PRICES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRICES_PARQUET_PATH, engine="pyarrow", index=False)

    out_df = out.set_index("Date").sort_index()
    if end_dt is None:
        return out_df.loc[start_dt:]
    return out_df.loc[start_dt:end_dt]


def get_sp500_prices_and_volume(
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      prices(pd.DataFrame)  Adj Close prices with DatetimeIndex
      volume(pd.DataFrame)  Volumes with DatetimeIndex

    Cached separately:
      - PRICES_PARQUET_PATH
      - VOLUME_PARQUET_PATH

    IMPORTANT:
    - Even when using cached parquet, we still slice to the requested [start, end] range.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    have_cache = PRICES_PARQUET_PATH.exists() and VOLUME_PARQUET_PATH.exists()

    # ---------- Load from cache ----------
    if have_cache and not force:
        px = pd.read_parquet(PRICES_PARQUET_PATH, engine="pyarrow")
        vol = pd.read_parquet(VOLUME_PARQUET_PATH, engine="pyarrow")

        if "Date" not in px.columns:
            raise RuntimeError(
                f"Cached prices parquet missing 'Date' column: {PRICES_PARQUET_PATH} "
                f"(columns={list(px.columns)})"
            )
        if "Date" not in vol.columns:
            raise RuntimeError(
                f"Cached volume parquet missing 'Date' column: {VOLUME_PARQUET_PATH} "
                f"(columns={list(vol.columns)})"
            )

        px["Date"] = pd.to_datetime(px["Date"])
        vol["Date"] = pd.to_datetime(vol["Date"])

        px = px.set_index("Date").sort_index()
        vol = vol.set_index("Date").sort_index()

        if end_dt is None:
            return px.loc[start_dt:], vol.loc[start_dt:]
        return px.loc[start_dt:end_dt], vol.loc[start_dt:end_dt]

    # ---------- Download fresh ----------
    tickers = load_sp500_tickers()
    prices, volume = download_adjclose_and_volume_panel(tickers, start=start, end=end)

    # Save cache in the schema we load: a Date column + ticker columns
    out_px = prices.copy()
    out_px.index = pd.to_datetime(out_px.index)
    out_px = out_px.reset_index().rename(columns={"index": "Date"})
    PRICES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_px.to_parquet(PRICES_PARQUET_PATH, engine="pyarrow", index=False)

    out_vol = volume.copy()
    out_vol.index = pd.to_datetime(out_vol.index)
    out_vol = out_vol.reset_index().rename(columns={"index": "Date"})
    VOLUME_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_vol.to_parquet(VOLUME_PARQUET_PATH, engine="pyarrow", index=False)

    px_df = out_px.set_index("Date").sort_index()
    vol_df = out_vol.set_index("Date").sort_index()

    if end_dt is None:
        return px_df.loc[start_dt:], vol_df.loc[start_dt:]
    return px_df.loc[start_dt:end_dt], vol_df.loc[start_dt:end_dt]


# ============================================================
# NEW: OHLCV loader for overnight-gap experiments (SP500)
# ============================================================

def download_ohlcv_panel(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download full daily OHLCV panel from yfinance in a single call.

    Returns a DataFrame with MultiIndex columns:
      level 0: ["Open","High","Low","Close","Adj Close","Volume"]
      level 1: ticker
    """
    if end is None:
        end = date.today().isoformat()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=True,
        threads=True,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned empty data. Network/rate limit?")

    if not isinstance(data.columns, pd.MultiIndex):
        # single ticker case -> make it MultiIndex-like
        t = tickers[0] if tickers else "TICKER"
        data.columns = pd.MultiIndex.from_product([data.columns, [t]])

    lvl0 = set(data.columns.get_level_values(0))
    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = sorted(required - lvl0)
    if missing:
        raise RuntimeError(f"Missing fields from yfinance download: {missing}. Got: {sorted(lvl0)}")

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def get_sp500_ohlcv(
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns OHLCV panels (each DataFrame indexed by date, columns=tickers):

      open_df, high_df, low_df, close_df, adjclose_df, volume_df

    Cached to a dedicated parquet file beside PRICES_PARQUET_PATH:
      <...>/sp500_ohlcv.parquet

    IMPORTANT:
    - Even when loading cache, we slice to requested [start, end].
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    ohlcv_parquet_path = PRICES_PARQUET_PATH.with_name("sp500_ohlcv.parquet")

    # ---------- Load from cache ----------
    if ohlcv_parquet_path.exists() and not force:
        df = pd.read_parquet(ohlcv_parquet_path, engine="pyarrow")
        if "Date" not in df.columns:
            raise RuntimeError(
                f"Cached OHLCV parquet missing 'Date' column: {ohlcv_parquet_path} "
                f"(columns={list(df.columns)[:20]}...)"
            )
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        if end_dt is None:
            df = df.loc[start_dt:]
        else:
            df = df.loc[start_dt:end_dt]

        # columns are flattened as "<FIELD>|<TICKER>"
        def _unpack(field: str) -> pd.DataFrame:
            cols = [c for c in df.columns if c.startswith(field + "|")]
            out = df[cols].copy()
            out.columns = [c.split("|", 1)[1] for c in cols]
            return out

        return (
            _unpack("Open"),
            _unpack("High"),
            _unpack("Low"),
            _unpack("Close"),
            _unpack("Adj Close"),
            _unpack("Volume"),
        )

    # ---------- Download fresh ----------
    tickers = load_sp500_tickers()
    panel = download_ohlcv_panel(tickers, start=start, end=end)

    open_df = panel["Open"].copy()
    high_df = panel["High"].copy()
    low_df = panel["Low"].copy()
    close_df = panel["Close"].copy()
    adj_df = panel["Adj Close"].copy()
    vol_df = panel["Volume"].copy()

    # Build a flat schema for parquet: Date + "Field|Ticker" columns
    out = pd.DataFrame(index=panel.index)
    for field, mat in [
        ("Open", open_df),
        ("High", high_df),
        ("Low", low_df),
        ("Close", close_df),
        ("Adj Close", adj_df),
        ("Volume", vol_df),
    ]:
        mat = mat.copy()
        mat.columns = [f"{field}|{c}" for c in mat.columns]
        out = out.join(mat, how="outer")

    out = out.reset_index().rename(columns={"index": "Date"})
    ohlcv_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ohlcv_parquet_path, engine="pyarrow", index=False)

    out = out.set_index("Date").sort_index()
    if end_dt is None:
        out = out.loc[start_dt:]
    else:
        out = out.loc[start_dt:end_dt]

    def _unpack(field: str) -> pd.DataFrame:
        cols = [c for c in out.columns if c.startswith(field + "|")]
        ret = out[cols].copy()
        ret.columns = [c.split("|", 1)[1] for c in cols]
        return ret

    return (
        _unpack("Open"),
        _unpack("High"),
        _unpack("Low"),
        _unpack("Close"),
        _unpack("Adj Close"),
        _unpack("Volume"),
    )


# ============================================================
# NEW: Benchmark loader (SPY OHLC) with caching
# ============================================================

def get_spy_ohlc(
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download (or load cached) daily SPY OHLC data.

    Returns a DataFrame indexed by Date with columns:
      ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    Cached beside PRICES_PARQUET_PATH as: spy_ohlc.parquet
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    spy_parquet_path = PRICES_PARQUET_PATH.with_name("spy_ohlc.parquet")

    if spy_parquet_path.exists() and not force:
        df = pd.read_parquet(spy_parquet_path, engine="pyarrow")
        if "Date" not in df.columns:
            raise RuntimeError(
                f"Cached SPY parquet missing 'Date' column: {spy_parquet_path} "
                f"(columns={list(df.columns)})"
            )
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        if end_dt is None:
            return df.loc[start_dt:]
        return df.loc[start_dt:end_dt]

    if end is None:
        end = date.today().isoformat()

    data = yf.download(
        tickers="SPY",
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned empty SPY data.")

    # Handle rare MultiIndex case
    if isinstance(data.columns, pd.MultiIndex):
        cols = {}
        for f in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            try:
                cols[f] = data[(f, "SPY")]
            except KeyError:
                t0 = data.columns.get_level_values(1)[0]
                cols[f] = data[(f, t0)]
        df = pd.DataFrame(cols)
    else:
        df = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    out = df.reset_index().rename(columns={"index": "Date"})
    spy_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(spy_parquet_path, engine="pyarrow", index=False)

    df = out.set_index("Date").sort_index()
    if end_dt is None:
        return df.loc[start_dt:]
    return df.loc[start_dt:end_dt]

