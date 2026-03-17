# src/playground/momo_rvrs/data_loader.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

import pandas as pd
import requests
import yfinance as yf

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber: Any = None


# ============================================================
# Paths / caching
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}
)


# ============================================================
# Universe specs
# ============================================================

@dataclass(frozen=True)
class UniverseSpec:
    code: str
    name: str
    source: str               # constituents source
    yfinance_suffix: str | None = None


UNIVERSES: dict[str, UniverseSpec] = {
    "SPX": UniverseSpec("SPX", "S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    "SX5E": UniverseSpec("SX5E", "EURO STOXX 50", "https://en.wikipedia.org/wiki/EURO_STOXX_50"),
    "FTSE100": UniverseSpec("FTSE100", "FTSE 100", "https://en.wikipedia.org/wiki/FTSE_100_Index", yfinance_suffix=".L"),
    "FTSE250": UniverseSpec("FTSE250", "FTSE 250", "https://en.wikipedia.org/wiki/FTSE_250_Index", yfinance_suffix=".L"),
    "NI225": UniverseSpec("NI225", "Nikkei 225", "https://en.wikipedia.org/wiki/Nikkei_225", yfinance_suffix=".T"),
    "JPX400": UniverseSpec(
        "JPX400",
        "JPX-Nikkei 400",
        "https://www.jpx.co.jp/english/markets/indices/jpx-nikkei400/tvdivq00000031dd-att/400_e.pdf",
        yfinance_suffix=".T",
    ),
    "SSE50": UniverseSpec("SSE50", "SSE 50", "https://en.wikipedia.org/wiki/SSE_50_Index", yfinance_suffix=".SS"),
    "CSI300": UniverseSpec("CSI300", "CSI 300", "https://en.wikipedia.org/wiki/CSI_300_Index"),
}

BENCHMARK_TICKER: dict[str, str] = {
    "SPX": "SPY",
    "SX5E": "^STOXX50E",
    "FTSE100": "^FTSE",
    "FTSE250": "^FTMC",
    "NI225": "^N225",
    "JPX400": "1592.T",
    "SSE50": "000016.SS",
    "CSI300": "000300.SS",
}


# ============================================================
# Small helpers
# ============================================================

def _wikipedia_render_url(url: str) -> str:
    """
    Convert Wikipedia URLs into the stable rendered HTML endpoint:
      https://en.wikipedia.org/wiki/Page_Name
      -> https://en.wikipedia.org/w/index.php?title=Page_Name&action=render
    """
    u = url.strip()

    if "/w/index.php" in u and "title=" in u:
        if "action=render" not in u:
            joiner = "&" if "?" in u else "?"
            u = u + f"{joiner}action=render"
        return u

    if "wikipedia.org/wiki/" in u:
        parsed = urlparse(u)
        title = parsed.path.split("/wiki/")[-1]
        title = unquote(title)
        title_q = quote(title, safe=":/()_-")
        return f"{parsed.scheme}://{parsed.netloc}/w/index.php?title={title_q}&action=render"

    parsed = urlparse(u)
    qs = parse_qs(parsed.query)
    if "title" in qs and len(qs["title"]) > 0:
        title = qs["title"][0]
        title_q = quote(title, safe=":/()_-")
        return f"{parsed.scheme}://{parsed.netloc}/w/index.php?title={title_q}&action=render"

    return u


def _read_html(url: str) -> str:
    target = _wikipedia_render_url(url) if "wikipedia.org" in url else url
    resp = SESSION.get(target, timeout=30)
    resp.raise_for_status()
    return resp.text


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    """
    Robust HTML table reader:
    - downloads via requests SESSION
    - Wikipedia: action=render
    - IMPORTANT: pass HTML via StringIO so pandas does NOT treat it as a filename/URL
      (this was the source of your huge stderr "yap")
    """
    target = _wikipedia_render_url(url) if "wikipedia.org" in url else url

    resp = SESSION.get(
        target,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
    )
    resp.raise_for_status()

    html = resp.text
    try:
        return pd.read_html(StringIO(html))
    except ValueError as e:
        # no tables found (don’t dump HTML)
        raise RuntimeError(f"pd.read_html found no tables for url={target}") from e
    except Exception as e:
        # any parser failures (don’t dump HTML)
        raise RuntimeError(f"pd.read_html failed for url={target}: {type(e).__name__}: {e}") from e

import pandas as pd
import numpy as np

_OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure df is indexed by DatetimeIndex (Date).
    Accepts either:
      - df with Date column
      - df with DatetimeIndex already
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = df.index.name or "Date"
        return df

    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df

    # Sometimes parquet stores index but loses name; if it’s not datetime we can’t guess.
    raise KeyError("No 'Date' column and index is not DatetimeIndex. Cannot infer dates.")


def _split_ohlcv_from_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a wide yfinance-style DataFrame into OHLCV panels.
    Handles MultiIndex columns in either order:
      (Field, Ticker)  OR  (Ticker, Field)
    Returns: open, high, low, close, adjclose, volume (all DataFrames date x ticker)
    """
    df = _ensure_datetime_index(df)

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns for wide OHLCV, got flat columns.")

    lvl0 = set(df.columns.get_level_values(0))
    lvl1 = set(df.columns.get_level_values(1))

    has_fields_lvl0 = set(_OHLCV_FIELDS).issubset(lvl0)
    has_fields_lvl1 = set(_OHLCV_FIELDS).issubset(lvl1)

    if has_fields_lvl0:
        # columns like ("Open","AAPL")
        panels = {f: df[f].copy() for f in _OHLCV_FIELDS}
    elif has_fields_lvl1:
        # columns like ("AAPL","Open")
        panels = {f: df.xs(f, level=1, axis=1).copy() for f in _OHLCV_FIELDS}
    else:
        raise ValueError(f"MultiIndex columns but cannot find OHLCV fields in either level. levels0 sample={list(sorted(lvl0))[:10]} levels1 sample={list(sorted(lvl1))[:10]}")

    # Standardize column names to tickers (strings)
    for k, v in panels.items():
        v.columns = v.columns.astype(str)
        panels[k] = v.sort_index(axis=1)

    o = panels["Open"]
    h = panels["High"]
    l = panels["Low"]
    c = panels["Close"]
    a = panels["Adj Close"]
    v = panels["Volume"]
    return o, h, l, c, a, v


def _chunked(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _normalize_suffix(t: str, suffix: str | None) -> str:
    t = str(t).strip()
    if not t or t.lower() == "nan":
        return ""
    if suffix is None:
        return t
    if "." in t:
        return t
    return f"{t}{suffix}"


# ============================================================
# Constituents fetchers
# ============================================================

def _tickers_spx() -> list[str]:
    tables = _read_html_tables(UNIVERSES["SPX"].source)
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    tickers = df[col].astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))


def _tickers_sx5e() -> list[str]:
    tables = _read_html_tables(UNIVERSES["SX5E"].source)
    for df in tables:
        if any(str(c).lower() == "ticker" for c in df.columns):
            tick_col = [c for c in df.columns if str(c).lower() == "ticker"][0]
            out = df[tick_col].astype(str).str.strip()
            out = out[out.notna() & (out.str.lower() != "nan")]
            return sorted(set(out.tolist()))
    raise RuntimeError("SX5E: could not find a 'Ticker' column on the page")


def _tickers_ftse(code: str) -> list[str]:
    spec = UNIVERSES[code]
    tables = _read_html_tables(spec.source)

    candidate_cols = ["ticker", "ticker symbol", "epic", "symbol", "security", "constituent", "company"]
    best: list[str] = []

    for df in tables:
        cols_l = {str(c).strip().lower(): c for c in df.columns}

        hit = None
        for cand in candidate_cols:
            if cand in cols_l:
                hit = cols_l[cand]
                break
        if hit is None:
            continue

        raw = df[hit].astype(str).str.strip()
        raw = raw[raw.notna() & (raw.str.lower() != "nan")]

        cleaned = []
        for t in raw.tolist():
            t2 = t.split()[0].strip()
            if len(t2) < 1 or len(t2) > 8:
                continue
            cleaned.append(t2)

        tickers = [_normalize_suffix(t, spec.yfinance_suffix) for t in cleaned]
        tickers = [t for t in tickers if t]

        if len(tickers) > len(best):
            best = tickers

    if not best:
        raise RuntimeError(f"{code}: could not find a plausible ticker column.")
    return sorted(set(best))


def _tickers_ni225() -> list[str]:
    html = _read_html(UNIVERSES["NI225"].source)
    codes = sorted(set(re.findall(r"www2\.jpx\.co\.jp[^\"']*?(\d{4})", html)))
    return [f"{c}.T" for c in codes]


def _tickers_jpx400() -> list[str]:
    if pdfplumber is None:
        raise RuntimeError("JPX400 requires pdfplumber. Install: pip install pdfplumber")

    pdf_url = UNIVERSES["JPX400"].source
    pdf_path = DATA_DIR / "jpx400_constituents.pdf"
    r = SESSION.get(pdf_url, timeout=60)
    r.raise_for_status()
    pdf_path.write_bytes(r.content)

    codes: set[str] = set()
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for m in re.finditer(r"(?m)^\s*(\d{4})\s+", text):
                codes.add(m.group(1))

    return [f"{c}.T" for c in sorted(codes)]


def _tickers_sse50() -> list[str]:
    html = _read_html(UNIVERSES["SSE50"].source)
    codes = sorted(set(re.findall(r"english\.sse\.com\.cn[^\"']*?(\d{6})", html)))
    return [f"{c}.SS" for c in codes]


def _tickers_csi300() -> list[str]:
    html = _read_html(UNIVERSES["CSI300"].source)
    sh = set(re.findall(r"english\.sse\.com\.cn[^\"']*?(\d{6})", html))
    sz = set(re.findall(r"szse\.cn[^\"']*?(\d{6})", html))
    return sorted({*(f"{c}.SS" for c in sh), *(f"{c}.SZ" for c in sz)})


def get_universe_tickers(universe: str, force_refresh: bool = False) -> list[str]:
    u = universe.strip().upper()
    if u not in UNIVERSES:
        raise ValueError(f"Unknown universe '{universe}'. Supported: {sorted(UNIVERSES)}")

    cache = DATA_DIR / f"{u.lower()}_tickers.json"
    if cache.exists() and not force_refresh:
        return json.loads(cache.read_text())

    if u == "SPX":
        tickers = _tickers_spx()
    elif u == "SX5E":
        tickers = _tickers_sx5e()
    elif u in {"FTSE100", "FTSE250"}:
        tickers = _tickers_ftse(u)
    elif u == "NI225":
        tickers = _tickers_ni225()
    elif u == "JPX400":
        tickers = _tickers_jpx400()
    elif u == "SSE50":
        tickers = _tickers_sse50()
    elif u == "CSI300":
        tickers = _tickers_csi300()
    else:
        raise RuntimeError(f"No fetcher implemented for {u}")

    cache.write_text(json.dumps(tickers, indent=2))
    return tickers


# ============================================================
# OHLCV download + cache
# ============================================================

_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _stack_yf(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Ensure output columns are MultiIndex (Field, Ticker).
    """
    if raw is None or raw.empty:
        # Return empty frame with expected structure
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0).astype(str)
        lv1 = raw.columns.get_level_values(1).astype(str)

        # If first level looks like tickers (not fields), swap to (Field, Ticker)
        if len(set(lv0) & set(_FIELDS)) == 0 and len(set(lv1) & set(_FIELDS)) > 0:
            raw = raw.swaplevel(0, 1, axis=1)

        raw.columns = pd.MultiIndex.from_tuples([(str(a), str(b)) for a, b in raw.columns])
        return raw

    # single ticker request returns single-level columns
    if len(tickers) != 1:
        raise RuntimeError("Unexpected yfinance output shape (single-level columns but multiple tickers).")

    t = tickers[0]
    out = raw.copy()
    out.columns = pd.MultiIndex.from_product([out.columns.astype(str).tolist(), [t]])
    return out


def _split_ohlcv(df: pd.DataFrame):
    def pick(field: str) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            return pd.DataFrame(index=df.index)
        if field not in df.columns.get_level_values(0):
            return pd.DataFrame(index=df.index)
        out = df[field].copy()
        out.columns = out.columns.astype(str)
        return out.sort_index(axis=1)

    return (
        pick("Open"),
        pick("High"),
        pick("Low"),
        pick("Close"),
        pick("Adj Close"),
        pick("Volume"),
    )


def get_universe_ohlcv(
    universe: str,
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
    chunk_size: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: open, high, low, close, adjclose, volume as (date x tickers) DataFrames.
    Cached per universe: data/<universe>_ohlcv.parquet

    Cache format (normalized):
      - Date in index (DatetimeIndex)
      - Columns are MultiIndex with OHLCV fields in level 0 and tickers in level 1:
          ("Open","AAPL"), ("High","AAPL"), ... etc.
    """
    u = universe.strip().upper()
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    p = DATA_DIR / f"{u.lower()}_ohlcv.parquet"

    # ----------------------------
    # helpers (local)
    # ----------------------------
    OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    def _slice_dates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        if end_dt is None:
            return df.loc[start_dt:]
        return df.loc[start_dt:end_dt]

    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        # Case 1: already datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.copy()
            out.index = pd.to_datetime(out.index)
            out.index.name = out.index.name or "Date"
            return out

        # Case 2: Date column exists
        if "Date" in df.columns:
            out = df.copy()
            out["Date"] = pd.to_datetime(out["Date"])
            out = out.set_index("Date")
            out = out.sort_index()
            return out

        raise KeyError("No 'Date' column and index is not DatetimeIndex.")

    def _normalize_wide_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure columns are MultiIndex (Field, Ticker) with OHLCV_FIELDS present.
        Supports:
          - (Field, Ticker)
          - (Ticker, Field)
        """
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns for wide OHLCV data.")

        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))

        has_fields_lvl0 = set(OHLCV_FIELDS).issubset(lvl0)
        has_fields_lvl1 = set(OHLCV_FIELDS).issubset(lvl1)

        if has_fields_lvl0:
            # already (Field, Ticker)
            out = df.copy()
            # standardize ticker strings
            out.columns = pd.MultiIndex.from_tuples(
                [(str(a), str(b)) for a, b in out.columns.to_list()],
                names=["Field", "Ticker"],
            )
            return out

        if has_fields_lvl1:
            # swap (Ticker, Field) -> (Field, Ticker)
            out = df.copy()
            out.columns = pd.MultiIndex.from_tuples(
                [(str(b), str(a)) for a, b in out.columns.to_list()],
                names=["Field", "Ticker"],
            )
            return out

        raise ValueError(
            "MultiIndex columns detected but cannot locate OHLCV fields "
            f"in either level. lvl0 sample={list(sorted(lvl0))[:10]} "
            f"lvl1 sample={list(sorted(lvl1))[:10]}"
        )

    def _split_ohlcv_from_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        df must be wide with MultiIndex columns normalized to (Field, Ticker) and Date index.
        """
        df = _ensure_datetime_index(df)
        df = _normalize_wide_multiindex(df)

        # Keep only expected fields (some feeds add weird extras)
        keep_cols = [c for c in df.columns if c[0] in OHLCV_FIELDS]
        df = df.loc[:, keep_cols]

        o = df["Open"].copy()
        h = df["High"].copy()
        l = df["Low"].copy()
        c = df["Close"].copy()
        a = df["Adj Close"].copy()
        v = df["Volume"].copy()

        # standardize columns to tickers, sorted
        for panel in (o, h, l, c, a, v):
            panel.columns = panel.columns.astype(str)

        o = o.sort_index(axis=1)
        h = h.sort_index(axis=1)
        l = l.sort_index(axis=1)
        c = c.sort_index(axis=1)
        a = a.sort_index(axis=1)
        v = v.sort_index(axis=1)

        return o, h, l, c, a, v

    # ----------------------------
    # load cache if present
    # ----------------------------
    if p.exists() and not force:
        df = pd.read_parquet(p, engine="pyarrow")

        # parquet may come back as:
        # - wide MultiIndex already (best case)
        # - flat with Date col (older cache)
        # If flat, you *cannot* reliably split without knowing schema,
        # but your project’s intent is wide; so we support both:
        df = _ensure_datetime_index(df)

        if isinstance(df.columns, pd.MultiIndex):
            df = _normalize_wide_multiindex(df)
            df = _slice_dates(df)
            return _split_ohlcv_from_wide(df)

        # If flat columns, try your existing splitter if it's designed for that.
        # BUT: this is exactly where your old Date bug came from.
        # Best move: treat flat-cache as legacy and force rebuild.
        raise RuntimeError(
            f"Cache {p} loaded with flat columns (not MultiIndex). "
            "This looks like a legacy/bad cache. Delete it or call with force=True."
        )

    # ----------------------------
    # fetch from yfinance (build wide MultiIndex)
    # ----------------------------
    tickers = get_universe_tickers(u, force_refresh=False)

    frames: list[pd.DataFrame] = []
    for chunk in _chunked(tickers, chunk_size):
        raw = yf.download(
            tickers=chunk,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d") if end_dt is not None else None,
            group_by="column",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
        )

        # raw from yf can come as:
        # - MultiIndex columns (Field, Ticker) for multiple tickers
        # - single-level columns for one ticker (rare depending on version/inputs)
        # Your _stack_yf() is presumably normalizing; keep using it.
        frames.append(_stack_yf(raw, chunk))

    # Combine as wide
    df = pd.concat(frames, axis=1)
    df = _ensure_datetime_index(df)

    # Guardrails
    df = df.loc[~df.index.duplicated(keep="first")].sort_index()
    df = df.loc[:, ~df.columns.duplicated()]

    # Normalize columns to (Field, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = _normalize_wide_multiindex(df)
    else:
        # If something returned flat columns, that’s not usable for multi-ticker OHLCV.
        raise RuntimeError(
            "yfinance returned flat columns unexpectedly (not MultiIndex). "
            "Check _stack_yf() output; it should produce wide MultiIndex columns."
        )

    # Slice requested range (safe even if yf returned more)
    df = _slice_dates(df)

    # Cache exactly in wide MultiIndex format (no Date column)
    df.to_parquet(p, engine="pyarrow")

    return _split_ohlcv_from_wide(df)


# ============================================================
# Benchmarks
# ============================================================

def _normalize_benchmark_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Force benchmark OHLCV columns to be single-level:
    Open/High/Low/Close/Adj Close/Volume
    Handles yfinance occasionally returning MultiIndex.
    """
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0).astype(str)
        lv1 = df.columns.get_level_values(1).astype(str)

        # if structure is (Ticker, Field), swap it
        if "Close" in set(lv1) and "Close" not in set(lv0):
            df = df.swaplevel(0, 1, axis=1)

        # now expected (Field, Ticker). pick our ticker if present, else first ticker
        if "Close" in df.columns.get_level_values(0):
            tickers = df.columns.get_level_values(1).unique().astype(str).tolist()
            t_use = ticker if ticker in tickers else tickers[0]
            df = df.xs(t_use, axis=1, level=1, drop_level=True)

    # Ensure column names are strings
    df.columns = df.columns.astype(str)
    return df


def get_benchmark_ohlc(
    universe: str,
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Returns OHLC for the benchmark ticker mapped to the universe.
    Cached per universe: data/<universe>_benchmark.parquet
    """
    u = universe.strip().upper()
    if u not in BENCHMARK_TICKER:
        raise ValueError(f"No benchmark configured for universe={u}")

    ticker = BENCHMARK_TICKER[u]
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    p = DATA_DIR / f"{u.lower()}_benchmark.parquet"
    if p.exists() and not force:
        df = pd.read_parquet(p, engine="pyarrow")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = _normalize_benchmark_df(df, ticker)
        return df.loc[start_dt:] if end_dt is None else df.loc[start_dt:end_dt]

    df = yf.download(
        tickers=ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d") if end_dt is not None else None,
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Benchmark download failed for {ticker}")

    df = _normalize_benchmark_df(df, ticker)
    df.to_parquet(p, engine="pyarrow")
    return df


# ============================================================
# Backward-compatible wrappers
# ============================================================

def get_sp500_ohlcv(start: str = "2000-01-01", end: str | None = None, force: bool = False):
    return get_universe_ohlcv("SPX", start=start, end=end, force=force)

def get_spy_ohlc(start: str = "2000-01-01", end: str | None = None, force: bool = False) -> pd.DataFrame:
    return get_benchmark_ohlc("SPX", start=start, end=end, force=force)

