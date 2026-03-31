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

# Local FTSE250 parquet files live here
LOCAL_FTSE250_DIR = DATA_DIR

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
    source: str
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

_OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _wikipedia_render_url(url: str) -> str:
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
        raise RuntimeError(f"pd.read_html found no tables for url={target}") from e
    except Exception as e:
        raise RuntimeError(f"pd.read_html failed for url={target}: {type(e).__name__}: {e}") from e


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
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

    raise KeyError("No 'Date' column and index is not DatetimeIndex. Cannot infer dates.")


def _split_ohlcv_from_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _ensure_datetime_index(df)

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns for wide OHLCV, got flat columns.")

    lvl0 = set(df.columns.get_level_values(0))
    lvl1 = set(df.columns.get_level_values(1))

    has_fields_lvl0 = set(_OHLCV_FIELDS).issubset(lvl0)
    has_fields_lvl1 = set(_OHLCV_FIELDS).issubset(lvl1)

    if has_fields_lvl0:
        panels = {f: df[f].copy() for f in _OHLCV_FIELDS}
    elif has_fields_lvl1:
        panels = {f: df.xs(f, level=1, axis=1).copy() for f in _OHLCV_FIELDS}
    else:
        raise ValueError(
            f"MultiIndex columns but cannot find OHLCV fields in either level. "
            f"levels0 sample={list(sorted(lvl0))[:10]} levels1 sample={list(sorted(lvl1))[:10]}"
        )

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
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0).astype(str)
        lv1 = raw.columns.get_level_values(1).astype(str)

        if len(set(lv0) & set(_FIELDS)) == 0 and len(set(lv1) & set(_FIELDS)) > 0:
            raw = raw.swaplevel(0, 1, axis=1)

        raw.columns = pd.MultiIndex.from_tuples([(str(a), str(b)) for a, b in raw.columns])
        return raw

    if len(tickers) != 1:
        raise RuntimeError("Unexpected yfinance output shape (single-level columns but multiple tickers).")

    t = tickers[0]
    out = raw.copy()
    out.columns = pd.MultiIndex.from_product([out.columns.astype(str).tolist(), [t]])
    return out


def get_universe_ohlcv(
    universe: str,
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
    chunk_size: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    u = universe.strip().upper()
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    p = DATA_DIR / f"{u.lower()}_ohlcv.parquet"
    OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    def _slice_dates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        if end_dt is None:
            return df.loc[start_dt:]
        return df.loc[start_dt:end_dt]

    def _normalize_wide_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns for wide OHLCV data.")

        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))

        has_fields_lvl0 = set(OHLCV_FIELDS).issubset(lvl0)
        has_fields_lvl1 = set(OHLCV_FIELDS).issubset(lvl1)

        if has_fields_lvl0:
            out = df.copy()
            out.columns = pd.MultiIndex.from_tuples(
                [(str(a), str(b)) for a, b in out.columns.to_list()],
                names=["Field", "Ticker"],
            )
            return out

        if has_fields_lvl1:
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

    def _split_ohlcv_local(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = _ensure_datetime_index(df)
        df = _normalize_wide_multiindex(df)

        keep_cols = [c for c in df.columns if c[0] in OHLCV_FIELDS]
        df = df.loc[:, keep_cols]

        o = df["Open"].copy()
        h = df["High"].copy()
        l = df["Low"].copy()
        c = df["Close"].copy()
        a = df["Adj Close"].copy()
        v = df["Volume"].copy()

        for panel in (o, h, l, c, a, v):
            panel.columns = panel.columns.astype(str)

        o = o.sort_index(axis=1)
        h = h.sort_index(axis=1)
        l = l.sort_index(axis=1)
        c = c.sort_index(axis=1)
        a = a.sort_index(axis=1)
        v = v.sort_index(axis=1)

        return o, h, l, c, a, v

    if p.exists() and not force:
        df = pd.read_parquet(p, engine="pyarrow")
        df = _ensure_datetime_index(df)

        if isinstance(df.columns, pd.MultiIndex):
            df = _normalize_wide_multiindex(df)
            df = _slice_dates(df)
            return _split_ohlcv_local(df)

        raise RuntimeError(
            f"Cache {p} loaded with flat columns (not MultiIndex). "
            "This looks like a legacy/bad cache. Delete it or call with force=True."
        )

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
        frames.append(_stack_yf(raw, chunk))

    df = pd.concat(frames, axis=1)
    df = _ensure_datetime_index(df)
    df = df.loc[~df.index.duplicated(keep="first")].sort_index()
    df = df.loc[:, ~df.columns.duplicated()]

    if isinstance(df.columns, pd.MultiIndex):
        df = _normalize_wide_multiindex(df)
    else:
        raise RuntimeError(
            "yfinance returned flat columns unexpectedly (not MultiIndex). "
            "Check _stack_yf() output; it should produce wide MultiIndex columns."
        )

    df = _slice_dates(df)
    df.to_parquet(p, engine="pyarrow")

    return _split_ohlcv_local(df)


# ============================================================
# Benchmarks
# ============================================================

def _normalize_benchmark_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0).astype(str)
        lv1 = df.columns.get_level_values(1).astype(str)

        if "Close" in set(lv1) and "Close" not in set(lv0):
            df = df.swaplevel(0, 1, axis=1)

        if "Close" in df.columns.get_level_values(0):
            tickers = df.columns.get_level_values(1).unique().astype(str).tolist()
            t_use = ticker if ticker in tickers else tickers[0]
            df = df.xs(t_use, axis=1, level=1, drop_level=True)

    df.columns = df.columns.astype(str)
    return df


def get_benchmark_ohlc(
    universe: str,
    start: str = "2000-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
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
# Local file-based FTSE250 loaders
# ============================================================

def load_ftse250_open_close_from_parquet(
    start: str = "2000-01-01",
    end: str | None = None,
    open_path: str | Path | None = None,
    close_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FTSE250 open/close panels from local parquet files.
    """
    if open_path is None:
        open_path = LOCAL_FTSE250_DIR / "open_prices.parquet"
    if close_path is None:
        close_path = LOCAL_FTSE250_DIR / "close_prices.parquet"

    open_df = pd.read_parquet(open_path)
    close_df = pd.read_parquet(close_path)

    open_df = _ensure_datetime_index(open_df)
    close_df = _ensure_datetime_index(close_df)

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    if end_dt is None:
        open_df = open_df.loc[start_dt:]
        close_df = close_df.loc[start_dt:]
    else:
        open_df = open_df.loc[start_dt:end_dt]
        close_df = close_df.loc[start_dt:end_dt]

    open_df.columns = open_df.columns.astype(str)
    close_df.columns = close_df.columns.astype(str)

    open_df = open_df.sort_index().sort_index(axis=1)
    close_df = close_df.sort_index().sort_index(axis=1)

    return open_df, close_df


def load_ftse250_precomputed_returns(
    start: str = "2000-01-01",
    end: str | None = None,
    log_ret_path: str | Path | None = None,
    gap_ret_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FTSE250 precomputed close-to-close and gap log returns from local parquet files.
    """
    if log_ret_path is None:
        log_ret_path = LOCAL_FTSE250_DIR / "log_returns.parquet"
    if gap_ret_path is None:
        gap_ret_path = LOCAL_FTSE250_DIR / "gap_returns.parquet"

    log_ret = pd.read_parquet(log_ret_path)
    gap_ret = pd.read_parquet(gap_ret_path)

    log_ret = _ensure_datetime_index(log_ret)
    gap_ret = _ensure_datetime_index(gap_ret)

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end is not None else None

    if end_dt is None:
        log_ret = log_ret.loc[start_dt:]
        gap_ret = gap_ret.loc[start_dt:]
    else:
        log_ret = log_ret.loc[start_dt:end_dt]
        gap_ret = gap_ret.loc[start_dt:end_dt]

    log_ret.columns = log_ret.columns.astype(str)
    gap_ret.columns = gap_ret.columns.astype(str)

    log_ret = log_ret.sort_index().sort_index(axis=1)
    gap_ret = gap_ret.sort_index().sort_index(axis=1)

    return log_ret, gap_ret


# ============================================================
# Backward-compatible wrappers
# ============================================================

def get_sp500_ohlcv(start: str = "2000-01-01", end: str | None = None, force: bool = False):
    return get_universe_ohlcv("SPX", start=start, end=end, force=force)


def get_spy_ohlc(start: str = "2000-01-01", end: str | None = None, force: bool = False) -> pd.DataFrame:
    return get_benchmark_ohlc("SPX", start=start, end=end, force=force)

