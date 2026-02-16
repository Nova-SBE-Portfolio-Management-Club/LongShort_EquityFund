"""
Network-effects demo on a fixed 5-stock basket from Yahoo Finance.

Steps
-----
- Download Adj Close for MSFT, AAPL, GOOGL, META, NVDA (2020 -> end_date).
- Compute log returns.
- Rolling correlation over a lookback window.
- Build Minimum Spanning Tree (MST) from distances (1 - corr).
- Degree centrality drives simple long/short selection.

This is a playground-only script (no cache, no S&P 500 universe).
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yfinance as yf


# ===================== CONFIG ==========================


@dataclass
class Config:
    start_date: str
    end_date: str
    window_size: int = 90  # correlation window in days
    rebalance_freq: str = "ME"  # month end
    metric_for_signal: str = "degree_centrality"
    long_q: float = 0.2
    short_q: float = 0.2


# ============== LOAD PRICE DATA ========================


def load_price_data(cfg: Config) -> pd.DataFrame:
    """
    Fetch Adj Close prices directly from Yahoo Finance for a fixed ticker set.
    """
    tickers = ["MSFT", "AAPL", "GOOGL", "META", "NVDA"]
    data = yf.download(
        tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    prices = data["Adj Close"]
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError("No prices downloaded; check tickers or connectivity.")
    print(f"[INFO] Loaded price matrix: {prices.shape[0]} rows x {prices.shape[1]} tickers.")
    return prices


# ================= RETORNOS / MST / MÉTRICAS ===========


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Computing log returns...")
    rets = np.log(price_df / price_df.shift(1))
    return rets.dropna(how="all")


def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    dist = np.sqrt(2 * (1 - corr))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def build_mst(distance_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    tickers = distance_df.index.tolist()
    for i, ti in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            tj = tickers[j]
            d = distance_df.iat[i, j]
            if not np.isnan(d):
                G.add_edge(ti, tj, weight=d)
    mst = nx.minimum_spanning_tree(G, weight="weight")
    return mst


def compute_rolling_corr_mst_metrics(
    returns_df: pd.DataFrame,
    window_size: int,
    rebalance_freq: str,
) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Rolling correlation -> distance -> MST -> node degree.
    """
    print("[INFO] Computing rolling MST metrics...")
    try:
        freq_alias = rebalance_freq.upper()
        if freq_alias == "M":
            freq_alias = "ME"
        freq = pd.tseries.frequencies.to_offset(freq_alias)
    except ValueError as exc:
        raise ValueError(
            f"Invalid rebalance frequency '{rebalance_freq}'. Use a valid pandas offset alias "
            "(e.g., 'M' for month-end, 'BM' for business month-end)."
        ) from exc

    rebalance_dates = returns_df.resample(freq).last().index

    corr_mats: Dict[pd.Timestamp, pd.DataFrame] = {}
    degree_rows = []

    for date in rebalance_dates:
        idx = returns_df.index.searchsorted(date, side="right") - 1
        if idx < 0:
            continue
        start = idx - window_size + 1
        if start < 0:
            continue

        window = returns_df.iloc[start : idx + 1]
        if window.isna().all().all():
            continue

        corr = window.corr()
        corr_mats[date] = corr

        dist = corr_to_distance(corr)
        mst = build_mst(dist)
        deg = nx.degree_centrality(mst)

        row = pd.Series(deg, name=date)
        degree_rows.append(row)

    if not degree_rows:
        raise RuntimeError("No MST metrics computed. Check window_size and data coverage.")

    degree_df = pd.DataFrame(degree_rows).sort_index()
    node_metrics = {"degree_centrality": degree_df}
    print(f"[INFO] Node metrics computed for {len(degree_df)} rebalance dates.")
    return corr_mats, node_metrics


# ====================== BACKTEST =======================


def backtest_long_short(
    returns_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    long_q: float,
    short_q: float,
) -> pd.Series:
    """
    Long: lowest centrality quantile (periphery).
    Short: highest centrality quantile (hubs).
    """
    print("[INFO] Backtesting strategy...")
    common = metric_df.index.intersection(returns_df.index)
    metric_df = metric_df.loc[common]
    returns_df = returns_df.loc[common]

    strategy = []

    for i in range(len(metric_df) - 1):
        d0 = metric_df.index[i]
        d1 = metric_df.index[i + 1]

        v = metric_df.iloc[i].dropna()
        if v.empty:
            continue

        lo = v.quantile(long_q)
        hi = v.quantile(1 - short_q)

        longs = v[v <= lo].index
        shorts = v[v >= hi].index

        if len(longs) == 0 or len(shorts) == 0:
            continue

        period = returns_df.loc[d0:d1].iloc[1:]
        if period.empty:
            continue

        cum_rets = period.add(1).prod() - 1
        long_ret = cum_rets[longs].mean()
        short_ret = cum_rets[shorts].mean()

        strat_ret = 0.5 * long_ret - 0.5 * short_ret
        strategy.append((d1, strat_ret))

    if not strategy:
        raise RuntimeError("No strategy returns computed. Check data and quantiles.")

    return pd.Series({d: r for d, r in strategy}, name="strategy_return").sort_index()


# ==================== PERFORMANCE ======================


def evaluate_performance(strategy_returns: pd.Series) -> Dict[str, float]:
    print("[INFO] Evaluating performance...")
    m = strategy_returns.mean()
    vol = strategy_returns.std()

    ann_ret = (1 + m) ** 12 - 1
    ann_vol = vol * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    curve = (1 + strategy_returns).cumprod()
    peak = curve.cummax()
    dd = (curve / peak) - 1
    max_dd = dd.min()

    return {
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def save_results(strategy_returns: pd.Series, perf: Dict[str, float], results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    sr_path = os.path.join(results_dir, "demo_strategy_returns.csv")
    pf_path = os.path.join(results_dir, "demo_performance_summary.csv")
    strategy_returns.to_csv(sr_path)
    pd.DataFrame([perf]).to_csv(pf_path, index=False)
    print(f"[INFO] Saved strategy returns to {sr_path}")
    print(f"[INFO] Saved performance summary to {pf_path}")


# ======================= MAIN =========================


def run_example(cfg: Config):
    print("[INFO] Running network-effects demo on: MSFT, AAPL, GOOGL, META, NVDA")

    price_df = load_price_data(cfg)
    returns_df = compute_log_returns(price_df)

    _, node_metrics = compute_rolling_corr_mst_metrics(
        returns_df=returns_df,
        window_size=cfg.window_size,
        rebalance_freq=cfg.rebalance_freq,
    )

    metric_df = node_metrics[cfg.metric_for_signal]

    strat_rets = backtest_long_short(
        returns_df=returns_df,
        metric_df=metric_df,
        long_q=cfg.long_q,
        short_q=cfg.short_q,
    )

    perf = evaluate_performance(strat_rets)

    print("\n====== DEMO STRATEGY PERFORMANCE ======")
    for k, v in perf.items():
        print(f"{k}: {v:.4f}")
    print("======================================\n")

    save_results(strat_rets, perf, results_dir="results")


if __name__ == "__main__":
    cfg = Config(
        start_date="2020-01-01",
        end_date=dt.date.today().strftime("%Y-%m-%d"),
    )
    run_example(cfg)
