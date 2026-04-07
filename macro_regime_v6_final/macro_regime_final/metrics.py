from __future__ import annotations

import numpy as np
import pandas as pd


def clean_returns(returns) -> pd.Series:
    return pd.Series(returns, dtype=float).dropna()


def total_return(returns) -> float:
    r = clean_returns(returns)
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def cagr(returns, periods: int = 4) -> float:
    r = clean_returns(returns)
    if r.empty:
        return float("nan")
    years = len(r) / periods
    if years <= 0:
        return float("nan")
    total = float((1.0 + r).prod())
    return float(total ** (1.0 / years) - 1.0)


def annualized_volatility(returns, periods: int = 4) -> float:
    r = clean_returns(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))


def annualized_downside_volatility(returns, periods: int = 4) -> float:
    r = clean_returns(returns)
    downside = r[r < 0]
    if len(downside) < 2:
        return float("nan")
    return float(downside.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns, periods: int = 4) -> float:
    ann_ret = cagr(returns, periods=periods)
    ann_vol = annualized_volatility(returns, periods=periods)
    if pd.isna(ann_vol) or ann_vol == 0:
        return float("nan")
    return float(ann_ret / ann_vol)


def sortino_ratio(returns, periods: int = 4) -> float:
    ann_ret = cagr(returns, periods=periods)
    down_vol = annualized_downside_volatility(returns, periods=periods)
    if pd.isna(down_vol) or down_vol == 0:
        return float("nan")
    return float(ann_ret / down_vol)
