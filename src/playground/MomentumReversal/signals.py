from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# Core / original functions (what you had first)
# ============================================================

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def formation_return(log_rets: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return log_rets.rolling(window=window).sum()


def rolling_volatility(log_rets: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling volatility of log returns (cross-sectional, per asset).

    Inputs:
        log_rets(pd.DataFrame)    Daily log returns, indexed by date, columns=tickers
        window(int)               Lookback window for volatility

    Outputs:
        vol(pd.DataFrame)         Rolling std-dev of log returns
    """
    return log_rets.rolling(window=window).std(ddof=1)


def vol_adjusted_formation_signal(
    log_rets: pd.DataFrame,
    formation_window: int = 20,
    vol_window: int = 20,
    vol_floor: float = 1e-8,
) -> pd.DataFrame:
    """
    Volatility-adjusted reversal signal:
        signal = formation_return(log_rets, formation_window) / rolling_volatility(log_rets, vol_window)

    Intuition:
        "How bad was recent performance relative to its typical daily risk?"

    Inputs:
        log_rets(pd.DataFrame)        Daily log returns
        formation_window(int)         Window used to compute trailing cumulative log return
        vol_window(int)               Window used to compute trailing volatility
        vol_floor(float)              Small floor to avoid division by zero / tiny vol blowups

    Outputs:
        signal(pd.DataFrame)          Vol-adjusted ranking signal
    """
    fr = formation_return(log_rets, window=formation_window)
    vol = rolling_volatility(log_rets, window=vol_window)

    # Avoid huge ratios when vol is ~0
    vol_safe = vol.clip(lower=vol_floor)

    return fr / vol_safe


def get_monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    month_ends = index.to_series().groupby(index.to_period("M")).last()
    return pd.DatetimeIndex(month_ends.values)


def get_weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # last trading day of each Friday-based week
    week_ends = index.to_series().groupby(index.to_period("W-FRI")).last()
    return pd.DatetimeIndex(week_ends.values)


def get_daily_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index)


def get_n_day_rebalance_dates(index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    if n <= 1:
        return get_daily_rebalance_dates(index)
    return pd.DatetimeIndex(index[::n])


def decile_long_short_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    weights = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)

    for date in rebalance_dates:
        if date not in signal.index:
            continue

        cross_section = signal.loc[date].dropna()
        weights.loc[date] = 0.0

        if len(cross_section) < n_quantiles:
            continue

        bins = pd.qcut(cross_section, q=n_quantiles, labels=False, duplicates="drop")
        long_names = bins[bins == bins.min()].index
        short_names = bins[bins == bins.max()].index

        if len(long_names) > 0:
            weights.loc[date, long_names] = 1.0 / len(long_names)
        if len(short_names) > 0:
            weights.loc[date, short_names] = -1.0 / len(short_names)

    return weights


def buffered_percentile_long_short_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    entry: float = 0.10,
    exit: float = 0.20,
) -> pd.DataFrame:
    if not (0 < entry < exit < 0.5):
        raise ValueError("Require 0 < entry < exit < 0.5")

    weights = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)

    prev_longs: set[str] = set()
    prev_shorts: set[str] = set()

    for dt in rebalance_dates:
        if dt not in signal.index:
            continue

        cross = signal.loc[dt].dropna()
        weights.loc[dt] = 0.0

        if cross.empty:
            prev_longs, prev_shorts = set(), set()
            continue

        ranks = cross.rank(pct=True, method="average")

        enter_longs = set(ranks[ranks <= entry].index)
        enter_shorts = set(ranks[ranks >= (1.0 - entry)].index)

        still_longs = {t for t in prev_longs if t in ranks.index and ranks[t] <= exit}
        still_shorts = {t for t in prev_shorts if t in ranks.index and ranks[t] >= (1.0 - exit)}

        longs = still_longs | enter_longs
        shorts = still_shorts | enter_shorts

        overlap = longs & shorts
        if overlap:
            longs -= overlap
            shorts -= overlap

        if longs:
            weights.loc[dt, list(longs)] = 1.0 / len(longs)
        if shorts:
            weights.loc[dt, list(shorts)] = -1.0 / len(shorts)

        prev_longs, prev_shorts = longs, shorts

    return weights


def forward_fill_weights(weights: pd.DataFrame) -> pd.DataFrame:
    return weights.ffill().fillna(0.0)


def compute_portfolio_returns(log_rets: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    return (weights.shift(1) * log_rets).sum(axis=1)


def equity_curve_from_log_returns(log_ret: pd.Series, start: float = 1.0) -> pd.Series:
    return start * np.exp(log_ret.fillna(0.0).cumsum())


def perf_metrics_from_log_returns(log_ret: pd.Series, periods_per_year: int = 252) -> dict:
    log_ret = log_ret.dropna()
    mu = log_ret.mean()
    sigma = log_ret.std(ddof=1)

    ann_return = float(np.exp(mu * periods_per_year) - 1.0)
    ann_vol = float(sigma * np.sqrt(periods_per_year))
    sharpe = float((mu / sigma) * np.sqrt(periods_per_year)) if sigma > 0 else np.nan

    equity = equity_curve_from_log_returns(log_ret)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    return {
        "mean_daily_log": float(mu),
        "vol_daily_log": float(sigma),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": float(equity.iloc[-1] - 1.0),
    }


# ============================================================
# Overlapping cohorts (what you had originally, kept together)
# ============================================================

@dataclass
class _Cohort:
    w: pd.Series
    expire_dt: pd.Timestamp


def _single_date_decile_weights(cross_section: pd.Series, n_quantiles: int = 10) -> pd.Series:
    cross = cross_section.dropna()
    if len(cross) < n_quantiles:
        return pd.Series(dtype=float)

    bins = pd.qcut(cross, q=n_quantiles, labels=False, duplicates="drop")
    long_names = bins[bins == bins.min()].index
    short_names = bins[bins == bins.max()].index

    w = pd.Series(0.0, index=cross.index)
    if len(long_names) > 0:
        w.loc[long_names] = 1.0 / len(long_names)
    if len(short_names) > 0:
        w.loc[short_names] = -1.0 / len(short_names)
    return w


def overlapping_cohort_long_short_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    hold_n_rebalances: int = 3,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    if hold_n_rebalances < 1:
        raise ValueError("hold_n_rebalances must be >= 1")

    rebalance_dates = pd.DatetimeIndex(sorted(set(rebalance_dates)))
    weights = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)
    active: deque[_Cohort] = deque()

    for i, dt in enumerate(rebalance_dates):
        if dt not in signal.index:
            continue

        while active and active[0].expire_dt <= dt:
            active.popleft()

        # form new cohort
        cross = signal.loc[dt]
        small = _single_date_decile_weights(cross, n_quantiles=n_quantiles)

        cohort_w = pd.Series(0.0, index=signal.columns)
        if not small.empty:
            cohort_w.loc[small.index] = small.values

        # expiry
        if i + hold_n_rebalances < len(rebalance_dates):
            expire_dt = pd.Timestamp(rebalance_dates[i + hold_n_rebalances])
        else:
            expire_dt = pd.Timestamp(signal.index.max()) + pd.Timedelta(days=1)

        active.append(_Cohort(w=cohort_w, expire_dt=expire_dt))

        # combine
        weights.loc[dt] = 0.0
        if len(active) > 0:
            combo = sum(c.w for c in active) / float(len(active))
            weights.loc[dt] = combo.values

    return weights


def overlapping_cohort_decile_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    hold_n_rebalances: int = 3,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """
    Overlapping cohort version of decile long/short.

    On each rebalance date:
      - form a fresh decile long/short portfolio from the cross-section
      - store it as a "cohort"
      - keep the last `hold_n_rebalances` cohorts
      - target portfolio = average of active cohorts

    This reduces turnover vs full reconstitution because only ~1/hold_n_rebalances
    of the book gets replaced each rebalance.

    Returns:
      weights panel with NaN on non-rebalance days (so forward_fill_weights carries).
    """
    if hold_n_rebalances < 1:
        raise ValueError("hold_n_rebalances must be >= 1")

    # weights on non-rebalance days are NaN (we forward fill later)
    weights = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)

    active_cohorts: list[pd.Series] = []

    for dt in rebalance_dates:
        if dt not in signal.index:
            continue

        cross = signal.loc[dt].dropna()

        # If nothing tradable, close all and reset cohorts
        if cross.empty or len(cross) < n_quantiles:
            weights.loc[dt] = 0.0
            active_cohorts = []
            continue

        # Build the NEW cohort (a Series of weights indexed by tickers)
        bins = pd.qcut(cross, q=n_quantiles, labels=False, duplicates="drop")
        long_names = bins[bins == bins.min()].index
        short_names = bins[bins == bins.max()].index

        cohort = pd.Series(0.0, index=signal.columns, dtype=float)

        if len(long_names) > 0:
            cohort.loc[long_names] = 1.0 / len(long_names)
        if len(short_names) > 0:
            cohort.loc[short_names] = -1.0 / len(short_names)

        # Add and keep only last K cohorts
        active_cohorts.append(cohort)
        if len(active_cohorts) > hold_n_rebalances:
            active_cohorts = active_cohorts[-hold_n_rebalances:]

        # Average active cohorts → target weights on dt
        avg = pd.concat(active_cohorts, axis=1).mean(axis=1)

        # Explicitly set all tickers (close stale names cleanly)
        weights.loc[dt] = 0.0
        weights.loc[dt, avg.index] = avg.values

    return weights


# ============================================================
# NEW: Sector-neutral weighting (everything added later goes here)
# ============================================================

def sector_neutral_decile_long_short_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    sector_map: pd.Series,  # index=ticker, value=sector str
    n_quantiles: int = 10,
    sector_weighting: str = "equal_sector",  # "equal_sector" | "proportional"
) -> pd.DataFrame:
    """
    Decile (or quantile) long/short, built WITHIN each sector.

    sector_weighting:
      - "equal_sector": each sector contributes equally to each side (long/short)
      - "proportional": each name equally weighted across all selected names (like non-sector-neutral)
    """
    if sector_weighting not in {"equal_sector", "proportional"}:
        raise ValueError("sector_weighting must be 'equal_sector' or 'proportional'")

    weights = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)

    sector_map = sector_map.reindex(signal.columns).dropna()

    for dt in rebalance_dates:
        if dt not in signal.index:
            continue

        cross = signal.loc[dt].dropna()
        weights.loc[dt] = 0.0
        if cross.empty:
            continue

        # keep only names with sector
        cross = cross[cross.index.isin(sector_map.index)]
        if cross.empty:
            continue

        longs_by_sector: dict[str, list[str]] = {}
        shorts_by_sector: dict[str, list[str]] = {}
        longs_all: list[str] = []
        shorts_all: list[str] = []

        sec_groups = sector_map.loc[cross.index].groupby(sector_map.loc[cross.index]).groups

        for sec, names in sec_groups.items():
            s = cross.loc[list(names)].dropna()
            if len(s) < n_quantiles:
                continue

            bins = pd.qcut(s, q=n_quantiles, labels=False, duplicates="drop")
            if bins.empty:
                continue

            long_names = bins[bins == bins.min()].index.tolist()
            short_names = bins[bins == bins.max()].index.tolist()

            if long_names:
                longs_by_sector[sec] = long_names
                longs_all.extend(long_names)
            if short_names:
                shorts_by_sector[sec] = short_names
                shorts_all.extend(short_names)

        if not longs_all and not shorts_all:
            continue

        if sector_weighting == "proportional":
            if longs_all:
                weights.loc[dt, longs_all] = 1.0 / len(longs_all)
            if shorts_all:
                weights.loc[dt, shorts_all] = -1.0 / len(shorts_all)
        else:
            if longs_by_sector:
                per_sec_long = 1.0 / len(longs_by_sector)
                for sec, names in longs_by_sector.items():
                    weights.loc[dt, names] = per_sec_long / len(names)

            if shorts_by_sector:
                per_sec_short = 1.0 / len(shorts_by_sector)
                for sec, names in shorts_by_sector.items():
                    weights.loc[dt, names] = -(per_sec_short / len(names))

    return weights


def sector_neutral_buffered_percentile_long_short_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    sector_map: pd.Series,  # index=ticker, value=sector str
    entry: float = 0.10,
    exit: float = 0.20,
    sector_weighting: str = "equal_sector",
) -> pd.DataFrame:
    """
    Buffered percentile long/short, built WITHIN each sector.
    State (prev_longs/prev_shorts) is tracked per sector.
    """
    if not (0 < entry < exit < 0.5):
        raise ValueError("Require 0 < entry < exit < 0.5")
    if sector_weighting not in {"equal_sector", "proportional"}:
        raise ValueError("sector_weighting must be 'equal_sector' or 'proportional'")

    weights = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)

    sector_map = sector_map.reindex(signal.columns).dropna()

    prev_longs: dict[str, set[str]] = {}
    prev_shorts: dict[str, set[str]] = {}

    for dt in rebalance_dates:
        if dt not in signal.index:
            continue

        cross = signal.loc[dt].dropna()
        weights.loc[dt] = 0.0

        if cross.empty:
            prev_longs, prev_shorts = {}, {}
            continue

        # keep only names with sector
        cross = cross[cross.index.isin(sector_map.index)]
        if cross.empty:
            prev_longs, prev_shorts = {}, {}
            continue

        longs_by_sector: dict[str, list[str]] = {}
        shorts_by_sector: dict[str, list[str]] = {}
        longs_all: list[str] = []
        shorts_all: list[str] = []

        sec_groups = sector_map.loc[cross.index].groupby(sector_map.loc[cross.index]).groups

        for sec, names in sec_groups.items():
            s = cross.loc[list(names)].dropna()
            if s.empty:
                continue

            ranks = s.rank(pct=True, method="average")

            pl = prev_longs.get(sec, set())
            ps = prev_shorts.get(sec, set())

            enter_longs = set(ranks[ranks <= entry].index)
            enter_shorts = set(ranks[ranks >= (1.0 - entry)].index)

            still_longs = {t for t in pl if t in ranks.index and ranks[t] <= exit}
            still_shorts = {t for t in ps if t in ranks.index and ranks[t] >= (1.0 - exit)}

            longs = still_longs | enter_longs
            shorts = still_shorts | enter_shorts

            overlap = longs & shorts
            if overlap:
                longs -= overlap
                shorts -= overlap

            prev_longs[sec] = longs
            prev_shorts[sec] = shorts

            if longs:
                ln = list(longs)
                longs_by_sector[sec] = ln
                longs_all.extend(ln)
            if shorts:
                sn = list(shorts)
                shorts_by_sector[sec] = sn
                shorts_all.extend(sn)

        if not longs_all and not shorts_all:
            continue

        if sector_weighting == "proportional":
            if longs_all:
                weights.loc[dt, longs_all] = 1.0 / len(longs_all)
            if shorts_all:
                weights.loc[dt, shorts_all] = -1.0 / len(shorts_all)
        else:
            if longs_by_sector:
                per_sec_long = 1.0 / len(longs_by_sector)
                for sec, names in longs_by_sector.items():
                    weights.loc[dt, names] = per_sec_long / len(names)

            if shorts_by_sector:
                per_sec_short = 1.0 / len(shorts_by_sector)
                for sec, names in shorts_by_sector.items():
                    weights.loc[dt, names] = -(per_sec_short / len(names))

    return weights

