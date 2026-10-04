"""Shared periodic-return backtesting and performance statistics.

At gross leverage 1, each period uses 50% long and 50% short exposure.
Inputs are underlying asset/book simple returns, including the short book.
See docs/BACKTESTING.md for alignment, capital, and metric conventions.
"""

from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

__all__ = [
    "backtest",
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "max_drawdown",
    "drawdown_curve",
    "summary_stats",
    "plot_backtest_results",
    "compare_strategies",
]


def _validate_returns(series: pd.Series, name: str = "returns") -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series of simple returns.")
    if series.empty:
        raise ValueError(f"{name} is empty.")
    if isinstance(series.index, pd.MultiIndex):
        raise ValueError(f"{name} must have a single period index.")
    if not series.index.is_unique:
        raise ValueError(f"{name} must have unique period labels; duplicate dates are ambiguous.")
    if series.index.hasnans:
        raise ValueError(f"{name} has missing period labels.")
    if (
        not is_numeric_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
    ):
        raise TypeError(f"{name} must contain real numeric returns.")
    result = series.astype(float).sort_index()
    observed = result.dropna()
    if not np.isfinite(observed).all():
        raise ValueError(f"{name} contains infinite returns.")
    if (observed < -1).any():
        raise ValueError(f"{name} contains a simple return below -100%.")
    return result


def _observed_returns(series: pd.Series) -> pd.Series:
    observed = _validate_returns(series).dropna()
    if observed.empty:
        raise ValueError("returns are empty after removing missing observations.")
    return observed


def _validate_periods(periods_per_year: int) -> None:
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, Integral)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer.")


def backtest(
    long_returns: pd.Series,
    short_returns: pd.Series,
    initial_capital: float = 100_000.0,
    leverage: float = 1.0,
) -> pd.DataFrame:
    """Compound a periodically rebalanced, equal-book long-short portfolio.

    Use sorted common period labels and drop observations missing from either
    book. The short input is the underlying book's return, before shorting.
    Net return is ``leverage * (long_return - short_return) / 2``.
    There are no modeled trading costs, cash interest, or financing charges.

    ``long_value`` and ``short_value`` are positive end-of-period target
    notionals after rebalancing, not separate equity accounts. Their sum is
    gross leverage times equity. Dropped input counts are stored in ``attrs``.
    """
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and greater than zero.")
    if not np.isfinite(leverage) or leverage < 0:
        raise ValueError("leverage must be finite and non-negative.")
    longs = _validate_returns(long_returns, "long_returns")
    shorts = _validate_returns(short_returns, "short_returns")
    aligned = (
        pd.concat([longs.rename("long"), shorts.rename("short")], axis=1, join="inner")
        .dropna()
        .sort_index()
    )
    if aligned.empty:
        raise ValueError("No common periods with valid returns in both books.")

    strategy_returns = (aligned["long"] - aligned["short"]) * (leverage / 2)
    if not np.isfinite(strategy_returns).all() or (strategy_returns < -1).any():
        raise ValueError("Portfolio return exceeds a 100% equity loss or is non-finite.")
    cumulative = (1 + strategy_returns).cumprod()
    equity = initial_capital * cumulative
    if not np.isfinite(equity).all():
        raise ValueError("Compounded portfolio equity is non-finite.")
    result = pd.DataFrame(
        {
            "long_value": equity * leverage / 2,
            "short_value": equity * leverage / 2,
            "portfolio_value": equity,
            "returns": strategy_returns,
            "cumulative_returns": cumulative,
        }
    )
    result.index.name = "date"
    result.attrs.update(
        initial_capital=float(initial_capital),
        gross_leverage=float(leverage),
        alignment="common periods; missing returns dropped",
        dropped_long_periods=len(longs) - len(aligned),
        dropped_short_periods=len(shorts) - len(aligned),
    )
    return result


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualized return (CAGR) over the observed periodic sample."""
    _validate_periods(periods_per_year)
    observed = _observed_returns(returns)
    if (observed == -1).any():
        return -1.0
    return float(np.expm1(np.log1p(observed).mean() * periods_per_year))


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize sample standard deviation; one observation yields NaN."""
    _validate_periods(periods_per_year)
    return float(_observed_returns(returns).std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Annualized arithmetic excess-return Sharpe with a constant cash rate.

    Convert the annual risk-free rate to a compounded per-period rate.
    Zero or undefined sample volatility yields NaN rather than a false zero.
    Square-root annualization assumes comparable, uncorrelated periods.
    """
    _validate_periods(periods_per_year)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -100%.")
    observed = _observed_returns(returns)
    vol = observed.std(ddof=1)
    if not np.isfinite(vol) or np.isclose(vol, 0, atol=1e-15, rtol=0):
        return float("nan")
    periodic_cash = np.expm1(np.log1p(risk_free_rate) / periods_per_year)
    return float((observed.mean() - periodic_cash) / vol * np.sqrt(periods_per_year))


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Drawdown from a running peak that includes initial wealth of one."""
    observed = _observed_returns(returns)
    cumulative = (1 + observed).cumprod()
    peak = cumulative.cummax().clip(lower=1.0)
    return cumulative / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst drawdown as a non-positive decimal, including an initial loss."""
    return float(drawdown_curve(returns).min())


def summary_stats(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Performance metrics on one consistent non-missing sample.

    Percentage metrics are decimals; ``Total Days`` counts observed periods
    even when the input is monthly or quarterly. Empty samples are rejected.
    """
    observed = _observed_returns(returns)
    winning = observed[observed > 0]
    losing = observed[observed < 0]
    return {
        "Total Return": float((1 + observed).prod() - 1),
        "Annualized Return": annualized_return(observed, periods_per_year),
        "Annualized Volatility": annualized_vol(observed, periods_per_year),
        "Sharpe Ratio": sharpe_ratio(observed, risk_free_rate, periods_per_year),
        "Max Drawdown": max_drawdown(observed),
        "Win Rate": float((observed > 0).mean()),
        "Avg Win": float(winning.mean()) if not winning.empty else 0.0,
        "Avg Loss": float(losing.mean()) if not losing.empty else 0.0,
        "Best Day": float(observed.max()),
        "Worst Day": float(observed.min()),
        "Total Days": len(observed),
    }


def plot_backtest_results(
    backtest_results: pd.DataFrame,
    title: str = "Long-Short Strategy Backtest Results",
) -> plt.Figure:
    """Plot equity, compounded returns, and drawdown for a backtest."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    axes[0].plot(backtest_results.index, backtest_results["portfolio_value"], color="#164e63")
    axes[0].set_ylabel("Portfolio equity ($)")
    axes[1].plot(
        backtest_results.index,
        (backtest_results["cumulative_returns"] - 1) * 100,
        color="#0f766e",
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("Cumulative return (%)")
    dd = drawdown_curve(backtest_results["returns"]) * 100
    axes[2].fill_between(dd.index, dd.values, 0, alpha=0.25, color="#b91c1c")
    axes[2].plot(dd.index, dd.values, color="#b91c1c")
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_xlabel("Period")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def compare_strategies(
    strategies: dict[str, pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compare metrics; percentage columns use percent units (10 means 10%)."""
    comparison = pd.DataFrame.from_dict(
        {
            name: summary_stats(returns, risk_free_rate, periods_per_year)
            for name, returns in strategies.items()
        },
        orient="index",
    )
    percentage_columns = [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Max Drawdown",
        "Win Rate",
        "Avg Win",
        "Avg Loss",
        "Best Day",
        "Worst Day",
    ]
    for column in percentage_columns:
        if column in comparison:
            comparison[column] *= 100
    return comparison
