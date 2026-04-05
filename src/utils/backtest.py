"""
Backtesting Module for Long-Short Equity Strategies

This module provides comprehensive backtesting functionality for analyzing
long-short equity strategies, including performance metrics, risk analysis,
and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List


def backtest(
    long_returns: pd.Series,
    short_returns: pd.Series,
    initial_capital: float = 100000.0,
    leverage: float = 1.0
) -> pd.DataFrame:
    """
    Perform backtesting for a long-short strategy.
    
    Parameters
    ----------
    long_returns : pd.Series
        Daily returns of the long portfolio
    short_returns : pd.Series
        Daily returns of the short portfolio
    initial_capital : float, default=100000.0
        Starting capital for the strategy
    leverage : float, default=1.0
        Leverage multiplier (1.0 = no leverage, 2.0 = 2x leverage)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, long_value, short_value, 
        portfolio_value, returns, cumulative_returns
    """
    # Align the series
    long_returns = long_returns.fillna(0)
    short_returns = short_returns.fillna(0)
    
    # Calculate strategy returns (long - short)
    strategy_returns = (long_returns - short_returns) * leverage
    
    # Calculate portfolio value over time
    cumulative_returns = (1 + strategy_returns).cumprod()
    portfolio_value = initial_capital * cumulative_returns
    
    # Calculate individual legs
    long_cumulative = (1 + long_returns).cumprod()
    short_cumulative = (1 + short_returns).cumprod()
    
    long_value = (initial_capital / 2) * long_cumulative * leverage
    short_value = (initial_capital / 2) * short_cumulative * leverage
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': strategy_returns.index,
        'long_value': long_value.values,
        'short_value': short_value.values,
        'portfolio_value': portfolio_value.values,
        'returns': strategy_returns.values,
        'cumulative_returns': cumulative_returns.values
    })
    
    results.set_index('date', inplace=True)
    
    return results


def alpha_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> Tuple[float, float]:
    """
    Compute alpha and beta versus a benchmark using a simple linear regression:
    returns = alpha + beta * benchmark + epsilon.

    Alpha is annualized by multiplying the daily intercept by periods_per_year.
    """
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if aligned.empty:
        return float('nan'), float('nan')
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    # fit y = a + b*x
    b, a = np.polyfit(x, y, 1)
    alpha_daily = a
    alpha_annual = alpha_daily * periods_per_year
    return alpha_annual, b


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns
    periods_per_year : int, default=252
        Number of periods in a year (252 for daily trading days)
    
    Returns
    -------
    float
        Annualized return as a decimal
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    
    if years == 0:
        return 0.0
    
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns
    periods_per_year : int, default=252
        Number of periods in a year
    
    Returns
    -------
    float
        Annualized volatility as a decimal
    """
    return returns.std() * np.sqrt(periods_per_year)


def _infer_periods_per_year(index: pd.Index) -> int:
    """
    Best-effort inference of observation frequency to annualize returns sensibly.
    Falls back to daily (252) if the frequency is unknown.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return 252

    freq = pd.infer_freq(index)
    if freq:
        code = freq.upper()
        if code.startswith(("D", "B")):
            return 252
        if code.startswith("W"):
            return 52
        if code.startswith("M"):
            return 12
        if code.startswith("Q"):
            return 4
        if code.startswith(("A", "Y")):
            return 1

    delta = index.to_series().diff().median()
    if pd.isna(delta):
        return 252
    days = delta / pd.Timedelta(days=1)
    if days <= 3:
        return 252
    if days <= 10:
        return 52
    if days <= 40:
        return 12
    if days <= 120:
        return 4
    return 1


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns
    risk_free_rate : float, default=0.02
        Annual risk-free rate as a decimal
    periods_per_year : int, default=252
        Number of periods in a year
    
    Returns
    -------
    float
        Sharpe ratio
    """
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_vol(returns, periods_per_year)
    
    if ann_vol == 0:
        return 0.0
    
    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns
    
    Returns
    -------
    float
        Maximum drawdown as a decimal (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """
    Calculate the drawdown curve over time.
    
    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns
    
    Returns
    -------
    pd.Series
        Drawdown at each point in time
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown


def summary_stats(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[int] = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive summary statistics.
    
    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns
    risk_free_rate : float, default=0.02
        Annual risk-free rate
    periods_per_year : Optional[int], default=None
        Number of periods in a year. If None, attempt to infer from the index
        (e.g., daily -> 252, weekly -> 52, monthly -> 12).
    benchmark_returns : Optional[pd.Series], default=None
        If provided, compute alpha/beta versus this benchmark.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing various performance metrics
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    total_return = (1 + returns).prod() - 1
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_vol(returns, periods_per_year)
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    max_dd = max_drawdown(returns)
    
    # Additional statistics
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    win_rate = winning_days / len(returns) if len(returns) > 0 else 0
    
    avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
    avg_loss = returns[returns < 0].mean() if losing_days > 0 else 0
    
    # Best and worst days
    best_day = returns.max()
    worst_day = returns.min()
    
    stats = {
        'Total Return': total_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Best Day': best_day,
        'Worst Day': worst_day,
        'Total Days': len(returns)
    }

    if benchmark_returns is not None:
        a, b = alpha_beta(returns, benchmark_returns, periods_per_year)
        stats['Alpha (annual)'] = a
        stats['Beta'] = b
    
    return stats


def plot_backtest_results(
    backtest_results: pd.DataFrame,
    title: str = "Long-Short Strategy Backtest Results"
) -> plt.Figure:
    """
    Create comprehensive visualization of backtest results.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Results from backtest() function
    title : str
        Title for the plot
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Portfolio Value Over Time
    axes[0].plot(backtest_results.index, backtest_results['portfolio_value'], 
                 label='Portfolio Value', color='blue', linewidth=2)
    axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
    axes[0].set_title('Portfolio Value Over Time', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Cumulative Returns
    axes[1].plot(backtest_results.index, 
                 (backtest_results['cumulative_returns'] - 1) * 100,
                 label='Cumulative Returns', color='green', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Cumulative Returns (%)', fontsize=12)
    axes[1].set_title('Cumulative Returns', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Drawdown
    dd = drawdown_curve(backtest_results['returns']) * 100
    axes[2].fill_between(dd.index, dd.values, 0, alpha=0.3, color='red')
    axes[2].plot(dd.index, dd.values, color='darkred', linewidth=1.5)
    axes[2].set_ylabel('Drawdown (%)', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_title('Drawdown', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def compare_strategies(
    strategies: Dict[str, pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    
    Parameters
    ----------
    strategies : Dict[str, pd.Series]
        Dictionary mapping strategy names to their return series
    risk_free_rate : float, default=0.02
        Annual risk-free rate
    periods_per_year : Optional[int], default=None
        Number of periods in a year; inferred from each series if None.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with performance metrics for each strategy
    """
    comparison = {}
    
    for name, returns in strategies.items():
        stats = summary_stats(returns, risk_free_rate, periods_per_year)
        comparison[name] = stats
    
    comparison_df = pd.DataFrame(comparison).T
    
    # Format percentages
    pct_cols = ['Total Return', 'Annualized Return', 'Annualized Volatility', 
                'Max Drawdown', 'Win Rate', 'Avg Win', 'Avg Loss', 
                'Best Day', 'Worst Day']
    
    for col in pct_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col] * 100
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    print("Backtest module loaded successfully!")
    print("Available functions:")
    print("  - backtest()")
    print("  - annualized_return()")
    print("  - annualized_vol()")
    print("  - sharpe_ratio()")
    print("  - max_drawdown()")
    print("  - drawdown_curve()")
    print("  - summary_stats()")
    print("  - plot_backtest_results()")
    print("  - compare_strategies()")
