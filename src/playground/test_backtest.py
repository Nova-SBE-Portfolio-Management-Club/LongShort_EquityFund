"""
Test/Demo script for the backtesting module

This script demonstrates how to use the backtest module to analyze
a long-short equity strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from utils import backtest, summary_stats, plot_backtest_results, compare_strategies


def generate_sample_data(n_days=252, seed=42):
    """
    Generate sample return data for testing.
    
    Parameters
    ----------
    n_days : int
        Number of trading days
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        (long_returns, short_returns, benchmark_returns) as pd.Series
    """
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate long portfolio returns (slightly positive drift)
    long_returns = pd.Series(
        np.random.normal(0.0005, 0.015, n_days),
        index=dates,
        name='Long Returns'
    )
    
    # Generate short portfolio returns (slightly negative drift)
    short_returns = pd.Series(
        np.random.normal(-0.0003, 0.012, n_days),
        index=dates,
        name='Short Returns'
    )
    
    # Simple benchmark series with modest positive drift
    benchmark_returns = pd.Series(
        np.random.normal(0.0002, 0.01, n_days),
        index=dates,
        name='Benchmark Returns'
    )
    
    return long_returns, short_returns, benchmark_returns


def main():
    """
    Main function to run backtest demonstration.
    """
    print("=" * 70)
    print("LONG-SHORT EQUITY STRATEGY BACKTEST DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Generate sample data
    print("Generating sample data...")
    long_returns, short_returns, benchmark_returns = generate_sample_data(n_days=252)
    print(f"✓ Generated {len(long_returns)} days of return data")
    print()
    
    # Run backtest
    print("Running backtest...")
    initial_capital = 100000.0
    leverage = 1.0
    
    results = backtest(
        long_returns=long_returns,
        short_returns=short_returns,
        initial_capital=initial_capital,
        leverage=leverage
    )
    
    print(f"✓ Backtest completed")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"  Total Return: {(results['cumulative_returns'].iloc[-1] - 1) * 100:.2f}%")
    print()
    
    # Calculate summary statistics
    print("=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    
    stats = summary_stats(results['returns'], risk_free_rate=0.02, benchmark_returns=benchmark_returns)
    
    print(f"\nReturns:")
    print(f"  Total Return:        {stats['Total Return'] * 100:>8.2f}%")
    print(f"  Annualized Return:   {stats['Annualized Return'] * 100:>8.2f}%")
    print(f"  Annualized Volatility: {stats['Annualized Volatility'] * 100:>8.2f}%")
    
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio:        {stats['Sharpe Ratio']:>8.2f}")
    print(f"  Max Drawdown:        {stats['Max Drawdown'] * 100:>8.2f}%")
    if 'Alpha (annual)' in stats and 'Beta' in stats:
        print(f"  Alpha (annual):      {stats['Alpha (annual)'] * 100:>8.2f}%")
        print(f"  Beta:                {stats['Beta']:>8.3f}")
    
    print(f"\nTrading Statistics:")
    print(f"  Total Days:          {stats['Total Days']:>8.0f}")
    print(f"  Win Rate:            {stats['Win Rate'] * 100:>8.2f}%")
    print(f"  Avg Win:             {stats['Avg Win'] * 100:>8.2f}%")
    print(f"  Avg Loss:            {stats['Avg Loss'] * 100:>8.2f}%")
    print(f"  Best Day:            {stats['Best Day'] * 100:>8.2f}%")
    print(f"  Worst Day:           {stats['Worst Day'] * 100:>8.2f}%")
    print()
    
    # Compare with different leverage levels
    print("=" * 70)
    print("LEVERAGE COMPARISON")
    print("=" * 70)
    print()
    
    strategies = {}
    
    for lev in [1.0, 1.5, 2.0]:
        lev_results = backtest(
            long_returns=long_returns,
            short_returns=short_returns,
            initial_capital=initial_capital,
            leverage=lev
        )
        strategies[f"Leverage {lev}x"] = lev_results['returns']
    
    comparison = compare_strategies(strategies)
    print(comparison.to_string())
    print()
    
    # Create visualization
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig = plot_backtest_results(results, title="Long-Short Strategy Backtest (1x Leverage)")
        
        # Save figure
        output_path = "backtest_results.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"⚠ Could not generate visualization: {e}")
    
    print()
    print("=" * 70)
    print("BACKTEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
