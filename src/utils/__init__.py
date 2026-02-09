"""
Utils Package
"""

import warnings

from .messages import *

try:
    from .corrcoint import cointegration, correlation
except Exception as exc:  # pragma: no cover - optional dependency guard
    def _missing_statsmodels(*args, **kwargs):
        raise ImportError(
            "statsmodels is required for utils.cointegration/utils.correlation"
        ) from exc

    warnings.warn(
        "statsmodels not installed; utils.cointegration/utils.correlation unavailable",
        RuntimeWarning,
    )
    cointegration = correlation = _missing_statsmodels

from .backtest import (
    backtest,
    annualized_return,
    annualized_vol,
    sharpe_ratio,
    max_drawdown,
    drawdown_curve,
    summary_stats,
    plot_backtest_results,
    compare_strategies,
)

try:
    from ..playground.network_effects_pipeline import (
        Config as NEConfig,
        backtest_long_short,
        build_mst,
        compute_log_returns,
        compute_rolling_corr_mst_metrics,
        corr_to_distance,
        evaluate_performance,
        load_price_data,
        run_example,
        save_results,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    def _missing_network_effects(*args, **kwargs):
        raise ImportError(
            "networkx/pandas/yfinance dependencies are required for network effects helpers"
        ) from exc

    warnings.warn(
        "network effects helpers unavailable (check networkx/pandas/yfinance dependencies)",
        RuntimeWarning,
    )
    NEConfig = backtest_long_short = build_mst = compute_log_returns = (
        compute_rolling_corr_mst_metrics
    ) = corr_to_distance = evaluate_performance = load_price_data = run_example = (
        save_results
    ) = _missing_network_effects

try:
    from .intraday_mean_reversion import (
        download_intraday_ohlcv,
        calculate_atr,
        add_intraday_vwap,
        backtest_intraday_mean_reversion,
        summarize_trades,
        run_nvda_example,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    def _missing_intraday(*args, **kwargs):
        raise ImportError(
            "yfinance/pandas are required for intraday_mean_reversion helpers"
        ) from exc

    warnings.warn(
        "intraday mean reversion helpers unavailable (check yfinance/pandas deps)",
        RuntimeWarning,
    )
    (
        download_intraday_ohlcv,
        calculate_atr,
        add_intraday_vwap,
        backtest_intraday_mean_reversion,
        summarize_trades,
        run_nvda_example,
    ) = (_missing_intraday,) * 6

__all__ = [
    'backtest',
    'annualized_return',
    'annualized_vol',
    'sharpe_ratio',
    'max_drawdown',
    'drawdown_curve',
    'summary_stats',
    'plot_backtest_results',
    'compare_strategies',
    'NEConfig',
    'backtest_long_short',
    'build_mst',
    'compute_log_returns',
    'compute_rolling_corr_mst_metrics',
    'corr_to_distance',
    'evaluate_performance',
    'load_price_data',
    'run_example',
    'save_results',
    'download_intraday_ohlcv',
    'calculate_atr',
    'add_intraday_vwap',
    'backtest_intraday_mean_reversion',
    'summarize_trades',
    'run_nvda_example',
]
