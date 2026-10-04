"""Compatibility imports for older earnings-surprise notebooks.

All calculations live in src/utils/backtest.py. This module also supports
notebooks opened directly inside the earnings-surprise folder.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parents[2])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.backtest import (  # noqa: E402
    annualized_return,
    annualized_vol,
    backtest,
    compare_strategies,
    drawdown_curve,
    max_drawdown,
    plot_backtest_results,
    sharpe_ratio,
    summary_stats,
)

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
