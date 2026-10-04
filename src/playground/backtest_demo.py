"""Reproducible synthetic demonstration of the shared long-short backtester."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.backtest import (  # noqa: E402
    backtest,
    compare_strategies,
    plot_backtest_results,
)


def generate_sample_data(n_days=252, seed=42):
    """Create synthetic book returns on business dates without global RNG changes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2023-01-02", periods=n_days)
    longs = pd.Series(rng.normal(0.0005, 0.015, n_days), index=dates, name="Long Returns")
    shorts = pd.Series(rng.normal(-0.0003, 0.012, n_days), index=dates, name="Short Returns")
    return longs, shorts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("backtest_results.png"))
    args = parser.parse_args(argv)
    longs, shorts = generate_sample_data()
    result = backtest(longs, shorts)
    print("SYNTHETIC BACKTEST DEMONSTRATION (seed=42; no historical strategy claim)")
    print("1x gross exposure means 50% long and 50% short, rebalanced each period.")
    print(f"Final portfolio equity: ${result['portfolio_value'].iloc[-1]:,.2f}")
    strategies = {
        f"Gross leverage {leverage}x": backtest(longs, shorts, leverage=leverage)["returns"]
        for leverage in (1.0, 1.5, 2.0)
    }
    print(compare_strategies(strategies).round(3).to_string())
    figure = plot_backtest_results(result, title="Synthetic long-short example | 1x gross exposure")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
