import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from utils.backtest import (
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


def returns(values, dates=None):
    if dates is None:
        dates = pd.date_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=pd.to_datetime(dates), dtype=float)


def test_initial_loss_counts_against_initial_capital():
    r = returns([-0.1, 0.0])
    assert max_drawdown(r) == pytest.approx(-0.1)
    assert_series_equal(drawdown_curve(r), returns([-0.1, -0.1]))


def test_drawdown_recovers_and_tracks_a_new_peak():
    r = returns([-0.1, 1 / 9, 0.2, -0.25, 1 / 3])
    np.testing.assert_allclose(drawdown_curve(r), [-0.1, 0, 0, -0.25, 0], atol=1e-14)
    assert max_drawdown(r) == pytest.approx(-0.25)


def test_complete_loss_has_a_finite_drawdown():
    r = returns([-1.0, 0.5])
    np.testing.assert_allclose(drawdown_curve(r), [-1, -1])
    assert annualized_return(r) == -1


def test_default_exposure_is_half_long_and_half_short():
    result = backtest(returns([0.1, 0]), returns([-0.1, 0.1]), initial_capital=100)
    np.testing.assert_allclose(result["returns"], [0.1, -0.05])
    np.testing.assert_allclose(result["portfolio_value"], [110, 104.5])
    np.testing.assert_allclose(result["long_value"], [55, 52.25])
    np.testing.assert_allclose(result["short_value"], [55, 52.25])


def test_leverage_scales_exposure_and_rebalances_each_period():
    result = backtest(returns([0.1]), returns([-0.1]), initial_capital=100, leverage=2)
    assert result["returns"].iloc[0] == pytest.approx(0.2)
    assert result["portfolio_value"].iloc[0] == pytest.approx(120)
    assert result["long_value"].iloc[0] + result["short_value"].iloc[0] == pytest.approx(240)


def test_zero_leverage_keeps_capital_flat():
    result = backtest(returns([0.2, -0.1]), returns([-0.3, 0.4]), leverage=0)
    np.testing.assert_array_equal(result["returns"], [0, 0])
    np.testing.assert_array_equal(result["portfolio_value"], [100_000, 100_000])
    np.testing.assert_array_equal(result[["long_value", "short_value"]], 0)


def test_alignment_uses_common_dates_without_mutating_inputs():
    longs = returns([0.1, 0.2, 0.3], ["2024-01-03", "2024-01-01", "2024-01-02"])
    shorts = returns([0.05, -0.1], ["2024-01-02", "2024-01-03"])
    before = longs.copy(), shorts.copy()
    result = backtest(longs, shorts)
    assert list(result.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    np.testing.assert_allclose(result["returns"], [0.125, 0.1])
    assert result.attrs["dropped_long_periods"] == 1
    assert result.attrs["dropped_short_periods"] == 0
    assert_series_equal(longs, before[0])
    assert_series_equal(shorts, before[1])


def test_missing_common_observations_are_dropped_instead_of_flattened():
    result = backtest(returns([0.1, np.nan, 0.2]), returns([0, 0.5, 0.1]))
    assert len(result) == 2
    np.testing.assert_allclose(result["returns"], [0.05, 0.05])
    assert result.attrs["dropped_long_periods"] == 1
    assert result.attrs["dropped_short_periods"] == 1


@pytest.mark.parametrize(
    "longs,shorts",
    [
        (returns([]), returns([])),
        (returns([0.1], ["2024-01-01"]), returns([0.1], ["2024-01-02"])),
        (returns([np.nan]), returns([0])),
    ],
)
def test_no_usable_overlap_has_a_clear_error(longs, shorts):
    with pytest.raises(ValueError, match="common|empty"):
        backtest(longs, shorts)


@pytest.mark.parametrize("values", [[np.inf], [-np.inf], [-1.01]])
def test_invalid_return_values_are_rejected(values):
    with pytest.raises(ValueError):
        backtest(returns(values), returns([0]))


def test_duplicate_dates_are_rejected():
    longs = returns([0.1, 0.2], ["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="unique|duplicate"):
        backtest(longs, returns([0]))


@pytest.mark.parametrize("capital", [0, -10, np.inf, np.nan])
def test_invalid_capital_is_rejected(capital):
    with pytest.raises(ValueError, match="initial_capital"):
        backtest(returns([0]), returns([0]), initial_capital=capital)


@pytest.mark.parametrize("leverage", [-1, np.inf, np.nan])
def test_invalid_leverage_is_rejected(leverage):
    with pytest.raises(ValueError, match="leverage"):
        backtest(returns([0]), returns([0]), leverage=leverage)


def test_loss_beyond_equity_is_rejected():
    with pytest.raises(ValueError, match="100%|equity"):
        backtest(returns([-1]), returns([1]), leverage=2)


def test_cagr_and_sample_volatility_have_known_answers():
    r = returns([0.1, -0.1])
    assert annualized_return(r, periods_per_year=2) == pytest.approx(-0.01)
    assert annualized_vol(r, periods_per_year=2) == pytest.approx(0.2)


def test_sharpe_uses_arithmetic_excess_returns():
    # 21% annual cash rate over two periods is exactly 10% per period.
    r = returns([0.2, 0.0])
    assert sharpe_ratio(r, risk_free_rate=0.21, periods_per_year=2) == pytest.approx(0)


def test_undefined_volatility_and_sharpe_are_nan():
    assert np.isnan(annualized_vol(returns([0.1])))
    assert np.isnan(sharpe_ratio(returns([0.0, 0.0])))


def test_metrics_use_the_same_non_missing_sample():
    r = returns([0.1, np.nan, -0.1])
    stats = summary_stats(r, periods_per_year=2)
    assert stats["Total Days"] == 2
    assert stats["Win Rate"] == 0.5
    assert stats["Total Return"] == pytest.approx(-0.01)
    assert stats["Max Drawdown"] == pytest.approx(-0.1)


@pytest.mark.parametrize("periods", [0, -1, 2.5])
def test_invalid_annualization_periods_are_rejected(periods):
    with pytest.raises(ValueError, match="periods_per_year"):
        summary_stats(returns([0.1, 0.2]), periods_per_year=periods)


def test_empty_metrics_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        summary_stats(returns([]))


def test_comparison_percentage_units_are_explicit():
    result = compare_strategies({"Example": returns([0.1, -0.1])}, periods_per_year=2)
    assert result.loc["Example", "Total Return"] == pytest.approx(-1)
    assert result.loc["Example", "Max Drawdown"] == pytest.approx(-10)


def test_legacy_earnings_import_reexports_the_shared_implementation():
    shared = importlib.import_module("utils.backtest")
    legacy = importlib.import_module("src.playground.earnings_surprise.backtest")
    for name in shared.__all__:
        assert getattr(legacy, name) is getattr(shared, name)


def test_plot_uses_corrected_drawdown(tmp_path):
    import matplotlib.pyplot as plt

    result = backtest(returns([-0.2, 0]), returns([0, 0]))
    figure = plot_backtest_results(result)
    np.testing.assert_allclose(figure.axes[2].lines[0].get_ydata(), [-10, -10])
    path = tmp_path / "backtest.png"
    figure.savefig(path)
    assert path.stat().st_size > 0
    plt.close(figure)


def test_summary_does_not_change_between_import_paths():
    legacy = importlib.import_module("src.playground.earnings_surprise.backtest")
    expected = compare_strategies({"Example": returns([-0.1, 0.1])})
    assert_frame_equal(legacy.compare_strategies({"Example": returns([-0.1, 0.1])}), expected)
