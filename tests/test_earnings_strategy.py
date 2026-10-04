import numpy as np
import pandas as pd
import pytest

from src.playground.earnings_surprise.best_earnsurp import (
    CONFIG,
    add_entry_exit_asym,
    build_signal_matrix,
    compute_portfolio_returns_long_anchored_50_50,
    compute_sue,
)


def portfolio(signals, min_longs=1):
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    prices = pd.DataFrame({"LONG": [100, 110], "SHORT": [100, 120]}, index=dates)
    sig = pd.DataFrame([signals], columns=prices.columns, index=dates[1:])
    config = dict(CONFIG, min_longs_for_shorts=min_longs, max_weight_long=0.2, max_weight_short=0.2)
    return compute_portfolio_returns_long_anchored_50_50(sig, prices, config)


def test_position_caps_and_short_sign_have_hand_calculated_returns():
    capped, uncapped, long, short, benchmark, _, _, long_exp, short_exp = portfolio([1, -1])
    assert capped.iloc[0] == pytest.approx(-0.02)
    assert uncapped.iloc[0] == pytest.approx(-0.05)
    assert long.iloc[0] == pytest.approx(0.02)
    assert short.iloc[0] == pytest.approx(-0.04)
    assert benchmark.iloc[0] == pytest.approx(0.15)
    assert long_exp.iloc[0] == 0.2
    assert short_exp.iloc[0] == -0.2


def test_shorts_are_suppressed_below_the_minimum_long_count():
    result = portfolio([1, -1], min_longs=3)
    assert result[0].iloc[0] == pytest.approx(0.02)
    assert result[6].iloc[0] == 0
    assert result[8].iloc[0] == 0


def test_short_only_signals_leave_the_strategy_flat():
    result = portfolio([0, -1])
    assert result[0].iloc[0] == 0
    assert result[7].iloc[0] == 0
    assert result[8].iloc[0] == 0


def test_entry_lag_and_exclusive_exit_window():
    event = pd.DataFrame(
        {"Security": ["EXAMPLE"], "AnnouncementDate": pd.to_datetime(["2024-01-05"]), "Signal": [1]}
    )
    config = dict(CONFIG, entry_lag_bdays=1, holding_long_bdays=2, start_date="2024-01-01")
    dated = add_entry_exit_asym(event, config)
    assert dated["Entry_Date"].iloc[0] == pd.Timestamp("2024-01-08")
    assert dated["Exit_Date"].iloc[0] == pd.Timestamp("2024-01-10")
    signal = build_signal_matrix(dated, config)
    assert signal["EXAMPLE"].tolist() == [1, 1, 0]


def test_sue_uses_only_preceding_forecast_errors():
    data = pd.DataFrame(
        {"Security": ["EXAMPLE"] * 6, "EPS_Actual": [1, 2, 3, 4, 10, 20], "EPS_Estimate": [0] * 6}
    )
    config = dict(CONFIG, sue_window_q=8)
    baseline = compute_sue(data, config)
    assert baseline["SUE"].iloc[4] == pytest.approx(10 / np.sqrt(5 / 3))
    data.loc[5, "EPS_Actual"] = 1000
    changed = compute_sue(data, config)
    pd.testing.assert_series_equal(baseline["SUE"].iloc[:5], changed["SUE"].iloc[:5])
