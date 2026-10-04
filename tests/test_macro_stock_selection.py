import numpy as np
import pandas as pd
import pytest

from src.playground.macro_regime.stock_model import select_stocks_within_sectors


@pytest.mark.parametrize(
    "history_months,dollar_volume,expected",
    [(24, 10_000_000, ["A"]), (36, 10_000_000, []), (24, 1_000_000, [])],
)
def test_stock_selection_applies_history_and_liquidity_gates(
    history_months, dollar_volume, expected
):
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    prices = pd.DataFrame({"A": np.linspace(100, 150, len(dates))}, index=dates)
    volumes = pd.DataFrame({"A": dollar_volume}, index=dates)
    picks = select_stocks_within_sectors(
        constituents=pd.DataFrame({"Symbol": ["A"], "Sector": ["Technology"]}),
        sector_names_pred=["Technology"],
        stock_prices_m=prices,
        stock_volume_d_m=volumes,
        min_avg_dollar_vol=5_000_000,
        top_n_per_sector=1,
        min_history_months=history_months,
    )
    assert picks == {"Technology": expected}
