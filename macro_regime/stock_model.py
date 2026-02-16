import pandas as pd
import numpy as np
try:
    from .features import stock_momentum_12_1
except ImportError:  # pragma: no cover - supports direct script execution
    from features import stock_momentum_12_1

def select_stocks_within_sectors(
    constituents: pd.DataFrame,
    sector_names_pred: list[str],
    stock_prices_m: pd.DataFrame,
    stock_volume_d_m: pd.DataFrame | None,
    min_avg_dollar_vol: float,
    top_n_per_sector: int,
    min_history_months: int = 24,
) -> dict[str, list[str]]:
    """
    sector_names_pred: e.g. ["Information Technology","Financials",...]
    Returns dict sector_name -> list of tickers
    """
    picks = {}

    # last 6m avg dollar volume filter (optional)
    if stock_volume_d_m is not None:
        dv = stock_volume_d_m.rolling(6).mean().iloc[-1]
    else:
        dv = pd.Series(index=stock_prices_m.columns, data=np.inf)

    for sec in sector_names_pred:
        tickers = constituents.loc[constituents["Sector"] == sec, "Symbol"].tolist()
        tickers = [t for t in tickers if t in stock_prices_m.columns]

        # Slice to recent history only (e.g. last 18 months) to avoid dropping stocks that didn't exist in 2000
        # We need 13 months for momentum.
        px_full = stock_prices_m[tickers]
        valid_len = px_full.notna().sum(axis=0)
        seasoned = valid_len[valid_len >= int(min_history_months)].index.tolist()
        px_full = px_full[seasoned] if seasoned else px_full
        if len(px_full) > 18:
            px_window = px_full.iloc[-18:]
        else:
            px_window = px_full

        # Drop columns that have any NaN in this recent window
        px = px_window.dropna(axis=1, how="any")

        # liquidity filter
        liquid = [t for t in px.columns if float(dv.get(t, 0.0)) >= min_avg_dollar_vol]
        px = px[liquid] if liquid else px

        if px.shape[1] == 0:
            picks[sec] = []
            continue

        mom = stock_momentum_12_1(px)
        mom = mom.sort_values(ascending=False)

        picks[sec] = mom.head(top_n_per_sector).index.tolist()

    return picks
