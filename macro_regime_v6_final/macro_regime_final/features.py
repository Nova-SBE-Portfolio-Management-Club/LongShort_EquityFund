import pandas as pd
import numpy as np

def sector_features(sector_rets_m: pd.DataFrame, spy_rets_m: pd.Series) -> pd.DataFrame:
    """
    Create monthly features for each sector:
      - 3m momentum
      - 6m momentum
      - 12m momentum
      - 6m volatility
      - relative strength vs SPY (3m)
    Output: MultiIndex columns (sector, feature)
    """
    feats = {}
    for s in sector_rets_m.columns:
        r = sector_rets_m[s]
        feats[(s,"mom3")] = r.rolling(3).mean()
        feats[(s,"mom6")] = r.rolling(6).mean()
        feats[(s,"mom12")] = r.rolling(12).mean()
        feats[(s,"vol6")] = r.rolling(6).std()
        feats[(s,"rel3")] = r.rolling(3).mean() - spy_rets_m.rolling(3).mean()

    X = pd.DataFrame(feats)
    X.columns = pd.MultiIndex.from_tuples(X.columns, names=["Sector","Feature"])
    # Do NOT dropna here, or we lose all history before the latest sector (XLC 2018) was born.
    return X

def make_sector_training_set(X_m: pd.DataFrame, sector_rets_q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align: use features available at quarter end T-1 to predict quarter return T.
    So:
      - X at quarter end dates (from monthly) -> X_q
      - label = sector_rets_q shifted -1 (next quarter return)
    """
    # features at quarter end
    X_q = X_m.resample("QE").last().dropna(how="all")
    Y_q_next = sector_rets_q.shift(-1)  # next quarter return

    common = X_q.index.intersection(Y_q_next.index)
    X_q = X_q.loc[common]
    Y_q_next = Y_q_next.loc[common]
    return X_q, Y_q_next

def stock_momentum_12_1(px_m: pd.DataFrame) -> pd.Series:
    """
    12-1 momentum: return from t-12 to t-1 months.
    """
    # px_m is monthly prices for stocks
    r_12 = px_m.pct_change(12)
    r_1 = px_m.pct_change(1)
    mom_12_1 = (1 + r_12) / (1 + r_1) - 1
    return mom_12_1.iloc[-1]


def stock_momentum_12_1_voladj(px_m: pd.DataFrame, vol_window: int = 6) -> pd.Series:
    """
    Volatility-adjusted 12-1 momentum: raw 12-1 return divided by trailing volatility.

    Equivalent to a Sharpe ratio of the recent trend. Ranks stocks by the
    consistency of their momentum, not just its magnitude. Stocks with high
    returns but erratic price paths score lower than stocks with steady trends.

    Theoretical basis: momentum crashes concentrate in high-vol/high-momentum
    names (Barroso & Santa-Clara 2015, Daniel & Moskowitz 2016). Vol-adjusting
    pre-filters noisy signals and preserves clean, persistent trends.

    vol_window: months of trailing returns used to estimate volatility (default 6).
    """
    r_12 = px_m.pct_change(12)
    r_1 = px_m.pct_change(1)
    mom_12_1 = (1 + r_12) / (1 + r_1) - 1

    trailing_vol = px_m.pct_change().rolling(vol_window).std().iloc[-1]
    trailing_vol = trailing_vol.replace(0, np.nan)

    return (mom_12_1.iloc[-1] / trailing_vol).fillna(0.0)
