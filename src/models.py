from __future__ import annotations
import numpy as np
import pandas as pd


class Ranking:
    """
    Ranks assets cross-sectionally using momentum / volatility.

    Parameters
    ----------
    lookback : int
        Rolling window length in trading days (e.g., 60).
    min_obs : int
        Minimum non-NaN observations required to compute a window value.
    winsor_pct : float | None
        If provided (e.g., 0.01), winsorizes row-wise scores to reduce outliers.
    """

    def __init__(self, lookback: int = 60, min_obs: int = 40, winsor_pct: float | None = None):
        self.lookback = int(lookback)
        self.min_obs = int(min_obs)
        self.winsor_pct = winsor_pct

    def _sanitize_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """ 
        Ensure time order and fill small gaps
        """
        prices = prices.sort_index().ffill().bfill()
        return prices

    def _returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Simple returns (you could use log returns if preferred)
        """
        return prices.pct_change()

    def _momentum(self, rets: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling mean of returns
        """
        return rets.rolling(self.lookback, min_periods=self.min_obs).mean()

    def _volatility(self, rets: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling std of returns
        """
        return rets.rolling(self.lookback, min_periods=self.min_obs).std()

    def score(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum-over-volatility scores per date and asset.
        """
        prices = self._sanitize_prices(prices)
        rets = self._returns(prices)
        mom = self._momentum(rets)
        vol = self._volatility(rets)
        scores = mom / vol.replace(0, np.nan)

        if self.winsor_pct is not None and 0 < self.winsor_pct < 0.5:
            """
            Winsorize row-wise to reduce the impact of extremes
            """
            lower = scores.quantile(self.winsor_pct, axis=1)
            upper = scores.quantile(1 - self.winsor_pct, axis=1)
            scores = scores.clip(lower=lower, upper=upper, axis=0)

        return scores

    def rank(self, prices: pd.DataFrame, ascending: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return scores and the cross-sectional rank for each date.
        1 = best when ascending=False.
        """
        scores = self.score(prices)
        ranks = scores.rank(axis=1, ascending=ascending, method="first")
        return scores, ranks
