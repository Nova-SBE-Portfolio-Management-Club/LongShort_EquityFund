"""
macro_data.py
=============
Pulls and prepares the four macro variables that feed the HMM regime engine.

Variables:
    1. HY OAS       — ICE BofA US High Yield OAS (BAMLH0A0HYM2)     [daily, FRED]
    2. Yield Curve  — 10Y minus 2Y Treasury spread (DGS10 - DGS2)   [daily, FRED]
    3. VIX          — CBOE Volatility Index (VIXCLS)                 [daily, FRED]
    4. HY OAS Mom   — 63-day rate of change of HY OAS, z-scored      [computed]
                      Positive = spreads widening (risk-off)
                      Negative = spreads tightening (risk-on)
                      Stationary by construction — no commodity supercycle issue

Z-scoring:
    Expanding window, shift(1) — no lookahead, no data leakage.
    Minimum 252 observations required before a z-score is trusted.

Period conventions (enforced throughout):
    BURN_IN_END   : 2006-12-31  — HMM never trained before this
    INSAMPLE_END  : 2018-12-31  — tune everything here, freeze params
    OOS_START     : 2019-01-01  — first real test, never touch again
    OOS_END       : 2022-12-31  — OOS evaluation window
    ROBUST_START  : 2023-01-01  — unseen robustness check"""

import os
import warnings
import numpy as np
import pandas as pd
from fredapi import Fred

warnings.filterwarnings("ignore")

# ─── Period boundaries ────────────────────────────────────────────────────────
DATA_START    = "2000-01-01"
BURN_IN_END   = "2006-12-31"
INSAMPLE_END  = "2018-12-31"
OOS_START     = "2019-01-01"
OOS_END       = "2022-12-31"
ROBUST_START  = "2023-01-01"

# ─── FRED series ──────────────────────────────────────────────────────────────
FRED_SERIES = {
    "hy_oas"    : "BAMLH0A0HYM2",      # ICE BofA US HY Index OAS (%) — public FRED series, runs to present
    "tsy_10y"   : "DGS10",             # 10Y Treasury yield (%)
    "tsy_2y"    : "DGS2",              # 2Y Treasury yield (%)
    "vix"       : "VIXCLS",            # VIX index
}

# Minimum observations before z-score is considered reliable
MIN_ZSCORE_OBS = 252

# HY OAS momentum window (~3 months of trading days)
# Frozen — do not tune to improve backtest results
MOMENTUM_WINDOW = 63


def _get_fred_client() -> Fred:
    """
    Resolve FRED API key.
    Priority: FRED_API_KEY env var → hardcoded fallback.
    Never commit live keys — set the env var in production.
    """
    key = os.environ.get("FRED_API_KEY", "dae2ef1813e221d9a3f2ba1764b2c8a9")
    if not key:
        raise EnvironmentError(
            "FRED API key not found. Set the FRED_API_KEY environment variable:\n"
            "  Windows PowerShell : $env:FRED_API_KEY = 'your_key'\n"
            "  Mac/Linux          : export FRED_API_KEY='your_key'"
        )
    return Fred(api_key=key)


def _pull_fred_series(fred: Fred, start: str) -> pd.DataFrame:
    """Pull all four FRED series and return as a daily DataFrame."""
    raw = {}
    for name, ticker in FRED_SERIES.items():
        s = fred.get_series(ticker, observation_start=start)
        s.name = name
        raw[name] = s

    df = pd.DataFrame(raw)

    # Yield curve = 10Y minus 2Y
    df["yield_curve"] = df["tsy_10y"] - df["tsy_2y"]
    df = df.drop(columns=["tsy_10y", "tsy_2y"])

    return df


def _compute_hy_oas_mom(df: pd.DataFrame) -> pd.Series:
    """
    Compute 63-day (≈3-month) rate of change of raw HY OAS.

    Positive value = spreads widening = risk-off signal.
    Negative value = spreads tightening = risk-on signal.

    Stationary by construction — avoids the commodity supercycle
    calibration problem of copper/gold (anchored to 2003-2007).
    """
    return df["hy_oas"].pct_change(MOMENTUM_WINDOW).rename("hy_oas_mom")


def _align_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to business day calendar, forward-fill gaps ≤ 3 days.
    Longer gaps (holidays, data outages) stay NaN.
    """
    bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(bdays)
    df = df.ffill(limit=3)
    return df


def _expanding_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score using expanding window with shift(1) lookahead guard.

    Formula:
        z_t = (x_t - mu_{t-1}) / sigma_{t-1}

    where mu and sigma are computed on all observations up to t-1.
    Observations before MIN_ZSCORE_OBS are set to NaN — insufficient
    history means the z-score is unreliable and should not feed the HMM.
    """
    mu    = series.expanding().mean().shift(1)
    sigma = series.expanding().std().shift(1)
    z     = (series - mu) / sigma

    # Mask early observations where expanding window is too short
    obs_count = series.expanding().count().shift(1)
    z = z.where(obs_count >= MIN_ZSCORE_OBS, np.nan)

    return z


def load_macro_features(start: str = DATA_START) -> pd.DataFrame:
    """
    Main entry point. Returns a daily DataFrame of z-scored macro features.

    Columns:
        hy_oas        — z-scored HY spread (higher = more stress)
        yield_curve   — z-scored 10Y-2Y slope (lower/negative = inversion)
        vix           — z-scored VIX (higher = more fear)
        hy_oas_mom    — z-scored 63-day rate of change of HY OAS
                        (positive = spreads widening = risk-off)

    Index: business day DatetimeIndex from ~2000 onward.
    NaN rows: first ~252 business days (burn-in) + momentum warm-up,
              and any days with data outages across any series.
    """
    fred = _get_fred_client()

    # Pull raw FRED series (hy_oas, yield_curve, vix)
    df = _pull_fred_series(fred, start)
    df = _align_to_business_days(df)

    # HY OAS 63-day momentum (rate of change, ~3 months)
    # Positive = spreads widening (risk-off), Negative = tightening (risk-on)
    # Stationary by construction — no commodity supercycle calibration issue
    df["hy_oas_mom"] = _compute_hy_oas_mom(df)

    # Z-score each variable independently
    feature_cols = ["hy_oas", "yield_curve", "vix", "hy_oas_mom"]
    z_df = pd.DataFrame(index=df.index)
    for col in feature_cols:
        z_df[col] = _expanding_zscore(df[col])

    # Drop rows where any feature is NaN
    # (early burn-in + momentum warm-up + occasional data gaps)
    z_df = z_df.dropna()

    # Attach period labels — useful for validation and reporting
    z_df = _attach_period_labels(z_df)

    print(f"[macro_data] Loaded {len(z_df)} daily observations "
          f"({z_df.index[0].date()} -> {z_df.index[-1].date()})")
    _print_coverage_report(z_df)

    return z_df


def _attach_period_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row with its period bucket.
    Used to enforce strict separation in the regime engine.

    Periods:
        burn_in     → before 2007 (HMM never trained here)
        in_sample   → 2007–2018 (build and tune)
        oos         → 2019–2022 (first real test, never tune on this)
        robustness  → 2023+     (unseen data, final check)
    """
    conditions = [
        df.index <= BURN_IN_END,
        (df.index > BURN_IN_END) & (df.index <= INSAMPLE_END),
        (df.index > INSAMPLE_END) & (df.index <= OOS_END),
        df.index > OOS_END,
    ]
    labels = ["burn_in", "in_sample", "oos", "robustness"]
    df["period"] = np.select(conditions, labels, default="unknown")
    return df


def _print_coverage_report(df: pd.DataFrame) -> None:
    """Print a summary of observations per period."""
    counts = df["period"].value_counts().reindex(
        ["burn_in", "in_sample", "oos", "robustness"], fill_value=0
    )
    print("\n[macro_data] Period coverage:")
    for period, count in counts.items():
        print(f"  {period:<12}: {count:>5} days")
    print()


# ─── Convenience accessors ────────────────────────────────────────────────────

def get_feature_cols() -> list[str]:
    """Return the ordered list of feature column names (excludes 'period')."""
    return ["hy_oas", "yield_curve", "vix", "hy_oas_mom"]


def get_insample(df: pd.DataFrame) -> pd.DataFrame:
    """Return only in-sample rows (burn_in excluded)."""
    return df[df["period"] == "in_sample"][get_feature_cols()]


def get_oos(df: pd.DataFrame) -> pd.DataFrame:
    """Return only out-of-sample rows."""
    return df[df["period"] == "oos"][get_feature_cols()]


def get_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Return only robustness-check rows."""
    return df[df["period"] == "robustness"][get_feature_cols()]


if __name__ == "__main__":
    df = load_macro_features()
    print(df.tail(10).to_string())
