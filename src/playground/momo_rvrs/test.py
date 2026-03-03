# src/playground/MomentumReversal/test.py
from __future__ import annotations

import os
import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.playground.momo_rvrs.data_loader import get_sp500_prices
from src.playground.momo_rvrs.signals import (
    compute_log_returns,
    formation_return,
    get_monthly_rebalance_dates,
    buffered_percentile_long_short_weights,
    forward_fill_weights,
    compute_portfolio_returns,
    perf_metrics_from_log_returns,
)

# ------------------------------------------------------------
# DEBUG: prove which files are actually running / imported
# ------------------------------------------------------------
print("RUNNING TEST.PY:", os.path.abspath(__file__))
import src.playground.momo_rvrs.data_loader as dl  # noqa: E402

print("DATA_LOADER IMPORTED FROM:", inspect.getfile(dl))
print("GET_SP500_PRICES DEFINED IN:", inspect.getfile(dl.get_sp500_prices))

# ============================================================
# CONFIG (edit these only)
# ============================================================

# Use a single fixed strategy (NO re-optimization) so regime comparisons are fair.
METHOD = "buffered_pct"
FORMATION_WINDOW = 1
REBALANCE = "monthly"
ENTRY = 0.05
EXIT = 0.175

# transaction costs (log-return space): 10bps=0.0010, 20bps=0.0020
COST_LEVELS = [0.0, 0.0010, 0.0020]

# Regime test periods (start/end are inclusive-ish; slicing is by trading dates)
REGIMES = [
    # Crashes / stress
    ("GFC Crash", "2008-09-01", "2009-03-31"),
    ("Euro/FlashCrash", "2010-05-01", "2010-09-30"),
    ("COVID Crash", "2020-02-15", "2020-04-30"),
    ("2022 Bear", "2022-01-01", "2022-10-31"),
    # Recoveries / strong trends
    ("Post-GFC Recovery", "2009-04-01", "2010-12-31"),
    ("Post-COVID MeltUp", "2020-05-01", "2021-12-31"),
    ("2023 AI Rally", "2023-01-01", "2023-12-31"),
    # Smooth bull / low vol
    ("2017 LowVol Bull", "2017-01-01", "2017-12-31"),
    # Choppy / mean-reverting-ish
    ("2015-16 Choppy", "2015-01-01", "2016-02-29"),
]

# We’ll fetch once for the full span needed, then slice per regime (much faster).
FETCH_START = min(s for _, s, _ in REGIMES)
FETCH_END = max(e for _, _, e in REGIMES)


# ============================================================
# Helpers
# ============================================================

@dataclass(frozen=True)
class PeriodResult:
    regime: str
    start: str
    end: str
    tc_bps: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_dd: float
    avg_daily_turnover: float
    avg_rebalance_turnover: float
    n_days: int
    n_names: int


def _slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    out = df.loc[start_dt:end_dt].copy()
    return out


def _run_one_period(
    prices_all: pd.DataFrame,
    regime: str,
    start: str,
    end: str,
) -> list[PeriodResult]:
    prices = _slice_df(prices_all, start, end)
    prices = prices.sort_index()

    # If the slice is too short, skip gracefully
    if len(prices) < 50:
        print(f"[SKIP] {regime}: only {len(prices)} rows in slice {start}->{end}")
        return []

    # Basic cleaning: drop tickers that are entirely missing in this slice
    prices = prices.dropna(axis=1, how="all")
    n_names = int(prices.shape[1])

    log_rets = compute_log_returns(prices)
    signal = formation_return(log_rets, window=FORMATION_WINDOW)

    rebalance_dates = get_monthly_rebalance_dates(signal.index)

    # --- weights (rebalance days only) ---
    w_rb = buffered_percentile_long_short_weights(
        signal=signal,
        rebalance_dates=rebalance_dates,
        entry=float(ENTRY),
        exit=float(EXIT),
    )

    # --- carry weights between rebalances ---
    weights = forward_fill_weights(w_rb)

    # --- turnover (one-way) ---
    turnover = 0.5 * weights.diff().abs().sum(axis=1).fillna(0.0)
    avg_daily_turnover = float(turnover.mean())

    rb_mask = weights.index.isin(rebalance_dates)
    avg_rebalance_turnover = float(turnover.loc[rb_mask].mean()) if rb_mask.any() else np.nan

    # --- gross strategy log returns ---
    gross = compute_portfolio_returns(log_rets, weights)

    out: list[PeriodResult] = []
    for tc in COST_LEVELS:
        net = gross - tc * turnover
        perf = perf_metrics_from_log_returns(net)

        out.append(
            PeriodResult(
                regime=regime,
                start=start,
                end=end,
                tc_bps=float(tc * 10000.0),
                ann_return=float(perf["ann_return"]),
                ann_vol=float(perf["ann_vol"]),
                sharpe=float(perf["sharpe"]),
                max_dd=float(perf["max_drawdown"]),
                avg_daily_turnover=avg_daily_turnover,
                avg_rebalance_turnover=float(avg_rebalance_turnover),
                n_days=int(net.dropna().shape[0]),
                n_names=n_names,
            )
        )

    return out


def main() -> None:
    print("\n=== REGIME TEST CONFIG ===")
    print(f"Universe: S&P 500 (Adj Close)")
    print(f"Method: {METHOD}")
    print(f"Formation window: {FORMATION_WINDOW} day(s)")
    print(f"Rebalance: {REBALANCE}")
    print(f"Buffered band: entry={ENTRY:.3f} exit={EXIT:.3f}")
    print(f"TC levels (bps): {[int(tc*10000) for tc in COST_LEVELS]}")
    print(f"Fetch range: {FETCH_START} -> {FETCH_END}\n")

    # ---------- Load once ----------
    prices_all = get_sp500_prices(start=FETCH_START, end=FETCH_END, force=False)
    prices_all.index = pd.to_datetime(prices_all.index)
    prices_all = prices_all.sort_index()

    print("PRICES RANGE:", prices_all.index.min().date(), "->", prices_all.index.max().date())
    print("NROWS:", len(prices_all), "NCOLS:", prices_all.shape[1])

    # ---------- Run all regimes ----------
    results: list[PeriodResult] = []
    for regime, start, end in REGIMES:
        results.extend(_run_one_period(prices_all, regime, start, end))

    if not results:
        print("\nNo results produced (all periods skipped).")
        return

    df = pd.DataFrame([r.__dict__ for r in results])

    # Pretty table: one line per regime per tc
    cols = [
        "regime",
        "start",
        "end",
        "tc_bps",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_dd",
        "avg_daily_turnover",
        "avg_rebalance_turnover",
        "n_days",
        "n_names",
    ]
    df = df[cols].sort_values(["regime", "tc_bps"])

    print("\n=== REGIME RESULTS (fixed strategy; compare apples-to-apples) ===")
    print(df.to_string(index=False))

    # Quick “recovery vs crash” feel: rank by Sharpe for tc=10bps
    df10 = df[df["tc_bps"] == 10.0].copy()
    if not df10.empty:
        print("\n=== RANKED BY SHARPE (tc=10 bps) ===")
        print(
            df10.sort_values("sharpe", ascending=False)[
                ["regime", "start", "end", "sharpe", "ann_return", "ann_vol", "max_dd"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

