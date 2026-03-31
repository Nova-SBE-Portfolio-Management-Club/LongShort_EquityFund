from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.playground.momo_rvrs.data_loader import (
    load_ftse250_open_close_from_parquet,
    load_ftse250_precomputed_returns,
    get_benchmark_ohlc,
)
from src.playground.momo_rvrs.signals import (
    vol_adjusted_formation_signal,
    forward_fill_weights,
    get_monthly_rebalance_dates,
    perf_metrics_from_log_returns,
)
from src.playground.momo_rvrs.locked_strategy import (
    tilted_buffered_long_short_weights,
    enforce_dollar_neutral_on_rebalance,
    portfolio_log_returns_same_day,
    turnover_on_rebalance,
    apply_total_cost_on_rebalance_days,
    equity_curve_from_log_returns,
    corr_and_beta,
    first_valid_date_from_panels,
)

# ============================================================
# CONFIG (your best strategy)
# ============================================================

FORMATION_WINDOW = 10

LONG_ENTRY = 0.10
LONG_EXIT = 0.30
SHORT_ENTRY = 0.90
SHORT_EXIT = 0.80

TC_BPS = 25.0
SLIPPAGE_BPS = 10.0

TARGET_GROSS = 2.0
MIN_NAMES = 200

OUT_DIR = Path("src/playground/momo_rvrs/sample_period_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RUN FUNCTION
# ============================================================

def run_sample(start: str, name: str):
    open_px, close_px = load_ftse250_open_close_from_parquet(start=start, end=None)
    cc_log, gap_log = load_ftse250_precomputed_returns(start=start, end=None)

    bench = get_benchmark_ohlc("FTSE250", start=start, end=None, force=False)
    bench_log = np.log(bench["Close"]).diff()

    # Align
    idx = gap_log.index.intersection(cc_log.index).intersection(bench_log.index)
    gap_log = gap_log.loc[idx]
    cc_log = cc_log.loc[idx]
    bench_log = bench_log.loc[idx]

    # Signal
    signal = vol_adjusted_formation_signal(
        log_rets=cc_log,
        formation_window=FORMATION_WINDOW,
        vol_window=20,
        vol_floor=1e-6,
    )

    # Filter dates
    valid = signal.notna().sum(axis=1) >= MIN_NAMES
    signal = signal.loc[valid]
    gap_log = gap_log.loc[valid]

    rebalance_dates = get_monthly_rebalance_dates(signal.index)

    # Weights
    w_on_reb = tilted_buffered_long_short_weights(
        signal=signal,
        rebalance_dates=rebalance_dates,
        long_entry=LONG_ENTRY,
        long_exit=LONG_EXIT,
        short_entry=SHORT_ENTRY,
        short_exit=SHORT_EXIT,
        min_names=MIN_NAMES,
    )

    w_on_reb = enforce_dollar_neutral_on_rebalance(w_on_reb, target_gross=TARGET_GROSS)

    # Daily weights
    w_daily = forward_fill_weights(w_on_reb).reindex(gap_log.index).ffill().fillna(0)

    # Returns
    gross = portfolio_log_returns_same_day(gap_log, w_daily)
    turnover = turnover_on_rebalance(w_on_reb)

    net = apply_total_cost_on_rebalance_days(
        gross,
        turnover,
        tc_bps_per_1turn=TC_BPS,
        close_slippage_bps_per_1turn=SLIPPAGE_BPS,
    )

    # Align final
    idx2 = net.index.intersection(bench_log.index)
    net = net.loc[idx2].dropna()
    bench_log = bench_log.loc[idx2].dropna()

    idx3 = net.index.intersection(bench_log.index)
    net = net.loc[idx3]
    bench_log = bench_log.loc[idx3]

    # Metrics
    m = perf_metrics_from_log_returns(net)
    corr, beta = corr_and_beta(net, bench_log)

    eq = equity_curve_from_log_returns(net)

    return {
        "sample": name,
        "start": start,
        "end": idx3.max().date().isoformat(),
        "n_days": len(idx3),
        "sharpe": m["sharpe"],
        "ann_return": m["ann_return"],
        "ann_vol": m["ann_vol"],
        "max_dd": m["max_drawdown"],
        "beta": beta,
        "corr": corr,
        "final_eq": eq.iloc[-1],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    open_px, close_px = load_ftse250_open_close_from_parquet(start="1900-01-01", end=None)
    full_start = first_valid_date_from_panels(open_px, close_px).date().isoformat()

    samples = [
        ("Full Sample", full_start),
        ("Post-GFC", "2009-07-01"),
        ("Post-QE", "2013-01-01"),
    ]

    results = []

    print("\n" + "=" * 80)
    print("SAMPLE PERIOD COMPARISON")
    print("=" * 80)

    for name, start in samples:
        print(f"Running: {name} ({start} -> today)")
        res = run_sample(start, name)
        results.append(res)

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    df.to_csv(OUT_DIR / "sample_period_comparison.csv", index=False)

    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
