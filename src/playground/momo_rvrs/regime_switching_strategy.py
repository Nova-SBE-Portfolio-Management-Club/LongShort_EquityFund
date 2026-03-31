# src/playground/momo_rvrs/regime_switching_strategy.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.playground.momo_rvrs.data_loader import (
    get_benchmark_ohlc,
    load_ftse250_open_close_from_parquet,
    load_ftse250_precomputed_returns,
)
from src.playground.momo_rvrs.locked_strategy import (
    TiltConfig,
    apply_exposure_scaling_on_rebalance,
    apply_total_cost_on_rebalance_days,
    corr_and_beta,
    enforce_dollar_neutral_on_rebalance,
    equity_curve_from_log_returns,
    first_valid_date_from_panels,
    gross_exposures,
    make_lambda_on_rebalance_dates,
    portfolio_log_returns_same_day,
    print_combined_variant,
    rolling_sharpe,
    tilted_buffered_long_short_weights,
    turnover_on_rebalance,
)
from src.playground.momo_rvrs.signals import (
    formation_return,
    forward_fill_weights,
    get_monthly_rebalance_dates,
    perf_metrics_from_log_returns,
    vol_adjusted_formation_signal,
)

# ============================================================
# Paths / config
# ============================================================

OUT_DIR = Path("src/playground/momo_rvrs/regime_switching_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = "FTSE250"
END = None
REBALANCE = "monthly"
VOL_WINDOW_SIGNAL = 20
TC_BPS_PER_1TURN = 25.0
CLOSE_SLIPPAGE_BPS_PER_1TURN = 10.0
TARGET_GROSS = 2.0
MIN_NAMES = 200

# Normal regime candidate from your full-sample grid
NORMAL_CFG = {
    "name": "normal_fw10_buffered3_scale_off",
    "formation_window": 10,
    "long_entry": 0.10,
    "long_exit": 0.30,
    "short_entry": 0.90,
    "short_exit": 0.80,
    "use_exposure_scaling": False,
    "lambda_earn_months": 1.0,
    "lambda_other_months": 1.0,
}

# Crisis regime candidate from your GFC table
CRISIS_CFG = {
    "name": "crisis_fw3_buffered3_scale_off",
    "formation_window": 3,
    "long_entry": 0.10,
    "long_exit": 0.30,
    "short_entry": 0.90,
    "short_exit": 0.80,
    "use_exposure_scaling": False,
    "lambda_earn_months": 1.0,
    "lambda_other_months": 1.0,
}

# Crisis rule
CRISIS_VOL_LOOKBACK = 20
CRISIS_VOL_THRESHOLD = 0.25   # 25% annualized realized vol
CRISIS_DD_THRESHOLD = -0.10   # benchmark drawdown <= -10%


# ============================================================
# Data classes
# ============================================================

@dataclass
class StrategyRun:
    name: str
    gross_log: pd.Series
    net_log: pd.Series
    bench_log: pd.Series
    reb_turn: pd.Series
    weights_on_reb: pd.DataFrame
    regime_on_reb: pd.Series
    metrics_gross: dict
    metrics_net: dict
    corr_to_bench: float
    beta_to_bench: float
    avg_reb_turnover: float
    avg_gross_long: float
    avg_gross_short: float
    avg_net: float
    final_equity_net: float


# ============================================================
# Helpers
# ============================================================

def compute_crisis_mask_from_benchmark(
    bench_close: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    vol_lookback: int = CRISIS_VOL_LOOKBACK,
    vol_threshold: float = CRISIS_VOL_THRESHOLD,
    dd_threshold: float = CRISIS_DD_THRESHOLD,
) -> pd.Series:
    """
    Crisis if rolling realized vol is high OR benchmark drawdown is deep.
    """
    cc_log = np.log(bench_close).diff()

    realized_vol = cc_log.rolling(vol_lookback).std() * np.sqrt(252.0)
    rolling_peak = bench_close.rolling(126, min_periods=20).max()
    drawdown = bench_close / rolling_peak - 1.0

    crisis_daily = (realized_vol >= vol_threshold) | (drawdown <= dd_threshold)
    crisis_daily = crisis_daily.fillna(False)

    crisis_on_reb = crisis_daily.reindex(rebalance_dates).ffill().fillna(False)
    crisis_on_reb = crisis_on_reb.astype(bool)
    crisis_on_reb.name = "is_crisis"
    return crisis_on_reb


def drawdown_curve(eq_curve: pd.Series) -> pd.Series:
    peak = eq_curve.cummax()
    return eq_curve / peak - 1.0


def make_comparison_plots(
    strategy_log: pd.Series,
    bench_log: pd.Series,
    prefix: str,
) -> None:
    common = strategy_log.dropna().index.intersection(bench_log.dropna().index)
    strategy_log = strategy_log.loc[common]
    bench_log = bench_log.loc[common]

    eq_strat = equity_curve_from_log_returns(strategy_log)
    eq_bench = equity_curve_from_log_returns(bench_log)

    dd_strat = drawdown_curve(eq_strat)
    dd_bench = drawdown_curve(eq_bench)

    roll_vol_strat = strategy_log.rolling(63).std() * np.sqrt(252.0)
    roll_vol_bench = bench_log.rolling(63).std() * np.sqrt(252.0)

    roll_sharpe_strat = rolling_sharpe(strategy_log, window=126)
    roll_sharpe_bench = rolling_sharpe(bench_log, window=126)

    monthly_strat = strategy_log.resample("ME").sum().pipe(np.exp) - 1.0
    monthly_bench = bench_log.resample("ME").sum().pipe(np.exp) - 1.0
    monthly_df = pd.DataFrame({"Strategy": monthly_strat, "FTSE250": monthly_bench})
    monthly_df.to_csv(OUT_DIR / f"{prefix}_monthly_returns.csv")

    plt.figure(figsize=(11, 6))
    plt.plot(eq_strat.index, eq_strat.values, label="Strategy")
    plt.plot(eq_bench.index, eq_bench.values, label="FTSE250")
    plt.title(f"Equity Curve: {prefix}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_equity_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(dd_strat.index, dd_strat.values, label="Strategy")
    plt.plot(dd_bench.index, dd_bench.values, label="FTSE250")
    plt.title(f"Drawdown: {prefix}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_drawdown.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(roll_vol_strat.index, roll_vol_strat.values, label="Strategy")
    plt.plot(roll_vol_bench.index, roll_vol_bench.values, label="FTSE250")
    plt.title(f"Rolling Volatility (63d): {prefix}")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_rolling_vol.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(roll_sharpe_strat.index, roll_sharpe_strat.values, label="Strategy")
    plt.plot(roll_sharpe_bench.index, roll_sharpe_bench.values, label="FTSE250")
    plt.title(f"Rolling Sharpe (126d): {prefix}")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_rolling_sharpe.png", dpi=150)
    plt.close()


def make_periods(
    strategy_log: pd.Series,
    bench_log: pd.Series,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    common = strategy_log.dropna().index.intersection(bench_log.dropna().index).sort_values()
    if len(common) == 0:
        return []

    full_start = pd.Timestamp(common.min())
    full_end = pd.Timestamp(common.max())

    raw_periods = [
        ("Full_Sample", full_start, full_end),
        ("Pre_GFC", pd.Timestamp("2002-01-01"), pd.Timestamp("2007-06-30")),
        ("GFC", pd.Timestamp("2007-07-01"), pd.Timestamp("2009-06-30")),
        ("Post_GFC_Recovery", pd.Timestamp("2009-07-01"), pd.Timestamp("2012-12-31")),
        ("QE_Expansion", pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")),
        ("COVID", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
        ("Post_COVID_Inflation", pd.Timestamp("2021-01-01"), pd.Timestamp("2099-12-31")),
    ]

    out: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for name, start, end in raw_periods:
        s = max(start, full_start)
        e = min(end, full_end)
        if s <= e:
            out.append((name, s, e))
    return out


def summarize_by_period(
    strategy_name: str,
    strategy_log: pd.Series,
    bench_log: pd.Series,
) -> pd.DataFrame:
    rows: list[dict] = []

    for period_name, start, end in make_periods(strategy_log, bench_log):
        s_log = strategy_log.loc[start:end].dropna()
        b_log = bench_log.loc[start:end].dropna()
        common = s_log.index.intersection(b_log.index)
        s_log = s_log.loc[common]
        b_log = b_log.loc[common]
        if len(common) < 20:
            continue

        s_m = perf_metrics_from_log_returns(s_log)
        corr, beta = corr_and_beta(s_log, b_log)

        rows.append(
            {
                "strategy": strategy_name,
                "period": period_name,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "n_days": int(len(common)),
                "sharpe": float(s_m.get("sharpe", np.nan)),
                "ann_return": float(s_m.get("ann_return", np.nan)),
                "ann_vol": float(s_m.get("ann_vol", np.nan)),
                "max_drawdown": float(s_m.get("max_drawdown", np.nan)),
                "beta_to_benchmark": float(beta),
                "corr_to_benchmark": float(corr),
                "final_equity": float(equity_curve_from_log_returns(s_log).iloc[-1]),
            }
        )

    return pd.DataFrame(rows)


def print_period_summary(df: pd.DataFrame, strategy_name: str) -> None:
    sub = df.loc[df["strategy"] == strategy_name].copy()
    if sub.empty:
        return

    print("\n" + "=" * 100)
    print(f"ROBUSTNESS SUMMARY | {strategy_name}")
    print("=" * 100)

    name_w = max(20, sub["period"].astype(str).map(len).max())
    for _, row in sub.iterrows():
        print(
            f"{row['period']:>{name_w}} | "
            f"{row['start']} -> {row['end']} | "
            f"n={int(row['n_days']):>4d} | "
            f"Sharpe={row['sharpe']:>6.3f} | "
            f"AnnRet={row['ann_return']:>6.3f} | "
            f"AnnVol={row['ann_vol']:>6.3f} | "
            f"MaxDD={row['max_drawdown']:>7.3f} | "
            f"Beta={row['beta_to_benchmark']:>7.3f}"
        )


def build_signal(
    cc_log: pd.DataFrame,
    formation_window: int,
) -> pd.DataFrame:
    if formation_window == 1:
        return vol_adjusted_formation_signal(
            log_rets=cc_log,
            formation_window=formation_window,
            vol_window=VOL_WINDOW_SIGNAL,
            vol_floor=1e-6,
        )
    return vol_adjusted_formation_signal(
        log_rets=cc_log,
        formation_window=formation_window,
        vol_window=VOL_WINDOW_SIGNAL,
        vol_floor=1e-6,
    )


def make_weights_for_config(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    cfg: dict,
) -> pd.DataFrame:
    weights = tilted_buffered_long_short_weights(
        signal=signal,
        rebalance_dates=rebalance_dates,
        long_entry=cfg["long_entry"],
        long_exit=cfg["long_exit"],
        short_entry=cfg["short_entry"],
        short_exit=cfg["short_exit"],
        min_names=MIN_NAMES,
        ew_within_side=True,
    )

    weights = enforce_dollar_neutral_on_rebalance(weights, target_gross=TARGET_GROSS)

    if cfg["use_exposure_scaling"]:
        lam = make_lambda_on_rebalance_dates(
            rebalance_dates,
            lambda_earn=cfg["lambda_earn_months"],
            lambda_other=cfg["lambda_other_months"],
        )
        weights = apply_exposure_scaling_on_rebalance(weights, lam)

    return weights


def evaluate_strategy(
    name: str,
    gap_log: pd.DataFrame,
    bench_log: pd.Series,
    weights_on_reb: pd.DataFrame,
    regime_on_reb: pd.Series | None = None,
) -> StrategyRun:
    w_daily = forward_fill_weights(weights_on_reb).reindex(gap_log.index).ffill().fillna(0.0)

    gross = portfolio_log_returns_same_day(gap_log, w_daily)
    reb_turn = turnover_on_rebalance(weights_on_reb)

    net = apply_total_cost_on_rebalance_days(
        gross,
        reb_turn,
        tc_bps_per_1turn=TC_BPS_PER_1TURN,
        close_slippage_bps_per_1turn=CLOSE_SLIPPAGE_BPS_PER_1TURN,
    )

    common = net.index.intersection(bench_log.index)
    gross = gross.loc[common].dropna()
    net = net.loc[common].dropna()
    bench = bench_log.loc[common].dropna()

    common2 = gross.index.intersection(net.index).intersection(bench.index)
    gross = gross.loc[common2]
    net = net.loc[common2]
    bench = bench.loc[common2]

    gross_m = perf_metrics_from_log_returns(gross)
    net_m = perf_metrics_from_log_returns(net)
    corr, beta = corr_and_beta(net, bench)

    eq_net = equity_curve_from_log_returns(net)
    gl, gs, n = gross_exposures(w_daily.reindex(common2))

    if regime_on_reb is None:
        regime_on_reb = pd.Series(False, index=weights_on_reb.index, name="is_crisis")

    return StrategyRun(
        name=name,
        gross_log=gross,
        net_log=net,
        bench_log=bench,
        reb_turn=reb_turn,
        weights_on_reb=weights_on_reb,
        regime_on_reb=regime_on_reb,
        metrics_gross=gross_m,
        metrics_net=net_m,
        corr_to_bench=corr,
        beta_to_bench=beta,
        avg_reb_turnover=float(reb_turn.mean()) if len(reb_turn) else np.nan,
        avg_gross_long=float(gl.mean()) if len(gl) else np.nan,
        avg_gross_short=float(gs.mean()) if len(gs) else np.nan,
        avg_net=float(n.mean()) if len(n) else np.nan,
        final_equity_net=float(eq_net.iloc[-1]) if len(eq_net) else np.nan,
    )


def print_run_summary(run: StrategyRun) -> None:
    print(f"\n--- {run.name} ---")
    print(f"Avg TURNOVER per REBALANCE: {run.avg_reb_turnover:.4f}")
    print(f"Avg gross long:            {run.avg_gross_long:.3f}")
    print(f"Avg gross short:           {run.avg_gross_short:.3f}")
    print(f"Avg net:                   {run.avg_net:.3f}")
    print("GROSS:")
    print(f"  Sharpe:    {run.metrics_gross['sharpe']:.4f}")
    print(f"  AnnRet:    {run.metrics_gross['ann_return']:.4f}")
    print(f"  AnnVol:    {run.metrics_gross['ann_vol']:.4f}")
    print(f"  MaxDD:     {run.metrics_gross['max_drawdown']:.4f}")
    print("NET:")
    print(f"  Sharpe:    {run.metrics_net['sharpe']:.4f}")
    print(f"  AnnRet:    {run.metrics_net['ann_return']:.4f}")
    print(f"  AnnVol:    {run.metrics_net['ann_vol']:.4f}")
    print(f"  MaxDD:     {run.metrics_net['max_drawdown']:.4f}")
    print(f"  Final Eq:  {run.final_equity_net:.4f}")
    print("vs Benchmark:")
    print(f"  Corr:      {run.corr_to_bench:.4f}")
    print(f"  Beta:      {run.beta_to_bench:.4f}")


def save_run_timeseries(run: StrategyRun, prefix: str) -> None:
    out = pd.DataFrame(
        {
            "strategy_log": run.net_log,
            "benchmark_log": run.bench_log,
            "strategy_eq": equity_curve_from_log_returns(run.net_log),
            "benchmark_eq": equity_curve_from_log_returns(run.bench_log),
        }
    )
    out.to_csv(OUT_DIR / f"{prefix}_timeseries.csv")


# ============================================================
# Main
# ============================================================

def main() -> None:
    open_px, close_px = load_ftse250_open_close_from_parquet(start="1900-01-01", end=None)
    cc_log, gap_log = load_ftse250_precomputed_returns(start="1900-01-01", end=None)

    requested_start = first_valid_date_from_panels(open_px, close_px).date().isoformat()

    bench = get_benchmark_ohlc(UNIVERSE, start=requested_start, end=END, force=False)
    bench_close = bench["Close"].copy()
    bench_log = np.log(bench_close).diff()

    idx = gap_log.index.intersection(cc_log.index).intersection(bench_close.index)
    gap_log = gap_log.loc[idx]
    cc_log = cc_log.loc[idx]
    bench_close = bench_close.loc[idx]
    bench_log = bench_log.loc[idx]

    rebalance_dates = get_monthly_rebalance_dates(gap_log.index)
    crisis_on_reb = compute_crisis_mask_from_benchmark(
        bench_close=bench_close,
        rebalance_dates=rebalance_dates,
        vol_lookback=CRISIS_VOL_LOOKBACK,
        vol_threshold=CRISIS_VOL_THRESHOLD,
        dd_threshold=CRISIS_DD_THRESHOLD,
    )

    print("\n" + "=" * 100)
    print("REGIME-SWITCHING STRATEGY | FTSE250")
    print("=" * 100)
    print(f"Requested start:        {requested_start}")
    print(f"Actual data start:      {idx.min().date().isoformat()}")
    print(f"Actual data end:        {idx.max().date().isoformat()}")
    print(f"Crisis vol lookback:    {CRISIS_VOL_LOOKBACK}")
    print(f"Crisis vol threshold:   {CRISIS_VOL_THRESHOLD:.2%}")
    print(f"Crisis dd threshold:    {CRISIS_DD_THRESHOLD:.1%}")
    print(f"Crisis rebalances:      {int(crisis_on_reb.sum())}")
    print(f"Normal rebalances:      {int((~crisis_on_reb).sum())}")
    print(f"Normal config:          {NORMAL_CFG['name']}")
    print(f"Crisis config:          {CRISIS_CFG['name']}")

    signal_normal = build_signal(cc_log, NORMAL_CFG["formation_window"])
    signal_crisis = build_signal(cc_log, CRISIS_CFG["formation_window"])

    weights_normal = make_weights_for_config(signal_normal, rebalance_dates, NORMAL_CFG)
    weights_crisis = make_weights_for_config(signal_crisis, rebalance_dates, CRISIS_CFG)

    switched_weights = weights_normal.copy()
    switched_weights.loc[crisis_on_reb[crisis_on_reb].index] = weights_crisis.loc[crisis_on_reb[crisis_on_reb].index]

    run_normal = evaluate_strategy(
        name="static_normal",
        gap_log=gap_log,
        bench_log=bench_log,
        weights_on_reb=weights_normal,
        regime_on_reb=pd.Series(False, index=weights_normal.index, name="is_crisis"),
    )

    run_crisis = evaluate_strategy(
        name="static_crisis",
        gap_log=gap_log,
        bench_log=bench_log,
        weights_on_reb=weights_crisis,
        regime_on_reb=pd.Series(True, index=weights_crisis.index, name="is_crisis"),
    )

    run_switch = evaluate_strategy(
        name="regime_switching",
        gap_log=gap_log,
        bench_log=bench_log,
        weights_on_reb=switched_weights,
        regime_on_reb=crisis_on_reb,
    )

    for run in [run_normal, run_crisis, run_switch]:
        print_run_summary(run)
        save_run_timeseries(run, run.name)
        make_comparison_plots(run.net_log, run.bench_log, run.name)

    summary_rows = []
    period_frames = []

    for run in [run_normal, run_crisis, run_switch]:
        summary_rows.append(
            {
                "strategy": run.name,
                "net_sharpe": float(run.metrics_net.get("sharpe", np.nan)),
                "net_annret": float(run.metrics_net.get("ann_return", np.nan)),
                "net_annvol": float(run.metrics_net.get("ann_vol", np.nan)),
                "net_maxdd": float(run.metrics_net.get("max_drawdown", np.nan)),
                "gross_sharpe": float(run.metrics_gross.get("sharpe", np.nan)),
                "avg_reb_turnover": float(run.avg_reb_turnover),
                "corr_to_bench": float(run.corr_to_bench),
                "beta_to_bench": float(run.beta_to_bench),
                "final_equity_net": float(run.final_equity_net),
            }
        )
        period_frames.append(summarize_by_period(run.name, run.net_log, run.bench_log))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["net_sharpe", "net_annret"], ascending=[False, False]
    ).reset_index(drop=True)
    period_df = pd.concat(period_frames, axis=0, ignore_index=True)

    summary_df.to_csv(OUT_DIR / "regime_switching_summary.csv", index=False)
    period_df.to_csv(OUT_DIR / "regime_switching_by_period.csv", index=False)

    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    for name in summary_df["strategy"]:
        print_period_summary(period_df, name)

    regime_df = pd.DataFrame(
        {
            "rebalance_date": crisis_on_reb.index,
            "is_crisis": crisis_on_reb.values,
        }
    )
    regime_df.to_csv(OUT_DIR / "regime_switching_flags.csv", index=False)

    print("\n" + "=" * 100)
    print("SAVED OUTPUTS")
    print("=" * 100)
    print(f"  {OUT_DIR / 'regime_switching_summary.csv'}")
    print(f"  {OUT_DIR / 'regime_switching_by_period.csv'}")
    print(f"  {OUT_DIR / 'regime_switching_flags.csv'}")
    print(f"  {OUT_DIR / 'static_normal_timeseries.csv'}")
    print(f"  {OUT_DIR / 'static_crisis_timeseries.csv'}")
    print(f"  {OUT_DIR / 'regime_switching_timeseries.csv'}")
    print(f"  {OUT_DIR / 'static_normal_equity_curve.png'}")
    print(f"  {OUT_DIR / 'static_crisis_equity_curve.png'}")
    print(f"  {OUT_DIR / 'regime_switching_equity_curve.png'}")
    print("=" * 100)


if __name__ == "__main__":
    main()

