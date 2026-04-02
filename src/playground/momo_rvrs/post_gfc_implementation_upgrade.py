from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.playground.momo_rvrs.data_loader import (
    get_benchmark_ohlc,
    load_ftse250_open_close_from_parquet,
    load_ftse250_precomputed_returns,
)
from src.playground.momo_rvrs.locked_strategy import (
    corr_and_beta,
    drawdown_curve,
    enforce_dollar_neutral_on_rebalance,
    equity_curve_from_log_returns,
    gross_exposures,
    portfolio_log_returns_same_day,
)
from src.playground.momo_rvrs.signals import (
    forward_fill_weights,
    get_monthly_rebalance_dates,
    perf_metrics_from_log_returns,
    vol_adjusted_formation_signal,
)

# ============================================================
# Config
# ============================================================

OUT_DIR = Path("src/playground/momo_rvrs/post_gfc_implementation_upgrade_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = "FTSE250"
START = "2009-07-01"
END = None

# Core signal locked from best post-GFC version
FORMATION_WINDOW = 10
VOL_WINDOW = 20
TARGET_GROSS = 2.0

# Separate minimums
MIN_NAMES_BASE = 200
MIN_NAMES_UPG = 125

# Baseline locked thresholds
BASE_LONG_ENTRY = 0.10
BASE_LONG_EXIT = 0.30
BASE_SHORT_ENTRY = 0.90
BASE_SHORT_EXIT = 0.80

# Turnover-reduction version: wider exits
UPG_LONG_ENTRY = 0.10
UPG_LONG_EXIT = 0.35
UPG_SHORT_ENTRY = 0.90
UPG_SHORT_EXIT = 0.75

# Liquidity filter
ADV_LOOKBACK = 60
TOP_LIQUID_NAMES = 175

# Cost model
BASE_TC_BPS = 10.0
BASE_SLIPPAGE_BPS = 10.0
LIQUIDITY_SLIPPAGE_MULTIPLIER_BPS = 25.0  # extra bps scaled by illiquidity rank

# Optional score-strength gate
USE_SIGNAL_STRENGTH_GATE = True
SIGNAL_GATE_LONG = 0.12
SIGNAL_GATE_SHORT = 0.88

# Exposure scaling
USE_EXPOSURE_SCALING = True
LAMBDA_EARN_MONTHS = 1.5
LAMBDA_OTHER_MONTHS = 0.2
EARN_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}


# ============================================================
# Data classes
# ============================================================

@dataclass
class StrategyRun:
    name: str
    gross_log: pd.Series
    net_log: pd.Series
    bench_log: pd.Series
    weights_on_reb: pd.DataFrame
    reb_turn: pd.Series
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

def prepare_data():
    open_px, close_px = load_ftse250_open_close_from_parquet(start=START, end=END)
    close_log, gap_log = load_ftse250_precomputed_returns(start=START, end=END)

    bench = get_benchmark_ohlc(UNIVERSE, start=START, end=END, force=False)
    bench_log = np.log(bench["Close"]).diff()

    idx = gap_log.index.intersection(close_log.index).intersection(bench_log.index)
    open_px = open_px.loc[idx]
    close_px = close_px.loc[idx]
    close_log = close_log.loc[idx]
    gap_log = gap_log.loc[idx]
    bench_log = bench_log.loc[idx]

    return open_px, close_px, close_log, gap_log, bench_log


def compute_adv_panel(close_px: pd.DataFrame) -> pd.DataFrame:
    """
    Rough liquidity proxy using rolling mean close price.
    Not true ADV, but usable as a cross-sectional liquidity ranking proxy
    when actual volume is unavailable in the local FTSE parquet setup.
    """
    return close_px.rolling(ADV_LOOKBACK, min_periods=20).mean()


def make_liquidity_mask(adv_panel: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, top_n: int) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=rebalance_dates, columns=adv_panel.columns)

    for dt in rebalance_dates:
        if dt not in adv_panel.index:
            continue
        row = adv_panel.loc[dt].dropna().sort_values(ascending=False)
        keep = row.index[:top_n]
        mask.loc[dt, keep] = True

    return mask


def row_percentiles(row: pd.Series) -> pd.Series:
    x = row.dropna()
    if len(x) < 2:
        return pd.Series(np.nan, index=row.index)
    r = x.rank(method="average", pct=True)
    out = pd.Series(np.nan, index=row.index)
    out.loc[r.index] = r.values
    return out


def turnover_on_rebalance(weights_on_reb: pd.DataFrame) -> pd.Series:
    w = weights_on_reb.fillna(0.0)
    dw = w.diff()
    turn = 0.5 * dw.abs().sum(axis=1)
    if len(turn) > 0:
        turn.iloc[0] = 0.0
    return turn


def build_buffered_weights_with_filters(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    liquid_mask_on_reb: pd.DataFrame,
    long_entry: float,
    long_exit: float,
    short_entry: float,
    short_exit: float,
    min_names: int,
    use_signal_strength_gate: bool,
    signal_gate_long: float,
    signal_gate_short: float,
) -> pd.DataFrame:
    rebalance_dates = pd.DatetimeIndex(rebalance_dates).intersection(signal.index)
    tickers = signal.columns
    w_out = pd.DataFrame(0.0, index=rebalance_dates, columns=tickers)

    state = pd.Series(0, index=tickers, dtype=int)

    for dt in rebalance_dates:
        row = signal.loc[dt].copy()

        if dt in liquid_mask_on_reb.index:
            row.loc[~liquid_mask_on_reb.loc[dt]] = np.nan

        n_ok = int(row.notna().sum())
        if n_ok < min_names:
            w_out.loc[dt] = 0.0
            continue

        pct = row_percentiles(row)

        is_long = state == 1
        is_short = state == -1

        exit_long = is_long & (pct > long_exit)
        state.loc[exit_long.index[exit_long]] = 0

        exit_short = is_short & (pct < short_exit)
        state.loc[exit_short.index[exit_short]] = 0

        if use_signal_strength_gate:
            enter_long = (state == 0) & (pct <= long_entry) & (pct <= signal_gate_long)
            enter_short = (state == 0) & (pct >= short_entry) & (pct >= signal_gate_short)
        else:
            enter_long = (state == 0) & (pct <= long_entry)
            enter_short = (state == 0) & (pct >= short_entry)

        state.loc[enter_long.index[enter_long]] = 1
        state.loc[enter_short.index[enter_short]] = -1

        # Anything not tradable this rebalance must be zeroed out
        active_liquid = row.notna()
        state.loc[~active_liquid] = 0

        longs = state[state == 1].index
        shorts = state[state == -1].index

        w = pd.Series(0.0, index=tickers, dtype=float)
        if len(longs) > 0:
            w.loc[longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            w.loc[shorts] = -1.0 / len(shorts)

        w_out.loc[dt] = w.values

    return w_out


def compute_liquidity_slippage_on_rebalances(
    weights_on_reb: pd.DataFrame,
    adv_panel: pd.DataFrame,
    multiplier_bps: float,
) -> pd.Series:
    """
    Extra slippage penalty based on illiquidity rank.
    Higher penalty when traded names are lower-liquidity names.
    """
    weights_on_reb = weights_on_reb.fillna(0.0)
    dw = weights_on_reb.diff().fillna(0.0).abs()

    penalties = []

    for dt in weights_on_reb.index:
        traded = dw.loc[dt]
        traded = traded[traded > 0]

        if traded.empty or dt not in adv_panel.index:
            penalties.append(0.0)
            continue

        adv_row = adv_panel.loc[dt].reindex(traded.index)
        valid = adv_row.dropna()
        if valid.empty:
            penalties.append(0.0)
            continue

        # lower adv => more illiquid => larger penalty
        ranks = valid.rank(pct=True, ascending=True)
        illiq_score = 1.0 - ranks

        traded_valid = traded.reindex(valid.index).fillna(0.0)
        if traded_valid.sum() <= 0:
            penalties.append(0.0)
            continue

        weighted_illiq = float((traded_valid * illiq_score).sum() / traded_valid.sum())
        penalties.append((multiplier_bps / 1e4) * weighted_illiq)

    return pd.Series(penalties, index=weights_on_reb.index, dtype=float)


def apply_cost_model(
    gross_log: pd.Series,
    reb_turnover: pd.Series,
    weights_on_reb: pd.DataFrame,
    adv_panel: pd.DataFrame,
    base_tc_bps: float,
    base_slippage_bps: float,
    liquidity_slippage_multiplier_bps: float,
) -> pd.Series:
    base_cost = ((base_tc_bps + base_slippage_bps) / 1e4) * reb_turnover
    extra_liq = compute_liquidity_slippage_on_rebalances(
        weights_on_reb=weights_on_reb,
        adv_panel=adv_panel,
        multiplier_bps=liquidity_slippage_multiplier_bps,
    )
    total_cost = base_cost.add(extra_liq, fill_value=0.0)
    total_cost = total_cost.reindex(gross_log.index).fillna(0.0)
    return gross_log - total_cost


def evaluate_run(
    name: str,
    gap_log: pd.DataFrame,
    bench_log: pd.Series,
    weights_on_reb: pd.DataFrame,
    adv_panel: pd.DataFrame,
    base_tc_bps: float,
    base_slippage_bps: float,
    liquidity_slippage_multiplier_bps: float,
) -> StrategyRun:
    w_daily = forward_fill_weights(weights_on_reb).reindex(gap_log.index).ffill().fillna(0.0)

    gross = portfolio_log_returns_same_day(gap_log, w_daily)
    reb_turn = turnover_on_rebalance(weights_on_reb)

    net = apply_cost_model(
        gross_log=gross,
        reb_turnover=reb_turn,
        weights_on_reb=weights_on_reb,
        adv_panel=adv_panel,
        base_tc_bps=base_tc_bps,
        base_slippage_bps=base_slippage_bps,
        liquidity_slippage_multiplier_bps=liquidity_slippage_multiplier_bps,
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

    return StrategyRun(
        name=name,
        gross_log=gross,
        net_log=net,
        bench_log=bench,
        weights_on_reb=weights_on_reb,
        reb_turn=reb_turn,
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


def summarize_by_period(strategy_name: str, strategy_log: pd.Series, bench_log: pd.Series) -> pd.DataFrame:
    periods = [
        ("Full_Post_GFC", pd.Timestamp("2009-07-01"), pd.Timestamp("2099-12-31")),
        ("Recovery_2009_2012", pd.Timestamp("2009-07-01"), pd.Timestamp("2012-12-31")),
        ("QE_2013_2019", pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")),
        ("COVID_2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
        ("Inflation_2021_now", pd.Timestamp("2021-01-01"), pd.Timestamp("2099-12-31")),
    ]

    rows = []
    for period_name, start, end in periods:
        s = strategy_log.loc[start:end].dropna()
        b = bench_log.loc[start:end].dropna()
        common = s.index.intersection(b.index)
        s = s.loc[common]
        b = b.loc[common]
        if len(common) < 20:
            continue

        m = perf_metrics_from_log_returns(s)
        corr, beta = corr_and_beta(s, b)
        eq = equity_curve_from_log_returns(s)

        rows.append(
            {
                "strategy": strategy_name,
                "period": period_name,
                "start": common.min().date().isoformat(),
                "end": common.max().date().isoformat(),
                "n_days": int(len(common)),
                "sharpe": float(m.get("sharpe", np.nan)),
                "ann_return": float(m.get("ann_return", np.nan)),
                "ann_vol": float(m.get("ann_vol", np.nan)),
                "max_drawdown": float(m.get("max_drawdown", np.nan)),
                "beta_to_benchmark": float(beta),
                "corr_to_benchmark": float(corr),
                "final_equity": float(eq.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


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


def print_period_summary(df: pd.DataFrame, strategy_name: str) -> None:
    sub = df.loc[df["strategy"] == strategy_name].copy()
    if sub.empty:
        return

    print("\n" + "=" * 100)
    print(f"ROBUSTNESS SUMMARY | {strategy_name}")
    print("=" * 100)
    for _, row in sub.iterrows():
        print(
            f"{row['period']:>18} | "
            f"{row['start']} -> {row['end']} | "
            f"n={int(row['n_days']):>4d} | "
            f"Sharpe={row['sharpe']:>6.3f} | "
            f"AnnRet={row['ann_return']:>6.3f} | "
            f"AnnVol={row['ann_vol']:>6.3f} | "
            f"MaxDD={row['max_drawdown']:>7.3f} | "
            f"Beta={row['beta_to_benchmark']:>7.3f}"
        )


def save_run_outputs(run: StrategyRun) -> None:
    ts = pd.DataFrame(
        {
            "strategy_log": run.net_log,
            "benchmark_log": run.bench_log,
            "strategy_eq": equity_curve_from_log_returns(run.net_log),
            "benchmark_eq": equity_curve_from_log_returns(run.bench_log),
            "strategy_dd": drawdown_curve(equity_curve_from_log_returns(run.net_log)),
            "benchmark_dd": drawdown_curve(equity_curve_from_log_returns(run.bench_log)),
        }
    )
    ts.to_csv(OUT_DIR / f"{run.name}_timeseries.csv")

    turn = pd.DataFrame(
        {
            "rebalance_date": run.reb_turn.index,
            "turnover": run.reb_turn.values,
        }
    )
    turn.to_csv(OUT_DIR / f"{run.name}_rebalance_turnover.csv", index=False)


def apply_exposure_scaling_on_rebalance(
    weights_on_reb: pd.DataFrame,
    earn_months: set[int],
    lambda_earn_months: float,
    lambda_other_months: float,
) -> pd.DataFrame:
    """
    Scale rebalance weights by month-level leverage regime.

    Busy/earnings-heavy months get higher exposure.
    Other months get lower exposure.
    """
    w = weights_on_reb.copy()

    for dt in w.index:
        lam = lambda_earn_months if dt.month in earn_months else lambda_other_months
        w.loc[dt] = w.loc[dt] * lam

    return w


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 100)
    print("POST-GFC IMPLEMENTATION UPGRADE TEST")
    print("=" * 100)
    print(f"Sample:                 {START} -> latest")
    print(f"Locked signal:          fw={FORMATION_WINDOW}")
    print(f"Base thresholds:        LE={BASE_LONG_ENTRY:.2f} LX={BASE_LONG_EXIT:.2f} SE={BASE_SHORT_ENTRY:.2f} SX={BASE_SHORT_EXIT:.2f}")
    print(f"Upgrade thresholds:     LE={UPG_LONG_ENTRY:.2f} LX={UPG_LONG_EXIT:.2f} SE={UPG_SHORT_ENTRY:.2f} SX={UPG_SHORT_EXIT:.2f}")
    print(f"ADV lookback:           {ADV_LOOKBACK}")
    print(f"Top liquid names:       {TOP_LIQUID_NAMES}")
    print(f"Min names base:         {MIN_NAMES_BASE}")
    print(f"Min names upgraded:     {MIN_NAMES_UPG}")
    print(f"Signal gate enabled:    {USE_SIGNAL_STRENGTH_GATE}")
    print(f"Cost model:             {BASE_TC_BPS:.1f} tc + {BASE_SLIPPAGE_BPS:.1f} slip + liquidity penalty")
    print(f"Exposure scaling:       {USE_EXPOSURE_SCALING}")
    if USE_EXPOSURE_SCALING:
        print(f"Lambda earn months:     {LAMBDA_EARN_MONTHS}")
        print(f"Lambda other months:    {LAMBDA_OTHER_MONTHS}")
        print(f"Earnings months proxy:  {sorted(EARN_MONTHS)}")

    open_px, close_px, close_log, gap_log, bench_log = prepare_data()

    signal = vol_adjusted_formation_signal(
        log_rets=close_log,
        formation_window=FORMATION_WINDOW,
        vol_window=VOL_WINDOW,
        vol_floor=1e-6,
    )

    # Common initial filter based on base minimum only
    valid_dates = gap_log.index[gap_log.notna().any(axis=1)]
    gap_log = gap_log.loc[valid_dates]
    signal = signal.loc[valid_dates]
    close_px = close_px.loc[valid_dates]

    tradable_counts = signal.notna().sum(axis=1)
    ok_dates = tradable_counts[tradable_counts >= MIN_NAMES_BASE].index
    gap_log = gap_log.loc[ok_dates]
    signal = signal.loc[ok_dates]
    close_px = close_px.loc[ok_dates]
    bench_log = bench_log.loc[ok_dates.intersection(bench_log.index)]

    rebalance_dates = get_monthly_rebalance_dates(signal.index)
    adv_panel = compute_adv_panel(close_px)
    liquid_mask_on_reb = make_liquidity_mask(adv_panel, rebalance_dates, TOP_LIQUID_NAMES)

    # Baseline: no liquidity filter, no signal-strength gate
    all_true_mask = pd.DataFrame(True, index=rebalance_dates, columns=signal.columns)

    weights_base = build_buffered_weights_with_filters(
        signal=signal,
        rebalance_dates=rebalance_dates,
        liquid_mask_on_reb=all_true_mask,
        long_entry=BASE_LONG_ENTRY,
        long_exit=BASE_LONG_EXIT,
        short_entry=BASE_SHORT_ENTRY,
        short_exit=BASE_SHORT_EXIT,
        min_names=MIN_NAMES_BASE,
        use_signal_strength_gate=False,
        signal_gate_long=SIGNAL_GATE_LONG,
        signal_gate_short=SIGNAL_GATE_SHORT,
    )
    weights_base = enforce_dollar_neutral_on_rebalance(weights_base, target_gross=TARGET_GROSS)

    if USE_EXPOSURE_SCALING:
        weights_base = apply_exposure_scaling_on_rebalance(
            weights_on_reb=weights_base,
            earn_months=EARN_MONTHS,
            lambda_earn_months=LAMBDA_EARN_MONTHS,
            lambda_other_months=LAMBDA_OTHER_MONTHS,
        )

    # Upgraded: liquidity filter + wider exits + optional score gate
    weights_upg = build_buffered_weights_with_filters(
        signal=signal,
        rebalance_dates=rebalance_dates,
        liquid_mask_on_reb=liquid_mask_on_reb,
        long_entry=UPG_LONG_ENTRY,
        long_exit=UPG_LONG_EXIT,
        short_entry=UPG_SHORT_ENTRY,
        short_exit=UPG_SHORT_EXIT,
        min_names=MIN_NAMES_UPG,
        use_signal_strength_gate=USE_SIGNAL_STRENGTH_GATE,
        signal_gate_long=SIGNAL_GATE_LONG,
        signal_gate_short=SIGNAL_GATE_SHORT,
    )
    weights_upg = enforce_dollar_neutral_on_rebalance(weights_upg, target_gross=TARGET_GROSS)

    if USE_EXPOSURE_SCALING:
        weights_upg = apply_exposure_scaling_on_rebalance(
            weights_on_reb=weights_upg,
            earn_months=EARN_MONTHS,
            lambda_earn_months=LAMBDA_EARN_MONTHS,
            lambda_other_months=LAMBDA_OTHER_MONTHS,
        )

    run_base = evaluate_run(
        name="baseline_locked_post_gfc",
        gap_log=gap_log,
        bench_log=bench_log,
        weights_on_reb=weights_base,
        adv_panel=adv_panel,
        base_tc_bps=BASE_TC_BPS,
        base_slippage_bps=BASE_SLIPPAGE_BPS,
        liquidity_slippage_multiplier_bps=0.0,
    )

    run_upg = evaluate_run(
        name="implementation_upgraded_post_gfc",
        gap_log=gap_log,
        bench_log=bench_log,
        weights_on_reb=weights_upg,
        adv_panel=adv_panel,
        base_tc_bps=BASE_TC_BPS,
        base_slippage_bps=BASE_SLIPPAGE_BPS,
        liquidity_slippage_multiplier_bps=LIQUIDITY_SLIPPAGE_MULTIPLIER_BPS,
    )

    for run in [run_base, run_upg]:
        print_run_summary(run)
        save_run_outputs(run)

    summary_df = pd.DataFrame(
        [
            {
                "strategy": run_base.name,
                "net_sharpe": float(run_base.metrics_net.get("sharpe", np.nan)),
                "net_annret": float(run_base.metrics_net.get("ann_return", np.nan)),
                "net_annvol": float(run_base.metrics_net.get("ann_vol", np.nan)),
                "net_maxdd": float(run_base.metrics_net.get("max_drawdown", np.nan)),
                "avg_reb_turnover": float(run_base.avg_reb_turnover),
                "beta_to_bench": float(run_base.beta_to_bench),
                "corr_to_bench": float(run_base.corr_to_bench),
                "final_equity_net": float(run_base.final_equity_net),
            },
            {
                "strategy": run_upg.name,
                "net_sharpe": float(run_upg.metrics_net.get("sharpe", np.nan)),
                "net_annret": float(run_upg.metrics_net.get("ann_return", np.nan)),
                "net_annvol": float(run_upg.metrics_net.get("ann_vol", np.nan)),
                "net_maxdd": float(run_upg.metrics_net.get("max_drawdown", np.nan)),
                "avg_reb_turnover": float(run_upg.avg_reb_turnover),
                "beta_to_bench": float(run_upg.beta_to_bench),
                "corr_to_bench": float(run_upg.corr_to_bench),
                "final_equity_net": float(run_upg.final_equity_net),
            },
        ]
    ).sort_values(["net_sharpe", "net_annret"], ascending=[False, False])

    summary_df.to_csv(OUT_DIR / "implementation_upgrade_summary.csv", index=False)

    period_df = pd.concat(
        [
            summarize_by_period(run_base.name, run_base.net_log, run_base.bench_log),
            summarize_by_period(run_upg.name, run_upg.net_log, run_upg.bench_log),
        ],
        axis=0,
        ignore_index=True,
    )
    period_df.to_csv(OUT_DIR / "implementation_upgrade_by_period.csv", index=False)

    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print_period_summary(period_df, run_base.name)
    print_period_summary(period_df, run_upg.name)

    print("\n" + "=" * 100)
    print("SAVED OUTPUTS")
    print("=" * 100)
    print(f"  {OUT_DIR / 'implementation_upgrade_summary.csv'}")
    print(f"  {OUT_DIR / 'implementation_upgrade_by_period.csv'}")
    print(f"  {OUT_DIR / 'baseline_locked_post_gfc_timeseries.csv'}")
    print(f"  {OUT_DIR / 'implementation_upgraded_post_gfc_timeseries.csv'}")
    print(f"  {OUT_DIR / 'baseline_locked_post_gfc_rebalance_turnover.csv'}")
    print(f"  {OUT_DIR / 'implementation_upgraded_post_gfc_rebalance_turnover.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()

