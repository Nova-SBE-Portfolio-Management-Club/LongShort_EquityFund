# src/playground/MomentumReversal/close_test.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.playground.momo_rvrs.data_loader import get_sp500_ohlcv, get_spy_ohlc
from src.playground.momo_rvrs.signals import (
    compute_log_returns,
    formation_return,
    get_weekly_rebalance_dates,
    get_monthly_rebalance_dates,
    buffered_percentile_long_short_weights,
    forward_fill_weights,
    perf_metrics_from_log_returns,
    vol_adjusted_formation_signal,
)

# ============================================================
# Helpers
# ============================================================

EARN_MONTHS = [1, 2, 4, 5, 7, 8, 10, 11]


def compute_gap_log_returns(open_px: pd.DataFrame, close_px: pd.DataFrame) -> pd.DataFrame:
    """gap_t = log(Open_{t+1}/Close_t) aligned to date t (close date)."""
    return np.log(open_px.shift(-1) / close_px)


def portfolio_log_returns_same_day(panel_log_rets: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Apply weights at date t to returns indexed at date t."""
    w = weights.reindex(panel_log_rets.index).fillna(0.0)
    r = panel_log_rets.reindex(weights.index)
    return (w * r).sum(axis=1)


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """turnover_t = 0.5 * sum_i |w_t - w_{t-1}|"""
    w = weights.fillna(0.0)
    dw = w.diff()
    turn = 0.5 * dw.abs().sum(axis=1)
    if len(turn) > 0:
        turn.iloc[0] = 0.0
    return turn


def equity_curve_from_log_returns(log_ret: pd.Series, start: float = 1.0) -> pd.Series:
    return start * np.exp(log_ret.fillna(0.0).cumsum())


def corr_and_beta(strategy_log: pd.Series, benchmark_log: pd.Series) -> tuple[float, float]:
    df = pd.concat([strategy_log.rename("strat"), benchmark_log.rename("bench")], axis=1).dropna()
    if len(df) < 50:
        return np.nan, np.nan
    corr = float(df["strat"].corr(df["bench"]))
    var = float(df["bench"].var(ddof=1))
    beta = float(df["strat"].cov(df["bench"]) / var) if var > 0 else np.nan
    return corr, beta


# ============================================================
# Earnings-month regime mask (month-level, good for monthly rebal)
# ============================================================

def earnings_month_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """True if date falls in an earnings-season month."""
    d = pd.DatetimeIndex(dates)
    return pd.Series(d.month.isin(EARN_MONTHS), index=dates, dtype=bool)


def make_exposure_multiplier(
    index: pd.DatetimeIndex,
    mask: pd.Series,
    lambda_true: float,
    lambda_false: float,
) -> pd.Series:
    """Build λ_t exposure multiplier series."""
    m = mask.reindex(index).fillna(False)
    lam = pd.Series(lambda_false, index=index, dtype=float)
    lam.loc[m.values] = float(lambda_true)
    return lam


def apply_exposure_scaling_on_rebalance(
    weights_on_reb: pd.DataFrame,
    lam_full: pd.Series,
) -> pd.DataFrame:
    """
    REBALANCE-ONLY leverage scaling:

    - Only scale the target weights on rebalance dates:
        w_reb_scaled[t] = lam[t] * w_reb_base[t]
    - Then forward-fill between rebalance dates (no daily lambda flip churn).
    """
    lam_reb = lam_full.reindex(weights_on_reb.index).fillna(1.0).astype(float)
    return weights_on_reb.mul(lam_reb, axis=0)


# ============================================================
# Result container + evaluator
# ============================================================

@dataclass
class VariantResult:
    name: str
    gross_metrics: dict
    net_metrics: dict
    avg_turnover: float
    corr_to_spy: float
    beta_to_spy: float
    final_equity_net: float


def evaluate_variant(
    name: str,
    gap_log: pd.DataFrame,
    weights: pd.DataFrame,
    tc_bps_per_1turn: float,
    spy_cc_log: pd.Series,
) -> VariantResult:
    gross = portfolio_log_returns_same_day(gap_log, weights)
    turnover = compute_turnover(weights)
    tc = (tc_bps_per_1turn / 1e4) * turnover
    net = gross - tc

    # Align to SPY sample for consistent corr/beta + metrics
    common = net.index.intersection(spy_cc_log.index)
    gross = gross.loc[common].dropna()
    net = net.loc[common].dropna()
    spy = spy_cc_log.loc[common].dropna()

    gross_m = perf_metrics_from_log_returns(gross)
    net_m = perf_metrics_from_log_returns(net)
    corr, beta = corr_and_beta(net, spy)

    eq_net = equity_curve_from_log_returns(net)
    final_eq = float(eq_net.iloc[-1]) if len(eq_net) else float("nan")

    return VariantResult(
        name=name,
        gross_metrics=gross_m,
        net_metrics=net_m,
        avg_turnover=float(turnover.loc[common].mean()) if len(common) else float(turnover.mean()),
        corr_to_spy=corr,
        beta_to_spy=beta,
        final_equity_net=final_eq,
    )


def print_variant(res: VariantResult) -> None:
    print(f"\n--- {res.name} ---")
    print(f"Avg daily turnover: {res.avg_turnover:.4f}")
    print("GROSS:")
    print(f"  Sharpe:    {res.gross_metrics['sharpe']:.4f}")
    print(f"  AnnRet:    {res.gross_metrics['ann_return']:.4f}")
    print(f"  AnnVol:    {res.gross_metrics['ann_vol']:.4f}")
    print(f"  MaxDD:     {res.gross_metrics['max_drawdown']:.4f}")
    print("NET:")
    print(f"  Sharpe:    {res.net_metrics['sharpe']:.4f}")
    print(f"  AnnRet:    {res.net_metrics['ann_return']:.4f}")
    print(f"  AnnVol:    {res.net_metrics['ann_vol']:.4f}")
    print(f"  MaxDD:     {res.net_metrics['max_drawdown']:.4f}")
    print(f"  Final Eq:  {res.final_equity_net:.4f}")
    print("vs SPY (close-to-close):")
    print(f"  Corr:      {res.corr_to_spy:.4f}")
    print(f"  Beta:      {res.beta_to_spy:.4f}")


# ============================================================
# Main runner: BASE vs REBALANCE-ONLY LEVERAGE (earnings months)
# ============================================================

def run_exposure_scaling_test(
    start: str = "2005-01-01",
    end: str | None = None,
    formation_window: int = 1,
    rebalance: str = "monthly",  # "weekly" | "monthly"
    entry: float = 0.05,
    exit: float = 0.10,
    tc_bps_per_1turn: float = 10.0,
    min_names: int = 200,
    use_vol_adjusted_signal: bool = True,
    vol_window: int = 20,
    lambda_earn_months: float = 1.50,
    lambda_other_months: float = 1.00,
) -> None:
    # ---------- Load SP500 OHLCV ----------
    open_px, high_px, low_px, close_px, adj_px, volume = get_sp500_ohlcv(start=start, end=end, force=False)
    open_px = open_px.sort_index()
    close_px = close_px.sort_index()

    idx = open_px.index.intersection(close_px.index)
    open_px = open_px.loc[idx]
    close_px = close_px.loc[idx]

    # ---------- Overnight returns (close -> next open) ----------
    gap_log = compute_gap_log_returns(open_px, close_px)

    # ---------- Close-to-close log returns for signal ----------
    cc_log = compute_log_returns(close_px)

    # ---------- Signal ----------
    if use_vol_adjusted_signal:
        signal = vol_adjusted_formation_signal(
            log_rets=cc_log,
            formation_window=formation_window,
            vol_window=vol_window,
            vol_floor=1e-6,
        )
    else:
        signal = formation_return(cc_log, window=formation_window)

    # keep dates with next open available
    valid_dates = gap_log.index[gap_log.notna().any(axis=1)]
    gap_log = gap_log.loc[valid_dates]
    signal = signal.loc[valid_dates]

    # filter days with enough names
    tradable_counts = signal.notna().sum(axis=1)
    ok_dates = tradable_counts[tradable_counts >= min_names].index
    gap_log = gap_log.loc[ok_dates]
    signal = signal.loc[ok_dates]

    if len(signal) < 252:
        raise RuntimeError("Not enough data after filtering.")

    # ---------- Rebalance dates ----------
    if rebalance == "weekly":
        rebalance_dates = get_weekly_rebalance_dates(signal.index)
    elif rebalance == "monthly":
        rebalance_dates = get_monthly_rebalance_dates(signal.index)
    else:
        raise ValueError("rebalance must be 'weekly' or 'monthly'")

    # ---------- Base targets on rebalance dates ----------
    weights_on_reb = buffered_percentile_long_short_weights(
        signal=signal,
        rebalance_dates=rebalance_dates,
        entry=entry,
        exit=exit,
    )
    base_w = forward_fill_weights(weights_on_reb)

    # ---------- Benchmark: SPY buy&hold close-to-close ----------
    spy = get_spy_ohlc(start=start, end=end, force=False)
    spy_cc_log = np.log(spy["Close"]).diff()
    spy_cc_log = spy_cc_log.reindex(base_w.index).dropna()

    # ---------- Earnings-month leverage regime ----------
    earn_month = earnings_month_mask(base_w.index)
    lam = make_exposure_multiplier(
        index=base_w.index,
        mask=earn_month,
        lambda_true=lambda_earn_months,
        lambda_false=lambda_other_months,
    )

    # ---------- Rebalance-only leverage scaling ----------
    scaled_targets = apply_exposure_scaling_on_rebalance(weights_on_reb, lam)
    scaled_w = forward_fill_weights(scaled_targets)

    # ---------- Evaluate ----------
    base_res = evaluate_variant("BASE (λ=1.0 always)", gap_log, base_w, tc_bps_per_1turn, spy_cc_log)
    scaled_res = evaluate_variant(
        f"REB-ONLY LEVERAGE (λ={lambda_earn_months:.2f} earnings-months, λ={lambda_other_months:.2f} other)",
        gap_log,
        scaled_w,
        tc_bps_per_1turn,
        spy_cc_log,
    )

    # ---------- Print ----------
    print("\n=== OVERNIGHT REVERSAL: BASE vs REBALANCE-ONLY LEVERAGE ===")
    print(f"start={start} end={end}")
    print(f"rebalance={rebalance} | formation_window={formation_window} | entry={entry:.3f} exit={exit:.3f}")
    print(f"tc_bps_per_1turn={tc_bps_per_1turn:.1f} | min_names={min_names}")
    if use_vol_adjusted_signal:
        print(f"signal=vol_adjusted | vol_window={vol_window}")
    else:
        print("signal=raw formation return")
    print(f"earnings-month fraction: {earn_month.mean():.2%}")
    print(f"lambda_earn_months={lambda_earn_months:.2f} | lambda_other_months={lambda_other_months:.2f}")

    print_variant(base_res)
    print_variant(scaled_res)

    # ---------- SPY benchmark ----------
    spy_m = perf_metrics_from_log_returns(spy_cc_log.dropna())
    print("\n--- SPY BUY&Hold (close-to-close) ---")
    print(f"  Sharpe:   {spy_m['sharpe']:.4f}")
    print(f"  AnnRet:   {spy_m['ann_return']:.4f}")
    print(f"  AnnVol:   {spy_m['ann_vol']:.4f}")
    print(f"  MaxDD:    {spy_m['max_drawdown']:.4f}")


# ============================================================
# Period-split runner (crisis vs recovery-ish regimes)
# ============================================================

def run_period_splits() -> None:
    """
    Run the exact same test across multiple market regimes to check robustness.

    Periods are chosen to capture:
    - GFC / high-vol stress
    - QE bull / low-vol recovery
    - COVID + post-COVID regime
    """
    periods: list[tuple[str, str | None, str]] = [
        ("2007-01-01", "2009-06-30", "2007-06/2009 (Crisis Core)"),
        ("2009-07-01", "2012-12-31", "2009–2012 (Post-crisis recovery)"),
        ("2020-01-01", None, "2020–present (COVID + new regime)"),
    ]

    for start, end, label in periods:
        print("\n" + "=" * 80)
        print(f"PERIOD: {label}")
        print("=" * 80)
        run_exposure_scaling_test(
            start=start,
            end=end,
            formation_window=1,
            rebalance="monthly",
            entry=0.05,
            exit=0.10,
            tc_bps_per_1turn=10.0,
            min_names=200,
            use_vol_adjusted_signal=True,
            vol_window=20,
            lambda_earn_months=1.50,
            lambda_other_months=0.20,
        )


if __name__ == "__main__":
    run_period_splits()

