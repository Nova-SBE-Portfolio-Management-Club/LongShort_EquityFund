from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.playground.momo_rvrs.data_loader import get_universe_ohlcv, get_benchmark_ohlc
from src.playground.momo_rvrs.signals import (
    compute_log_returns,
    formation_return,
    get_weekly_rebalance_dates,
    get_monthly_rebalance_dates,
    forward_fill_weights,
    perf_metrics_from_log_returns,
    vol_adjusted_formation_signal,
)

# ============================================================
# Locked strategy
# ============================================================

LOCKED_NAME = "FTSE250_locked_v1"
LOCKED_LONG_ENTRY = 0.12
LOCKED_LONG_EXIT = 0.25
LOCKED_SHORT_ENTRY = 0.92
LOCKED_SHORT_EXIT = 0.70

# ============================================================
# Config
# ============================================================

UNIVERSE = "FTSE250"
START = "2000-01-01"
END = None

FORMATION_WINDOW = 1
REBALANCE = "monthly"
TC_BPS_PER_1TURN = 10.0
USE_VOL_ADJUSTED_SIGNAL = True
VOL_WINDOW = 20
DOLLAR_NEUTRALIZE = True
TARGET_GROSS = 2.0
USE_EXPOSURE_SCALING = True
LAMBDA_EARN_MONTHS = 1.5
LAMBDA_OTHER_MONTHS = 0.2

# Lower than 200 because older FTSE250 history can be patchier.
MIN_NAMES = 120

OUT_DIR = Path("src/playground/momo_rvrs/period_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helpers
# ============================================================


def compute_gap_log_returns(open_px: pd.DataFrame, close_px: pd.DataFrame) -> pd.DataFrame:
    """gap_t = log(Open_{t+1}/Close_t) aligned to date t (close date)."""
    return np.log(open_px.shift(-1) / close_px)


def portfolio_log_returns_same_day(panel_log_rets: pd.DataFrame, weights_daily: pd.DataFrame) -> pd.Series:
    """Apply weights at date t to returns indexed at date t."""
    w = weights_daily.reindex(panel_log_rets.index).fillna(0.0)
    r = panel_log_rets.reindex(panel_log_rets.index)
    return (w * r).sum(axis=1)


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


def gross_exposures(weights_daily: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (gross_long, gross_short, net) time series. gross_short is positive abs()."""
    w = weights_daily.fillna(0.0)
    gross_long = w.clip(lower=0.0).sum(axis=1)
    gross_short = (-w.clip(upper=0.0)).sum(axis=1)
    net = gross_long - gross_short
    return gross_long, gross_short, net


def turnover_on_rebalance(weights_on_reb: pd.DataFrame) -> pd.Series:
    """turnover_reb[t] = 0.5 * sum_i |w_t - w_{t-1}|, computed on rebalance dates."""
    w = weights_on_reb.fillna(0.0)
    dw = w.diff()
    turn = 0.5 * dw.abs().sum(axis=1)
    if len(turn) > 0:
        turn.iloc[0] = 0.0
    return turn


def apply_tc_on_rebalance_days(gross_log: pd.Series, reb_turnover: pd.Series, tc_bps_per_1turn: float) -> pd.Series:
    """Apply transaction costs ONLY on rebalance dates as a one-day log-return hit."""
    tc = (tc_bps_per_1turn / 1e4) * reb_turnover
    tc = tc.reindex(gross_log.index).fillna(0.0)
    return gross_log - tc


def enforce_dollar_neutral_on_rebalance(weights_on_reb: pd.DataFrame, target_gross: float = 2.0) -> pd.DataFrame:
    """Dollar-neutralize ONLY on rebalance dates: gross_long==gross_short==target_gross/2."""
    w = weights_on_reb.fillna(0.0).copy()
    pos = w.clip(lower=0.0)
    neg = w.clip(upper=0.0)

    gl = pos.sum(axis=1)
    gs = (-neg).sum(axis=1)

    target_side = 0.5 * float(target_gross)

    scale_long = pd.Series(0.0, index=w.index, dtype=float)
    scale_short = pd.Series(0.0, index=w.index, dtype=float)

    has_long = gl > 0
    has_short = gs > 0

    scale_long.loc[has_long] = target_side / gl.loc[has_long]
    scale_short.loc[has_short] = target_side / gs.loc[has_short]

    pos2 = pos.mul(scale_long, axis=0)
    neg2 = neg.mul(scale_short, axis=0)

    return (pos2 + neg2).fillna(0.0)


def summarize_panel(panel: pd.DataFrame) -> tuple[int, float]:
    n_names = int(panel.shape[1])
    miss_frac = float(panel.isna().mean().mean()) if panel.size else float("nan")
    return n_names, miss_frac


# ============================================================
# Exposure scaling (rebalance-only)
# ============================================================


def is_earnings_heavy_month(d: pd.Timestamp) -> bool:
    return int(d.month) in {1, 2, 4, 5, 7, 8, 10, 11}


def make_lambda_on_rebalance_dates(
    rebalance_dates: pd.DatetimeIndex,
    lambda_earn: float = 1.5,
    lambda_other: float = 0.2,
) -> pd.Series:
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    lam = pd.Series(lambda_other, index=rebalance_dates, dtype=float)
    earn_mask = rebalance_dates.map(lambda x: is_earnings_heavy_month(pd.Timestamp(x)))
    lam.loc[earn_mask] = float(lambda_earn)
    return lam


def apply_exposure_scaling_on_rebalance(weights_on_reb: pd.DataFrame, lam_on_reb: pd.Series) -> pd.DataFrame:
    lam_on_reb = lam_on_reb.reindex(weights_on_reb.index).fillna(1.0)
    return weights_on_reb.mul(lam_on_reb, axis=0)


# ============================================================
# Tilting selection
# ============================================================


def _row_percentiles(row: pd.Series) -> pd.Series:
    x = row.dropna()
    if len(x) < 2:
        return row * np.nan
    r = x.rank(method="average", pct=True)
    out = pd.Series(np.nan, index=row.index)
    out.loc[r.index] = r.values
    return out


def tilted_buffered_long_short_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    long_entry: float,
    long_exit: float,
    short_entry: float,
    short_exit: float,
    min_names: int = 120,
    ew_within_side: bool = True,
) -> pd.DataFrame:
    rebalance_dates = pd.DatetimeIndex(rebalance_dates).intersection(signal.index)
    if len(rebalance_dates) == 0:
        raise ValueError("No rebalance dates overlap signal index.")

    tickers = signal.columns
    w_out = pd.DataFrame(0.0, index=rebalance_dates, columns=tickers)

    state = pd.Series(0, index=tickers, dtype=int)

    for dt in rebalance_dates:
        row = signal.loc[dt]
        n_ok = int(row.notna().sum())
        if n_ok < min_names:
            w_out.loc[dt] = 0.0
            continue

        pct = _row_percentiles(row)

        is_long = state == 1
        is_short = state == -1

        exit_long = is_long & (pct > long_exit)
        state.loc[exit_long.index[exit_long]] = 0

        exit_short = is_short & (pct < short_exit)
        state.loc[exit_short.index[exit_short]] = 0

        enter_long = (state == 0) & (pct <= long_entry)
        state.loc[enter_long.index[enter_long]] = 1

        enter_short = (state == 0) & (pct >= short_entry)
        state.loc[enter_short.index[enter_short]] = -1

        longs = state[state == 1].index
        shorts = state[state == -1].index

        w = pd.Series(0.0, index=tickers, dtype=float)
        if ew_within_side:
            if len(longs) > 0:
                w.loc[longs] = 1.0 / float(len(longs))
            if len(shorts) > 0:
                w.loc[shorts] = -1.0 / float(len(shorts))
        else:
            w.loc[longs] = 1.0
            w.loc[shorts] = -1.0

        w_out.loc[dt] = w.values

    return w_out


# ============================================================
# Results containers
# ============================================================


@dataclass
class PeriodResult:
    period: str
    start: str
    end: str | None
    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float
    gross_sharpe: float
    gross_annret: float
    gross_annvol: float
    gross_maxdd: float
    net_sharpe: float
    net_annret: float
    net_annvol: float
    net_maxdd: float
    final_eq_net: float
    avg_reb_turnover: float
    avg_gross_long: float
    avg_gross_short: float
    avg_net: float
    corr_to_bench: float
    beta_to_bench: float
    n_days: int
    median_tradable: float
    n_names: int
    miss_open: float
    miss_close: float
    error: str = ""


# ============================================================
# Universe prep
# ============================================================


def prepare_universe(
    universe: str,
    start: str,
    end: str | None,
    *,
    formation_window: int,
    rebalance: str,
    min_names: int,
    use_vol_adjusted_signal: bool,
    vol_window: int,
    lambda_earn_months: float,
    lambda_other_months: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.Series, pd.Series, dict]:
    u = universe.strip().upper()

    open_px, _, _, close_px, _, _ = get_universe_ohlcv(u, start=start, end=end, force=False)
    open_px = open_px.sort_index()
    close_px = close_px.sort_index()

    idx = open_px.index.intersection(close_px.index)
    open_px = open_px.loc[idx]
    close_px = close_px.loc[idx]

    _, miss_open = summarize_panel(open_px)
    n_names_close, miss_close = summarize_panel(close_px)

    gap_log = compute_gap_log_returns(open_px, close_px)
    cc_log = compute_log_returns(close_px)

    if use_vol_adjusted_signal:
        signal = vol_adjusted_formation_signal(
            log_rets=cc_log,
            formation_window=formation_window,
            vol_window=vol_window,
            vol_floor=1e-6,
        )
    else:
        signal = formation_return(cc_log, window=formation_window)

    valid_dates = gap_log.index[gap_log.notna().any(axis=1)]
    gap_log = gap_log.loc[valid_dates]
    signal = signal.loc[valid_dates]

    tradable_counts = signal.notna().sum(axis=1)
    ok_dates = tradable_counts[tradable_counts >= min_names].index
    gap_log = gap_log.loc[ok_dates]
    signal = signal.loc[ok_dates]

    if len(signal) < 120:
        raise RuntimeError(f"{u}: Not enough data after filtering (need >=120, got {len(signal)}).")

    if rebalance == "weekly":
        rebalance_dates = get_weekly_rebalance_dates(signal.index)
    elif rebalance == "monthly":
        rebalance_dates = get_monthly_rebalance_dates(signal.index)
    else:
        raise ValueError("rebalance must be 'weekly' or 'monthly'")

    lam_on_reb = make_lambda_on_rebalance_dates(
        rebalance_dates,
        lambda_earn=lambda_earn_months,
        lambda_other=lambda_other_months,
    )

    bench = get_benchmark_ohlc(u, start=start, end=end, force=False)
    bench_cc_log_daily = np.log(bench["Close"]).diff().reindex(gap_log.index).dropna()

    meta = {
        "n_days": int(len(signal)),
        "median_tradable": float(tradable_counts.loc[signal.index].median()),
        "n_names": int(n_names_close),
        "miss_open": float(miss_open),
        "miss_close": float(miss_close),
    }
    return gap_log, signal, rebalance_dates, lam_on_reb, bench_cc_log_daily, meta


# ============================================================
# Single-period run
# ============================================================


def run_locked_strategy_for_period(
    period_name: str,
    start: str,
    end: str | None,
) -> PeriodResult:
    gap_log, signal, rebalance_dates, lam_on_reb, bench_cc_log_daily, meta = prepare_universe(
        UNIVERSE,
        start=start,
        end=end,
        formation_window=FORMATION_WINDOW,
        rebalance=REBALANCE,
        min_names=MIN_NAMES,
        use_vol_adjusted_signal=USE_VOL_ADJUSTED_SIGNAL,
        vol_window=VOL_WINDOW,
        lambda_earn_months=LAMBDA_EARN_MONTHS,
        lambda_other_months=LAMBDA_OTHER_MONTHS,
    )

    w_on_reb = tilted_buffered_long_short_weights(
        signal=signal,
        rebalance_dates=rebalance_dates,
        long_entry=LOCKED_LONG_ENTRY,
        long_exit=LOCKED_LONG_EXIT,
        short_entry=LOCKED_SHORT_ENTRY,
        short_exit=LOCKED_SHORT_EXIT,
        min_names=MIN_NAMES,
        ew_within_side=True,
    )

    if DOLLAR_NEUTRALIZE:
        w_on_reb = enforce_dollar_neutral_on_rebalance(w_on_reb, target_gross=TARGET_GROSS)

    if USE_EXPOSURE_SCALING:
        w_on_reb = apply_exposure_scaling_on_rebalance(w_on_reb, lam_on_reb)

    w_daily = forward_fill_weights(w_on_reb).reindex(gap_log.index).ffill().fillna(0.0)

    gross = portfolio_log_returns_same_day(gap_log, w_daily)
    reb_turn = turnover_on_rebalance(w_on_reb)
    net = apply_tc_on_rebalance_days(gross, reb_turn, TC_BPS_PER_1TURN)

    common = net.index.intersection(bench_cc_log_daily.index)
    gross = gross.loc[common].dropna()
    net = net.loc[common].dropna()
    bench = bench_cc_log_daily.loc[common].dropna()

    common2 = gross.index.intersection(net.index).intersection(bench.index)
    gross = gross.loc[common2]
    net = net.loc[common2]
    bench = bench.loc[common2]

    gross_m = perf_metrics_from_log_returns(gross)
    net_m = perf_metrics_from_log_returns(net)
    corr, beta = corr_and_beta(net, bench)

    eq_net = equity_curve_from_log_returns(net)
    final_eq = float(eq_net.iloc[-1]) if len(eq_net) else float("nan")

    gl, gs, n = gross_exposures(w_daily.reindex(common2))

    return PeriodResult(
        period=period_name,
        start=start,
        end=end,
        long_entry=LOCKED_LONG_ENTRY,
        long_exit=LOCKED_LONG_EXIT,
        short_entry=LOCKED_SHORT_ENTRY,
        short_exit=LOCKED_SHORT_EXIT,
        gross_sharpe=float(gross_m.get("sharpe", np.nan)),
        gross_annret=float(gross_m.get("ann_return", np.nan)),
        gross_annvol=float(gross_m.get("ann_vol", np.nan)),
        gross_maxdd=float(gross_m.get("max_drawdown", np.nan)),
        net_sharpe=float(net_m.get("sharpe", np.nan)),
        net_annret=float(net_m.get("ann_return", np.nan)),
        net_annvol=float(net_m.get("ann_vol", np.nan)),
        net_maxdd=float(net_m.get("max_drawdown", np.nan)),
        final_eq_net=float(final_eq),
        avg_reb_turnover=float(reb_turn.mean()) if len(reb_turn) else float("nan"),
        avg_gross_long=float(gl.mean()) if len(gl) else float("nan"),
        avg_gross_short=float(gs.mean()) if len(gs) else float("nan"),
        avg_net=float(n.mean()) if len(n) else float("nan"),
        corr_to_bench=float(corr),
        beta_to_bench=float(beta),
        n_days=int(meta["n_days"]),
        median_tradable=float(meta["median_tradable"]),
        n_names=int(meta["n_names"]),
        miss_open=float(meta["miss_open"]),
        miss_close=float(meta["miss_close"]),
        error="",
    )


# ============================================================
# Reporting
# ============================================================


def period_schedule() -> list[tuple[str, str, str | None]]:
    return [
        ("dotcom_unwind", "2000-01-01", "2002-12-31"),
        ("pre_gfc_expansion", "2003-01-01", "2006-12-31"),
        ("gfc", "2007-01-01", "2009-12-31"),
        ("post_crisis_qe", "2010-01-01", "2014-12-31"),
        ("late_cycle_pre_covid", "2015-01-01", "2019-12-31"),
        ("covid_rebound", "2020-01-01", "2021-12-31"),
        ("inflation_tightening", "2022-01-01", "2022-12-31"),
        ("recent_regime", "2023-01-01", None),
        ("full_sample", "2000-01-01", None),
    ]


def run_all_periods() -> pd.DataFrame:
    rows: list[dict] = []

    for period_name, start, end in period_schedule():
        print("\n" + "=" * 80)
        print(f"RUNNING PERIOD: {period_name} | start={start} | end={end}")
        print("=" * 80)

        try:
            res = run_locked_strategy_for_period(period_name, start, end)
            row = res.__dict__.copy()
            rows.append(row)

            print(f"NET Sharpe:   {res.net_sharpe:.4f}")
            print(f"NET AnnRet:   {res.net_annret:.4f}")
            print(f"NET AnnVol:   {res.net_annvol:.4f}")
            print(f"NET MaxDD:    {res.net_maxdd:.4f}")
            print(f"Final Eq:     {res.final_eq_net:.4f}")
            print(f"Corr:         {res.corr_to_bench:.4f}")
            print(f"Beta:         {res.beta_to_bench:.4f}")
            print(f"Avg Reb Turn: {res.avg_reb_turnover:.4f}")
            print(f"Median names: {res.median_tradable:.1f}")

        except Exception as e:
            rows.append(
                {
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "long_entry": LOCKED_LONG_ENTRY,
                    "long_exit": LOCKED_LONG_EXIT,
                    "short_entry": LOCKED_SHORT_ENTRY,
                    "short_exit": LOCKED_SHORT_EXIT,
                    "gross_sharpe": np.nan,
                    "gross_annret": np.nan,
                    "gross_annvol": np.nan,
                    "gross_maxdd": np.nan,
                    "net_sharpe": np.nan,
                    "net_annret": np.nan,
                    "net_annvol": np.nan,
                    "net_maxdd": np.nan,
                    "final_eq_net": np.nan,
                    "avg_reb_turnover": np.nan,
                    "avg_gross_long": np.nan,
                    "avg_gross_short": np.nan,
                    "avg_net": np.nan,
                    "corr_to_bench": np.nan,
                    "beta_to_bench": np.nan,
                    "n_days": np.nan,
                    "median_tradable": np.nan,
                    "n_names": np.nan,
                    "miss_open": np.nan,
                    "miss_close": np.nan,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            print(f"ERROR: {type(e).__name__}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "locked_strategy_by_period.csv", index=False)

    return out


def print_summary_table(df: pd.DataFrame) -> None:
    display_cols = [
        "period",
        "start",
        "end",
        "net_sharpe",
        "net_annret",
        "net_annvol",
        "net_maxdd",
        "final_eq_net",
        "avg_reb_turnover",
        "corr_to_bench",
        "beta_to_bench",
        "median_tradable",
        "n_days",
        "error",
    ]
    print("\n" + "=" * 80)
    print("LOCKED STRATEGY ACROSS REGIMES")
    print("=" * 80)
    print(df[display_cols].to_string(index=False))


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 80)
    print("FTSE250 LOCKED STRATEGY | PERIOD ANALYSIS")
    print("=" * 80)
    print(f"Universe:            {UNIVERSE}")
    print(f"Start:               {START}")
    print(f"End:                 {END}")
    print(f"Rebalance:           {REBALANCE}")
    print(f"Formation window:    {FORMATION_WINDOW}")
    print(f"Vol-adjusted signal: {USE_VOL_ADJUSTED_SIGNAL}")
    print(f"Vol window:          {VOL_WINDOW}")
    print(f"TC bps / 1 turn:     {TC_BPS_PER_1TURN}")
    print(f"Min names:           {MIN_NAMES}")
    print()
    print("LOCKED STRATEGY")
    print(f"  long_entry  = {LOCKED_LONG_ENTRY:.3f}")
    print(f"  long_exit   = {LOCKED_LONG_EXIT:.3f}")
    print(f"  short_entry = {LOCKED_SHORT_ENTRY:.3f}")
    print(f"  short_exit  = {LOCKED_SHORT_EXIT:.3f}")

    df = run_all_periods()
    print_summary_table(df)

    print("\n" + "=" * 80)
    print(f"Saved: {OUT_DIR / 'locked_strategy_by_period.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
