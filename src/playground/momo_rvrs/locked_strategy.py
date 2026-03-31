from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.playground.momo_rvrs.data_loader import (
    get_benchmark_ohlc,
    get_universe_ohlcv,
    load_ftse250_open_close_from_parquet,
    load_ftse250_precomputed_returns,
)
from src.playground.momo_rvrs.signals import (
    compute_log_returns,
    formation_return,
    forward_fill_weights,
    get_monthly_rebalance_dates,
    get_weekly_rebalance_dates,
    perf_metrics_from_log_returns,
    vol_adjusted_formation_signal,
)

# ============================================================
# Output paths
# ============================================================

OUT_DIR = Path("src/playground/momo_rvrs/locked_strategy_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def compute_gap_log_returns(open_px: pd.DataFrame, close_px: pd.DataFrame) -> pd.DataFrame:
    """
    Small description:
        Compute overnight gap log returns aligned to close date t:
        gap_t = log(Open_{t+1} / Close_t)

    Inputs:
        open_px(pd.DataFrame)       Open price panel
        close_px(pd.DataFrame)      Close price panel

    Outputs:
        gap_log(pd.DataFrame)       Gap log return panel
    """
    return np.log(open_px.shift(-1) / close_px)


def portfolio_log_returns_same_day(panel_log_rets: pd.DataFrame, weights_daily: pd.DataFrame) -> pd.Series:
    """
    Small description:
        Apply weights at date t to returns indexed at date t.

    Inputs:
        panel_log_rets(pd.DataFrame)    Return panel
        weights_daily(pd.DataFrame)     Daily weights

    Outputs:
        port_log(pd.Series)             Portfolio log returns
    """
    w = weights_daily.reindex(panel_log_rets.index).fillna(0.0)
    r = panel_log_rets.reindex(panel_log_rets.index)
    return (w * r).sum(axis=1)


def equity_curve_from_log_returns(log_ret: pd.Series, start: float = 1.0) -> pd.Series:
    """
    Small description:
        Convert log returns into cumulative equity curve.

    Inputs:
        log_ret(pd.Series)          Log return series
        start(float)               Starting equity

    Outputs:
        eq(pd.Series)              Equity curve
    """
    return start * np.exp(log_ret.fillna(0.0).cumsum())


def corr_and_beta(strategy_log: pd.Series, benchmark_log: pd.Series) -> tuple[float, float]:
    """
    Small description:
        Compute correlation and beta of strategy vs benchmark.

    Inputs:
        strategy_log(pd.Series)    Strategy log returns
        benchmark_log(pd.Series)   Benchmark log returns

    Outputs:
        corr(float)                Correlation
        beta(float)                Beta
    """
    df = pd.concat([strategy_log.rename("strat"), benchmark_log.rename("bench")], axis=1).dropna()
    if len(df) < 50:
        return np.nan, np.nan
    corr = float(df["strat"].corr(df["bench"]))
    var = float(df["bench"].var(ddof=1))
    beta = float(df["strat"].cov(df["bench"]) / var) if var > 0 else np.nan
    return corr, beta


def gross_exposures(weights_daily: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Small description:
        Compute gross long, gross short, and net exposure time series.

    Inputs:
        weights_daily(pd.DataFrame)    Daily weights

    Outputs:
        gross_long(pd.Series)          Gross long exposure
        gross_short(pd.Series)         Gross short exposure
        net(pd.Series)                 Net exposure
    """
    w = weights_daily.fillna(0.0)
    gross_long = w.clip(lower=0.0).sum(axis=1)
    gross_short = (-w.clip(upper=0.0)).sum(axis=1)
    net = gross_long - gross_short
    return gross_long, gross_short, net


def turnover_on_rebalance(weights_on_reb: pd.DataFrame) -> pd.Series:
    """
    Small description:
        Compute turnover on rebalance dates.

    Inputs:
        weights_on_reb(pd.DataFrame)   Rebalance weights

    Outputs:
        turn(pd.Series)                Turnover by rebalance date
    """
    w = weights_on_reb.fillna(0.0)
    dw = w.diff()
    turn = 0.5 * dw.abs().sum(axis=1)
    if len(turn) > 0:
        turn.iloc[0] = 0.0
    return turn


def apply_total_cost_on_rebalance_days(
    gross_log: pd.Series,
    reb_turnover: pd.Series,
    tc_bps_per_1turn: float,
    close_slippage_bps_per_1turn: float = 0.0,
) -> pd.Series:
    """
    Small description:
        Apply implementation costs only on rebalance days.

    Inputs:
        gross_log(pd.Series)                   Gross strategy log returns
        reb_turnover(pd.Series)                Rebalance turnover
        tc_bps_per_1turn(float)                Transaction cost in bps per 1-turn
        close_slippage_bps_per_1turn(float)    Slippage in bps per 1-turn

    Outputs:
        net_log(pd.Series)                     Net strategy log returns
    """
    total_bps = float(tc_bps_per_1turn) + float(close_slippage_bps_per_1turn)
    tc = (total_bps / 1e4) * reb_turnover
    tc = tc.reindex(gross_log.index).fillna(0.0)
    return gross_log - tc


def enforce_dollar_neutral_on_rebalance(weights_on_reb: pd.DataFrame, target_gross: float = 2.0) -> pd.DataFrame:
    """
    Small description:
        Dollar-neutralize rebalance weights to target gross exposure.

    Inputs:
        weights_on_reb(pd.DataFrame)   Rebalance weights
        target_gross(float)            Target total gross exposure

    Outputs:
        w_neutral(pd.DataFrame)        Dollar-neutralized weights
    """
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
    """
    Small description:
        Summarize panel width and average missing fraction.

    Inputs:
        panel(pd.DataFrame)            Data panel

    Outputs:
        n_names(int)                   Number of columns
        miss_frac(float)               Average NaN fraction
    """
    n_names = int(panel.shape[1])
    miss_frac = float(panel.isna().mean().mean()) if panel.size else float("nan")
    return n_names, miss_frac


def common_intersection_index(series_list: list[pd.Series]) -> pd.DatetimeIndex:
    """
    Small description:
        Find common non-null date intersection across series.

    Inputs:
        series_list(list[pd.Series])   List of time series

    Outputs:
        idx(pd.DatetimeIndex)          Common date index
    """
    if len(series_list) == 0:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(series_list[0].dropna().index)
    for s in series_list[1:]:
        idx = idx.intersection(pd.DatetimeIndex(s.dropna().index))
    return idx.sort_values()


def drawdown_curve(eq_curve: pd.Series) -> pd.Series:
    """
    Small description:
        Compute drawdown series from an equity curve.

    Inputs:
        eq_curve(pd.Series)            Equity curve

    Outputs:
        dd(pd.Series)                  Drawdown curve
    """
    peak = eq_curve.cummax()
    return eq_curve / peak - 1.0


def rolling_sharpe(log_ret: pd.Series, window: int = 126) -> pd.Series:
    """
    Small description:
        Compute rolling annualized Sharpe.

    Inputs:
        log_ret(pd.Series)             Log return series
        window(int)                    Rolling window length

    Outputs:
        rs(pd.Series)                  Rolling Sharpe series
    """
    ann_mean = log_ret.rolling(window).mean() * 252.0
    ann_vol = log_ret.rolling(window).std() * np.sqrt(252.0)
    return ann_mean / ann_vol


def first_valid_date_from_panels(*panels: pd.DataFrame) -> pd.Timestamp:
    """
    Small description:
        Find earliest common date with at least one valid observation in all panels.

    Inputs:
        panels(*pd.DataFrame)          One or more panels

    Outputs:
        first_date(pd.Timestamp)       Earliest valid common date
    """
    idx = None
    for p in panels:
        p_idx = p.index[p.notna().any(axis=1)]
        idx = p_idx if idx is None else idx.intersection(p_idx)
    if idx is None or len(idx) == 0:
        raise RuntimeError("Could not determine first valid common date from panels.")
    return pd.Timestamp(idx.min())


# ============================================================
# Exposure scaling (rebalance-only)
# ============================================================

def is_earnings_heavy_month(d: pd.Timestamp) -> bool:
    """
    Small description:
        Flag months treated as earnings-heavy.

    Inputs:
        d(pd.Timestamp)                Date

    Outputs:
        is_heavy(bool)                 Whether month is earnings-heavy
    """
    return int(d.month) in {1, 2, 4, 5, 7, 8, 10, 11}


def make_lambda_on_rebalance_dates(
    rebalance_dates: pd.DatetimeIndex,
    lambda_earn: float = 1.5,
    lambda_other: float = 0.2,
) -> pd.Series:
    """
    Small description:
        Build leverage multipliers on rebalance dates.

    Inputs:
        rebalance_dates(pd.DatetimeIndex)  Rebalance dates
        lambda_earn(float)                 Multiplier for earnings-heavy months
        lambda_other(float)                Multiplier for other months

    Outputs:
        lam(pd.Series)                     Lambda series on rebalance dates
    """
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    lam = pd.Series(lambda_other, index=rebalance_dates, dtype=float)
    earn_mask = rebalance_dates.map(lambda x: is_earnings_heavy_month(pd.Timestamp(x)))
    lam.loc[earn_mask] = float(lambda_earn)
    return lam


def apply_exposure_scaling_on_rebalance(weights_on_reb: pd.DataFrame, lam_on_reb: pd.Series) -> pd.DataFrame:
    """
    Small description:
        Apply leverage multipliers to rebalance weights.

    Inputs:
        weights_on_reb(pd.DataFrame)   Rebalance weights
        lam_on_reb(pd.Series)          Lambda series

    Outputs:
        w_scaled(pd.DataFrame)         Scaled weights
    """
    lam_on_reb = lam_on_reb.reindex(weights_on_reb.index).fillna(1.0)
    return weights_on_reb.mul(lam_on_reb, axis=0)


# ============================================================
# Tilting selection
# ============================================================

def _row_percentiles(row: pd.Series) -> pd.Series:
    """
    Small description:
        Convert row values into cross-sectional percentiles.

    Inputs:
        row(pd.Series)                 Signal row

    Outputs:
        pct(pd.Series)                 Percentile-ranked row
    """
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
    min_names: int = 200,
    ew_within_side: bool = True,
) -> pd.DataFrame:
    """
    Small description:
        Build buffered long-short weights using percentile thresholds.

    Inputs:
        signal(pd.DataFrame)               Signal panel
        rebalance_dates(pd.DatetimeIndex)  Rebalance dates
        long_entry(float)                  Long entry percentile threshold
        long_exit(float)                   Long exit percentile threshold
        short_entry(float)                 Short entry percentile threshold
        short_exit(float)                  Short exit percentile threshold
        min_names(int)                     Minimum valid names required
        ew_within_side(bool)               Equal-weight within long/short sides

    Outputs:
        w_out(pd.DataFrame)                Rebalance weights
    """
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
# Strategy generation
# ============================================================

@dataclass(frozen=True)
class TiltConfig:
    name: str
    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float


# ============================================================
# Results containers
# ============================================================

@dataclass
class VariantResult:
    universe: str
    name: str
    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float
    gross_metrics: dict
    net_metrics: dict
    avg_reb_turnover: float
    corr_to_bench: float
    beta_to_bench: float
    final_equity_net: float
    avg_gross_long: float
    avg_gross_short: float
    avg_net: float
    n_days: int
    median_tradable: float
    n_names: int
    miss_open: float
    miss_close: float


@dataclass
class VariantPayload:
    result: VariantResult
    gross_log: pd.Series
    net_log: pd.Series
    bench_log: pd.Series
    reb_turn: pd.Series


@dataclass
class CombinedVariantResult:
    name: str
    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float
    market_results: dict[str, VariantResult]
    combined_gross_metrics: dict
    combined_net_metrics: dict
    combined_corr_to_bench: float
    combined_beta_to_bench: float
    combined_final_equity_net: float
    avg_market_reb_turnover: float
    combined_gross_log: pd.Series
    combined_net_log: pd.Series
    combined_bench_log: pd.Series


# ============================================================
# Evaluation
# ============================================================

def evaluate_variant_rebalance_only(
    universe: str,
    name: str,
    gap_log: pd.DataFrame,
    weights_on_reb: pd.DataFrame,
    tc_bps_per_1turn: float,
    bench_cc_log_daily: pd.Series,
    long_entry: float,
    long_exit: float,
    short_entry: float,
    short_exit: float,
    *,
    meta: dict,
    close_slippage_bps_per_1turn: float = 0.0,
) -> VariantPayload:
    """
    Small description:
        Evaluate one strategy variant.

    Inputs:
        universe(str)                          Universe code
        name(str)                              Variant name
        gap_log(pd.DataFrame)                  Gap return panel
        weights_on_reb(pd.DataFrame)           Rebalance weights
        tc_bps_per_1turn(float)                Transaction cost bps
        bench_cc_log_daily(pd.Series)          Benchmark log returns
        long_entry(float)                      Long entry threshold
        long_exit(float)                       Long exit threshold
        short_entry(float)                     Short entry threshold
        short_exit(float)                      Short exit threshold
        meta(dict)                             Metadata dict
        close_slippage_bps_per_1turn(float)    Slippage bps

    Outputs:
        payload(VariantPayload)                Evaluation payload
    """
    w_daily = forward_fill_weights(weights_on_reb).reindex(gap_log.index).ffill().fillna(0.0)

    gross = portfolio_log_returns_same_day(gap_log, w_daily)

    reb_turn = turnover_on_rebalance(weights_on_reb)
    net = apply_total_cost_on_rebalance_days(
        gross,
        reb_turn,
        tc_bps_per_1turn=tc_bps_per_1turn,
        close_slippage_bps_per_1turn=close_slippage_bps_per_1turn,
    )

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

    result = VariantResult(
        universe=universe,
        name=name,
        long_entry=long_entry,
        long_exit=long_exit,
        short_entry=short_entry,
        short_exit=short_exit,
        gross_metrics=gross_m,
        net_metrics=net_m,
        avg_reb_turnover=float(reb_turn.mean()) if len(reb_turn) else float("nan"),
        corr_to_bench=corr,
        beta_to_bench=beta,
        final_equity_net=final_eq,
        avg_gross_long=float(gl.mean()) if len(gl) else float("nan"),
        avg_gross_short=float(gs.mean()) if len(gs) else float("nan"),
        avg_net=float(n.mean()) if len(n) else float("nan"),
        n_days=int(meta["n_days"]),
        median_tradable=float(meta["median_tradable"]),
        n_names=int(meta["n_names"]),
        miss_open=float(meta["miss_open"]),
        miss_close=float(meta["miss_close"]),
    )
    return VariantPayload(
        result=result,
        gross_log=gross,
        net_log=net,
        bench_log=bench,
        reb_turn=reb_turn,
    )


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
    """
    Small description:
        Prepare universe data, signal, benchmark, and metadata.

    Inputs:
        universe(str)                          Universe code
        start(str)                             Start date
        end(str | None)                        End date
        formation_window(int)                  Formation window
        rebalance(str)                         Rebalance frequency
        min_names(int)                         Minimum valid names
        use_vol_adjusted_signal(bool)          Whether to use vol-adjusted signal
        vol_window(int)                        Volatility lookback window
        lambda_earn_months(float)              Earnings-month leverage
        lambda_other_months(float)             Other-month leverage

    Outputs:
        gap_log(pd.DataFrame)                  Gap return panel
        signal(pd.DataFrame)                   Signal panel
        rebalance_dates(pd.DatetimeIndex)      Rebalance dates
        lam_on_reb(pd.Series)                  Lambda on rebalance dates
        bench_cc_log_daily(pd.Series)          Benchmark close-to-close log returns
        meta(dict)                             Metadata dict
    """
    u = universe.strip().upper()

    if u == "FTSE250":
        open_px, close_px = load_ftse250_open_close_from_parquet(
            start=start,
            end=end,
        )

        cc_log, gap_log = load_ftse250_precomputed_returns(
            start=start,
            end=end,
        )
    else:
        open_px, _, _, close_px, _, _ = get_universe_ohlcv(u, start=start, end=end, force=False)
        open_px = open_px.sort_index()
        close_px = close_px.sort_index()

        idx = open_px.index.intersection(close_px.index)
        open_px = open_px.loc[idx]
        close_px = close_px.loc[idx]

        gap_log = compute_gap_log_returns(open_px, close_px)
        cc_log = compute_log_returns(close_px)

    open_px = open_px.sort_index()
    close_px = close_px.sort_index()

    idx = open_px.index.intersection(close_px.index)
    open_px = open_px.loc[idx]
    close_px = close_px.loc[idx]

    idx2 = cc_log.index.intersection(gap_log.index).intersection(idx)
    cc_log = cc_log.loc[idx2]
    gap_log = gap_log.loc[idx2]

    _, miss_open = summarize_panel(open_px)
    n_names_close, miss_close = summarize_panel(close_px)

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
    tradable_counts = tradable_counts.loc[ok_dates]

    if len(signal) < 252:
        raise RuntimeError(f"{u}: Not enough data after filtering (need >=252, got {len(signal)}).")

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
        "median_tradable": float(tradable_counts.median()),
        "n_names": int(n_names_close),
        "miss_open": float(miss_open),
        "miss_close": float(miss_close),
    }
    return gap_log, signal, rebalance_dates, lam_on_reb, bench_cc_log_daily, meta


# ============================================================
# Printing
# ============================================================

def print_combined_variant(res: CombinedVariantResult, market_order: list[str]) -> None:
    """
    Small description:
        Print top combined strategy summary.

    Inputs:
        res(CombinedVariantResult)     Combined result
        market_order(list[str])        Market order

    Outputs:
        None
    """
    print(f"\n--- {res.name} ---")
    print("Thresholds:")
    print(f"  long_entry  = {res.long_entry:.3f}")
    print(f"  long_exit   = {res.long_exit:.3f}")
    print(f"  short_entry = {res.short_entry:.3f}")
    print(f"  short_exit  = {res.short_exit:.3f}")

    print("\nCOMBINED (equal-weight across markets, common dates):")
    print(f"  Avg market rebalance turnover: {res.avg_market_reb_turnover:.4f}")
    print("  GROSS:")
    print(f"    Sharpe:   {res.combined_gross_metrics['sharpe']:.4f}")
    print(f"    AnnRet:   {res.combined_gross_metrics['ann_return']:.4f}")
    print(f"    AnnVol:   {res.combined_gross_metrics['ann_vol']:.4f}")
    print(f"    MaxDD:    {res.combined_gross_metrics['max_drawdown']:.4f}")
    print("  NET:")
    print(f"    Sharpe:   {res.combined_net_metrics['sharpe']:.4f}")
    print(f"    AnnRet:   {res.combined_net_metrics['ann_return']:.4f}")
    print(f"    AnnVol:   {res.combined_net_metrics['ann_vol']:.4f}")
    print(f"    MaxDD:    {res.combined_net_metrics['max_drawdown']:.4f}")
    print(f"    Final Eq: {res.combined_final_equity_net:.4f}")
    print("  vs EW benchmark:")
    print(f"    Corr:     {res.combined_corr_to_bench:.4f}")
    print(f"    Beta:     {res.combined_beta_to_bench:.4f}")

    for u in market_order:
        mr = res.market_results[u]
        print(f"\n  {u}:")
        print(f"    NET Sharpe:   {mr.net_metrics['sharpe']:.4f}")
        print(f"    NET AnnRet:   {mr.net_metrics['ann_return']:.4f}")
        print(f"    NET AnnVol:   {mr.net_metrics['ann_vol']:.4f}")
        print(f"    NET MaxDD:    {mr.net_metrics['max_drawdown']:.4f}")
        print(f"    Final Eq:     {mr.final_equity_net:.4f}")
        print(f"    Corr:         {mr.corr_to_bench:.4f}")
        print(f"    Beta:         {mr.beta_to_bench:.4f}")
        print(f"    Avg Reb Turn: {mr.avg_reb_turnover:.4f}")


# ============================================================
# Market-level run
# ============================================================

def run_single_market(
    universe: str,
    tilts: list[TiltConfig],
    start: str,
    end: str | None,
    *,
    formation_window: int,
    rebalance: str,
    tc_bps_per_1turn: float,
    use_vol_adjusted_signal: bool,
    vol_window: int,
    dollar_neutralize: bool,
    target_gross: float,
    use_exposure_scaling: bool,
    lambda_earn_months: float,
    lambda_other_months: float,
    min_names: int,
    close_slippage_bps_per_1turn: float = 0.0,
    verbose: bool = True,
) -> tuple[dict[str, VariantPayload], dict, pd.Series]:
    """
    Small description:
        Run all strategy variants for one market.

    Inputs:
        universe(str)                          Universe code
        tilts(list[TiltConfig])                Strategy variants
        start(str)                             Start date
        end(str | None)                        End date
        formation_window(int)                  Formation window
        rebalance(str)                         Rebalance frequency
        tc_bps_per_1turn(float)                Transaction cost bps
        use_vol_adjusted_signal(bool)          Use vol-adjusted signal
        vol_window(int)                        Vol lookback
        dollar_neutralize(bool)                Whether to dollar-neutralize
        target_gross(float)                    Target gross
        use_exposure_scaling(bool)             Whether to scale exposure
        lambda_earn_months(float)              Earnings-month leverage
        lambda_other_months(float)             Other-month leverage
        min_names(int)                         Minimum valid names
        close_slippage_bps_per_1turn(float)    Slippage bps
        verbose(bool)                          Whether to print

    Outputs:
        payloads(dict[str, VariantPayload])    Results by variant
        meta(dict)                             Metadata
        bench_cc_log_daily(pd.Series)          Benchmark returns
    """
    u = universe.strip().upper()

    if verbose:
        print("\n" + "=" * 80)
        print(
            f"UNIVERSE: {u} | start={start} end={end} | rebalance={rebalance} | "
            f"min_names={min_names} | tc_bps={tc_bps_per_1turn} | close_slip_bps={close_slippage_bps_per_1turn}"
        )
        print("=" * 80)

    gap_log, signal, rebalance_dates, lam_on_reb, bench_cc_log_daily, meta = prepare_universe(
        u,
        start=start,
        end=end,
        formation_window=formation_window,
        rebalance=rebalance,
        min_names=min_names,
        use_vol_adjusted_signal=use_vol_adjusted_signal,
        vol_window=vol_window,
        lambda_earn_months=lambda_earn_months,
        lambda_other_months=lambda_other_months,
    )

    if verbose:
        print("\n--- DATA SUMMARY ---")
        print(f"names={meta['n_names']} | avg NaN frac (open)={meta['miss_open']:.3%} | (close)={meta['miss_close']:.3%}")
        print(f"sample days after filters: {meta['n_days']}")
        print(f"median tradable names/day: {meta['median_tradable']:.1f}")

    payloads: dict[str, VariantPayload] = {}

    for tilt in tqdm(tilts, desc=f"{u} tilts", leave=verbose):
        w_on_reb = tilted_buffered_long_short_weights(
            signal=signal,
            rebalance_dates=rebalance_dates,
            long_entry=tilt.long_entry,
            long_exit=tilt.long_exit,
            short_entry=tilt.short_entry,
            short_exit=tilt.short_exit,
            min_names=min_names,
            ew_within_side=True,
        )

        if dollar_neutralize:
            w_on_reb = enforce_dollar_neutral_on_rebalance(w_on_reb, target_gross=target_gross)
        if use_exposure_scaling:
            w_on_reb = apply_exposure_scaling_on_rebalance(w_on_reb, lam_on_reb)

        payload = evaluate_variant_rebalance_only(
            universe=u,
            name=tilt.name,
            gap_log=gap_log,
            weights_on_reb=w_on_reb,
            tc_bps_per_1turn=tc_bps_per_1turn,
            bench_cc_log_daily=bench_cc_log_daily,
            long_entry=tilt.long_entry,
            long_exit=tilt.long_exit,
            short_entry=tilt.short_entry,
            short_exit=tilt.short_exit,
            meta=meta,
            close_slippage_bps_per_1turn=close_slippage_bps_per_1turn,
        )
        payloads[tilt.name] = payload

    if verbose:
        bench_m = perf_metrics_from_log_returns(bench_cc_log_daily.dropna())
        print("\n--- BENCH BUY&Hold (close-to-close) ---")
        print(f"  Sharpe:   {bench_m['sharpe']:.4f}")
        print(f"  AnnRet:   {bench_m['ann_return']:.4f}")
        print(f"  AnnVol:   {bench_m['ann_vol']:.4f}")
        print(f"  MaxDD:    {bench_m['max_drawdown']:.4f}")

    return payloads, meta, bench_cc_log_daily


# ============================================================
# Cross-market combination
# ============================================================

def combine_variant_across_markets(
    tilt: TiltConfig,
    payloads_by_market: dict[str, dict[str, VariantPayload]],
    market_order: list[str],
) -> CombinedVariantResult:
    """
    Small description:
        Combine one variant across markets.

    Inputs:
        tilt(TiltConfig)                           Variant config
        payloads_by_market(dict[str, dict])        Payloads by market and variant
        market_order(list[str])                    Market order

    Outputs:
        cres(CombinedVariantResult)                Combined result
    """
    chosen_payloads = [payloads_by_market[u][tilt.name] for u in market_order]

    common_idx = common_intersection_index([p.net_log for p in chosen_payloads] + [p.bench_log for p in chosen_payloads])
    if len(common_idx) < 50:
        raise RuntimeError(f"{tilt.name}: not enough common dates across markets ({len(common_idx)}).")

    net_df = pd.concat([p.net_log.reindex(common_idx).rename(u) for p, u in zip(chosen_payloads, market_order)], axis=1)
    gross_df = pd.concat([p.gross_log.reindex(common_idx).rename(u) for p, u in zip(chosen_payloads, market_order)], axis=1)
    bench_df = pd.concat([p.bench_log.reindex(common_idx).rename(u) for p, u in zip(chosen_payloads, market_order)], axis=1)

    combined_net = net_df.mean(axis=1)
    combined_gross = gross_df.mean(axis=1)
    combined_bench = bench_df.mean(axis=1)

    combined_gross_m = perf_metrics_from_log_returns(combined_gross)
    combined_net_m = perf_metrics_from_log_returns(combined_net)
    combined_corr, combined_beta = corr_and_beta(combined_net, combined_bench)
    combined_eq = equity_curve_from_log_returns(combined_net)
    combined_final_eq = float(combined_eq.iloc[-1]) if len(combined_eq) else float("nan")

    market_results = {u: payloads_by_market[u][tilt.name].result for u in market_order}
    avg_turn = float(np.nanmean([market_results[u].avg_reb_turnover for u in market_order]))

    return CombinedVariantResult(
        name=tilt.name,
        long_entry=tilt.long_entry,
        long_exit=tilt.long_exit,
        short_entry=tilt.short_entry,
        short_exit=tilt.short_exit,
        market_results=market_results,
        combined_gross_metrics=combined_gross_m,
        combined_net_metrics=combined_net_m,
        combined_corr_to_bench=combined_corr,
        combined_beta_to_bench=combined_beta,
        combined_final_equity_net=combined_final_eq,
        avg_market_reb_turnover=avg_turn,
        combined_gross_log=combined_gross,
        combined_net_log=combined_net,
        combined_bench_log=combined_bench,
    )


def run_cross_market_combined(
    universes: list[str],
    tilts: list[TiltConfig],
    start: str,
    end: str | None,
    *,
    formation_window: int,
    rebalance: str,
    tc_bps_per_1turn: float,
    use_vol_adjusted_signal: bool,
    vol_window: int,
    dollar_neutralize: bool,
    target_gross: float,
    use_exposure_scaling: bool,
    lambda_earn_months: float,
    lambda_other_months: float,
    min_names_by_universe: dict[str, int],
    top_k_print: int = 10,
    close_slippage_bps_per_1turn: float = 0.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[CombinedVariantResult]]:
    """
    Small description:
        Run all variants across all markets and combine them.

    Inputs:
        universes(list[str])                       Universe list
        tilts(list[TiltConfig])                   Strategy variants
        start(str)                                Start date
        end(str | None)                           End date
        formation_window(int)                     Formation window
        rebalance(str)                            Rebalance frequency
        tc_bps_per_1turn(float)                   Transaction cost bps
        use_vol_adjusted_signal(bool)             Use vol-adjusted signal
        vol_window(int)                           Vol lookback
        dollar_neutralize(bool)                   Whether to dollar-neutralize
        target_gross(float)                       Target gross
        use_exposure_scaling(bool)                Whether to scale exposure
        lambda_earn_months(float)                 Earnings-month leverage
        lambda_other_months(float)                Other-month leverage
        min_names_by_universe(dict[str, int])     Minimum names by universe
        top_k_print(int)                          Number of top variants to print
        close_slippage_bps_per_1turn(float)       Slippage bps
        verbose(bool)                             Whether to print

    Outputs:
        df(pd.DataFrame)                          Summary dataframe
        ranked(list[CombinedVariantResult])       Ranked combined results
    """
    market_order = [u.strip().upper() for u in universes]
    payloads_by_market: dict[str, dict[str, VariantPayload]] = {}

    for u in tqdm(market_order, desc="Markets", leave=verbose):
        min_names = int(min_names_by_universe.get(u, 200))
        payloads, _, _ = run_single_market(
            universe=u,
            tilts=tilts,
            start=start,
            end=end,
            formation_window=formation_window,
            rebalance=rebalance,
            tc_bps_per_1turn=tc_bps_per_1turn,
            use_vol_adjusted_signal=use_vol_adjusted_signal,
            vol_window=vol_window,
            dollar_neutralize=dollar_neutralize,
            target_gross=target_gross,
            use_exposure_scaling=use_exposure_scaling,
            lambda_earn_months=lambda_earn_months,
            lambda_other_months=lambda_other_months,
            min_names=min_names,
            close_slippage_bps_per_1turn=close_slippage_bps_per_1turn,
            verbose=verbose,
        )
        payloads_by_market[u] = payloads

    combined_results: list[CombinedVariantResult] = []
    rows: list[dict] = []

    for tilt in tqdm(tilts, desc="Combining", leave=verbose):
        cres = combine_variant_across_markets(
            tilt=tilt,
            payloads_by_market=payloads_by_market,
            market_order=market_order,
        )
        combined_results.append(cres)

        rows.append({
            "variant": cres.name,
            "combined_net_sharpe": cres.combined_net_metrics.get("sharpe", np.nan),
        })

    ranked = sorted(
        combined_results,
        key=lambda r: float(r.combined_net_metrics.get("sharpe", np.nan)),
        reverse=True,
    )

    if verbose and len(ranked) > 0:
        print("\n" + "=" * 80)
        print("TOP STRATEGY")
        print("=" * 80)
        print_combined_variant(ranked[0], market_order)

    return pd.DataFrame(rows), ranked


# ============================================================
# Plotting / reporting
# ============================================================

def save_summary_stats(strategy_log: pd.Series, bench_log: pd.Series) -> pd.DataFrame:
    """
    Small description:
        Save full-sample strategy vs benchmark summary stats.

    Inputs:
        strategy_log(pd.Series)        Strategy log returns
        bench_log(pd.Series)           Benchmark log returns

    Outputs:
        summary(pd.DataFrame)          Summary table
    """
    strat_metrics = perf_metrics_from_log_returns(strategy_log.dropna())
    bench_metrics = perf_metrics_from_log_returns(bench_log.dropna())
    corr, beta = corr_and_beta(strategy_log, bench_log)

    summary = pd.DataFrame(
        [
            {
                "series": "Strategy",
                "ann_return": float(strat_metrics.get("ann_return", np.nan)),
                "ann_vol": float(strat_metrics.get("ann_vol", np.nan)),
                "sharpe": float(strat_metrics.get("sharpe", np.nan)),
                "max_drawdown": float(strat_metrics.get("max_drawdown", np.nan)),
                "corr_to_ftse250": float(corr),
                "beta_to_ftse250": float(beta),
            },
            {
                "series": "FTSE250",
                "ann_return": float(bench_metrics.get("ann_return", np.nan)),
                "ann_vol": float(bench_metrics.get("ann_vol", np.nan)),
                "sharpe": float(bench_metrics.get("sharpe", np.nan)),
                "max_drawdown": float(bench_metrics.get("max_drawdown", np.nan)),
                "corr_to_ftse250": 1.0,
                "beta_to_ftse250": 1.0,
            },
        ]
    )

    summary.to_csv(OUT_DIR / "strategy_vs_ftse250_summary.csv", index=False)
    return summary


def make_comparison_plots(strategy_log: pd.Series, bench_log: pd.Series) -> None:
    """
    Small description:
        Create full-sample plots and monthly return CSV.

    Inputs:
        strategy_log(pd.Series)        Strategy log returns
        bench_log(pd.Series)           Benchmark log returns

    Outputs:
        None
    """
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

    monthly_df = pd.DataFrame(
        {
            "Strategy": monthly_strat,
            "FTSE250": monthly_bench,
        }
    )
    monthly_df.to_csv(OUT_DIR / "monthly_returns_strategy_vs_ftse250.csv")

    plt.figure(figsize=(11, 6))
    plt.plot(eq_strat.index, eq_strat.values, label="Strategy")
    plt.plot(eq_bench.index, eq_bench.values, label="FTSE250")
    plt.title("Equity Curve: Strategy vs FTSE250")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "equity_curve_strategy_vs_ftse250.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(dd_strat.index, dd_strat.values, label="Strategy")
    plt.plot(dd_bench.index, dd_bench.values, label="FTSE250")
    plt.title("Drawdown: Strategy vs FTSE250")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "drawdown_strategy_vs_ftse250.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(roll_vol_strat.index, roll_vol_strat.values, label="Strategy")
    plt.plot(roll_vol_bench.index, roll_vol_bench.values, label="FTSE250")
    plt.title("Rolling Volatility (63d): Strategy vs FTSE250")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rolling_vol_strategy_vs_ftse250.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(roll_sharpe_strat.index, roll_sharpe_strat.values, label="Strategy")
    plt.plot(roll_sharpe_bench.index, roll_sharpe_bench.values, label="FTSE250")
    plt.title("Rolling Sharpe (126d): Strategy vs FTSE250")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rolling_sharpe_strategy_vs_ftse250.png", dpi=150)
    plt.close()

    monthly_df_plot = monthly_df.dropna().copy()
    if not monthly_df_plot.empty:
        x = np.arange(len(monthly_df_plot))
        width = 0.4

        plt.figure(figsize=(14, 6))
        plt.bar(x - width / 2, monthly_df_plot["Strategy"].values, width=width, label="Strategy")
        plt.bar(x + width / 2, monthly_df_plot["FTSE250"].values, width=width, label="FTSE250")
        plt.title("Monthly Returns: Strategy vs FTSE250")
        plt.xlabel("Month")
        plt.ylabel("Monthly Return")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)

        tick_idx = np.arange(0, len(monthly_df_plot), max(1, len(monthly_df_plot) // 12))
        tick_labels = [monthly_df_plot.index[i].strftime("%Y-%m") for i in tick_idx]
        plt.xticks(tick_idx, tick_labels, rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(OUT_DIR / "monthly_returns_strategy_vs_ftse250.png", dpi=150)
        plt.close()


def make_full_sample_periods_from_logs(
    strategy_log: pd.Series,
    bench_log: pd.Series,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Small description:
        Build regime periods from actual common strategy/benchmark sample.

    Inputs:
        strategy_log(pd.Series)        Strategy log returns
        bench_log(pd.Series)           Benchmark log returns

    Outputs:
        periods(list[tuple])           List of named periods
    """
    common = strategy_log.dropna().index.intersection(bench_log.dropna().index).sort_values()
    if len(common) == 0:
        raise RuntimeError("No common dates between strategy and benchmark logs.")

    full_start = pd.Timestamp(common.min())
    full_end = pd.Timestamp(common.max())

    raw_periods = [
        ("Pre_GFC", pd.Timestamp("2002-01-01"), pd.Timestamp("2007-06-30")),
        ("GFC", pd.Timestamp("2007-07-01"), pd.Timestamp("2009-06-30")),
        ("Post_GFC_Recovery", pd.Timestamp("2009-07-01"), pd.Timestamp("2012-12-31")),
        ("QE_Expansion", pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")),
        ("COVID", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
        ("Post_COVID_Inflation", pd.Timestamp("2021-01-01"), pd.Timestamp("2099-12-31")),
    ]

    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]] = [
        ("Full_Sample", full_start, full_end)
    ]

    for name, start, end in raw_periods:
        s = max(start, full_start)
        e = min(end, full_end)
        if s <= e:
            periods.append((name, s, e))

    return periods


def summarize_by_periods(
    strategy_log: pd.Series,
    bench_log: pd.Series,
    out_path: Path,
) -> pd.DataFrame:
    """
    Small description:
        Build detailed period-by-period table for strategy and benchmark.

    Inputs:
        strategy_log(pd.Series)        Strategy log returns
        bench_log(pd.Series)           Benchmark log returns
        out_path(Path)                 Output CSV path

    Outputs:
        out(pd.DataFrame)              Detailed period table
    """
    common = strategy_log.dropna().index.intersection(bench_log.dropna().index)
    strategy_log = strategy_log.loc[common].sort_index()
    bench_log = bench_log.loc[common].sort_index()

    if len(common) == 0:
        raise RuntimeError("No common dates between strategy and benchmark logs.")

    periods = make_full_sample_periods_from_logs(strategy_log, bench_log)

    rows: list[dict] = []

    for period_name, start, end in periods:
        s_log = strategy_log.loc[start:end].dropna()
        b_log = bench_log.loc[start:end].dropna()

        common_period = s_log.index.intersection(b_log.index)
        s_log = s_log.loc[common_period]
        b_log = b_log.loc[common_period]

        if len(common_period) < 20:
            continue

        s_m = perf_metrics_from_log_returns(s_log)
        b_m = perf_metrics_from_log_returns(b_log)
        corr, beta = corr_and_beta(s_log, b_log)

        rows.append(
            {
                "period": period_name,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "n_days": int(len(common_period)),
                "series": "Strategy",
                "ann_return": float(s_m.get("ann_return", np.nan)),
                "ann_vol": float(s_m.get("ann_vol", np.nan)),
                "sharpe": float(s_m.get("sharpe", np.nan)),
                "max_drawdown": float(s_m.get("max_drawdown", np.nan)),
                "corr_to_benchmark": float(corr),
                "beta_to_benchmark": float(beta),
                "final_equity": float(equity_curve_from_log_returns(s_log).iloc[-1]),
            }
        )

        rows.append(
            {
                "period": period_name,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "n_days": int(len(common_period)),
                "series": "Benchmark",
                "ann_return": float(b_m.get("ann_return", np.nan)),
                "ann_vol": float(b_m.get("ann_vol", np.nan)),
                "sharpe": float(b_m.get("sharpe", np.nan)),
                "max_drawdown": float(b_m.get("max_drawdown", np.nan)),
                "corr_to_benchmark": 1.0,
                "beta_to_benchmark": 1.0,
                "final_equity": float(equity_curve_from_log_returns(b_log).iloc[-1]),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def compute_period_metric_row(
    strategy_log: pd.Series,
    bench_log: pd.Series,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict | None:
    """
    Small description:
        Compute compact strategy-only metrics for one period.

    Inputs:
        strategy_log(pd.Series)        Strategy log returns
        bench_log(pd.Series)           Benchmark log returns
        period_name(str)               Period label
        start(pd.Timestamp)            Period start
        end(pd.Timestamp)              Period end

    Outputs:
        row(dict | None)               Compact metrics row or None
    """
    s_log = strategy_log.loc[start:end].dropna()
    b_log = bench_log.loc[start:end].dropna()

    common = s_log.index.intersection(b_log.index)
    s_log = s_log.loc[common]
    b_log = b_log.loc[common]

    if len(common) < 20:
        return None

    s_m = perf_metrics_from_log_returns(s_log)
    corr, beta = corr_and_beta(s_log, b_log)

    return {
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


def build_period_compact_table(
    strategy_log: pd.Series,
    bench_log: pd.Series,
    out_path: Path,
) -> pd.DataFrame:
    """
    Small description:
        Build compact one-row-per-period strategy table.

    Inputs:
        strategy_log(pd.Series)        Strategy log returns
        bench_log(pd.Series)           Benchmark log returns
        out_path(Path)                 Output CSV path

    Outputs:
        out(pd.DataFrame)              Compact period table
    """
    periods = make_full_sample_periods_from_logs(strategy_log, bench_log)

    rows: list[dict] = []
    for period_name, start, end in periods:
        row = compute_period_metric_row(strategy_log, bench_log, period_name, start, end)
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def print_period_compact_table(period_df: pd.DataFrame) -> None:
    """
    Small description:
        Print compact period summary in robustness style.

    Inputs:
        period_df(pd.DataFrame)        Compact period table

    Outputs:
        None
    """
    if period_df.empty:
        print("No valid period rows to print.")
        return

    print("\n" + "=" * 80)
    print("PERIOD SUMMARY")
    print("=" * 80)

    name_w = max(12, period_df["period"].astype(str).map(len).max())

    for _, row in period_df.iterrows():
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


def run_robustness_tests(
    universes: list[str],
    tilts: list[TiltConfig],
    min_names_by_universe: dict[str, int],
    start: str,
) -> pd.DataFrame:
    """
    Small description:
        Run cost/slippage robustness tests from a common sample start.

    Inputs:
        universes(list[str])                       Universe list
        tilts(list[TiltConfig])                   Strategy variants
        min_names_by_universe(dict[str, int])     Minimum names by universe
        start(str)                                Start date

    Outputs:
        out(pd.DataFrame)                         Robustness table
    """
    scenarios = [
        {"scenario": "base_realistic", "tc": 10.0, "slip": 10.0},
        {"scenario": "tc_25bps", "tc": 25.0, "slip": 10.0},
        {"scenario": "tc_50bps", "tc": 50.0, "slip": 10.0},
        {"scenario": "tc_75bps", "tc": 75.0, "slip": 10.0},
        {"scenario": "tc_100bps", "tc": 100.0, "slip": 10.0},
    ]

    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)

    rows = []
    name_w = 15

    for sc in scenarios:
        _, ranked = run_cross_market_combined(
            universes=universes,
            tilts=tilts,
            start=start,
            end=None,
            formation_window=1,
            rebalance="monthly",
            tc_bps_per_1turn=sc["tc"],
            use_vol_adjusted_signal=True,
            vol_window=20,
            dollar_neutralize=True,
            target_gross=2.0,
            use_exposure_scaling=True,
            lambda_earn_months=1.5,
            lambda_other_months=0.2,
            min_names_by_universe=min_names_by_universe,
            top_k_print=1,
            close_slippage_bps_per_1turn=sc["slip"],
            verbose=False,
        )

        best = ranked[0]

        sharpe = best.combined_net_metrics["sharpe"]
        annret = best.combined_net_metrics["ann_return"]
        maxdd = best.combined_net_metrics["max_drawdown"]
        beta = best.combined_beta_to_bench
        annvol = best.combined_net_metrics["ann_vol"]
        corr = best.combined_corr_to_bench

        print(
            f"{sc['scenario']:>{name_w}} | "
            f"tc={sc['tc']:>5.1f} | "
            f"slip={sc['slip']:>5.1f} | "
            f"Sharpe={sharpe:>6.3f} | "
            f"AnnRet={annret:>6.3f} | "
            f"MaxDD={maxdd:>7.3f} | "
            f"Beta={beta:>7.3f}"
        )

        rows.append({
            "scenario": sc["scenario"],
            "tc_bps": sc["tc"],
            "slip_bps": sc["slip"],
            "sharpe": sharpe,
            "ann_return": annret,
            "max_dd": maxdd,
            "beta_to_ftse250": beta,
            "ann_vol": annvol,
            "corr_to_ftse250": corr,
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Small description:
        Run locked FTSE250 strategy on full available local sample.

    Inputs:
        None

    Outputs:
        None
    """
    ALL_MARKETS = ["FTSE250"]

    TILTS = [
        TiltConfig(
            name="LOCKED_FTSE250",
            long_entry=0.120,
            long_exit=0.250,
            short_entry=0.920,
            short_exit=0.700,
        )
    ]

    MIN_NAMES = {
        "FTSE250": 200,
    }

    # Earliest available date from local parquet panels
    open_px, close_px = load_ftse250_open_close_from_parquet(start="1900-01-01", end=None)
    requested_start = first_valid_date_from_panels(open_px, close_px).date().isoformat()

    print("\n" + "=" * 80)
    print(f"RUNNING FULL SAMPLE FROM {requested_start} TO LATEST AVAILABLE DATE")
    print("=" * 80)

    df, ranked = run_cross_market_combined(
        universes=ALL_MARKETS,
        tilts=TILTS,
        start=requested_start,
        end=None,
        formation_window=1,
        rebalance="monthly",
        tc_bps_per_1turn=25.0,
        use_vol_adjusted_signal=True,
        vol_window=20,
        dollar_neutralize=True,
        target_gross=2.0,
        use_exposure_scaling=True,
        lambda_earn_months=1.5,
        lambda_other_months=0.2,
        min_names_by_universe=MIN_NAMES,
        top_k_print=1,
        close_slippage_bps_per_1turn=10.0,
    )

    out_csv = OUT_DIR / "cross_market_combined_random_search_full_sample.csv"
    df.to_csv(out_csv, index=False)

    if len(ranked) == 0:
        raise RuntimeError("No ranked strategy results were produced.")

    best = ranked[0]
    strategy_log = best.combined_net_log.rename("Strategy")
    bench_log = best.combined_bench_log.rename("FTSE250")

    combined_ts = pd.DataFrame(
        {
            "strategy_log": strategy_log,
            "ftse250_log": bench_log,
            "strategy_eq": equity_curve_from_log_returns(strategy_log),
            "ftse250_eq": equity_curve_from_log_returns(bench_log),
        }
    )
    combined_ts.to_csv(OUT_DIR / "strategy_vs_ftse250_timeseries_full_sample.csv")

    summary = save_summary_stats(strategy_log, bench_log)
    make_comparison_plots(strategy_log, bench_log)

    period_detail_df = summarize_by_periods(
        strategy_log=strategy_log,
        bench_log=bench_log,
        out_path=OUT_DIR / "strategy_vs_ftse250_by_period_detailed.csv",
    )

    period_compact_df = build_period_compact_table(
        strategy_log=strategy_log,
        bench_log=bench_log,
        out_path=OUT_DIR / "strategy_vs_ftse250_by_period_compact.csv",
    )

    robustness_df = run_robustness_tests(
        universes=ALL_MARKETS,
        tilts=TILTS,
        min_names_by_universe=MIN_NAMES,
        start=requested_start,
    )
    robustness_df.to_csv(OUT_DIR / "robustness_tests_close_slippage_and_tc.csv", index=False)

    actual_common = strategy_log.dropna().index.intersection(bench_log.dropna().index).sort_values()
    actual_start = actual_common.min().date().isoformat()
    actual_end = actual_common.max().date().isoformat()

    print("\n" + "=" * 80)
    print("SUMMARY STATS | FULL SAMPLE")
    print("=" * 80)
    print(f"Requested start: {requested_start}")
    print(f"Actual common sample used: {actual_start} -> {actual_end}")
    print(summary.to_string(index=False))

    print_period_compact_table(period_compact_df)

    print("\n" + "=" * 80)
    print("DETAILED PERIOD TABLE")
    print("=" * 80)
    print(period_detail_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Strategies tested: {len(TILTS)}")
    print(f"Requested start used: {requested_start}")
    print(f"Actual common sample used: {actual_start} -> {actual_end}")
    print(f"Saved main CSV: {out_csv}")
    print(f"Saved outputs to: {OUT_DIR}")
    print("Generated files:")
    print(f"  {OUT_DIR / 'strategy_vs_ftse250_summary.csv'}")
    print(f"  {OUT_DIR / 'strategy_vs_ftse250_timeseries_full_sample.csv'}")
    print(f"  {OUT_DIR / 'strategy_vs_ftse250_by_period_detailed.csv'}")
    print(f"  {OUT_DIR / 'strategy_vs_ftse250_by_period_compact.csv'}")
    print(f"  {OUT_DIR / 'equity_curve_strategy_vs_ftse250.png'}")
    print(f"  {OUT_DIR / 'drawdown_strategy_vs_ftse250.png'}")
    print(f"  {OUT_DIR / 'rolling_vol_strategy_vs_ftse250.png'}")
    print(f"  {OUT_DIR / 'rolling_sharpe_strategy_vs_ftse250.png'}")
    print(f"  {OUT_DIR / 'monthly_returns_strategy_vs_ftse250.csv'}")
    print(f"  {OUT_DIR / 'monthly_returns_strategy_vs_ftse250.png'}")
    print(f"  {OUT_DIR / 'robustness_tests_close_slippage_and_tc.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

