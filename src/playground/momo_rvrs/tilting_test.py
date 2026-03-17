from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def common_intersection_index(series_list: list[pd.Series]) -> pd.DatetimeIndex:
    if len(series_list) == 0:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(series_list[0].dropna().index)
    for s in series_list[1:]:
        idx = idx.intersection(pd.DatetimeIndex(s.dropna().index))
    return idx.sort_values()


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
    min_names: int = 200,
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
# Strategy generation
# ============================================================


@dataclass(frozen=True)
class TiltConfig:
    name: str
    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float


def generate_random_tilts(
    n_random: int,
    seed: int = 42,
    include_base: bool = True,
) -> list[TiltConfig]:
    """
    Generate valid buffered long/short tilts.

    Constraints enforced:
      long_entry  < long_exit
      short_exit  < short_entry
      long_entry  in [0.01, 0.15]
      long_exit   in [0.05, 0.35]
      short_exit  in [0.65, 0.99]
      short_entry in [0.75, 0.99]
    """
    rng = np.random.default_rng(seed)
    tilts: list[TiltConfig] = []
    seen: set[tuple[float, float, float, float]] = set()

    if include_base:
        base = TiltConfig("BASE symmetric", 0.050, 0.100, 0.950, 0.900)
        tilts.append(base)
        seen.add((base.long_entry, base.long_exit, base.short_entry, base.short_exit))

    tries = 0
    max_tries = max(10000, n_random * 50)

    while len(tilts) < n_random + int(include_base) and tries < max_tries:
        tries += 1

        le = float(np.round(rng.uniform(0.01, 0.15), 3))
        lx = float(np.round(rng.uniform(max(le + 0.01, 0.05), 0.35), 3))

        sx = float(np.round(rng.uniform(0.65, 0.97), 3))
        se_low = max(sx + 0.01, 0.75)
        if se_low >= 0.99:
            continue
        se = float(np.round(rng.uniform(se_low, 0.99), 3))

        key = (le, lx, se, sx)
        if key in seen:
            continue

        seen.add(key)
        tilts.append(TiltConfig(f"RandTilt_{len(tilts):05d}", le, lx, se, sx))

    if len(tilts) < n_random + int(include_base):
        raise RuntimeError(
            f"Could only generate {len(tilts)} unique tilts "
            f"(requested {n_random + int(include_base)})."
        )

    return tilts


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
) -> VariantPayload:
    w_daily = forward_fill_weights(weights_on_reb).reindex(gap_log.index).ffill().fillna(0.0)

    gross = portfolio_log_returns_same_day(gap_log, w_daily)

    reb_turn = turnover_on_rebalance(weights_on_reb)
    net = apply_tc_on_rebalance_days(gross, reb_turn, tc_bps_per_1turn)

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
        "median_tradable": float(tradable_counts.loc[signal.index].median()),
        "n_names": int(n_names_close),
        "miss_open": float(miss_open),
        "miss_close": float(miss_close),
    }
    return gap_log, signal, rebalance_dates, lam_on_reb, bench_cc_log_daily, meta


# ============================================================
# Printing
# ============================================================


def print_variant_like_before(res: VariantResult) -> None:
    print(f"\n--- {res.name} | {res.universe} ---")
    print("Thresholds:")
    print(f"  long_entry  = {res.long_entry:.3f}")
    print(f"  long_exit   = {res.long_exit:.3f}")
    print(f"  short_entry = {res.short_entry:.3f}")
    print(f"  short_exit  = {res.short_exit:.3f}")
    print(f"Avg TURNOVER per REBALANCE: {res.avg_reb_turnover:.4f}")
    print(f"Avg gross long:            {res.avg_gross_long:.3f}")
    print(f"Avg gross short:           {res.avg_gross_short:.3f}")
    print(f"Avg net:                   {res.avg_net:.3f}")
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
    print("vs Benchmark (close-to-close):")
    print(f"  Corr:      {res.corr_to_bench:.4f}")
    print(f"  Beta:      {res.beta_to_bench:.4f}")


def print_combined_variant(res: CombinedVariantResult, market_order: list[str]) -> None:
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
) -> tuple[dict[str, VariantPayload], dict, pd.Series]:
    u = universe.strip().upper()

    print("\n" + "=" * 80)
    print(f"UNIVERSE: {u} | start={start} end={end} | rebalance={rebalance} | min_names={min_names}")
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

    print("\n--- DATA SUMMARY ---")
    print(f"names={meta['n_names']} | avg NaN frac (open)={meta['miss_open']:.3%} | (close)={meta['miss_close']:.3%}")
    print(f"sample days after filters: {meta['n_days']}")
    print(f"median tradable names/day: {meta['median_tradable']:.1f}")

    payloads: dict[str, VariantPayload] = {}

    for tilt in tqdm(tilts, desc=f"{u} tilts", leave=True):
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
        )
        payloads[tilt.name] = payload

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
) -> pd.DataFrame:
    market_order = [u.strip().upper() for u in universes]
    payloads_by_market: dict[str, dict[str, VariantPayload]] = {}

    for u in tqdm(market_order, desc="Markets", leave=True):
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
        )
        payloads_by_market[u] = payloads

    combined_results: list[CombinedVariantResult] = []
    rows: list[dict] = []

    for tilt in tqdm(tilts, desc="Combining across markets", leave=True):
        try:
            cres = combine_variant_across_markets(
                tilt=tilt,
                payloads_by_market=payloads_by_market,
                market_order=market_order,
            )
            combined_results.append(cres)

            row = {
                "variant": cres.name,
                "LE": cres.long_entry,
                "LX": cres.long_exit,
                "SE": cres.short_entry,
                "SX": cres.short_exit,
                "combined_net_sharpe": float(cres.combined_net_metrics.get("sharpe", np.nan)),
                "combined_net_annret": float(cres.combined_net_metrics.get("ann_return", np.nan)),
                "combined_net_annvol": float(cres.combined_net_metrics.get("ann_vol", np.nan)),
                "combined_net_maxdd": float(cres.combined_net_metrics.get("max_drawdown", np.nan)),
                "combined_gross_sharpe": float(cres.combined_gross_metrics.get("sharpe", np.nan)),
                "combined_corr_to_bench": float(cres.combined_corr_to_bench),
                "combined_beta_to_bench": float(cres.combined_beta_to_bench),
                "combined_final_eq_net": float(cres.combined_final_equity_net),
                "avg_market_reb_turnover": float(cres.avg_market_reb_turnover),
            }

            for u in market_order:
                mr = cres.market_results[u]
                prefix = u.lower()
                row[f"{prefix}_net_sharpe"] = float(mr.net_metrics.get("sharpe", np.nan))
                row[f"{prefix}_net_annret"] = float(mr.net_metrics.get("ann_return", np.nan))
                row[f"{prefix}_net_annvol"] = float(mr.net_metrics.get("ann_vol", np.nan))
                row[f"{prefix}_net_maxdd"] = float(mr.net_metrics.get("max_drawdown", np.nan))
                row[f"{prefix}_gross_sharpe"] = float(mr.gross_metrics.get("sharpe", np.nan))
                row[f"{prefix}_corr_to_bench"] = float(mr.corr_to_bench)
                row[f"{prefix}_beta_to_bench"] = float(mr.beta_to_bench)
                row[f"{prefix}_final_eq_net"] = float(mr.final_equity_net)
                row[f"{prefix}_avg_reb_turn"] = float(mr.avg_reb_turnover)
                row[f"{prefix}_n_days"] = int(mr.n_days)
                row[f"{prefix}_median_tradable"] = float(mr.median_tradable)
                row[f"{prefix}_n_names"] = int(mr.n_names)
                row[f"{prefix}_miss_open"] = float(mr.miss_open)
                row[f"{prefix}_miss_close"] = float(mr.miss_close)

            row["error"] = ""
            rows.append(row)

        except Exception as e:
            rows.append(
                {
                    "variant": tilt.name,
                    "LE": tilt.long_entry,
                    "LX": tilt.long_exit,
                    "SE": tilt.short_entry,
                    "SX": tilt.short_exit,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    ranked = sorted(
        combined_results,
        key=lambda r: float(r.combined_net_metrics.get("sharpe", np.nan)),
        reverse=True,
    )

    print("\n" + "=" * 80)
    print(f"TOP {top_k_print} STRATEGIES BY COMBINED NET SHARPE")
    print(f"Combined across: {', '.join(market_order)} | equal-weight | common dates only")
    print("=" * 80)

    for k, res in enumerate(ranked[:top_k_print], 1):
        print(f"\n#{k}")
        print_combined_variant(res, market_order)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================


def main() -> None:
    ALL_MARKETS = ["FTSE250"]

    N_RANDOM_TILTS = 2000
    RNG_SEED = 42

    TILTS = generate_random_tilts(
        n_random=N_RANDOM_TILTS,
        seed=RNG_SEED,
        include_base=True,
    )

    MIN_NAMES = {
        "FTSE250": 200,
    }

    df = run_cross_market_combined(
        universes=ALL_MARKETS,
        tilts=TILTS,
        start="2020-01-01",
        end=None,
        formation_window=1,
        rebalance="monthly",
        tc_bps_per_1turn=10.0,
        use_vol_adjusted_signal=True,
        vol_window=20,
        dollar_neutralize=True,
        target_gross=2.0,
        use_exposure_scaling=True,
        lambda_earn_months=1.5,
        lambda_other_months=0.2,
        min_names_by_universe=MIN_NAMES,
        top_k_print=10,
    )

    out_csv = "src/playground/momo_rvrs/cross_market_combined_random_search.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"Random tilts tested: {len(TILTS)}")
    print(f"Saved: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

