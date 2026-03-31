from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
# Locked strategy from cluster center
# ============================================================

LOCKED_LONG_ENTRY = 0.100
LOCKED_LONG_EXIT = 0.250
LOCKED_SHORT_ENTRY = 0.920
LOCKED_SHORT_EXIT = 0.750

# ============================================================
# Paths / config
# ============================================================

RANDOM_SEARCH_CSV = Path("src/playground/momo_rvrs/cross_market_combined_random_search.csv")
OUT_DIR = Path("src/playground/momo_rvrs/robustness_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = "FTSE250"
START = "2020-01-01"
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
MIN_NAMES = 200

# ============================================================
# Helpers (copied/aligned with your current file logic)
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
# Data classes
# ============================================================


@dataclass
class StrategyRunResult:
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
    gross_log: pd.Series
    net_log: pd.Series
    bench_log: pd.Series
    reb_turn: pd.Series


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
# Strategy run
# ============================================================


def run_strategy(
    universe: str,
    start: str,
    end: str | None,
    *,
    long_entry: float,
    long_exit: float,
    short_entry: float,
    short_exit: float,
    formation_window: int = FORMATION_WINDOW,
    rebalance: str = REBALANCE,
    tc_bps_per_1turn: float = TC_BPS_PER_1TURN,
    use_vol_adjusted_signal: bool = USE_VOL_ADJUSTED_SIGNAL,
    vol_window: int = VOL_WINDOW,
    dollar_neutralize: bool = DOLLAR_NEUTRALIZE,
    target_gross: float = TARGET_GROSS,
    use_exposure_scaling: bool = USE_EXPOSURE_SCALING,
    lambda_earn_months: float = LAMBDA_EARN_MONTHS,
    lambda_other_months: float = LAMBDA_OTHER_MONTHS,
    min_names: int = MIN_NAMES,
    name: str = "locked_strategy",
) -> StrategyRunResult:
    gap_log, signal, rebalance_dates, lam_on_reb, bench_cc_log_daily, meta = prepare_universe(
        universe,
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

    w_on_reb = tilted_buffered_long_short_weights(
        signal=signal,
        rebalance_dates=rebalance_dates,
        long_entry=long_entry,
        long_exit=long_exit,
        short_entry=short_entry,
        short_exit=short_exit,
        min_names=min_names,
        ew_within_side=True,
    )

    if dollar_neutralize:
        w_on_reb = enforce_dollar_neutral_on_rebalance(w_on_reb, target_gross=target_gross)

    if use_exposure_scaling:
        w_on_reb = apply_exposure_scaling_on_rebalance(w_on_reb, lam_on_reb)

    w_daily = forward_fill_weights(w_on_reb).reindex(gap_log.index).ffill().fillna(0.0)

    gross = portfolio_log_returns_same_day(gap_log, w_daily)

    reb_turn = turnover_on_rebalance(w_on_reb)
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

    return StrategyRunResult(
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
        gross_log=gross,
        net_log=net,
        bench_log=bench,
        reb_turn=reb_turn,
    )


# ============================================================
# Random-search robustness summary
# ============================================================


def summarize_random_search(csv_path: Path, top_n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads your random-search CSV and summarizes the top-N strategies.

    Expected threshold columns from your file:
      LE, LX, SE, SX
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find random-search CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "combined_net_sharpe" in df.columns:
        sharpe_col = "combined_net_sharpe"
    elif "ftse250_net_sharpe" in df.columns:
        sharpe_col = "ftse250_net_sharpe"
    else:
        raise ValueError("Could not find a usable Sharpe column in the CSV.")

    required = ["LE", "LX", "SE", "SX", sharpe_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in random search CSV: {missing}")

    top_df = df.sort_values(sharpe_col, ascending=False).head(top_n).copy()

    summary = top_df[["LE", "LX", "SE", "SX"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    summary["range"] = summary["max"] - summary["min"]

    top_df.to_csv(OUT_DIR / f"top_{top_n}_random_search_rows.csv", index=False)
    summary.to_csv(OUT_DIR / f"top_{top_n}_random_search_summary.csv")

    print("\n" + "=" * 80)
    print(f"TOP {top_n} PARAMETER CLUSTER SUMMARY")
    print("=" * 80)
    print(summary)

    return top_df, summary


# ============================================================
# Subperiod robustness
# ============================================================


def run_subperiod_analysis() -> pd.DataFrame:
    periods = [
        ("2020_2021", "2020-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023_2024", "2023-01-01", "2024-12-31"),
        ("2025_plus", "2025-01-01", None),
        ("full_sample", START, END),
    ]

    rows: list[dict] = []

    for label, start, end in periods:
        try:
            res = run_strategy(
                universe=UNIVERSE,
                start=start,
                end=end,
                long_entry=LOCKED_LONG_ENTRY,
                long_exit=LOCKED_LONG_EXIT,
                short_entry=LOCKED_SHORT_ENTRY,
                short_exit=LOCKED_SHORT_EXIT,
                name=label,
            )

            rows.append(
                {
                    "period": label,
                    "start": start,
                    "end": end,
                    "LE": res.long_entry,
                    "LX": res.long_exit,
                    "SE": res.short_entry,
                    "SX": res.short_exit,
                    "gross_sharpe": float(res.gross_metrics.get("sharpe", np.nan)),
                    "gross_annret": float(res.gross_metrics.get("ann_return", np.nan)),
                    "gross_annvol": float(res.gross_metrics.get("ann_vol", np.nan)),
                    "gross_maxdd": float(res.gross_metrics.get("max_drawdown", np.nan)),
                    "net_sharpe": float(res.net_metrics.get("sharpe", np.nan)),
                    "net_annret": float(res.net_metrics.get("ann_return", np.nan)),
                    "net_annvol": float(res.net_metrics.get("ann_vol", np.nan)),
                    "net_maxdd": float(res.net_metrics.get("max_drawdown", np.nan)),
                    "final_eq_net": float(res.final_equity_net),
                    "avg_reb_turnover": float(res.avg_reb_turnover),
                    "corr_to_bench": float(res.corr_to_bench),
                    "beta_to_bench": float(res.beta_to_bench),
                    "n_days": int(res.n_days),
                    "median_tradable": float(res.median_tradable),
                }
            )

        except Exception as e:
            rows.append(
                {
                    "period": label,
                    "start": start,
                    "end": end,
                    "LE": LOCKED_LONG_ENTRY,
                    "LX": LOCKED_LONG_EXIT,
                    "SE": LOCKED_SHORT_ENTRY,
                    "SX": LOCKED_SHORT_EXIT,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "locked_strategy_subperiods.csv", index=False)

    print("\n" + "=" * 80)
    print("SUBPERIOD ROBUSTNESS")
    print("=" * 80)
    print(out.to_string(index=False))

    return out


# ============================================================
# Local grid around locked strategy
# ============================================================


def run_local_grid() -> pd.DataFrame:
    long_entries = [0.08, 0.10, 0.12]
    long_exits = [0.20, 0.25, 0.30]
    short_entries = [0.90, 0.92, 0.94]
    short_exits = [0.70, 0.75, 0.80]

    rows: list[dict] = []
    total = len(long_entries) * len(long_exits) * len(short_entries) * len(short_exits)

    i = 0
    for le in long_entries:
        for lx in long_exits:
            for se in short_entries:
                for sx in short_exits:
                    i += 1
                    try:
                        res = run_strategy(
                            universe=UNIVERSE,
                            start=START,
                            end=END,
                            long_entry=le,
                            long_exit=lx,
                            short_entry=se,
                            short_exit=sx,
                            name=f"grid_{i:03d}",
                        )

                        rows.append(
                            {
                                "name": res.name,
                                "LE": le,
                                "LX": lx,
                                "SE": se,
                                "SX": sx,
                                "gross_sharpe": float(res.gross_metrics.get("sharpe", np.nan)),
                                "gross_annret": float(res.gross_metrics.get("ann_return", np.nan)),
                                "gross_annvol": float(res.gross_metrics.get("ann_vol", np.nan)),
                                "gross_maxdd": float(res.gross_metrics.get("max_drawdown", np.nan)),
                                "net_sharpe": float(res.net_metrics.get("sharpe", np.nan)),
                                "net_annret": float(res.net_metrics.get("ann_return", np.nan)),
                                "net_annvol": float(res.net_metrics.get("ann_vol", np.nan)),
                                "net_maxdd": float(res.net_metrics.get("max_drawdown", np.nan)),
                                "final_eq_net": float(res.final_equity_net),
                                "avg_reb_turnover": float(res.avg_reb_turnover),
                                "corr_to_bench": float(res.corr_to_bench),
                                "beta_to_bench": float(res.beta_to_bench),
                            }
                        )

                        print(
                            f"[grid {i:03d}/{total}] "
                            f"LE={le:.2f} LX={lx:.2f} SE={se:.2f} SX={sx:.2f} "
                            f"| net_sharpe={float(res.net_metrics.get('sharpe', np.nan)):.4f}"
                        )

                    except Exception as e:
                        rows.append(
                            {
                                "name": f"grid_{i:03d}",
                                "LE": le,
                                "LX": lx,
                                "SE": se,
                                "SX": sx,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )

    out = pd.DataFrame(rows)
    if "net_sharpe" in out.columns:
        out = out.sort_values("net_sharpe", ascending=False, na_position="last").reset_index(drop=True)

    out.to_csv(OUT_DIR / "locked_strategy_local_grid.csv", index=False)

    print("\n" + "=" * 80)
    print("TOP 20 LOCAL GRID RESULTS")
    print("=" * 80)
    print(out.head(20).to_string(index=False))

    return out


# ============================================================
# Single locked-strategy full sample
# ============================================================


def run_locked_strategy_full_sample() -> pd.DataFrame:
    res = run_strategy(
        universe=UNIVERSE,
        start=START,
        end=END,
        long_entry=LOCKED_LONG_ENTRY,
        long_exit=LOCKED_LONG_EXIT,
        short_entry=LOCKED_SHORT_ENTRY,
        short_exit=LOCKED_SHORT_EXIT,
        name="locked_cluster_center",
    )

    row = {
        "name": res.name,
        "LE": res.long_entry,
        "LX": res.long_exit,
        "SE": res.short_entry,
        "SX": res.short_exit,
        "gross_sharpe": float(res.gross_metrics.get("sharpe", np.nan)),
        "gross_annret": float(res.gross_metrics.get("ann_return", np.nan)),
        "gross_annvol": float(res.gross_metrics.get("ann_vol", np.nan)),
        "gross_maxdd": float(res.gross_metrics.get("max_drawdown", np.nan)),
        "net_sharpe": float(res.net_metrics.get("sharpe", np.nan)),
        "net_annret": float(res.net_metrics.get("ann_return", np.nan)),
        "net_annvol": float(res.net_metrics.get("ann_vol", np.nan)),
        "net_maxdd": float(res.net_metrics.get("max_drawdown", np.nan)),
        "final_eq_net": float(res.final_equity_net),
        "avg_reb_turnover": float(res.avg_reb_turnover),
        "avg_gross_long": float(res.avg_gross_long),
        "avg_gross_short": float(res.avg_gross_short),
        "avg_net": float(res.avg_net),
        "corr_to_bench": float(res.corr_to_bench),
        "beta_to_bench": float(res.beta_to_bench),
        "n_days": int(res.n_days),
        "median_tradable": float(res.median_tradable),
        "n_names": int(res.n_names),
        "miss_open": float(res.miss_open),
        "miss_close": float(res.miss_close),
    }

    out = pd.DataFrame([row])
    out.to_csv(OUT_DIR / "locked_strategy_full_sample.csv", index=False)

    eq_df = pd.DataFrame(
        {
            "gross_log": res.gross_log,
            "net_log": res.net_log,
            "bench_log": res.bench_log.reindex(res.net_log.index),
            "gross_eq": equity_curve_from_log_returns(res.gross_log),
            "net_eq": equity_curve_from_log_returns(res.net_log),
            "bench_eq": equity_curve_from_log_returns(res.bench_log.reindex(res.net_log.index).fillna(0.0)),
        }
    )
    eq_df.to_csv(OUT_DIR / "locked_strategy_equity_curve.csv")

    print("\n" + "=" * 80)
    print("LOCKED STRATEGY | FULL SAMPLE")
    print("=" * 80)
    print(out.to_string(index=False))

    return out


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 80)
    print("ROBUSTNESS ANALYSIS | FTSE250 Momentum Reversal")
    print("=" * 80)
    print(f"Universe:            {UNIVERSE}")
    print(f"Start:               {START}")
    print(f"End:                 {END}")
    print(f"Min names:           {MIN_NAMES}")
    print(f"Rebalance:           {REBALANCE}")
    print(f"Formation window:    {FORMATION_WINDOW}")
    print(f"Vol-adjusted signal: {USE_VOL_ADJUSTED_SIGNAL}")
    print(f"Vol window:          {VOL_WINDOW}")
    print(f"TC bps / 1 turn:     {TC_BPS_PER_1TURN}")
    print()
    print("LOCKED STRATEGY")
    print(f"  long_entry  = {LOCKED_LONG_ENTRY:.3f}")
    print(f"  long_exit   = {LOCKED_LONG_EXIT:.3f}")
    print(f"  short_entry = {LOCKED_SHORT_ENTRY:.3f}")
    print(f"  short_exit  = {LOCKED_SHORT_EXIT:.3f}")

    print("\n[1/4] Summarizing top random-search strategies...")
    summarize_random_search(RANDOM_SEARCH_CSV, top_n=100)

    print("\n[2/4] Running locked strategy on full sample...")
    run_locked_strategy_full_sample()

    print("\n[3/4] Running subperiod robustness...")
    run_subperiod_analysis()

    print("\n[4/4] Running local parameter grid...")
    run_local_grid()

    print("\n" + "=" * 80)
    print("Saved outputs:")
    print(f"  {OUT_DIR / 'top_100_random_search_rows.csv'}")
    print(f"  {OUT_DIR / 'top_100_random_search_summary.csv'}")
    print(f"  {OUT_DIR / 'locked_strategy_full_sample.csv'}")
    print(f"  {OUT_DIR / 'locked_strategy_equity_curve.csv'}")
    print(f"  {OUT_DIR / 'locked_strategy_subperiods.csv'}")
    print(f"  {OUT_DIR / 'locked_strategy_local_grid.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
