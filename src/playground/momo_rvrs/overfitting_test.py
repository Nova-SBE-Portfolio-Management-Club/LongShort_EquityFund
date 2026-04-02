from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.playground.momo_rvrs.data_loader import (
    get_benchmark_ohlc,
    load_ftse250_open_close_from_parquet,
    load_ftse250_precomputed_returns,
)
from src.playground.momo_rvrs.locked_strategy import (
    apply_total_cost_on_rebalance_days,
    corr_and_beta,
    enforce_dollar_neutral_on_rebalance,
    equity_curve_from_log_returns,
    gross_exposures,
    portfolio_log_returns_same_day,
    tilted_buffered_long_short_weights,
    turnover_on_rebalance,
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

UNIVERSE = "FTSE250"
START = "2009-07-01"
END = None

MIN_NAMES = 200
TARGET_GROSS = 2.0

TC_BPS = 25.0
SLIPPAGE_BPS = 10.0

OUT_DIR = Path("src/playground/momo_rvrs/overfitting_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of CSCV blocks. Must be even.
N_BLOCKS = 8


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class StrategySpec:
    formation_window: int
    vol_window: int
    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float
    min_names: int = MIN_NAMES
    target_gross: float = TARGET_GROSS

    @property
    def name(self) -> str:
        return (
            f"fw={self.formation_window}"
            f"_vw={self.vol_window}"
            f"_le={self.long_entry:.2f}"
            f"_lx={self.long_exit:.2f}"
            f"_se={self.short_entry:.2f}"
            f"_sx={self.short_exit:.2f}"
        )


@dataclass
class CandidateRun:
    name: str
    net_log: pd.Series
    gross_log: pd.Series
    weights_on_reb: pd.DataFrame
    reb_turn: pd.Series
    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    beta_to_bench: float
    corr_to_bench: float
    avg_reb_turnover: float
    final_equity: float


@dataclass
class PBOResult:
    pbo: float
    n_candidates: int
    n_splits: int
    lambda_logits: pd.Series
    oos_pct_ranks: pd.Series
    split_summary: pd.DataFrame


# ============================================================
# Helpers
# ============================================================

def sharpe_ratio(log_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio from log returns.
    """
    x = log_returns.dropna()
    if len(x) < 2:
        return np.nan

    mu = x.mean()
    sigma = x.std(ddof=1)

    if sigma == 0 or np.isnan(sigma):
        return np.nan

    return float(np.sqrt(periods_per_year) * mu / sigma)


def safe_pct_rank(scores: pd.Series, selected_name: str) -> float:
    """
    Percentile rank in (0,1) of selected_name within scores.

    Higher score is better.
    """
    s = scores.dropna().sort_values()
    if len(s) == 0 or selected_name not in s.index:
        return np.nan

    # sort ascending, so best has highest index
    loc = s.index.get_loc(selected_name)
    return float((loc + 0.5) / len(s))


def split_into_equal_blocks(df: pd.DataFrame, n_blocks: int) -> list[pd.DataFrame]:
    """
    Split a time series DataFrame into contiguous equal-sized blocks.
    Any leftover rows are dropped from the front.
    """
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even.")

    t = len(df)
    block_len = t // n_blocks
    if block_len < 2:
        raise ValueError("Not enough data for requested number of blocks.")

    trimmed = df.iloc[-(block_len * n_blocks):].copy()

    blocks: list[pd.DataFrame] = []
    for i in range(n_blocks):
        start = i * block_len
        end = (i + 1) * block_len
        blocks.append(trimmed.iloc[start:end])

    return blocks


# ============================================================
# Data prep
# ============================================================

def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load FTSE250 open/close-derived return panels and benchmark log returns.
    """
    open_px, close_px = load_ftse250_open_close_from_parquet(start=START, end=END)
    cc_log, gap_log = load_ftse250_precomputed_returns(start=START, end=END)

    bench = get_benchmark_ohlc(UNIVERSE, start=START, end=END, force=False)
    bench_log = np.log(bench["Close"]).diff()

    idx = gap_log.index.intersection(cc_log.index).intersection(bench_log.index)

    gap_log = gap_log.loc[idx].sort_index()
    cc_log = cc_log.loc[idx].sort_index()
    bench_log = bench_log.loc[idx].sort_index()

    return gap_log, cc_log, bench_log


# ============================================================
# Strategy runner
# ============================================================

def build_weights_from_spec(
    gap_log: pd.DataFrame,
    cc_log: pd.DataFrame,
    spec: StrategySpec,
) -> pd.DataFrame:
    """
    Build rebalance-date weights for one strategy specification.
    """
    signal = vol_adjusted_formation_signal(
        log_rets=cc_log,
        formation_window=spec.formation_window,
        vol_window=spec.vol_window,
        vol_floor=1e-6,
    )

    valid_dates = gap_log.index[gap_log.notna().any(axis=1)]
    signal = signal.loc[signal.index.intersection(valid_dates)]

    tradable_counts = signal.notna().sum(axis=1)
    ok_dates = tradable_counts[tradable_counts >= spec.min_names].index
    signal = signal.loc[ok_dates]

    rebalance_dates = get_monthly_rebalance_dates(signal.index)

    w_on_reb = tilted_buffered_long_short_weights(
        signal=signal,
        rebalance_dates=rebalance_dates,
        long_entry=spec.long_entry,
        long_exit=spec.long_exit,
        short_entry=spec.short_entry,
        short_exit=spec.short_exit,
        min_names=spec.min_names,
        ew_within_side=True,
    )

    w_on_reb = enforce_dollar_neutral_on_rebalance(
        w_on_reb,
        target_gross=spec.target_gross,
    )

    return w_on_reb


def evaluate_spec(
    spec: StrategySpec,
    gap_log: pd.DataFrame,
    cc_log: pd.DataFrame,
    bench_log: pd.Series,
    tc_bps: float,
    slippage_bps: float,
) -> CandidateRun:
    """
    Run one candidate strategy and return its results.
    """
    weights_on_reb = build_weights_from_spec(gap_log, cc_log, spec)

    w_daily = forward_fill_weights(weights_on_reb).reindex(gap_log.index).ffill().fillna(0.0)

    gross = portfolio_log_returns_same_day(gap_log, w_daily)
    reb_turn = turnover_on_rebalance(weights_on_reb)

    net = apply_total_cost_on_rebalance_days(
        gross,
        reb_turn,
        tc_bps_per_1turn=tc_bps,
        close_slippage_bps_per_1turn=slippage_bps,
    )

    common = net.index.intersection(bench_log.index)
    gross = gross.loc[common].dropna()
    net = net.loc[common].dropna()
    bench = bench_log.loc[common].dropna()

    common2 = gross.index.intersection(net.index).intersection(bench.index)
    gross = gross.loc[common2]
    net = net.loc[common2]
    bench = bench.loc[common2]

    net_metrics = perf_metrics_from_log_returns(net)
    corr, beta = corr_and_beta(net, bench)
    eq = equity_curve_from_log_returns(net)

    return CandidateRun(
        name=spec.name,
        net_log=net,
        gross_log=gross,
        weights_on_reb=weights_on_reb,
        reb_turn=reb_turn,
        sharpe=float(net_metrics.get("sharpe", np.nan)),
        ann_return=float(net_metrics.get("ann_return", np.nan)),
        ann_vol=float(net_metrics.get("ann_vol", np.nan)),
        max_drawdown=float(net_metrics.get("max_drawdown", np.nan)),
        beta_to_bench=float(beta),
        corr_to_bench=float(corr),
        avg_reb_turnover=float(reb_turn.mean()) if len(reb_turn) else np.nan,
        final_equity=float(eq.iloc[-1]) if len(eq) else np.nan,
    )


def generate_candidate_specs() -> list[StrategySpec]:
    """
    Generate a realistic grid of strategy variants.

    Keep this grid aligned with the variants you genuinely explored.
    """
    specs: list[StrategySpec] = []

    formation_windows = [5, 10, 15, 20]
    vol_windows = [10, 20, 40]

    long_entries = [0.05, 0.10, 0.15]
    long_exits = [0.25, 0.35, 0.45]

    short_entries = [0.85, 0.90, 0.95]
    short_exits = [0.65, 0.75, 0.85]

    for fw, vw, le, lx, se, sx in product(
        formation_windows,
        vol_windows,
        long_entries,
        long_exits,
        short_entries,
        short_exits,
    ):
        # Must make economic sense:
        # long-entry < long-exit, and short-exit < short-entry
        if not (le < lx):
            continue
        if not (sx < se):
            continue

        specs.append(
            StrategySpec(
                formation_window=fw,
                vol_window=vw,
                long_entry=le,
                long_exit=lx,
                short_entry=se,
                short_exit=sx,
            )
        )

    return specs


def generate_candidate_returns(
    gap_log: pd.DataFrame,
    cc_log: pd.DataFrame,
    bench_log: pd.Series,
    tc_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all candidate specs and return:
    1) return matrix: dates x strategy variants
    2) summary table per candidate
    """
    specs = generate_candidate_specs()

    returns_dict: dict[str, pd.Series] = {}
    summary_rows: list[dict] = []

    print("\nGenerating candidate strategies...")
    print(f"Total candidate specs: {len(specs)}")

    for i, spec in enumerate(specs, start=1):
        try:
            run = evaluate_spec(
                spec=spec,
                gap_log=gap_log,
                cc_log=cc_log,
                bench_log=bench_log,
                tc_bps=tc_bps,
                slippage_bps=slippage_bps,
            )

            returns_dict[run.name] = run.net_log
            summary_rows.append(
                {
                    "name": run.name,
                    "formation_window": spec.formation_window,
                    "vol_window": spec.vol_window,
                    "long_entry": spec.long_entry,
                    "long_exit": spec.long_exit,
                    "short_entry": spec.short_entry,
                    "short_exit": spec.short_exit,
                    "sharpe": run.sharpe,
                    "ann_return": run.ann_return,
                    "ann_vol": run.ann_vol,
                    "max_drawdown": run.max_drawdown,
                    "beta_to_bench": run.beta_to_bench,
                    "corr_to_bench": run.corr_to_bench,
                    "avg_reb_turnover": run.avg_reb_turnover,
                    "final_equity": run.final_equity,
                }
            )

            if i % 25 == 0 or i == len(specs):
                print(f"  Completed {i}/{len(specs)}")

        except Exception as e:
            print(f"  Skipped {spec.name} | error: {e}")

    candidate_returns = pd.DataFrame(returns_dict).sort_index()
    candidate_summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)

    return candidate_returns, candidate_summary


# ============================================================
# PBO / CSCV
# ============================================================

def compute_cscv_pbo(
    candidate_returns: pd.DataFrame,
    n_blocks: int = N_BLOCKS,
    metric_func: Callable[[pd.Series], float] | None = None,
) -> PBOResult:
    """
    Compute Probability of Backtest Overfitting via CSCV.

    candidate_returns:
        index = dates
        columns = candidate strategy names
        values = return series
    """
    if metric_func is None:
        metric_func = lambda x: sharpe_ratio(x, periods_per_year=252)

    if candidate_returns.shape[1] < 2:
        raise ValueError("Need at least two candidate strategies.")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even.")

    blocks = split_into_equal_blocks(candidate_returns, n_blocks)
    half = n_blocks // 2

    train_splits = list(combinations(range(n_blocks), half))

    lambda_list: list[float] = []
    pct_rank_list: list[float] = []
    split_rows: list[dict] = []

    for split_id, train_idx in enumerate(train_splits, start=1):
        test_idx = [i for i in range(n_blocks) if i not in train_idx]

        train_df = pd.concat([blocks[i] for i in train_idx], axis=0)
        test_df = pd.concat([blocks[i] for i in test_idx], axis=0)

        train_scores = train_df.apply(metric_func, axis=0).replace([np.inf, -np.inf], np.nan)
        test_scores = test_df.apply(metric_func, axis=0).replace([np.inf, -np.inf], np.nan)

        if train_scores.dropna().empty or test_scores.dropna().empty:
            continue

        best_in_sample = train_scores.idxmax()

        pct_rank = safe_pct_rank(test_scores, best_in_sample)
        if np.isnan(pct_rank):
            continue

        lam = float(np.log(pct_rank / (1.0 - pct_rank)))

        lambda_list.append(lam)
        pct_rank_list.append(pct_rank)

        split_rows.append(
            {
                "split_id": split_id,
                "train_blocks": str(train_idx),
                "test_blocks": str(tuple(test_idx)),
                "best_in_sample": best_in_sample,
                "train_score": float(train_scores.loc[best_in_sample]),
                "test_score": float(test_scores.loc[best_in_sample]),
                "oos_pct_rank": pct_rank,
                "lambda_logit": lam,
                "is_overfit_flag": int(lam < 0),
            }
        )

    if len(lambda_list) == 0:
        raise ValueError("No valid CSCV splits were evaluated.")

    lambda_s = pd.Series(lambda_list, name="lambda_logit")
    pct_rank_s = pd.Series(pct_rank_list, name="oos_pct_rank")
    split_summary = pd.DataFrame(split_rows)

    pbo = float((lambda_s < 0).mean())

    return PBOResult(
        pbo=pbo,
        n_candidates=candidate_returns.shape[1],
        n_splits=len(lambda_s),
        lambda_logits=lambda_s,
        oos_pct_ranks=pct_rank_s,
        split_summary=split_summary,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 90)
    print("OVERFITTING TEST | FTSE250 MOMENTUM REVERSAL")
    print("=" * 90)
    print(f"Universe:        {UNIVERSE}")
    print(f"Sample:          {START} -> latest")
    print(f"Min names:       {MIN_NAMES}")
    print(f"Target gross:    {TARGET_GROSS}")
    print(f"Costs:           {TC_BPS} bps TC + {SLIPPAGE_BPS} bps slippage")
    print(f"CSCV blocks:     {N_BLOCKS}")
    print("=" * 90)

    gap_log, cc_log, bench_log = prepare_data()

    candidate_returns, candidate_summary = generate_candidate_returns(
        gap_log=gap_log,
        cc_log=cc_log,
        bench_log=bench_log,
        tc_bps=TC_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )

    if candidate_returns.empty or candidate_returns.shape[1] < 2:
        raise ValueError("Not enough successful candidate strategies to run PBO.")

    # Save raw candidate outputs
    candidate_returns.to_csv(OUT_DIR / "candidate_returns.csv")
    candidate_summary.to_csv(OUT_DIR / "candidate_summary.csv", index=False)

    pbo_result = compute_cscv_pbo(
        candidate_returns=candidate_returns,
        n_blocks=N_BLOCKS,
        metric_func=lambda x: sharpe_ratio(x, periods_per_year=252),
    )

    pbo_result.split_summary.to_csv(OUT_DIR / "pbo_split_summary.csv", index=False)
    pbo_result.lambda_logits.to_csv(OUT_DIR / "pbo_lambda_logits.csv", index=False)
    pbo_result.oos_pct_ranks.to_csv(OUT_DIR / "pbo_oos_pct_ranks.csv", index=False)

    top10 = candidate_summary.head(10).copy()

    print("\n" + "=" * 90)
    print("TOP 10 CANDIDATES BY FULL-SAMPLE SHARPE")
    print("=" * 90)
    cols = [
        "name",
        "sharpe",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "avg_reb_turnover",
    ]
    print(top10[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("PBO RESULT")
    print("=" * 90)
    print(f"Candidates evaluated:   {pbo_result.n_candidates}")
    print(f"Valid CSCV splits:      {pbo_result.n_splits}")
    print(f"PBO:                    {pbo_result.pbo:.4f}")
    print(f"Avg OOS pct rank:       {pbo_result.oos_pct_ranks.mean():.4f}")
    print(f"Median OOS pct rank:    {pbo_result.oos_pct_ranks.median():.4f}")

    print("\nInterpretation:")
    if pbo_result.pbo < 0.20:
        print("  Low overfitting risk signal.")
    elif pbo_result.pbo < 0.50:
        print("  Moderate overfitting risk. Be cautious.")
    else:
        print("  High overfitting risk. Strategy search likely overfit.")

    print("\nSaved files:")
    print(f"  {OUT_DIR / 'candidate_returns.csv'}")
    print(f"  {OUT_DIR / 'candidate_summary.csv'}")
    print(f"  {OUT_DIR / 'pbo_split_summary.csv'}")
    print(f"  {OUT_DIR / 'pbo_lambda_logits.csv'}")
    print(f"  {OUT_DIR / 'pbo_oos_pct_ranks.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()

