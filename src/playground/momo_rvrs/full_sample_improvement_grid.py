# src/playground/momo_rvrs/full_sample_improvement_grid.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.playground.momo_rvrs.data_loader import load_ftse250_open_close_from_parquet
from src.playground.momo_rvrs.locked_strategy import (
    TiltConfig,
    corr_and_beta,
    equity_curve_from_log_returns,
    first_valid_date_from_panels,
    run_single_market,
)

# ============================================================
# Paths / config
# ============================================================

OUT_DIR = Path("src/playground/momo_rvrs/full_sample_improvement_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = "FTSE250"
END = None

REBALANCE = "monthly"
TC_BPS_PER_1TURN = 25.0
CLOSE_SLIPPAGE_BPS_PER_1TURN = 10.0
USE_VOL_ADJUSTED_SIGNAL = True
VOL_WINDOW = 20
DOLLAR_NEUTRALIZE = True
TARGET_GROSS = 2.0

TOP_N_FULL_SAMPLE_PRINT = 12
TOP_N_BY_PERIOD_PRINT = 8


# ============================================================
# Candidate spec
# ============================================================

@dataclass(frozen=True)
class CandidateSpec:
    name: str
    formation_window: int
    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float
    use_exposure_scaling: bool
    lambda_earn_months: float
    lambda_other_months: float
    min_names: int


# ============================================================
# Candidate grid
# ============================================================

def build_candidate_specs() -> list[CandidateSpec]:
    """
    Small description:
        Build a targeted improvement grid for the full-sample FTSE250 test.

    Inputs:
        None

    Outputs:
        specs(list[CandidateSpec])     Candidate strategy configurations
    """
    threshold_sets = [
        ("current_locked", 0.120, 0.250, 0.920, 0.700),
        ("buffered_1",     0.100, 0.250, 0.920, 0.750),
        ("buffered_2",     0.080, 0.200, 0.920, 0.800),
        ("buffered_3",     0.100, 0.300, 0.900, 0.800),
    ]

    formation_windows = [1, 3, 5, 10]

    scaling_modes = [
        ("scale_on_strong", True, 1.5, 0.2),
        ("scale_on_mild",   True, 1.2, 0.6),
        ("scale_off",       False, 1.0, 1.0),
    ]

    min_names_list = [200, 250]

    specs: list[CandidateSpec] = []

    for fw in formation_windows:
        for th_name, le, lx, se, sx in threshold_sets:
            for scale_name, use_scale, lam_earn, lam_other in scaling_modes:
                for min_names in min_names_list:
                    name = (
                        f"fw{fw}_"
                        f"{th_name}_"
                        f"{scale_name}_"
                        f"mn{min_names}"
                    )

                    specs.append(
                        CandidateSpec(
                            name=name,
                            formation_window=fw,
                            long_entry=le,
                            long_exit=lx,
                            short_entry=se,
                            short_exit=sx,
                            use_exposure_scaling=use_scale,
                            lambda_earn_months=lam_earn,
                            lambda_other_months=lam_other,
                            min_names=min_names,
                        )
                    )

    return specs


# ============================================================
# Period helpers
# ============================================================

def make_regime_periods_for_logs(
    strategy_log: pd.Series,
    bench_log: pd.Series,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Small description:
        Build regime periods clipped to the actual common sample for the given run.

    Inputs:
        strategy_log(pd.Series)        Strategy net log returns
        bench_log(pd.Series)           Benchmark log returns

    Outputs:
        periods(list[tuple])           Named periods with clipped dates
    """
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

    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for name, start, end in raw_periods:
        s = max(start, full_start)
        e = min(end, full_end)
        if s <= e:
            periods.append((name, s, e))

    return periods


def compute_period_row(
    candidate_name: str,
    strategy_log: pd.Series,
    bench_log: pd.Series,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict | None:
    """
    Small description:
        Compute compact period metrics for one candidate.

    Inputs:
        candidate_name(str)            Candidate name
        strategy_log(pd.Series)        Strategy net log returns
        bench_log(pd.Series)           Benchmark log returns
        period_name(str)               Period name
        start(pd.Timestamp)            Start date
        end(pd.Timestamp)              End date

    Outputs:
        row(dict | None)               Period summary row
    """
    s_log = strategy_log.loc[start:end].dropna()
    b_log = bench_log.loc[start:end].dropna()

    common = s_log.index.intersection(b_log.index)
    s_log = s_log.loc[common]
    b_log = b_log.loc[common]

    if len(common) < 20:
        return None

    ann_mean = float(s_log.mean() * 252.0)
    ann_vol = float(s_log.std(ddof=1) * np.sqrt(252.0))
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan

    eq = equity_curve_from_log_returns(s_log)
    peak = eq.cummax()
    maxdd = float((eq / peak - 1.0).min()) if len(eq) else np.nan

    corr, beta = corr_and_beta(s_log, b_log)

    return {
        "candidate": candidate_name,
        "period": period_name,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "n_days": int(len(common)),
        "sharpe": float(sharpe),
        "ann_return": float(ann_mean),
        "ann_vol": float(ann_vol),
        "max_drawdown": float(maxdd),
        "corr_to_benchmark": float(corr),
        "beta_to_benchmark": float(beta),
        "final_equity": float(eq.iloc[-1]) if len(eq) else np.nan,
    }


# ============================================================
# Formatting helpers
# ============================================================

def print_top_full_sample(df: pd.DataFrame, n: int) -> None:
    """
    Small description:
        Print top-N full-sample candidates in a clean compact format.

    Inputs:
        df(pd.DataFrame)               Full-sample ranking table
        n(int)                         Number of rows to print

    Outputs:
        None
    """
    print("\n" + "=" * 100)
    print("TOP FULL-SAMPLE CANDIDATES")
    print("=" * 100)

    show = df.head(n).copy()
    if show.empty:
        print("No rows to print.")
        return

    name_w = max(18, show["candidate"].astype(str).map(len).max())

    for _, row in show.iterrows():
        print(
            f"{row['candidate']:>{name_w}} | "
            f"fw={int(row['formation_window']):>2d} | "
            f"scale={'on ' if bool(row['use_exposure_scaling']) else 'off'} | "
            f"mn={int(row['min_names']):>3d} | "
            f"LE={row['long_entry']:.2f} LX={row['long_exit']:.2f} "
            f"SE={row['short_entry']:.2f} SX={row['short_exit']:.2f} | "
            f"Sharpe={row['net_sharpe']:>6.3f} | "
            f"AnnRet={row['net_annret']:>6.3f} | "
            f"AnnVol={row['net_annvol']:>6.3f} | "
            f"MaxDD={row['net_maxdd']:>7.3f} | "
            f"Turn={row['avg_reb_turnover']:>6.3f}"
        )


def print_top_by_period(period_df: pd.DataFrame, n: int) -> None:
    """
    Small description:
        Print top-N candidates for each period in robustness-summary style.

    Inputs:
        period_df(pd.DataFrame)        Period summary table
        n(int)                         Number of rows to print per period

    Outputs:
        None
    """
    periods = [
        "Full_Sample",
        "Pre_GFC",
        "GFC",
        "Post_GFC_Recovery",
        "QE_Expansion",
        "COVID",
        "Post_COVID_Inflation",
    ]

    for period in periods:
        sub = period_df.loc[period_df["period"] == period].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(
            ["sharpe", "ann_return", "max_drawdown"],
            ascending=[False, False, False],
        ).head(n)

        print("\n" + "=" * 100)
        print(f"ROBUSTNESS SUMMARY | {period}")
        print("=" * 100)

        name_w = max(18, sub["candidate"].astype(str).map(len).max())

        for _, row in sub.iterrows():
            print(
                f"{row['candidate']:>{name_w}} | "
                f"{row['start']} -> {row['end']} | "
                f"n={int(row['n_days']):>4d} | "
                f"Sharpe={row['sharpe']:>6.3f} | "
                f"AnnRet={row['ann_return']:>6.3f} | "
                f"AnnVol={row['ann_vol']:>6.3f} | "
                f"MaxDD={row['max_drawdown']:>7.3f} | "
                f"Beta={row['beta_to_benchmark']:>7.3f}"
            )


# ============================================================
# Main runner
# ============================================================

def main() -> None:
    """
    Small description:
        Run a targeted improvement grid on the full FTSE250 sample and print
        clean robustness summaries for full sample and each major regime.

    Inputs:
        None

    Outputs:
        None
    """
    open_px, close_px = load_ftse250_open_close_from_parquet(start="1900-01-01", end=None)
    requested_start = first_valid_date_from_panels(open_px, close_px).date().isoformat()

    specs = build_candidate_specs()

    print("\n" + "=" * 100)
    print("FULL-SAMPLE IMPROVEMENT GRID | FTSE250")
    print("=" * 100)
    print(f"Universe:              {UNIVERSE}")
    print(f"Requested start:       {requested_start}")
    print(f"End:                   {END}")
    print(f"Rebalance:             {REBALANCE}")
    print(f"Transaction cost:      {TC_BPS_PER_1TURN} bps / 1-turn")
    print(f"Close slippage:        {CLOSE_SLIPPAGE_BPS_PER_1TURN} bps / 1-turn")
    print(f"Vol-adjusted signal:   {USE_VOL_ADJUSTED_SIGNAL}")
    print(f"Vol window:            {VOL_WINDOW}")
    print(f"Dollar neutralize:     {DOLLAR_NEUTRALIZE}")
    print(f"Target gross:          {TARGET_GROSS}")
    print(f"Candidates to test:    {len(specs)}")

    full_rows: list[dict] = []
    period_rows: list[dict] = []

    for spec in tqdm(specs, desc="Improvement grid", leave=True):
        tilt = TiltConfig(
            name=spec.name,
            long_entry=spec.long_entry,
            long_exit=spec.long_exit,
            short_entry=spec.short_entry,
            short_exit=spec.short_exit,
        )

        try:
            payloads, _, _ = run_single_market(
                universe=UNIVERSE,
                tilts=[tilt],
                start=requested_start,
                end=END,
                formation_window=spec.formation_window,
                rebalance=REBALANCE,
                tc_bps_per_1turn=TC_BPS_PER_1TURN,
                use_vol_adjusted_signal=USE_VOL_ADJUSTED_SIGNAL,
                vol_window=VOL_WINDOW,
                dollar_neutralize=DOLLAR_NEUTRALIZE,
                target_gross=TARGET_GROSS,
                use_exposure_scaling=spec.use_exposure_scaling,
                lambda_earn_months=spec.lambda_earn_months,
                lambda_other_months=spec.lambda_other_months,
                min_names=spec.min_names,
                close_slippage_bps_per_1turn=CLOSE_SLIPPAGE_BPS_PER_1TURN,
                verbose=False,
            )

            payload = payloads[spec.name]
            res = payload.result

            full_rows.append(
                {
                    "candidate": spec.name,
                    "formation_window": spec.formation_window,
                    "long_entry": spec.long_entry,
                    "long_exit": spec.long_exit,
                    "short_entry": spec.short_entry,
                    "short_exit": spec.short_exit,
                    "use_exposure_scaling": spec.use_exposure_scaling,
                    "lambda_earn_months": spec.lambda_earn_months,
                    "lambda_other_months": spec.lambda_other_months,
                    "min_names": spec.min_names,
                    "net_sharpe": float(res.net_metrics.get("sharpe", np.nan)),
                    "net_annret": float(res.net_metrics.get("ann_return", np.nan)),
                    "net_annvol": float(res.net_metrics.get("ann_vol", np.nan)),
                    "net_maxdd": float(res.net_metrics.get("max_drawdown", np.nan)),
                    "gross_sharpe": float(res.gross_metrics.get("sharpe", np.nan)),
                    "avg_reb_turnover": float(res.avg_reb_turnover),
                    "beta_to_bench": float(res.beta_to_bench),
                    "corr_to_bench": float(res.corr_to_bench),
                    "final_equity_net": float(res.final_equity_net),
                    "n_days": int(res.n_days),
                    "median_tradable": float(res.median_tradable),
                    "n_names": int(res.n_names),
                }
            )

            periods = make_regime_periods_for_logs(payload.net_log, payload.bench_log)
            for period_name, start, end in periods:
                row = compute_period_row(
                    candidate_name=spec.name,
                    strategy_log=payload.net_log,
                    bench_log=payload.bench_log,
                    period_name=period_name,
                    start=start,
                    end=end,
                )
                if row is not None:
                    period_rows.append(row)

        except Exception as exc:
            full_rows.append(
                {
                    "candidate": spec.name,
                    "formation_window": spec.formation_window,
                    "long_entry": spec.long_entry,
                    "long_exit": spec.long_exit,
                    "short_entry": spec.short_entry,
                    "short_exit": spec.short_exit,
                    "use_exposure_scaling": spec.use_exposure_scaling,
                    "lambda_earn_months": spec.lambda_earn_months,
                    "lambda_other_months": spec.lambda_other_months,
                    "min_names": spec.min_names,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    full_df = pd.DataFrame(full_rows)
    period_df = pd.DataFrame(period_rows)

    ok_full_df = full_df.loc[~full_df.get("net_sharpe", pd.Series(index=full_df.index, dtype=float)).isna()].copy()
    ok_full_df = ok_full_df.sort_values(
        ["net_sharpe", "net_annret", "avg_reb_turnover"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    full_df.to_csv(OUT_DIR / "full_sample_improvement_grid_all.csv", index=False)
    ok_full_df.to_csv(OUT_DIR / "full_sample_improvement_grid_ranked.csv", index=False)
    period_df.to_csv(OUT_DIR / "full_sample_improvement_grid_by_period.csv", index=False)

    print_top_full_sample(ok_full_df, TOP_N_FULL_SAMPLE_PRINT)
    print_top_by_period(period_df, TOP_N_BY_PERIOD_PRINT)

    print("\n" + "=" * 100)
    print("SAVED OUTPUTS")
    print("=" * 100)
    print(f"  {OUT_DIR / 'full_sample_improvement_grid_all.csv'}")
    print(f"  {OUT_DIR / 'full_sample_improvement_grid_ranked.csv'}")
    print(f"  {OUT_DIR / 'full_sample_improvement_grid_by_period.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()

