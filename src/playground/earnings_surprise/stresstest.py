"""
PEAD Strategy — Stress Testing & Robustness Analysis

Tests:
1. Sub-period performance (non-overlapping windows)
2. Rolling Sharpe ratio over time
3. Year-by-year returns breakdown
4. Market regime analysis (bull vs bear vs sideways)
5. Monthly return heatmap
6. Earnings season vs off-season performance
7. Parameter sensitivity (threshold grid)
8. Drawdown recovery analysis

Uses the same strategy code — just wraps it in diagnostic frameworks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from copy import deepcopy

# ============================================================
# IMPORT STRATEGY FUNCTIONS FROM MAIN FILE
# ============================================================

from best_earnsurp import (
    CONFIG,
    load_earnings,
    load_prices,
    compute_sue,
    compute_zscore,
    compute_signals_strategy_A,
    add_entry_exit_asym,
    build_signal_matrix,
    compute_portfolio_returns_long_anchored_50_50,
)


# ============================================================
# HELPER: RUN STRATEGY WITH CUSTOM CONFIG (RETURNS ONLY)
# ============================================================

def run_strategy_returns(config: dict):
    """
    Run the full pipeline and return daily return series.
    Lightweight version — no prints, no plots.
    """
    earnings = load_earnings(config)
    prices   = load_prices(config)

    earnings = compute_sue(earnings, config)
    earnings = compute_zscore(earnings, config)
    earnings = compute_signals_strategy_A(earnings, config)

    events = add_entry_exit_asym(earnings, config)
    sigmat = build_signal_matrix(events, config)

    (port_ret, port_ret_uncap, long_ret, short_ret, bench_ret,
     n_long, n_short_eff, long_exposure, short_exposure) = \
        compute_portfolio_returns_long_anchored_50_50(sigmat, prices, config)

    return port_ret, long_ret, short_ret, bench_ret, n_long, n_short_eff


def calc_metrics_dict(s: pd.Series) -> dict:
    """Compute key metrics as a dict (for tabulation)."""
    s = s.dropna()
    if len(s) == 0:
        return {"Ann.Ret%": np.nan, "Vol%": np.nan, "Sharpe": np.nan,
                "WinRate%": np.nan, "MaxDD%": np.nan, "Calmar": np.nan}

    ann_ret  = s.mean() * 252
    ann_vol  = s.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

    active   = s[s != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else np.nan

    cum    = (1 + s).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Ann.Ret%": ann_ret * 100,
        "Vol%":     ann_vol * 100,
        "Sharpe":   sharpe,
        "WinRate%": win_rate * 100 if not np.isnan(win_rate) else np.nan,
        "MaxDD%":   max_dd * 100,
        "Calmar":   calmar,
    }


# ============================================================
# TEST 1: SUB-PERIOD ANALYSIS
# ============================================================

def test_subperiod(port_ret: pd.Series, bench_ret: pd.Series):
    """
    Split returns into non-overlapping sub-periods and compare metrics.
    Tests whether the strategy works across different market environments.
    """
    periods = {
        "2005-2007 (Pre-GFC)":     ("2005-01-01", "2007-12-31"),
        "2008-2009 (GFC)":         ("2008-01-01", "2009-12-31"),
        "2010-2012 (Recovery)":    ("2010-01-01", "2012-12-31"),
        "2013-2015 (Bull)":        ("2013-01-01", "2015-12-31"),
        "2016-2019 (Late Cycle)":  ("2016-01-01", "2019-12-31"),
        "2020 (COVID)":            ("2020-01-01", "2020-12-31"),
        "2021-2022 (Post-COVID)":  ("2021-01-01", "2022-12-31"),
        "2023-2026 (Recent)":      ("2023-01-01", "2026-12-31"),
    }

    print("\n" + "=" * 100)
    print("TEST 1: SUB-PERIOD PERFORMANCE")
    print("=" * 100)

    rows = []
    for label, (start, end) in periods.items():
        mask = (port_ret.index >= start) & (port_ret.index <= end)
        if mask.sum() == 0:
            continue

        strat_sub = port_ret[mask]
        bench_sub = bench_ret[mask]

        sm = calc_metrics_dict(strat_sub)
        bm = calc_metrics_dict(bench_sub)

        rows.append({
            "Period": label,
            "Strat Ret%": sm["Ann.Ret%"],
            "Strat Vol%": sm["Vol%"],
            "Strat Sharpe": sm["Sharpe"],
            "Strat MaxDD%": sm["MaxDD%"],
            "Bench Ret%": bm["Ann.Ret%"],
            "Bench Sharpe": bm["Sharpe"],
            "Alpha%": sm["Ann.Ret%"] - bm["Ann.Ret%"],
        })

    df = pd.DataFrame(rows).set_index("Period")

    for _, row in df.iterrows():
        print(f"  {row.name:28s} | Ret {row['Strat Ret%']:6.1f}%  Vol {row['Strat Vol%']:5.1f}%  "
              f"Sharpe {row['Strat Sharpe']:5.2f}  MaxDD {row['Strat MaxDD%']:6.1f}%  "
              f"Alpha {row['Alpha%']:+6.1f}%  (Bench Sharpe {row['Bench Sharpe']:5.2f})")

    # Flag problem periods
    negative_sharpe = df[df["Strat Sharpe"] < 0]
    if not negative_sharpe.empty:
        print(f"\n  ⚠ WARNING: Negative Sharpe in: {', '.join(negative_sharpe.index)}")
    else:
        print(f"\n  ✓ Positive Sharpe in all sub-periods")

    underperform = df[df["Alpha%"] < 0]
    if not underperform.empty:
        print(f"  ⚠ Underperforms benchmark in: {', '.join(underperform.index)}")
    else:
        print(f"  ✓ Outperforms benchmark in all sub-periods")

    return df


# ============================================================
# TEST 2: ROLLING SHARPE
# ============================================================

def test_rolling_sharpe(port_ret: pd.Series, bench_ret: pd.Series,
                        windows=[126, 252]):
    """
    Plot rolling Sharpe ratio over time.
    Checks whether alpha is persistent or comes in bursts.
    """
    print("\n" + "=" * 100)
    print("TEST 2: ROLLING SHARPE ANALYSIS")
    print("=" * 100)

    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 4 * len(windows)), sharex=True)
    if len(windows) == 1:
        axes = [axes]

    for ax, w in zip(axes, windows):
        label = f"{w // 21}M" if w >= 21 else f"{w}d"

        roll_mean_s = port_ret.rolling(w).mean() * 252
        roll_vol_s  = port_ret.rolling(w).std() * np.sqrt(252)
        roll_sharpe_s = roll_mean_s / roll_vol_s

        roll_mean_b = bench_ret.rolling(w).mean() * 252
        roll_vol_b  = bench_ret.rolling(w).std() * np.sqrt(252)
        roll_sharpe_b = roll_mean_b / roll_vol_b

        ax.plot(roll_sharpe_s, color="black", lw=1.2, label="Strategy")
        ax.plot(roll_sharpe_b, color="blue", lw=1.0, alpha=0.6, ls="--", label="Benchmark")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.axhline(1, color="green", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-1, color="red", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"Rolling {label} Sharpe Ratio")
        ax.set_ylabel("Sharpe")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Diagnostics
        pct_negative = (roll_sharpe_s.dropna() < 0).mean() * 100
        pct_above_1  = (roll_sharpe_s.dropna() > 1).mean() * 100
        print(f"  Rolling {label}: {pct_negative:.1f}% of time Sharpe < 0, "
              f"{pct_above_1:.1f}% of time Sharpe > 1")

    plt.tight_layout()
    plt.show()


# ============================================================
# TEST 3: YEAR-BY-YEAR RETURNS
# ============================================================

def test_yearly_returns(port_ret: pd.Series, bench_ret: pd.Series):
    """
    Show year-by-year annualized returns + Sharpe.
    Checks for consistency vs. single-year outliers driving the backtest.
    """
    print("\n" + "=" * 100)
    print("TEST 3: YEAR-BY-YEAR PERFORMANCE")
    print("=" * 100)

    years = sorted(port_ret.index.year.unique())

    strat_annual = []
    bench_annual = []

    print(f"  {'Year':6s} | {'Strat Ret':>10}  {'Strat Vol':>10}  {'Sharpe':>8}  "
          f"{'Bench Ret':>10}  {'Alpha':>8}  {'MaxDD':>8}")
    print("  " + "-" * 80)

    for y in years:
        mask = port_ret.index.year == y
        s = port_ret[mask]
        b = bench_ret[mask]

        if len(s) < 20:
            continue

        sm = calc_metrics_dict(s)
        bm = calc_metrics_dict(b)

        alpha = sm["Ann.Ret%"] - bm["Ann.Ret%"]
        strat_annual.append(sm["Ann.Ret%"])
        bench_annual.append(bm["Ann.Ret%"])

        flag = " ⚠" if sm["Sharpe"] < 0 else ""
        print(f"  {y:6d} | {sm['Ann.Ret%']:9.1f}%  {sm['Vol%']:9.1f}%  "
              f"{sm['Sharpe']:7.2f}  {bm['Ann.Ret%']:9.1f}%  {alpha:+7.1f}%  "
              f"{sm['MaxDD%']:7.1f}%{flag}")

    print("  " + "-" * 80)

    strat_annual = np.array(strat_annual)
    bench_annual = np.array(bench_annual)

    winning_years = (strat_annual > bench_annual).sum()
    total_years   = len(strat_annual)

    print(f"\n  Outperforms benchmark: {winning_years}/{total_years} years "
          f"({winning_years/total_years*100:.0f}%)")
    print(f"  Positive Sharpe years: {(strat_annual > 0).sum()}/{total_years} "
          f"({(strat_annual > 0).mean()*100:.0f}%)")
    print(f"  Best year:  {strat_annual.max():+.1f}%")
    print(f"  Worst year: {strat_annual.min():+.1f}%")
    print(f"  Std of annual returns: {strat_annual.std():.1f}%")

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(years[:len(strat_annual)]))
    width = 0.35

    ax.bar(x - width/2, strat_annual, width, label="Strategy", color="black", alpha=0.8)
    ax.bar(x + width/2, bench_annual, width, label="Benchmark", color="blue", alpha=0.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years[:len(strat_annual)], rotation=45)
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Year-by-Year Returns: Strategy vs Benchmark")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# ============================================================
# TEST 4: MARKET REGIME ANALYSIS
# ============================================================

def test_market_regimes(port_ret: pd.Series, bench_ret: pd.Series):
    """
    Classify each day by market regime based on benchmark rolling return,
    then measure strategy performance in each regime.
    """
    print("\n" + "=" * 100)
    print("TEST 4: MARKET REGIME ANALYSIS")
    print("=" * 100)

    # 63-day (~3M) rolling benchmark return to classify regime
    bench_roll = bench_ret.rolling(63).sum()

    # Define regimes
    regimes = pd.Series(index=port_ret.index, dtype="object")
    regimes[bench_roll > 0.10]                              = "Strong Bull (>10% / 3M)"
    regimes[(bench_roll > 0.02) & (bench_roll <= 0.10)]     = "Mild Bull (2-10% / 3M)"
    regimes[(bench_roll >= -0.02) & (bench_roll <= 0.02)]   = "Sideways (±2% / 3M)"
    regimes[(bench_roll >= -0.10) & (bench_roll < -0.02)]   = "Mild Bear (-10 to -2% / 3M)"
    regimes[bench_roll < -0.10]                             = "Crash (<-10% / 3M)"

    # Drop NaN (first 63 days)
    valid = regimes.dropna()

    regime_order = [
        "Strong Bull (>10% / 3M)",
        "Mild Bull (2-10% / 3M)",
        "Sideways (±2% / 3M)",
        "Mild Bear (-10 to -2% / 3M)",
        "Crash (<-10% / 3M)",
    ]

    print(f"\n  {'Regime':35s} | {'Days':>6}  {'Strat Ret':>10}  {'Sharpe':>8}  "
          f"{'Bench Ret':>10}  {'Alpha':>8}")
    print("  " + "-" * 90)

    for regime in regime_order:
        mask = regimes == regime
        if mask.sum() == 0:
            continue

        sm = calc_metrics_dict(port_ret[mask])
        bm = calc_metrics_dict(bench_ret[mask])
        alpha = sm["Ann.Ret%"] - bm["Ann.Ret%"]

        print(f"  {regime:35s} | {mask.sum():5d}  {sm['Ann.Ret%']:9.1f}%  "
              f"{sm['Sharpe']:7.2f}  {bm['Ann.Ret%']:9.1f}%  {alpha:+7.1f}%")

    # Key question: does strategy lose money in crashes?
    crash_mask = regimes == "Crash (<-10% / 3M)"
    if crash_mask.sum() > 0:
        crash_ret = port_ret[crash_mask].mean() * 252
        if crash_ret > 0:
            print(f"\n  ✓ Strategy is POSITIVE during crashes ({crash_ret:.1f}% ann.)")
        else:
            print(f"\n  ⚠ Strategy is NEGATIVE during crashes ({crash_ret:.1f}% ann.)")


# ============================================================
# TEST 5: MONTHLY RETURN HEATMAP
# ============================================================

def test_monthly_heatmap(port_ret: pd.Series):
    """
    Heatmap of monthly returns to spot seasonality.
    Earnings seasons (Jan/Apr/Jul/Oct) should show stronger returns.
    """
    print("\n" + "=" * 100)
    print("TEST 5: MONTHLY RETURN HEATMAP")
    print("=" * 100)

    # Compute monthly returns
    monthly = port_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Pivot: rows = years, columns = months
    monthly_df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Ret":   monthly.values * 100,
    })

    pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Ret", aggfunc="first")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Print average by month
    avg_by_month = pivot.mean()
    print("\n  Average monthly return by calendar month:")
    for m, v in avg_by_month.items():
        flag = " ← earnings season" if m in ["Jan", "Feb", "Apr", "May", "Jul", "Aug", "Oct", "Nov"] else ""
        print(f"    {m:3s}: {v:+.2f}%{flag}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.4)))

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-5, vmax=5)

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.astype(int))

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 3 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color)

    ax.set_title("Monthly Returns Heatmap (%)")
    fig.colorbar(im, ax=ax, label="Return %", shrink=0.6)

    plt.tight_layout()
    plt.show()


# ============================================================
# TEST 6: DRAWDOWN RECOVERY ANALYSIS
# ============================================================

def test_drawdown_analysis(port_ret: pd.Series, top_n=5):
    """
    Identify the worst drawdowns, their duration, and recovery time.
    """
    print("\n" + "=" * 100)
    print(f"TEST 6: TOP {top_n} DRAWDOWN EPISODES")
    print("=" * 100)

    cum = (1 + port_ret.fillna(0)).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max

    # Find drawdown episodes
    is_dd = drawdown < 0
    dd_starts = is_dd & ~is_dd.shift(1, fill_value=False)
    dd_ends   = ~is_dd & is_dd.shift(1, fill_value=False)

    start_dates = drawdown.index[dd_starts]
    end_dates   = drawdown.index[dd_ends]

    # Match starts to ends
    episodes = []
    for s in start_dates:
        matching_ends = end_dates[end_dates > s]
        if len(matching_ends) > 0:
            e = matching_ends[0]
            dd_slice = drawdown[s:e]
            trough_date = dd_slice.idxmin()
            trough_val  = dd_slice.min()

            episodes.append({
                "Start":       s.strftime("%Y-%m-%d"),
                "Trough":      trough_date.strftime("%Y-%m-%d"),
                "Recovery":    e.strftime("%Y-%m-%d"),
                "Depth%":      trough_val * 100,
                "Days to Trough": np.busday_count(s.date(), trough_date.date()),
                "Days to Recover": np.busday_count(s.date(), e.date()),
            })

    # Sort by depth, take top N
    episodes = sorted(episodes, key=lambda x: x["Depth%"])[:top_n]

    print(f"\n  {'#':3s} {'Start':12s} {'Trough':12s} {'Recovery':12s} "
          f"{'Depth':>8s}  {'To Trough':>10s}  {'To Recover':>11s}")
    print("  " + "-" * 80)

    for i, ep in enumerate(episodes):
        print(f"  {i+1:2d}. {ep['Start']:12s} {ep['Trough']:12s} {ep['Recovery']:12s} "
              f"{ep['Depth%']:7.2f}%  {ep['Days to Trough']:8d}d  "
              f"{ep['Days to Recover']:9d}d")

    avg_recovery = np.mean([ep["Days to Recover"] for ep in episodes])
    print(f"\n  Average recovery time (top {top_n}): {avg_recovery:.0f} business days")


# ============================================================
# TEST 7: PARAMETER SENSITIVITY
# ============================================================

def test_parameter_sensitivity(base_config: dict):
    """
    Grid search over key parameters to check stability.
    If small changes in thresholds cause large Sharpe swings, the strategy
    is fragile and likely overfit.
    """
    print("\n" + "=" * 100)
    print("TEST 7: PARAMETER SENSITIVITY (THRESHOLD GRID)")
    print("=" * 100)

    sue_thresholds = [1.0, 1.5, 2.0, 2.5]
    z_thresholds   = [0.5, 1.0, 1.5, 2.0]

    results = []

    total = len(sue_thresholds) * len(z_thresholds)
    count = 0

    for sue_t in sue_thresholds:
        for z_t in z_thresholds:
            count += 1
            print(f"  Running {count}/{total}: SUE ±{sue_t}, Z ±{z_t} ...", end="")

            cfg = deepcopy(base_config)
            cfg["sue_thresh_long"]  = sue_t
            cfg["sue_thresh_short"] = -sue_t
            cfg["z_thresh_long"]    = z_t
            cfg["z_thresh_short"]   = -z_t

            try:
                port_ret, _, _, _, _, _ = run_strategy_returns(cfg)
                m = calc_metrics_dict(port_ret)
                results.append({
                    "SUE": sue_t, "Z": z_t,
                    "Sharpe": m["Sharpe"],
                    "Ret%": m["Ann.Ret%"],
                    "Vol%": m["Vol%"],
                    "MaxDD%": m["MaxDD%"],
                })
                print(f" Sharpe={m['Sharpe']:.2f}, Ret={m['Ann.Ret%']:.1f}%")
            except Exception as e:
                print(f" FAILED: {e}")
                results.append({
                    "SUE": sue_t, "Z": z_t,
                    "Sharpe": np.nan, "Ret%": np.nan,
                    "Vol%": np.nan, "MaxDD%": np.nan,
                })

    df = pd.DataFrame(results)

    # Sharpe heatmap
    sharpe_pivot = df.pivot_table(index="SUE", columns="Z", values="Sharpe")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sharpe
    ax = axes[0]
    im = ax.imshow(sharpe_pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=sharpe_pivot.values[~np.isnan(sharpe_pivot.values)].max())
    ax.set_xticks(range(len(z_thresholds)))
    ax.set_xticklabels([f"±{z}" for z in z_thresholds])
    ax.set_yticks(range(len(sue_thresholds)))
    ax.set_yticklabels([f"±{s}" for s in sue_thresholds])
    ax.set_xlabel("Z-Score Threshold")
    ax.set_ylabel("SUE Threshold")
    ax.set_title("Sharpe Ratio (Symmetric Thresholds)")
    for i in range(len(sue_thresholds)):
        for j in range(len(z_thresholds)):
            val = sharpe_pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # MaxDD
    dd_pivot = df.pivot_table(index="SUE", columns="Z", values="MaxDD%")
    ax = axes[1]
    im = ax.imshow(dd_pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=dd_pivot.values[~np.isnan(dd_pivot.values)].min(), vmax=0)
    ax.set_xticks(range(len(z_thresholds)))
    ax.set_xticklabels([f"±{z}" for z in z_thresholds])
    ax.set_yticks(range(len(sue_thresholds)))
    ax.set_yticklabels([f"±{s}" for s in sue_thresholds])
    ax.set_xlabel("Z-Score Threshold")
    ax.set_ylabel("SUE Threshold")
    ax.set_title("Max Drawdown % (Symmetric Thresholds)")
    for i in range(len(sue_thresholds)):
        for j in range(len(z_thresholds)):
            val = dd_pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()

    # Stability analysis
    valid_sharpes = df["Sharpe"].dropna()
    print(f"\n  Sharpe range across grid: {valid_sharpes.min():.2f} — {valid_sharpes.max():.2f}")
    print(f"  Sharpe std:  {valid_sharpes.std():.2f}")
    print(f"  Sharpe mean: {valid_sharpes.mean():.2f}")

    if valid_sharpes.std() < 0.3:
        print("  ✓ Low sensitivity — strategy is robust to threshold changes")
    elif valid_sharpes.std() < 0.5:
        print("  ~ Moderate sensitivity — some parameter dependence")
    else:
        print("  ⚠ High sensitivity — strategy may be overfit to specific thresholds")


# ============================================================
# TEST 8: TAIL RISK ANALYSIS
# ============================================================

def test_tail_risk(port_ret: pd.Series, bench_ret: pd.Series):
    """
    Analyze tail behavior: worst days, skewness, kurtosis,
    and conditional tail metrics (CVaR).
    """
    print("\n" + "=" * 100)
    print("TEST 8: TAIL RISK ANALYSIS")
    print("=" * 100)

    s = port_ret.dropna()
    b = bench_ret.dropna()

    # Distribution stats
    print(f"\n  {'Metric':30s} {'Strategy':>12s}  {'Benchmark':>12s}")
    print("  " + "-" * 58)
    print(f"  {'Skewness':30s} {s.skew():11.3f}  {b.skew():11.3f}")
    print(f"  {'Kurtosis (excess)':30s} {s.kurtosis():11.3f}  {b.kurtosis():11.3f}")
    print(f"  {'VaR 1% (daily)':30s} {s.quantile(0.01)*100:10.2f}%  {b.quantile(0.01)*100:10.2f}%")
    print(f"  {'VaR 5% (daily)':30s} {s.quantile(0.05)*100:10.2f}%  {b.quantile(0.05)*100:10.2f}%")

    # CVaR (expected shortfall)
    cvar_1_s = s[s <= s.quantile(0.01)].mean() * 100
    cvar_1_b = b[b <= b.quantile(0.01)].mean() * 100
    cvar_5_s = s[s <= s.quantile(0.05)].mean() * 100
    cvar_5_b = b[b <= b.quantile(0.05)].mean() * 100

    print(f"  {'CVaR 1% (daily)':30s} {cvar_1_s:10.2f}%  {cvar_1_b:10.2f}%")
    print(f"  {'CVaR 5% (daily)':30s} {cvar_5_s:10.2f}%  {cvar_5_b:10.2f}%")

    # Worst days
    print(f"\n  Top 5 worst days (strategy):")
    worst = s.nsmallest(5)
    for d, v in worst.items():
        bench_same_day = bench_ret.get(d, np.nan)
        print(f"    {d.strftime('%Y-%m-%d')}: {v*100:+.2f}%  "
              f"(benchmark: {bench_same_day*100:+.2f}%)")

    # Best days
    print(f"\n  Top 5 best days (strategy):")
    best = s.nlargest(5)
    for d, v in best.items():
        bench_same_day = bench_ret.get(d, np.nan)
        print(f"    {d.strftime('%Y-%m-%d')}: {v*100:+.2f}%  "
              f"(benchmark: {bench_same_day*100:+.2f}%)")


# ============================================================
# MAIN — RUN ALL STRESS TESTS
# ============================================================

def run_all_stress_tests(config: dict = CONFIG):
    """
    Run the full stress test suite.
    """
    print("\n" + "#" * 100)
    print("#  PEAD STRATEGY A — COMPREHENSIVE STRESS TEST & ROBUSTNESS ANALYSIS")
    print("#" * 100)

    print(f"\nConfig: SUE ±{config['sue_thresh_long']}, Z ±{config['z_thresh_long']}, "
          f"Long HP {config['holding_long_bdays']}d, Short HP {config['holding_short_bdays']}d, "
          f"Max Wt {config['max_weight_long']*100:.0f}%, "
          f"Min Longs {config['min_longs_for_shorts']}, "
          f"Start {config['start_date']}")

    # Run strategy
    port_ret, long_ret, short_ret, bench_ret, n_long, n_short = run_strategy_returns(config)

    # Run tests
    test_subperiod(port_ret, bench_ret)
    test_rolling_sharpe(port_ret, bench_ret)
    test_yearly_returns(port_ret, bench_ret)
    test_market_regimes(port_ret, bench_ret)
    test_monthly_heatmap(port_ret)
    test_drawdown_analysis(port_ret)
    test_parameter_sensitivity(config)
    test_tail_risk(port_ret, bench_ret)

    print("\n" + "#" * 100)
    print("#  STRESS TEST COMPLETE")
    print("#" * 100)


if __name__ == "__main__":
    run_all_stress_tests()