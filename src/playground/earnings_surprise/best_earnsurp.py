"""
PEAD Strategy A (SUE & Z-Score) with Long-Anchored Short Overlay
+ Position Concentration Cap (Improvement #1)

- Signals:
    * Strategy A: SUE AND Z-Score must both be extreme
- Portfolio construction:
    * Long-anchored overlay:
        - Shorts only allowed when there are active longs
        - No short-only days
        - Short exposure is capped by long exposure (never net short)
    * Separate holding periods for longs and shorts (configurable)
    * NEW: Per-name weight cap — no single position can exceed a max
      percentage of its book. Excess weight goes to cash (reduces exposure).

This file is designed to be upload-ready for a quant research repo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # Paths
    "earnings_path"   : "/Users/noah/Documents/Nova SBE/PMC/earningssurprise_S&Pall.xlsx",
    "earnings_sheet"  : "Data",
    "prices_path"     : "/Users/noah/Documents/Nova SBE/PMC/S&P_dailyprices_clean.csv",

    # SUE parameters
    "sue_window_q"    : 8,      # rolling quarters for forecast error std
    "sue_thresh_long" : 1.0,
    "sue_thresh_short": -2.0,

    # Z-score parameters
    "z_window_q"      : 8,
    "z_thresh_long"   : 1.0,
    "z_thresh_short"  : -1.5,

    # Holding periods (business days)
    "entry_lag_bdays"    : 1,
    "holding_long_bdays" : 7,   # longs tendenziell länger halten
    "holding_short_bdays": 3,   # shorts tendenziell kürzer halten

    # Long-anchored overlay parameters
    "min_longs_for_shorts": 3,  # Mindestanzahl Longs, damit Shorts erlaubt sind

    # --- NEW: Position concentration cap ---
    "max_weight_long"  : 0.2,  # max 5% of total portfolio per long name
    "max_weight_short" : 0.2,  # max 5% of total portfolio per short name

    # Backtest start
    "start_date"      : "2010-01-01",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_earnings(config: dict) -> pd.DataFrame:
    """
    Load and clean earnings data.

    - Reads EPS actual/estimate and computes percentage surprise.
    - Filters out technical issues (NaN, inf, extreme surprises).
    - Sorts by Security and AnnouncementDate.
    """
    df = pd.read_excel(config["earnings_path"], sheet_name=config["earnings_sheet"])
    df = df[["Security", "AnnouncementDate", "EPS_Actual", "EPS_Estimate"]].copy()

    df["AnnouncementDate"] = pd.to_datetime(df["AnnouncementDate"], dayfirst=True)
    df["EPS_Actual"]       = pd.to_numeric(df["EPS_Actual"], errors="coerce")
    df["EPS_Estimate"]     = pd.to_numeric(df["EPS_Estimate"], errors="coerce")

    df["Surprise_pct"] = (df["EPS_Actual"] - df["EPS_Estimate"]) / df["EPS_Estimate"].abs()

    mask_bad = (
        df["Surprise_pct"].isna() |
        np.isinf(df["Surprise_pct"]) |
        (df["Surprise_pct"].abs() > 5)
    )
    df.loc[mask_bad, "Surprise_pct"] = np.nan

    df = df.sort_values(["Security", "AnnouncementDate"]).reset_index(drop=True)
    return df


def load_prices(config: dict) -> pd.DataFrame:
    """
    Load daily price data and ensure a clean, sorted DateTime index.
    """
    prices = pd.read_csv(config["prices_path"], index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


# ============================================================
# SUE CALCULATION
# ============================================================

def compute_sue(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute Standardized Unexpected Earnings (SUE):

    - Forecast_Error = EPS_Actual - EPS_Estimate
    - SUE = Forecast_Error / rolling std of past forecast errors (per security)
    - Uses a rolling window over past quarters (sue_window_q).
    """
    df = df.copy()
    df["Forecast_Error"] = df["EPS_Actual"] - df["EPS_Estimate"]

    df["FE_shift"] = df.groupby("Security")["Forecast_Error"].shift(1)

    df["FE_std"] = df.groupby("Security")["FE_shift"].transform(
        lambda x: x.rolling(config["sue_window_q"], min_periods=4).std()
    )

    df["SUE"] = df["Forecast_Error"] / df["FE_std"].replace(0, np.nan)
    return df


# ============================================================
# Z-SCORE CALCULATION
# ============================================================

def compute_zscore(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute Z-Score of percentage surprise per security:

    - Surprise_pct is shifted by 1 to avoid look-ahead.
    - Rolling mean/std over z_window_q quarters.
    - Z_Score = (Surprise_pct - roll_mean) / roll_std
    """
    df = df.copy()

    df["roll_mean"] = df.groupby("Security")["Surprise_pct"].transform(
        lambda x: x.shift(1).rolling(config["z_window_q"], min_periods=4).mean()
    )
    df["roll_std"] = df.groupby("Security")["Surprise_pct"].transform(
        lambda x: x.shift(1).rolling(config["z_window_q"], min_periods=4).std()
    )

    df["Z_Score"] = (df["Surprise_pct"] - df["roll_mean"]) / df["roll_std"].replace(0, np.nan)
    return df


# ============================================================
# STRATEGY A — SUE AND Z-SCORE MUST BOTH BE EXTREME
# ============================================================

def compute_signals_strategy_A(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Strategy A signal logic:

    - Long if SUE > sue_thresh_long AND Z_Score > z_thresh_long
    - Short if SUE < sue_thresh_short AND Z_Score < z_thresh_short
    - Otherwise: no position (Signal = 0)
    """
    df = df.copy()

    long_condition  = (
        (df["SUE"]     > config["sue_thresh_long"]) &
        (df["Z_Score"] > config["z_thresh_long"])
    )
    short_condition = (
        (df["SUE"]     < config["sue_thresh_short"]) &
        (df["Z_Score"] < config["z_thresh_short"])
    )

    df["Signal"] = np.where(long_condition,   1,
                   np.where(short_condition, -1, 0))

    return df


# ============================================================
# ENTRY / EXIT DATES (ASYMMETRIC HOLDING PERIODS)
# ============================================================

def add_entry_exit_asym(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add entry and exit dates with asymmetric holding periods:

    - Entry_Date = AnnouncementDate + entry_lag_bdays
    - Longs: Exit_Date = Entry_Date + holding_long_bdays
    - Shorts: Exit_Date = Entry_Date + holding_short_bdays

    Only non-zero signals are kept.
    """
    df = df[df["Signal"] != 0].copy()

    df["Entry_Date"] = df["AnnouncementDate"] + pd.offsets.BDay(config["entry_lag_bdays"])

    is_long  = df["Signal"] == 1
    is_short = df["Signal"] == -1

    df["Exit_Date"] = pd.NaT
    df.loc[is_long,  "Exit_Date"] = df.loc[is_long,  "Entry_Date"] + pd.offsets.BDay(config["holding_long_bdays"])
    df.loc[is_short, "Exit_Date"] = df.loc[is_short, "Entry_Date"] + pd.offsets.BDay(config["holding_short_bdays"])

    return df


# ============================================================
# SIGNAL MATRIX
# ============================================================

def build_signal_matrix(events: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Expand event-level signals into a daily signal matrix:

    - Rows: business days
    - Columns: tickers
    - Values: -1 (short), 0 (flat), +1 (long)
    - Exit date is exclusive: [Entry_Date, Exit_Date)
    """
    active = events[events["Entry_Date"] >= config["start_date"]].copy()
    if active.empty:
        raise ValueError("No active events after start_date.")

    tickers = sorted(active["Security"].unique())
    days    = pd.bdate_range(active["Entry_Date"].min(), active["Exit_Date"].max())

    ticker_idx = {t: i for i, t in enumerate(tickers)}
    day_idx    = {d: i for i, d in enumerate(days)}

    mat = np.zeros((len(days), len(tickers)), dtype=np.int8)

    for _, row in active.iterrows():
        t  = ticker_idx[row["Security"]]
        d0 = day_idx.get(row["Entry_Date"])
        d1 = day_idx.get(row["Exit_Date"])
        if d0 is not None and d1 is not None:
            mat[d0:d1, t] = row["Signal"]

    return pd.DataFrame(mat, index=days, columns=tickers)


# ============================================================
# LONG-ANCHORED PORTFOLIO CONSTRUCTION + POSITION CAP
# ============================================================

def compute_portfolio_returns_long_anchored_50_50(
    sig: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict
):
    """
    Long-anchored 50/50 portfolio construction with position caps:

    Step 1 — Equal-weight within each book (same as before):
    - If at least one long exists:
        * Longs get +50% exposure (equal-weighted)
        * Shorts get -50% exposure (equal-weighted)
    - If no longs exist: flat
    - Shorts only allowed when n_long >= min_longs_for_shorts

    Step 2 — Position cap (NEW):
    - Each long name is capped at max_weight_long of total portfolio
    - Each short name is capped at max_weight_short of total portfolio
    - Excess weight goes to cash (total book exposure shrinks)
    - This means on days with few signals, exposure is reduced rather
      than concentrating into a handful of names

    Example: 2 longs, 50% book → 25% each → capped to 5% each → 10% total long
             This dramatically reduces idiosyncratic risk on low-signal days.

    Returns:
        port_ret       : capped L/S portfolio returns
        port_ret_uncap : uncapped L/S portfolio returns (for comparison)
        long_ret       : capped long-leg returns
        short_ret      : capped short-leg returns
        bench_ret      : equal-weight benchmark across all tickers
        n_long         : number of active longs per day
        n_short        : number of active (allowed) shorts per day
        long_exposure  : total long weight per day (after cap)
        short_exposure : total short weight per day (after cap)
    """

    returns = prices.pct_change()

    common_dates   = sig.index.intersection(returns.index)
    common_tickers = sig.columns.intersection(returns.columns)

    sig = sig.loc[common_dates, common_tickers]
    ret = returns.loc[common_dates, common_tickers]

    long_mask_raw  = sig == 1
    short_mask_raw = sig == -1

    n_long  = long_mask_raw.sum(axis=1)
    n_short = short_mask_raw.sum(axis=1)

    # --- Uncapped weights (original logic) ---
    n_long_safe = n_long.replace(0, np.nan)
    long_w_uncap = long_mask_raw.div(n_long_safe, axis=0).fillna(0) * 0.5

    min_longs = config.get("min_longs_for_shorts", 1)
    allow_shorts_day = n_long >= min_longs

    short_mask = short_mask_raw & allow_shorts_day.values[:, None]
    short_mask = short_mask & (n_long > 0).values[:, None]

    n_short_eff = short_mask.sum(axis=1)
    n_short_safe = n_short_eff.replace(0, np.nan)

    short_w_uncap = short_mask.div(n_short_safe, axis=0).fillna(0) * -0.5

    # Uncapped portfolio return (for comparison)
    port_ret_uncap = (ret * (long_w_uncap + short_w_uncap)).sum(axis=1)

    # --- Step 2: Apply position cap (NEW) ---
    max_w_long  = config.get("max_weight_long",  0.05)
    max_w_short = config.get("max_weight_short", 0.05)

    # Cap longs: each name max +max_w_long
    long_w = long_w_uncap.clip(upper=max_w_long)

    # Cap shorts: each name max -max_w_short (remember shorts are negative)
    short_w = short_w_uncap.clip(lower=-max_w_short)

    # Compute exposure after capping
    long_exposure  = long_w.sum(axis=1)
    short_exposure = short_w.sum(axis=1)  # negative values

    # Portfolio returns (capped)
    port_ret  = (ret * (long_w + short_w)).sum(axis=1)
    long_ret  = (ret * long_w).sum(axis=1)
    short_ret = (ret * short_w).sum(axis=1)
    bench_ret = returns.loc[common_dates].mean(axis=1)

    return (port_ret, port_ret_uncap, long_ret, short_ret, bench_ret,
            n_long, n_short_eff, long_exposure, short_exposure)


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

def analyze_overlap(sigmat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overlap statistics:

    - n_long, n_short
    - hedged (both long & short)
    - long_only, short_only, flat
    """
    long_mask  = sigmat == 1
    short_mask = sigmat == -1

    n_long  = long_mask.sum(axis=1)
    n_short = short_mask.sum(axis=1)

    overlap_df = pd.DataFrame({
        "n_long": n_long,
        "n_short": n_short,
        "hedged": ((n_long > 0) & (n_short > 0)).astype(int),
        "long_only": ((n_long > 0) & (n_short == 0)).astype(int),
        "short_only": ((n_short > 0) & (n_long == 0)).astype(int),
        "flat": ((n_long == 0) & (n_short == 0)).astype(int),
    })

    return overlap_df


# ============================================================
# METRICS
# ============================================================

def metrics(series: pd.Series, label: str):
    """
    Print standard performance metrics:

    - Annualized return, volatility, Sharpe
    - Active win rate (excludes zero-return / flat days)
    - Max drawdown
    - Calmar ratio
    """
    s        = series.dropna()
    ann_ret  = s.mean() * 252
    ann_vol  = s.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Active-day win rate: only count days where the portfolio had exposure
    active   = s[s != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else np.nan
    pct_active = len(active) / len(s) * 100 if len(s) > 0 else 0

    cum      = (1 + s).cumprod()
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    print(f"{label:25s} | Ret {ann_ret*100:6.2f}%  Vol {ann_vol*100:5.2f}%  "
          f"Sharpe {sharpe:5.2f}  WinRate {win_rate*100:4.1f}% ({pct_active:.0f}% active)  "
          f"MaxDD {max_dd*100:6.2f}%  Calmar {calmar:5.2f}")


# ============================================================
# PLOTS
# ============================================================

def plot_equity_curves(port_ret, port_ret_uncap, long_ret, short_ret, bench_ret, config, title_suffix=""):
    """
    Plot cumulative performance of:

    - Capped L/S portfolio (main)
    - Uncapped L/S portfolio (for comparison)
    - Long-only leg (capped)
    - Short leg (capped)
    - Benchmark (equal-weight)
    - Drawdown of capped portfolio
    """
    cum_ls      = (1 + port_ret.fillna(0)).cumprod()
    cum_ls_raw  = (1 + port_ret_uncap.fillna(0)).cumprod()
    cum_long    = (1 + long_ret.fillna(0)).cumprod()
    cum_short   = (1 + short_ret.fillna(0)).cumprod()
    cum_bench   = (1 + bench_ret.fillna(0)).cumprod()

    rolling_max = cum_ls.cummax()
    drawdown    = (cum_ls - rolling_max) / rolling_max

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(cum_ls,      label="L/S Capped",              color="black",  lw=1.8)
    ax.plot(cum_ls_raw,  label="L/S Uncapped (original)",  color="gray",   lw=1.2, alpha=0.6, ls=":")
    ax.plot(cum_long,    label="Long Book (capped)",       color="green",  lw=1.2, alpha=0.8)
    ax.plot(cum_short,   label="Short Book (capped)",      color="red",    lw=1.2, alpha=0.8)
    ax.plot(cum_bench,   label="Benchmark (EW S&P)",       color="blue",   lw=1.2, alpha=0.8, ls="--")
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.set_title(
        f"PEAD Strategy A (SUE & Z-Score, Long-Anchored, Position Cap) {title_suffix}\n"
        f"SUE>{config['sue_thresh_long']} ({config['sue_window_q']}Q), "
        f"Z>{config['z_thresh_long']} ({config['z_window_q']}Q) | "
        f"Long HP {config['holding_long_bdays']}d, Short HP {config['holding_short_bdays']}d | "
        f"Entry Lag {config['entry_lag_bdays']}d | "
        f"Max Wt {config['max_weight_long']*100:.0f}%",
        fontsize=12
    )
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_ylim(drawdown.min() * 1.2, 0.05)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.show()


def plot_exposure_timeline(n_long: pd.Series, n_short: pd.Series,
                           long_exposure: pd.Series, short_exposure: pd.Series):
    """
    Plot the number of active positions AND effective exposure over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Position count
    ax = axes[0]
    ax.plot(n_long.index,  n_long,  label="n_long",  color="green", alpha=0.8)
    ax.plot(n_short.index, n_short, label="n_short (allowed)", color="red", alpha=0.8)
    ax.set_title("Position Count (Long-Anchored)")
    ax.set_ylabel("# Positions")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Effective exposure after cap
    ax2 = axes[1]
    ax2.plot(long_exposure.index,  long_exposure,         label="Long exposure",  color="green", alpha=0.8)
    ax2.plot(short_exposure.index, short_exposure.abs(),   label="Short exposure (abs)", color="red", alpha=0.8)
    ax2.axhline(0.50, color="gray", ls="--", lw=0.8, label="Uncapped max (50%)")
    ax2.set_title("Effective Book Exposure After Position Cap")
    ax2.set_ylabel("Exposure (% of portfolio)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    for a in axes:
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PIPELINE — STRATEGY A + LONG-ANCHORED + POSITION CAP
# ============================================================

def run_strategy_A_long_anchored(config: dict = CONFIG):
    """
    Full pipeline for:

    - Strategy A (SUE & Z-Score both extreme)
    - Long-anchored portfolio construction
    - Asymmetric holding periods for longs and shorts
    - NEW: Position concentration cap
    - Performance metrics + plots + exposure timeline
    """
    print("\n===== RUNNING STRATEGY A (SUE & Z-SCORE, LONG-ANCHORED, POSITION CAP) =====")
    print(f"Max weight per name: Long {config['max_weight_long']*100:.0f}%, "
          f"Short {config['max_weight_short']*100:.0f}%")

    earnings = load_earnings(config)
    prices   = load_prices(config)

    earnings = compute_sue(earnings, config)
    earnings = compute_zscore(earnings, config)
    earnings = compute_signals_strategy_A(earnings, config)

    events = add_entry_exit_asym(earnings, config)
    sigmat = build_signal_matrix(events, config)
    print(f"Tickers in signal matrix: {len(sigmat.columns)}")

    (port_ret, port_ret_uncap, long_ret, short_ret, bench_ret,
     n_long, n_short_eff, long_exposure, short_exposure) = \
        compute_portfolio_returns_long_anchored_50_50(sigmat, prices, config)

    print(f"\n{'Strategy':25s} | {'Ann.Ret':>9}  {'Vol':>7}  {'Sharpe':>8}  "
          f"{'WinRate':>8}  {'MaxDD':>8}  {'Calmar':>7}")
    print("-" * 95)
    metrics(port_ret,        "L/S Capped")
    metrics(port_ret_uncap,  "L/S Uncapped (orig)")
    metrics(long_ret,        "Long Book (capped)")
    metrics(short_ret,       "Short Book (capped)")
    print("-" * 95)
    metrics(bench_ret,       "Benchmark (EW S&P)")

    plot_equity_curves(port_ret, port_ret_uncap, long_ret, short_ret, bench_ret, config)

    # Overlap analysis — RAW signals 
    # (before long-anchored overlay)
    overlap = analyze_overlap(sigmat)

    # Overlap analysis — EFFECTIVE positions (after overlay + cap)
    # Use n_long and n_short_eff which reflect the actual portfolio
    eff_hedged     = ((n_long > 0) & (n_short_eff > 0)).mean() * 100
    eff_long_only  = ((n_long > 0) & (n_short_eff == 0)).mean() * 100
    eff_short_only = ((n_short_eff > 0) & (n_long == 0)).mean() * 100
    eff_flat       = ((n_long == 0) & (n_short_eff == 0)).mean() * 100

    print("\n=== Overlap Analysis: EFFECTIVE Positions (after overlay + cap) ===")
    print(f"Hedged days:      {eff_hedged:.1f}%")
    print(f"Long-only days:   {eff_long_only:.1f}%")
    print(f"Short-only days:  {eff_short_only:.1f}%  ← should be 0%")
    print(f"Flat days:        {eff_flat:.1f}%")

    # How many short signals got killed by the overlay?
    raw_short_only_days = ((overlap['n_short'] > 0) & (overlap['n_long'] == 0)).sum()
    print(f"\nShort signals suppressed by overlay: {raw_short_only_days} days")

    # Position cap diagnostics
    print("\n=== Position Cap Diagnostics ===")
    print(f"Avg long exposure:   {long_exposure.mean()*100:.1f}%  (uncapped max: 50%)")
    print(f"Avg short exposure:  {short_exposure.abs().mean()*100:.1f}%  (uncapped max: 50%)")
    print(f"Median long exposure:  {long_exposure.median()*100:.1f}%")
    print(f"Median short exposure: {short_exposure.abs().median()*100:.1f}%")

    # How often is the cap binding?
    # Cap binds when equal-weight per name > max_weight
    # For longs: 0.5/n_long > max_weight → n_long < 0.5/max_weight
    max_w = config["max_weight_long"]
    cap_threshold_long = 0.5 / max_w  # e.g. 0.5/0.05 = 10
    cap_binding_long = (n_long < cap_threshold_long) & (n_long > 0)

    max_w_s = config["max_weight_short"]
    cap_threshold_short = 0.5 / max_w_s
    cap_binding_short = (n_short_eff < cap_threshold_short) & (n_short_eff > 0)

    print(f"\nCap binding threshold: <{cap_threshold_long:.0f} longs, <{cap_threshold_short:.0f} shorts")
    print(f"% days cap binds (longs):  {cap_binding_long.mean()*100:.1f}%")
    print(f"% days cap binds (shorts): {cap_binding_short.mean()*100:.1f}%")

    # Exposure timeline
    plot_exposure_timeline(n_long, n_short_eff, long_exposure, short_exposure)

    return port_ret, port_ret_uncap, long_ret, short_ret, bench_ret, sigmat


if __name__ == "__main__":
    run_strategy_A_long_anchored()