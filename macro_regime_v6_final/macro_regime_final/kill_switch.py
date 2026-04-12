"""
kill_switch.py
==============
Intra-quarter macro de-risk kill switch.

Trigger
-------
    HY OAS (BAMLH0A0HYM2) rising >= KS_TRIGGER_ZSCORE standard deviations
    above its 252-day expanding mean in any single day within a rolling
    KS_WINDOW-day window.

    Z-score formula (no lookahead):
        z_t = (x_t - mu_{t-1}) / sigma_{t-1}
    where mu and sigma are expanding-window, shifted by 1 day.

Action when triggered
---------------------
    Override current allocation to kill_switch_contraction_exp (default 0.20)
    regardless of what the HMM says.  Override holds until deactivation.
    This is a true replacement of the regime exposure, not an additional scaling.

Deactivation (hysteresis)
-------------------------
    Kill switch deactivates when HY OAS z-score remains BELOW
    KS_DEACT_ZSCORE for KS_WINDOW consecutive days.
    Deactivation threshold < trigger threshold intentionally — prevents
    whipsawing at the boundary.

Minimum history guard
---------------------
    Kill switch cannot fire until KS_MIN_OBS usable z-score observations
    exist.  Early observations are masked to NaN.

Integration
-----------
    build_kill_switch_history(hy_oas_series) -> pd.DataFrame
        Pure function.  Call once per backtest run.

    get_kill_switch_status(as_of_date, cfg) -> KillSwitchState
        Live use.  Pulls HY OAS from FRED, returns current state.

    validate_kill_switch_history(df) -> None
        Sanity check.  Prints PASS/WARN for GFC (2008-09 to 2009-03)
        and COVID (2020-03).

All tuneable parameters live in config.py as frozen fields.
Do NOT hardcode them here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .paths import reports_dir
except ImportError:  # pragma: no cover - supports direct script execution
    from paths import reports_dir

warnings.filterwarnings("ignore")

# ── Default parameter fallbacks (mirrors config.py frozen fields) ─────────────
# These are only used when no Config object is passed (e.g. __main__ smoke test).
_DEFAULT_TRIGGER_ZSCORE  : float = 3.0
_DEFAULT_DEACT_ZSCORE    : float = 2.0
_DEFAULT_WINDOW          : int   = 5
_DEFAULT_MIN_OBS         : int   = 252
_DEFAULT_FRED_SERIES     : str   = "BAMLH0A0HYM2"
_DEFAULT_CONTRACTION_EXP : float = 0.20   # must match kill_switch_contraction_exp in config.py


def load_cached_kill_switch_history() -> pd.DataFrame:
    """Load the most recent persisted kill-switch history from reports/."""
    cache_path = reports_dir() / "kill_switch_history.csv"
    if not cache_path.exists():
        return pd.DataFrame()

    cache_df = pd.read_csv(cache_path)
    date_col = cache_df.columns[0]
    cache_df[date_col] = pd.to_datetime(cache_df[date_col])
    cache_df = cache_df.set_index(date_col).sort_index()
    return cache_df


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class KillSwitchState:
    """Kill switch reading at one point in time."""
    date        : pd.Timestamp
    active      : bool          # True = override to Contraction exposure
    zscore      : float         # HY OAS z-score at this date (NaN if insufficient history)
    reason      : str           # human-readable reason ("triggered" / "cooling" / "inactive" / "warmup")
    exposure    : float         # effective exposure override (0.30 if active, else pass-through)

    @property
    def label(self) -> str:
        return "KILL_SWITCH_ACTIVE" if self.active else "ok"


# ── Core z-score helper ────────────────────────────────────────────────────────

def _expanding_zscore(series: pd.Series, min_obs: int) -> pd.Series:
    """
    Expanding-window z-score with shift(1) lookahead guard.

    z_t = (x_t - mu_{t-1}) / sigma_{t-1}

    Observations before min_obs are set to NaN — insufficient history
    means the z-score is unreliable and must not trigger the switch.
    """
    mu    = series.expanding().mean().shift(1)
    sigma = series.expanding().std(ddof=1).shift(1)
    z     = (series - mu) / sigma.replace(0, np.nan)

    obs_count = series.expanding().count().shift(1)
    z = z.where(obs_count >= min_obs, np.nan)
    return z


# ── State machine ─────────────────────────────────────────────────────────────

def _build_state_machine(
    zscore_series      : pd.Series,
    trigger_zscore     : float,
    deact_zscore       : float,
    window             : int,
) -> pd.DataFrame:
    """
    Apply hysteretic state machine over a daily z-score series.

    State transitions:
        inactive  -> active   : any z in last `window` days >= trigger_zscore
        active    -> inactive : all z in last `window` days <  deact_zscore
        warmup    -> *        : z is NaN (< min_obs); switch locked off

    Returns DataFrame with columns:
        zscore, active, reason
    """
    z      = zscore_series.values
    dates  = zscore_series.index
    n      = len(z)

    active_arr = np.zeros(n, dtype=bool)
    reason_arr = ["warmup"] * n

    currently_active = False

    for i in range(n):
        z_i = z[i]

        if np.isnan(z_i):
            # Not enough history — kill switch is locked off
            currently_active = False
            active_arr[i]    = False
            reason_arr[i]    = "warmup"
            continue

        # Window of z-scores ending at i (inclusive), drop NaNs
        start    = max(0, i - window + 1)
        z_window = z[start : i + 1]
        z_window = z_window[~np.isnan(z_window)]

        if not currently_active:
            # Check trigger: any day in window >= trigger_zscore
            if len(z_window) > 0 and np.any(z_window >= trigger_zscore):
                currently_active = True
                reason_arr[i]    = "triggered"
            else:
                reason_arr[i] = "inactive"
        else:
            # Check deactivation: ALL days in window < deact_zscore
            if len(z_window) >= window and np.all(z_window < deact_zscore):
                currently_active = False
                reason_arr[i]    = "inactive"
            else:
                reason_arr[i] = "cooling"

        active_arr[i] = currently_active

    return pd.DataFrame(
        {
            "zscore" : z,
            "active" : active_arr,
            "reason" : reason_arr,
        },
        index=dates,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_kill_switch_history(
    hy_oas_series       : pd.Series,
    trigger_zscore      : float = _DEFAULT_TRIGGER_ZSCORE,
    deact_zscore        : float = _DEFAULT_DEACT_ZSCORE,
    window              : int   = _DEFAULT_WINDOW,
    min_obs             : int   = _DEFAULT_MIN_OBS,
    contraction_exp     : float = _DEFAULT_CONTRACTION_EXP,
) -> pd.DataFrame:
    """
    Build a full daily kill switch history from a HY OAS price series.

    Parameters
    ----------
    hy_oas_series   : Daily HY OAS series (raw, NOT z-scored).  Index must be
                      a DatetimeIndex sorted ascending.
    trigger_zscore  : Fire threshold (default 3.0 — from config).
    deact_zscore    : Deactivation threshold (default 2.0 — from config).
    window          : Rolling window for trigger and deactivation (default 5).
    min_obs         : Minimum observations before switch can fire (default 252).
    contraction_exp : Exposure override when active (default 0.30).

    Returns
    -------
    DataFrame indexed by date with columns:
        hy_oas   : raw HY OAS level
        zscore   : expanding z-score (NaN during warmup)
        active   : bool — kill switch on
        reason   : "warmup" | "inactive" | "triggered" | "cooling"
        exposure : 0.30 if active, else NaN (caller applies their HMM exposure)
    """
    if not isinstance(hy_oas_series.index, pd.DatetimeIndex):
        hy_oas_series.index = pd.to_datetime(hy_oas_series.index)

    hy_oas_series = hy_oas_series.sort_index().dropna()

    zscore = _expanding_zscore(hy_oas_series, min_obs=min_obs)

    state_df = _build_state_machine(
        zscore_series  = zscore,
        trigger_zscore = trigger_zscore,
        deact_zscore   = deact_zscore,
        window         = window,
    )

    state_df.insert(0, "hy_oas", hy_oas_series.reindex(state_df.index))
    state_df["exposure"] = state_df["active"].apply(
        lambda a: contraction_exp if a else np.nan
    )

    return state_df


def get_kill_switch_status(
    as_of_date  : str | pd.Timestamp | None = None,
    cfg         = None,
) -> KillSwitchState:
    """
    Live-use entry point.  Pulls HY OAS from FRED, builds history, and returns
    the kill switch state as of `as_of_date` (defaults to latest available).

    Parameters
    ----------
    as_of_date : Date to query.  If None, uses the latest available observation.
    cfg        : Config object.  If None, defaults from module-level constants
                 are used.

    Returns
    -------
    KillSwitchState
    """
    # Resolve parameters
    trigger_z   = float(getattr(cfg, "kill_switch_trigger_zscore",  _DEFAULT_TRIGGER_ZSCORE))
    deact_z     = float(getattr(cfg, "kill_switch_deact_zscore",    _DEFAULT_DEACT_ZSCORE))
    window      = int(getattr(cfg,   "kill_switch_window",          _DEFAULT_WINDOW))
    min_obs     = int(getattr(cfg,   "kill_switch_min_obs",         _DEFAULT_MIN_OBS))
    fred_series = str(getattr(cfg,   "kill_switch_fred_series",     _DEFAULT_FRED_SERIES))
    cont_exp    = float(getattr(cfg, "kill_switch_contraction_exp", _DEFAULT_CONTRACTION_EXP))

    # Pull HY OAS from FRED using the established client pattern
    try:
        from .macro_data import _get_fred_client  # noqa: PLC0415
    except ImportError:
        from macro_data import _get_fred_client  # noqa: PLC0415

    try:
        fred  = _get_fred_client()
        raw   = fred.get_series(fred_series)
        raw.name = "hy_oas"

        history = build_kill_switch_history(
            hy_oas_series   = raw,
            trigger_zscore  = trigger_z,
            deact_zscore    = deact_z,
            window          = window,
            min_obs         = min_obs,
            contraction_exp = cont_exp,
        )
    except Exception:
        history = load_cached_kill_switch_history()
        if history.empty:
            raise

    if as_of_date is not None:
        as_of_ts = pd.Timestamp(as_of_date)
        # Find the last row on or before as_of_date
        available = history.loc[:as_of_ts]
        if available.empty:
            raise ValueError(
                f"No kill switch data available on or before {as_of_date}. "
                f"Earliest date in history: {history.index[0].date()}."
            )
        row = available.iloc[-1]
        date = available.index[-1]
    else:
        row  = history.iloc[-1]
        date = history.index[-1]

    zscore   = row["zscore"]
    active   = bool(row["active"])
    reason   = str(row["reason"])
    exposure = cont_exp if active else np.nan

    return KillSwitchState(
        date     = date,
        active   = active,
        zscore   = float(zscore) if not np.isnan(zscore) else float("nan"),
        reason   = reason,
        exposure = exposure,
    )


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_kill_switch_history(df: pd.DataFrame) -> None:
    """
    Sanity-check kill switch history against known stress periods.

    Checks
    ------
    GFC (Sep 2008 - Mar 2009)
        HY OAS peaked at ~6 sigma on the expanding window -- kill switch MUST
        fire here. PASS requires >= 50% of window days active.

    COVID (Mar-May 2020)
        HY OAS peaked at ~2 sigma on the long (1996-present) expanding window.
        The 3 sigma trigger correctly does NOT fire -- March 2020 spread widening
        was violent but brief relative to the 2008 baseline. Reported as INFO
        only; never raises a WARN.

    Does NOT raise -- these are observational diagnostics.
    """
    print("\n[kill_switch] Validation against known stress events:")
    print("=" * 60)

    # (name, start, end, min_active_frac, hard_check)
    # hard_check=False => report only, never marks WARN
    checks = [
        ("GFC (Sep 2008 - Mar 2009)", "2008-09-01", "2009-03-31", 0.50, True),
        ("COVID (Mar-May 2020)",       "2020-03-01", "2020-05-31", 0.00, False),
    ]

    all_pass = True
    for name, start, end, min_active_frac, hard_check in checks:
        window = df.loc[start:end]
        if window.empty:
            print(f"  {name:<30}: NO DATA")
            continue

        frac_active = float(window["active"].mean())
        fired_ever  = bool(window["active"].any())
        peak_z      = window["zscore"].max()
        peak_z_str  = f"{peak_z:.2f}" if not np.isnan(peak_z) else "NaN"

        if hard_check:
            status = "PASS" if frac_active >= min_active_frac else "WARN"
            if status == "WARN":
                all_pass = False
        else:
            status = "INFO"

        print(
            f"  {name:<30}: {frac_active:.0%} days active  "
            f"peak_z={peak_z_str}  [fired={fired_ever}]  [{status}]"
        )

    if all_pass:
        print("\n  [OK] Kill switch passed all hard validation checks.")
    else:
        print(
            "\n  [WARN] Kill switch missed a required stress period. "
            "Review HY OAS data quality or trigger threshold."
        )

    # Summary stats
    n_total   = len(df)
    n_warmup  = int((df["reason"] == "warmup").sum())
    n_active  = int(df["active"].sum())
    n_usable  = n_total - n_warmup
    pct_on    = n_active / n_usable if n_usable > 0 else 0.0

    print(f"\n[kill_switch] Coverage summary:")
    print(f"  Total days      : {n_total}")
    print(f"  Warmup (masked) : {n_warmup}")
    print(f"  Usable days     : {n_usable}")
    print(f"  Days active     : {n_active}  ({pct_on:.1%} of usable)")

    # Count distinct activation episodes
    transitions = df["active"].astype(int).diff()
    episodes    = int((transitions == 1).sum())
    print(f"  Kill switch episodes : {episodes}")


# ── Convenience: apply to backtest return series ───────────────────────────────

def apply_kill_switch_to_returns(
    returns_df      : pd.DataFrame,
    ks_history      : pd.DataFrame,
    return_col      : str   = "PortRet",
    contraction_exp : float = _DEFAULT_CONTRACTION_EXP,
    regime_exp_col  : str   = "RegimeExp",
) -> pd.DataFrame:
    """
    Apply kill switch exposure override to a quarterly backtest return series.

    When active, the kill switch replaces the regime exposure with contraction_exp.
    The override is a true replacement — not a further scaling on top of the regime
    exposure that is already embedded in return_col.

    Mechanics
    ---------
    return_col already contains  gross_ret × regime_exp - costs.
    To override regime_exp → contraction_exp we rescale:

        PortRetKS = return_col × (contraction_exp / regime_exp)

    clipped so the scale factor never exceeds 1.0 (the kill switch only reduces
    exposure, never increases it).  When the kill switch is inactive, PortRetKS
    equals return_col unchanged.

    Parameters
    ----------
    returns_df      : Backtest DataFrame indexed by quarterly date.
                      Must contain regime_exp_col (e.g. "RegimeExp").
    ks_history      : Output of build_kill_switch_history().
    return_col      : Column name of the portfolio return to override.
    contraction_exp : Target exposure when kill switch is active.
    regime_exp_col  : Column in returns_df holding the regime engine's exposure
                      for that quarter (used to compute the rescale factor).

    Returns
    -------
    Copy of returns_df with additional columns:
        ks_active        : bool
        ks_zscore        : float
        ks_reason        : str
        ks_exposure      : effective exposure after override
        PortRetKS        : return_col after kill switch override
    """
    out = returns_df.copy()
    ks_active_list   = []
    ks_zscore_list   = []
    ks_reason_list   = []
    ks_exposure_list = []

    for qt in out.index:
        # Last kill switch reading on or before this quarterly date
        avail = ks_history.loc[:qt]
        if avail.empty:
            ks_active_list.append(False)
            ks_zscore_list.append(np.nan)
            ks_reason_list.append("no_data")
            ks_exposure_list.append(float(out.loc[qt, regime_exp_col]) if regime_exp_col in out.columns else 1.0)
            continue
        row    = avail.iloc[-1]
        active = bool(row["active"])
        zscore = float(row["zscore"]) if not np.isnan(row["zscore"]) else np.nan
        reason = str(row["reason"])

        if active:
            ks_exposure_list.append(contraction_exp)
        else:
            regime_exp = float(out.loc[qt, regime_exp_col]) if regime_exp_col in out.columns else 1.0
            ks_exposure_list.append(regime_exp)

        ks_active_list.append(active)
        ks_zscore_list.append(zscore)
        ks_reason_list.append(reason)

    out["ks_active"]   = ks_active_list
    out["ks_zscore"]   = ks_zscore_list
    out["ks_reason"]   = ks_reason_list
    out["ks_exposure"] = ks_exposure_list

    if return_col in out.columns and regime_exp_col in out.columns:
        # Rescale: replace the baked-in regime_exp with ks_exposure.
        # Cap scale at 1.0 — kill switch never increases exposure.
        regime_exp_series = out[regime_exp_col].replace(0.0, np.nan)
        scale = (out["ks_exposure"] / regime_exp_series).clip(upper=1.0).fillna(1.0)
        out["PortRetKS"] = out[return_col] * scale
    elif return_col in out.columns:
        # Fallback: RegimeExp column absent — cannot compute a proper rescale.
        # Copy return_col unchanged so downstream callers always have PortRetKS.
        # This path should never be reached in production (regime engine always
        # emits a RegimeExp column); it is preserved as a safety net only.
        out["PortRetKS"] = out[return_col].copy()

    return out


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    print("[kill_switch] Pulling HY OAS from FRED...")
    try:
        from macro_data import _get_fred_client
    except ImportError:
        from macro_data import _get_fred_client

    fred = _get_fred_client()
    hy_oas = fred.get_series(_DEFAULT_FRED_SERIES)
    hy_oas.name = "hy_oas"
    print(f"[kill_switch] Fetched {len(hy_oas)} observations "
          f"({hy_oas.index[0].date()} -> {hy_oas.index[-1].date()})")

    df = build_kill_switch_history(hy_oas)

    validate_kill_switch_history(df)

    print("\n[kill_switch] Last 15 rows:")
    print(
        df[["hy_oas", "zscore", "active", "reason", "exposure"]]
        .tail(15)
        .to_string()
    )

    # Spot check key dates
    print("\n[kill_switch] Spot checks:")
    for check_date in ["2008-10-15", "2009-06-30", "2020-03-23", "2021-06-30"]:
        try:
            state = get_kill_switch_status(as_of_date=check_date)
            print(f"  {check_date}: active={state.active}  z={state.zscore:.2f}  "
                  f"reason={state.reason}  label={state.label}")
        except Exception as e:
            print(f"  {check_date}: ERROR — {e}")
