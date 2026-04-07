"""
live_predict.py
===============
Live signal dashboard. Run before market open any day.

Outputs a single terminal summary covering every layer of the strategy:

    Layer 1 — Macro regime   (HMM: Expansion / Transition / Contraction)
    Layer 2 — Kill switch    (HY OAS > 3σ → emergency de-risk)
    Layer 3 — Sector signal  (Ridge: top 3 sectors for next quarter)
    Layer 4 — Stock picks    (Momentum: top 10 stocks per sector)
    Layer 5 — Position sizes (regime exposure × equal weight)
    Layer 6 — Action summary (plain English: what to do today)

Usage
-----
    python live_predict.py              # full dashboard
    python live_predict.py --json       # machine-readable JSON output
    python live_predict.py --save       # save to reports/live_signal_YYYYMMDD.txt

All data pulled live: FRED (regime), yfinance (sector ETFs + stocks).
No stale data. Every run reflects the current market state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve project root (works whether run directly or as module) ─────────────
_HERE = Path(__file__).resolve().parent

try:
    from .config import Config
    from .features import sector_features, make_sector_training_set
    from .macro_data import load_macro_features, get_feature_cols
    from .paths import ensure_layout, input_file, reports_dir
    from .prices import (
        download_monthly_adjclose,
        monthly_returns_from_prices,
        quarter_returns_from_monthly,
    )
    from .regime_engine import (
        build_full_regime_history,
        _train_hmm,
        _label_states,
        EXPOSURE_MAP,
        CONFIRM_WINDOW,
    )
    from .sector_model import SectorReturnModel
    from .stock_model import select_stocks_within_sectors
    from .universe import (
        apply_history_events,
        constituents_asof,
        load_constituents_history,
        load_sp500_constituents,
    )
    from .kill_switch import get_kill_switch_status
except ImportError:
    from config import Config
    from features import sector_features, make_sector_training_set
    from macro_data import load_macro_features, get_feature_cols
    from paths import ensure_layout, input_file, reports_dir
    from prices import (
        download_monthly_adjclose,
        monthly_returns_from_prices,
        quarter_returns_from_monthly,
    )
    from regime_engine import (
        build_full_regime_history,
        _train_hmm,
        _label_states,
        EXPOSURE_MAP,
        CONFIRM_WINDOW,
    )
    from sector_model import SectorReturnModel
    from stock_model import select_stocks_within_sectors
    from universe import (
        apply_history_events,
        constituents_asof,
        load_constituents_history,
        load_sp500_constituents,
    )
    from kill_switch import get_kill_switch_status


# ── Constants ─────────────────────────────────────────────────────────────────

SECTOR_ETF_TO_GICS = {
    "XLC" : "Communication Services",
    "XLY" : "Consumer Discretionary",
    "XLP" : "Consumer Staples",
    "XLE" : "Energy",
    "XLF" : "Financials",
    "XLV" : "Health Care",
    "XLI" : "Industrials",
    "XLB" : "Materials",
    "XLRE": "Real Estate",
    "XLK" : "Information Technology",
    "XLU" : "Utilities",
}

REGIME_EMOJI = {
    "Expansion"   : "▲",
    "Transition"  : "→",
    "Contraction" : "▼",
    "Unknown"     : "?",
}

REGIME_DESCRIPTION = {
    "Expansion": (
        "Risk-on. Credit spreads tight and tightening, markets calm. "
        "Momentum works strongly. Full exposure justified."
    ),
    "Transition": (
        "Normal growth regime. No acute stress signals. "
        "Run momentum strategy at full exposure."
    ),
    "Contraction": (
        "Macro stress detected. HY OAS momentum rising, risk-off conditions. "
        "Exposure reduced to protect capital."
    ),
    "Unknown": "Regime could not be classified — defaulting to full exposure.",
}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RegimeReading:
    regime          : str   = "Unknown"
    confidence      : float = 1.0
    exposure        : float = 1.0
    is_stable       : bool  = True
    prob_expansion  : float = 0.0
    prob_transition : float = 0.0
    prob_contraction: float = 0.0
    hy_oas_zscore   : float = float("nan")


@dataclass
class KillSwitchReading:
    active  : bool  = False
    zscore  : float = float("nan")
    reason  : str   = "inactive"
    exposure: float = 1.0


@dataclass
class SectorCall:
    rank        : int
    etf         : str
    name        : str
    predicted_q : float   # predicted next-quarter return (relative score)
    stocks      : list[str] = field(default_factory=list)


@dataclass
class PositionEntry:
    ticker  : str
    sector  : str
    weight  : float   # % of portfolio
    size_1k : float   # position size at £1,000 portfolio (scale linearly)


@dataclass
class LiveSignal:
    signal_date        : str
    next_quarter_end   : str
    regime             : RegimeReading
    kill_switch        : KillSwitchReading
    effective_exposure : float
    sectors            : list[SectorCall]
    positions          : list[PositionEntry]
    action             : str
    warnings           : list[str] = field(default_factory=list)
    errors             : list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts.to_period("Q") + 1).to_timestamp(how="end").normalize()


def _download_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    px = download_monthly_adjclose(tickers, start=start, end=end)
    return px.loc[:, ~px.columns.duplicated()].sort_index()


def _bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    filled = int(round(value / max_val * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


# ── Layer 1: Regime ───────────────────────────────────────────────────────────

def get_regime(cfg: Config, warn_list: list[str]) -> RegimeReading:
    """
    Pull live macro features from FRED, train HMM on all available
    in-sample data, classify current regime using last CONFIRM_WINDOW days.
    """
    try:
        print("  [1/5] Loading macro features from FRED...", end=" ", flush=True)
        features_df  = load_macro_features()
        feature_cols = get_feature_cols()
        print(f"OK ({len(features_df)} obs, latest {features_df.index[-1].date()})")

        # Train HMM on all in-sample data (frozen at INSAMPLE_END)
        from macro_data import INSAMPLE_END
        X_insample   = features_df.loc[:INSAMPLE_END, feature_cols].dropna()
        print(f"  [1/5] Training HMM on {len(X_insample)} in-sample days...", end=" ", flush=True)
        model        = _train_hmm(X_insample.values)
        state_labels = _label_states(model, feature_cols)
        print("OK")

        # Classify using last CONFIRM_WINDOW days
        X_recent  = features_df[feature_cols].dropna().values[-CONFIRM_WINDOW:]
        posteriors = model.predict_proba(X_recent)

        # Stability: all CONFIRM_WINDOW days must agree
        daily_regimes = [
            state_labels[int(np.argmax(posteriors[t]))]
            for t in range(len(posteriors))
        ]
        is_stable     = len(set(daily_regimes)) == 1
        last_probs    = posteriors[-1]
        winning_state = int(np.argmax(last_probs))
        regime        = state_labels[winning_state] if is_stable else "Transition"
        confidence    = float(last_probs[winning_state])

        if not is_stable:
            warn_list.append(
                f"Regime unstable: {set(daily_regimes)} over last {CONFIRM_WINDOW} days "
                "→ defaulting to Transition (conservative)"
            )

        probs_dict = {state_labels[i]: float(last_probs[i]) for i in range(3)}

        # HY OAS z-score for context (latest reading)
        hy_z = float(features_df["hy_oas"].dropna().iloc[-1])

        return RegimeReading(
            regime           = regime,
            confidence       = confidence,
            exposure         = float(EXPOSURE_MAP.get(regime, 1.0)),
            is_stable        = is_stable,
            prob_expansion   = probs_dict.get("Expansion",    0.0),
            prob_transition  = probs_dict.get("Transition",   0.0),
            prob_contraction = probs_dict.get("Contraction",  0.0),
            hy_oas_zscore    = hy_z,
        )

    except Exception as exc:
        warn_list.append(f"Regime engine failed ({exc}) — defaulting to Transition, 1.0x exposure")
        return RegimeReading(regime="Unknown", confidence=1.0, exposure=1.0)


# ── Layer 2: Kill switch ──────────────────────────────────────────────────────

def get_kill_switch(cfg: Config, warn_list: list[str]) -> KillSwitchReading:
    """Pull live HY OAS from FRED and check kill switch status."""
    try:
        print("  [2/5] Checking kill switch...", end=" ", flush=True)
        ks = get_kill_switch_status(cfg=cfg)
        print(f"OK  active={ks.active}  z={ks.zscore:.2f}  reason={ks.reason}")
        if ks.active:
            warn_list.append(
                f"KILL SWITCH ACTIVE — HY OAS z-score {ks.zscore:.2f} > 3.0σ. "
                "Portfolio forced to Contraction exposure (0.30x)."
            )
        return KillSwitchReading(
            active   = ks.active,
            zscore   = ks.zscore,
            reason   = ks.reason,
            exposure = 0.30 if ks.active else 1.0,
        )
    except Exception as exc:
        warn_list.append(f"Kill switch check failed ({exc}) — assuming inactive")
        return KillSwitchReading()


# ── Layer 3: Sector signal ────────────────────────────────────────────────────

def get_sector_signal(cfg: Config, warn_list: list[str]) -> tuple[list[SectorCall], pd.Timestamp]:
    """
    Train Ridge model on live sector price data and predict
    next-quarter returns for all 11 SPDR sectors.
    """
    print("  [3/5] Downloading sector prices...", end=" ", flush=True)
    end_date     = (date.today() + timedelta(days=1)).isoformat()
    sector_px_m  = _download_panel(cfg.sector_tickers, cfg.start_date, end_date)
    spy_px_m     = _download_panel([cfg.spy_ticker],    cfg.start_date, end_date)

    if sector_px_m.empty or spy_px_m.empty:
        warn_list.append("Could not download sector prices — no sector signal available")
        return [], pd.Timestamp.today()

    sector_rets_m = monthly_returns_from_prices(sector_px_m)
    spy_rets_m    = monthly_returns_from_prices(spy_px_m)[cfg.spy_ticker]
    sector_rets_q = quarter_returns_from_monthly(sector_px_m)
    feature_dt    = sector_rets_m.index.max()
    print(f"OK (latest {feature_dt.date()})")

    print("  [3/5] Training Ridge sector model...", end=" ", flush=True)
    X_m          = sector_features(sector_rets_m, spy_rets_m)
    X_q, Y_q     = make_sector_training_set(X_m, sector_rets_q)
    train_mask   = Y_q.notna().all(axis=1)
    X_train      = X_q.loc[train_mask]
    Y_train      = Y_q.loc[train_mask]

    model = SectorReturnModel(alpha=10.0)
    model.fit(X_train, Y_train)
    preds = model.predict_one(X_q.loc[[feature_dt]])
    print(f"OK ({len(preds)} sector predictions)")

    pred_sorted = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    calls = []
    for rank, (etf, score) in enumerate(pred_sorted, 1):
        name = SECTOR_ETF_TO_GICS.get(etf, etf)
        calls.append(SectorCall(
            rank=rank, etf=etf, name=name,
            predicted_q=float(score), stocks=[]
        ))

    return calls, feature_dt


# ── Layer 4: Stock picks ──────────────────────────────────────────────────────

def get_stock_picks(
    cfg              : Config,
    sector_calls     : list[SectorCall],
    feature_dt       : pd.Timestamp,
    warn_list        : list[str],
) -> list[SectorCall]:
    """Select top momentum stocks within the predicted sectors."""
    top_sectors   = [s for s in sector_calls if s.rank <= cfg.top_k_sectors]
    top_names     = [s.name for s in top_sectors]

    print("  [4/5] Loading constituents and stock prices...", end=" ", flush=True)
    constituents  = load_sp500_constituents(str(input_file("sp500_constituents.csv")))
    history_path  = input_file("constituents_history_template.csv")
    history       = load_constituents_history(str(history_path)) if history_path.exists() else None

    current_const = constituents_asof(constituents, feature_dt)
    current_const = apply_history_events(current_const, history, feature_dt)
    if current_const.empty:
        current_const = constituents.copy()

    sector_tickers = (
        current_const
        .loc[current_const["Sector"].isin(top_names), "Symbol"]
        .dropna().unique().tolist()
    )

    end_date   = (date.today() + timedelta(days=1)).isoformat()
    stock_px_m = _download_panel(sector_tickers, cfg.start_date, end_date)
    print(f"OK ({len(sector_tickers)} sector members, {stock_px_m.shape[1]} downloaded)")

    print("  [4/5] Ranking stocks by 12-1 momentum...", end=" ", flush=True)
    picks = select_stocks_within_sectors(
        constituents         = current_const,
        sector_names_pred    = top_names,
        stock_prices_m       = stock_px_m.loc[:feature_dt],
        stock_volume_d_m     = None,
        min_avg_dollar_vol   = cfg.min_avg_dollar_vol,
        top_n_per_sector     = cfg.top_n_stocks_per_sector,
        min_history_months   = cfg.min_history_months_for_stock_selection,
    )
    print("OK")

    # Attach picks to sector calls
    for call in top_sectors:
        call.stocks = picks.get(call.name, [])

    return sector_calls


# ── Layer 5: Position sizes ───────────────────────────────────────────────────

def build_positions(
    sector_calls       : list[SectorCall],
    effective_exposure : float,
    cfg                : Config,
) -> list[PositionEntry]:
    """
    Equal-weight positions within the selected sectors,
    scaled by the effective regime exposure.
    """
    top_sectors = [s for s in sector_calls if s.rank <= cfg.top_k_sectors]
    all_picks   = [
        (ticker, call.name)
        for call in top_sectors
        for ticker in call.stocks
    ]

    if not all_picks:
        return []

    n           = len(all_picks)
    weight_each = effective_exposure / n      # fraction of total portfolio
    size_1k     = weight_each * 1_000         # position size at £1k portfolio

    positions = []
    for ticker, sector in all_picks:
        positions.append(PositionEntry(
            ticker  = ticker,
            sector  = sector,
            weight  = weight_each * 100,       # as %
            size_1k = size_1k,
        ))

    return sorted(positions, key=lambda x: x.sector)


# ── Layer 6: Action summary ───────────────────────────────────────────────────

def build_action_summary(
    regime      : RegimeReading,
    kill_switch : KillSwitchReading,
    exposure    : float,
    sectors     : list[SectorCall],
    cfg         : Config,
) -> str:
    top = [s for s in sectors if s.rank <= cfg.top_k_sectors]

    if kill_switch.active:
        return (
            f"KILL SWITCH ACTIVE. Reduce all equity positions to {exposure:.0%} of normal size. "
            f"HY OAS is {kill_switch.zscore:.1f}σ above long-run average — acute credit stress. "
            "Hold short-duration bonds or cash for the remainder of the quarter. "
            "Do not initiate new positions until kill switch deactivates."
        )

    if regime.regime == "Contraction":
        sector_str = ", ".join(s.name for s in top)
        return (
            f"CONTRACTION REGIME — run at {exposure:.0%} exposure. "
            f"Hold {sector_str} but size positions at {exposure:.0%} of normal. "
            f"Confidence: {regime.confidence:.0%}. "
            "Keep remaining capital in short-duration bonds or cash. "
            "Review at next quarterly rebalance."
        )

    if regime.regime in ("Expansion", "Transition"):
        sector_str = ", ".join(s.name for s in top)
        stock_count = sum(len(s.stocks) for s in top)
        return (
            f"{'EXPANSION' if regime.regime == 'Expansion' else 'TRANSITION'} REGIME — "
            f"run at full {exposure:.0%} exposure. "
            f"Buy equal-weight across {stock_count} stocks in {sector_str}. "
            f"HMM confidence: {regime.confidence:.0%}. "
            "Hold until next quarterly rebalance signal."
        )

    return (
        "Regime unclear. Hold current positions. "
        "Do not rebalance until regime stabilises over 5 consecutive days."
    )


# ── Dashboard printer ─────────────────────────────────────────────────────────

def print_dashboard(signal: LiveSignal) -> None:
    W  = 68
    hr = "─" * W

    def box(title: str) -> None:
        print(f"\n┌{hr}┐")
        pad = (W - len(title)) // 2
        print(f"│{' ' * pad}{title}{' ' * (W - pad - len(title))}│")
        print(f"└{hr}┘")

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("═" * (W + 2))
    title = "MACRO REGIME MOMENTUM STRATEGY — LIVE SIGNAL"
    print(f"  {title}")
    print(f"  Signal date : {signal.signal_date}")
    print(f"  For quarter : {signal.next_quarter_end}")
    print("═" * (W + 2))

    # ── Layer 1: Regime ───────────────────────────────────────────────────────
    box("LAYER 1 — MACRO REGIME (HMM)")
    r = signal.regime
    icon = REGIME_EMOJI.get(r.regime, "?")
    print(f"  Regime      : {icon} {r.regime.upper()}")
    print(f"  Confidence  : {r.confidence:.1%}  {_bar(r.confidence)}")
    print(f"  Stable      : {'Yes' if r.is_stable else 'No — forced to Transition'}")
    print(f"  HY OAS z    : {r.hy_oas_zscore:>+.2f}σ")
    print()
    print(f"  State probabilities:")
    for label, prob in [
        ("Expansion",    r.prob_expansion),
        ("Transition",   r.prob_transition),
        ("Contraction",  r.prob_contraction),
    ]:
        marker = " ◄" if label == r.regime else ""
        print(f"    {label:<14}: {prob:.1%}  {_bar(prob, 25)}{marker}")
    print()
    desc = REGIME_DESCRIPTION.get(r.regime, "")
    # Wrap description at W-4 chars
    words = desc.split()
    line, lines = [], []
    for w in words:
        if len(" ".join(line + [w])) > W - 4:
            lines.append("  " + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))

    # ── Layer 2: Kill switch ──────────────────────────────────────────────────
    box("LAYER 2 — KILL SWITCH (HY OAS LEVEL)")
    ks = signal.kill_switch
    status = "🔴 ACTIVE" if ks.active else "🟢 INACTIVE"
    print(f"  Status      : {status}")
    print(f"  HY OAS z    : {ks.zscore:>+.2f}σ  (trigger > 3.0σ, deactivate < 2.0σ)")
    print(f"  Reason      : {ks.reason}")
    if ks.active:
        print(f"  Override    : Forces exposure to 0.30x regardless of HMM")

    # ── Effective exposure ────────────────────────────────────────────────────
    box("EFFECTIVE EXPOSURE")
    exp = signal.effective_exposure
    print(f"  HMM exposure    : {signal.regime.exposure:.2f}x")
    if signal.kill_switch.active:
        print(f"  Kill switch     : 0.30x  (OVERRIDE — takes precedence)")
    else:
        print(f"  Kill switch     : inactive (no override)")
    print(f"  ─────────────────────────────")
    print(f"  FINAL EXPOSURE  : {exp:.2f}x  {_bar(exp, 30)}")
    print(f"  Cash/bonds      : {1-exp:.0%} of portfolio")

    # ── Layer 3: Sector signal ────────────────────────────────────────────────
    box("LAYER 3 — SECTOR SIGNAL (RIDGE MODEL)")
    print(f"  {'Rank':<6} {'ETF':<6} {'Sector':<30} {'Score':>8}  {'Selected'}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*28}  {'─'*8}  {'─'*8}")
    for s in signal.sectors:
        selected = "✓ HOLD" if s.rank <= 3 else ""
        # Normalise scores for display (min-max)
        print(f"  #{s.rank:<4} {s.etf:<6} {s.name:<30} {s.predicted_q:>+8.4f}  {selected}")

    # ── Layer 4: Stock picks ──────────────────────────────────────────────────
    box("LAYER 4 — STOCK PICKS (12-1 MOMENTUM)")
    for s in signal.sectors:
        if s.rank > 3:
            continue
        print(f"  {s.name} ({s.etf})")
        if s.stocks:
            cols = [s.stocks[i:i+5] for i in range(0, len(s.stocks), 5)]
            for col in cols:
                print("    " + "  ".join(f"{t:<8}" for t in col))
        else:
            print("    (no picks available)")
        print()

    # ── Layer 5: Position sizes ───────────────────────────────────────────────
    box("LAYER 5 — POSITION SIZES")
    if signal.positions:
        print(f"  {'Ticker':<8} {'Sector':<30} {'Weight':>7}  {'£1k → £':>10}  {'£100k → £':>12}")
        print(f"  {'─'*6}  {'─'*28}  {'─'*7}  {'─'*10}  {'─'*12}")
        prev_sector = None
        for p in signal.positions:
            if p.sector != prev_sector:
                if prev_sector is not None:
                    print()
                prev_sector = p.sector
            print(
                f"  {p.ticker:<8} {p.sector:<30} {p.weight:>6.2f}%  "
                f"£{p.size_1k:>8.2f}   £{p.size_1k*100:>10,.0f}"
            )
        print()
        print(f"  Total exposure: {signal.effective_exposure:.0%} of portfolio")
        print(f"  Cash/bonds:     {1-signal.effective_exposure:.0%} of portfolio")
    else:
        print("  No positions — hold cash.")

    # ── Layer 6: Action ───────────────────────────────────────────────────────
    box("LAYER 6 — ACTION SUMMARY")
    words = signal.action.split()
    line, lines = [], []
    for w in words:
        if len(" ".join(line + [w])) > W - 4:
            lines.append("  " + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))

    # ── Warnings ──────────────────────────────────────────────────────────────
    if signal.warnings:
        print()
        print(f"  ⚠  WARNINGS ({len(signal.warnings)})")
        for w in signal.warnings:
            print(f"     • {w}")

    print()
    print("═" * (W + 2))
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_live_signal(
    cfg        : Config,
    output_dir : str | Path | None = None,
) -> LiveSignal:

    ensure_layout()
    out_dir = Path(output_dir) if output_dir is not None else reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    warn_list = []
    print()
    print("Building live signal...")
    print()

    # Layer 1 — Regime
    regime = get_regime(cfg, warn_list)

    # Layer 2 — Kill switch
    kill_switch = get_kill_switch(cfg, warn_list)

    # Effective exposure: kill switch overrides HMM
    if kill_switch.active:
        effective_exposure = 0.30
    else:
        if regime.regime == "Contraction":
            # Soft scale: base × confidence
            effective_exposure = regime.exposure * regime.confidence
        else:
            effective_exposure = regime.exposure

    # Layer 3 — Sector signal
    sector_calls, feature_dt = get_sector_signal(cfg, warn_list)

    # Layer 4 — Stock picks (only for top 3 sectors)
    if sector_calls:
        sector_calls = get_stock_picks(cfg, sector_calls, feature_dt, warn_list)

    # Layer 5 — Positions
    print("  [5/5] Computing position sizes...", end=" ", flush=True)
    positions = build_positions(sector_calls, effective_exposure, cfg)
    print(f"OK ({len(positions)} positions at {effective_exposure:.0%} exposure)")

    # Layer 6 — Action
    action = build_action_summary(regime, kill_switch, effective_exposure, sector_calls, cfg)

    signal = LiveSignal(
        signal_date        = pd.Timestamp.today().strftime("%Y-%m-%d"),
        next_quarter_end   = str(_next_quarter_end(feature_dt).date()),
        regime             = regime,
        kill_switch        = kill_switch,
        effective_exposure = effective_exposure,
        sectors            = sector_calls,
        positions          = positions,
        action             = action,
        warnings           = warn_list,
    )

    # Save outputs
    _save_signal(signal, out_dir)

    return signal


def _save_signal(signal: LiveSignal, out_dir: Path) -> None:
    """Save machine-readable JSON and CSV outputs."""
    today = signal.signal_date

    # JSON (full signal)
    def _serialise(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    json_path = out_dir / f"live_signal_{today}.json"
    json_path.write_text(
        json.dumps(asdict(signal), indent=2), encoding="utf-8"
    )

    # Sector predictions CSV
    sector_rows = [
        {
            "rank"       : s.rank,
            "etf"        : s.etf,
            "sector"     : s.name,
            "predicted_q": s.predicted_q,
            "selected"   : s.rank <= 3,
        }
        for s in signal.sectors
    ]
    pd.DataFrame(sector_rows).to_csv(
        out_dir / f"live_sectors_{today}.csv", index=False
    )

    # Position sizes CSV
    if signal.positions:
        pos_rows = [
            {
                "ticker"  : p.ticker,
                "sector"  : p.sector,
                "weight_pct": p.weight,
                "size_per_1k_GBP": p.size_1k,
            }
            for p in signal.positions
        ]
        pd.DataFrame(pos_rows).to_csv(
            out_dir / f"live_positions_{today}.csv", index=False
        )

    # Summary text
    summary_lines = [
        f"Live Macro Regime Signal — {today}",
        f"Next quarter end       : {signal.next_quarter_end}",
        f"Regime                 : {signal.regime.regime} "
        f"(conf={signal.regime.confidence:.1%}, stable={signal.regime.is_stable})",
        f"HY OAS z-score         : {signal.regime.hy_oas_zscore:+.2f}σ",
        f"Kill switch            : {'ACTIVE' if signal.kill_switch.active else 'inactive'}",
        f"Kill switch z          : {signal.kill_switch.zscore:+.2f}σ",
        f"Effective exposure     : {signal.effective_exposure:.0%}",
        "",
        "Top 3 sectors:",
    ]
    for s in signal.sectors[:3]:
        summary_lines.append(f"  {s.rank}. {s.name} ({s.etf}) — picks: {', '.join(s.stocks)}")

    summary_lines += ["", "Action:", f"  {signal.action}"]

    if signal.warnings:
        summary_lines += ["", "Warnings:"]
        for w in signal.warnings:
            summary_lines.append(f"  • {w}")

    (out_dir / f"live_summary_{today}.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print(f"\n  Saved: {json_path.name}, live_sectors_{today}.csv, "
          f"live_positions_{today}.csv, live_summary_{today}.txt")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live macro regime signal dashboard."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for saved outputs (default: reports/)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable JSON to stdout instead of dashboard"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages (dashboard still prints)"
    )
    args = parser.parse_args()

    if args.quiet:
        sys.stdout = open(os.devnull, "w")

    cfg    = Config()
    signal = generate_live_signal(cfg=cfg, output_dir=args.output_dir)

    if args.quiet:
        sys.stdout = sys.__stdout__

    if args.json:
        print(json.dumps(asdict(signal), indent=2))
    else:
        print_dashboard(signal)


if __name__ == "__main__":
    main()
