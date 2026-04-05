import numpy as np
import pandas as pd

try:
    from .backtest import run_walkforward_backtest
    from .universe import constituents_asof, apply_history_events
except ImportError:  # pragma: no cover - supports direct script execution
    from backtest import run_walkforward_backtest
    from universe import constituents_asof, apply_history_events


def _max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())


def _to_metrics(name: str, rets: pd.Series, spy: pd.Series) -> dict:
    if rets.empty:
        return {
            "Test": name,
            "NQuarters": 0,
            "AnnRet": np.nan,
            "AnnVol": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "HitRateVsSPY": np.nan,
            "AvgAlphaQ": np.nan,
        }

    common = rets.index.intersection(spy.index)
    r = rets.loc[common].dropna()
    s = spy.loc[common].dropna()
    common2 = r.index.intersection(s.index)
    r = r.loc[common2]
    s = s.loc[common2]
    if r.empty:
        return {
            "Test": name,
            "NQuarters": 0,
            "AnnRet": np.nan,
            "AnnVol": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "HitRateVsSPY": np.nan,
            "AvgAlphaQ": np.nan,
        }

    mean_q = float(r.mean())
    vol_q = float(r.std())
    ann_ret = (1 + mean_q) ** 4 - 1
    ann_vol = vol_q * np.sqrt(4)
    eq = (1 + r).cumprod()
    sharpe = ann_ret / (ann_vol + 1e-12)
    hit = float((r > s).mean())
    alpha = float((r - s).mean())
    return {
        "Test": name,
        "NQuarters": int(len(r)),
        "AnnRet": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": _max_drawdown(eq),
        "HitRateVsSPY": hit,
        "AvgAlphaQ": alpha,
    }


def _run_variant(
    name: str,
    cfg,
    sector_features_q: pd.DataFrame,
    sector_labels_q: pd.DataFrame,
    sector_returns_q: pd.DataFrame,
    spy_returns_q: pd.Series,
    constituents: pd.DataFrame,
    stock_prices_m: pd.DataFrame,
    stock_returns_q: pd.DataFrame,
    constituents_history: pd.DataFrame | None = None,
) -> dict:
    bt_df, _ = run_walkforward_backtest(
        cfg=cfg,
        sector_features_q=sector_features_q,
        sector_labels_q=sector_labels_q,
        sector_returns_q=sector_returns_q,
        spy_returns_q=spy_returns_q,
        constituents=constituents,
        stock_prices_m=stock_prices_m,
        stock_returns_q=stock_returns_q,
        stock_dollarvol_m=None,
        sector_etf_to_gics=None,
        constituents_history=constituents_history,
    )
    if bt_df.empty:
        return _to_metrics(name, pd.Series(dtype=float), spy_returns_q)
    return _to_metrics(name, bt_df["PortRet"], bt_df["SPYRet"])


def _equal_weight_all_stocks_metrics(stock_returns_q: pd.DataFrame, spy_returns_q: pd.Series) -> dict:
    ew = stock_returns_q.mean(axis=1, skipna=True).dropna()
    return _to_metrics("EqualWeight_AllStocks", ew, spy_returns_q)


def _random_sector_baseline(
    cfg,
    sector_returns_q: pd.DataFrame,
    constituents: pd.DataFrame,
    stock_prices_m: pd.DataFrame,
    stock_returns_q: pd.DataFrame,
    spy_returns_q: pd.Series,
    n_trials: int = 20,
    constituents_history: pd.DataFrame | None = None,
) -> dict:
    q_dates = sector_returns_q.index.sort_values()
    min_train = max(20, cfg.lookback_months_sector // 3)
    sym_to_sector = dict(zip(constituents["Symbol"], constituents["Sector"]))

    trial_sharpes = []
    trial_annrets = []
    trial_alpha_q = []

    sector_names = list(sector_returns_q.columns)
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        port = []
        spy = []
        dates = []
        for i in range(min_train, len(q_dates) - 1):
            dt = q_dates[i]
            next_dt = q_dates[i + 1]
            if next_dt not in stock_returns_q.index:
                continue

            k = min(cfg.top_k_sectors, len(sector_names))
            chosen_secs = list(rng.choice(sector_names, size=k, replace=False))

            current_constituents = constituents_asof(constituents, dt)
            current_constituents = apply_history_events(current_constituents, constituents_history, dt)
            if current_constituents.empty:
                current_constituents = constituents
            picks = []
            for sec in chosen_secs:
                tickers = current_constituents.loc[current_constituents["Sector"] == sec, "Symbol"].tolist()
                tickers = [t for t in tickers if t in stock_prices_m.columns]
                if not tickers:
                    continue
                sub_px = stock_prices_m.loc[:dt, tickers]
                if len(sub_px) > 18:
                    sub_px = sub_px.iloc[-18:]
                sub_px = sub_px.dropna(axis=1, how="any")
                if sub_px.shape[1] == 0:
                    continue
                # random stock picks within sector
                nn = min(cfg.top_n_stocks_per_sector, sub_px.shape[1])
                picks.extend(list(rng.choice(sub_px.columns, size=nn, replace=False)))

            if picks:
                pick_set = sorted(set(picks))
                r = stock_returns_q.loc[next_dt, pick_set].dropna()
                port_ret = float(r.mean()) if not r.empty else 0.0
            else:
                port_ret = 0.0
            port_ret = port_ret - (cfg.txn_cost_bps / 1e4)
            spy_ret = float(spy_returns_q.loc[next_dt]) if next_dt in spy_returns_q.index else 0.0

            # Guard against impossible sector map drift
            if picks:
                _ = [sym_to_sector.get(t) for t in pick_set]

            dates.append(next_dt)
            port.append(port_ret)
            spy.append(spy_ret)

        if not dates:
            continue
        idx = pd.Index(dates)
        p = pd.Series(port, index=idx).sort_index()
        s = pd.Series(spy, index=idx).sort_index()
        m = _to_metrics("RandomSector_Baseline", p, s)
        trial_sharpes.append(m["Sharpe"])
        trial_annrets.append(m["AnnRet"])
        trial_alpha_q.append(m["AvgAlphaQ"])

    if not trial_sharpes:
        return {
            "Test": "RandomSector_Baseline",
            "NQuarters": 0,
            "AnnRet": np.nan,
            "AnnVol": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "HitRateVsSPY": np.nan,
            "AvgAlphaQ": np.nan,
        }

    return {
        "Test": "RandomSector_Baseline",
        "NQuarters": np.nan,
        "AnnRet": float(np.nanmean(trial_annrets)),
        "AnnVol": np.nan,
        "Sharpe": float(np.nanmean(trial_sharpes)),
        "MaxDD": np.nan,
        "HitRateVsSPY": np.nan,
        "AvgAlphaQ": float(np.nanmean(trial_alpha_q)),
    }


def run_bias_diagnostics(
    cfg,
    strategy_df: pd.DataFrame,
    sector_features_q: pd.DataFrame,
    sector_labels_q: pd.DataFrame,
    sector_returns_q: pd.DataFrame,
    spy_returns_q: pd.Series,
    constituents: pd.DataFrame,
    stock_prices_m: pd.DataFrame,
    stock_returns_q: pd.DataFrame,
    benchmark_note: str = "",
    universe_note: str = "",
    constituents_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    rows = []
    rows.append(_to_metrics("Strategy_Main", strategy_df.get("PortRet", pd.Series(dtype=float)), strategy_df.get("SPYRet", pd.Series(dtype=float))))
    rows.append(_equal_weight_all_stocks_metrics(stock_returns_q, spy_returns_q))
    rows.append(
        _random_sector_baseline(
            cfg=cfg,
            sector_returns_q=sector_returns_q,
            constituents=constituents,
            stock_prices_m=stock_prices_m,
            stock_returns_q=stock_returns_q,
            spy_returns_q=spy_returns_q,
            n_trials=20,
            constituents_history=constituents_history,
        )
    )

    stale_X = sector_features_q.shift(1).dropna(how="all")
    rows.append(
        _run_variant(
            "StaleFeatures_ShiftPlus1Q",
            cfg,
            stale_X,
            sector_labels_q,
            sector_returns_q,
            spy_returns_q,
            constituents,
            stock_prices_m,
            stock_returns_q,
            constituents_history,
        )
    )

    leaked_X = sector_features_q.shift(-1).dropna(how="all")
    rows.append(
        _run_variant(
            "LeakedFeatures_ShiftMinus1Q",
            cfg,
            leaked_X,
            sector_labels_q,
            sector_returns_q,
            spy_returns_q,
            constituents,
            stock_prices_m,
            stock_returns_q,
            constituents_history,
        )
    )

    rng = np.random.default_rng(42)
    shuffled_Y = pd.DataFrame(
        {c: pd.Series(rng.permutation(sector_labels_q[c].values), index=sector_labels_q.index) for c in sector_labels_q.columns},
        index=sector_labels_q.index,
    )
    rows.append(
        _run_variant(
            "ShuffledLabels_Sanity",
            cfg,
            sector_features_q,
            shuffled_Y,
            sector_returns_q,
            spy_returns_q,
            constituents,
            stock_prices_m,
            stock_returns_q,
            constituents_history,
        )
    )

    diag_df = pd.DataFrame(rows)
    for col in ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "HitRateVsSPY", "AvgAlphaQ"]:
        if col in diag_df.columns:
            diag_df[col] = pd.to_numeric(diag_df[col], errors="coerce")

    lookup = {r["Test"]: r for r in rows}
    s_main = lookup.get("Strategy_Main", {})
    stale = lookup.get("StaleFeatures_ShiftPlus1Q", {})
    leaked = lookup.get("LeakedFeatures_ShiftMinus1Q", {})
    shuffled = lookup.get("ShuffledLabels_Sanity", {})
    ew = lookup.get("EqualWeight_AllStocks", {})
    rnd = lookup.get("RandomSector_Baseline", {})

    flags = []
    main_sharpe = float(s_main.get("Sharpe", np.nan))
    if np.isfinite(main_sharpe):
        stale_sharpe = float(stale.get("Sharpe", np.nan))
        if np.isfinite(stale_sharpe) and stale_sharpe >= main_sharpe * 0.9:
            flags.append("Stale-feature test did not degrade much; check feature/label date alignment.")

        shuffled_sharpe = float(shuffled.get("Sharpe", np.nan))
        if np.isfinite(shuffled_sharpe) and shuffled_sharpe >= main_sharpe * 0.8:
            flags.append("Shuffled-label performance remains high; possible overfitting or leakage.")

        leaked_sharpe = float(leaked.get("Sharpe", np.nan))
        if np.isfinite(leaked_sharpe) and leaked_sharpe < main_sharpe:
            flags.append("Future-feature test did not improve vs main; inspect shift directions carefully.")

        ew_sharpe = float(ew.get("Sharpe", np.nan))
        if np.isfinite(ew_sharpe) and main_sharpe <= ew_sharpe:
            flags.append("Strategy Sharpe does not beat equal-weight stock baseline.")

        rnd_sharpe = float(rnd.get("Sharpe", np.nan))
        if np.isfinite(rnd_sharpe) and main_sharpe <= rnd_sharpe:
            flags.append("Strategy Sharpe does not beat random-sector baseline.")

    lines = []
    lines.append("Bias/Leakage Diagnostics Summary")
    if benchmark_note:
        lines.append(benchmark_note)
    if universe_note:
        lines.append(universe_note)
    lines.append(f"Main strategy Sharpe: {main_sharpe:.3f}" if np.isfinite(main_sharpe) else "Main strategy Sharpe: NaN")
    lines.append("Tests run: Strategy, EqualWeight baseline, RandomSector baseline, StaleFeatures, LeakedFeatures, ShuffledLabels")
    if flags:
        lines.append("Flags:")
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("Flags: none triggered by current heuristics.")
    lines.append("")
    if "added/removed history events" in universe_note.lower():
        lines.append("Important residual risk: event file is sparse/template-level; use full constituent history and delisting returns for complete bias control.")
    elif "point-in-time filter enabled" in universe_note.lower():
        lines.append("Important residual risk: this uses DateAdded only; add DateRemoved/delist history for full point-in-time membership.")
    else:
        lines.append("Important residual risk: static constituents introduce survivorship bias unless replaced by point-in-time membership.")

    return diag_df, "\n".join(lines)
