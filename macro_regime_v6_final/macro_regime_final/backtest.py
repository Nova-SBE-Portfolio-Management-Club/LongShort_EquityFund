import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

try:
    from .metrics import annualized_volatility, cagr, sharpe_ratio
    from .paths import reports_dir
    from .sector_model import SectorReturnModel
    from .stock_model import select_stocks_within_sectors
    from .universe import constituents_asof, apply_history_events
    from .kill_switch import build_kill_switch_history, apply_kill_switch_to_returns
    from .macro_data import load_macro_features
    from .regime_engine import run_walkforward_regimes, classify_oos
except ImportError:  # pragma: no cover - supports direct script execution
    from metrics import annualized_volatility, cagr, sharpe_ratio
    from paths import reports_dir
    from sector_model import SectorReturnModel
    from stock_model import select_stocks_within_sectors
    from universe import constituents_asof, apply_history_events
    from kill_switch import build_kill_switch_history, apply_kill_switch_to_returns
    from macro_data import load_macro_features
    from regime_engine import run_walkforward_regimes, classify_oos

_GLOBAL_REGIME_DF_CACHE = None


def _load_cached_regime_history() -> pd.DataFrame:
    """Reuse the most recent successful regime classification if live rebuild fails."""
    cache_path = reports_dir() / "regime_classification_diagnostics.csv"
    if not cache_path.exists():
        return pd.DataFrame()

    cache_df = pd.read_csv(cache_path)
    if "Date" not in cache_df.columns:
        return pd.DataFrame()

    cache_df["Date"] = pd.to_datetime(cache_df["Date"])
    cache_df = cache_df.set_index("Date").sort_index()
    return cache_df


def _load_cached_kill_switch_history() -> pd.DataFrame:
    """Reuse the last successful HY OAS kill-switch history if FRED is unavailable."""
    cache_path = reports_dir() / "kill_switch_history.csv"
    if not cache_path.exists():
        return pd.DataFrame()

    cache_df = pd.read_csv(cache_path)
    date_col = cache_df.columns[0]
    cache_df[date_col] = pd.to_datetime(cache_df[date_col])
    cache_df = cache_df.set_index(date_col).sort_index()
    return cache_df

@dataclass
class QuarterResult:
    date: pd.Timestamp
    predicted_sectors: list[str]          # ETF tickers
    predicted_sector_names: list[str]     # GICS names
    realized_sector_ranks: list[str]      # top realized ETF tickers in that quarter
    selected_stocks: dict[str, list[str]] # sector_name -> stocks
    n_holdings: int
    turnover: float
    gross_return: float
    txn_cost_bps_applied: float
    financing_cost_return: float
    financing_rate_annual: float
    borrowed_notional_usd: float
    portfolio_return: float
    spy_return: float
    top_stocks_quarter: list[str]         # top performing stocks overall
    top_stocks_in_predicted_sectors: int  # count
    regime: str                           # final regime used by strategy = raw last-day HMM winner
    regime_confidence: float              # confidence of final/raw regime on last day
    regime_exposure: float                # 1.0, 0.65, 0.30
    raw_regime: str = "Unknown"           # raw last-day HMM winner (same as final regime)
    raw_regime_confidence: float = 1.0    # confidence of raw last-day HMM winner
    smoothed_regime: str = "Unknown"      # averaged-posterior regime over confirm window
    smoothed_regime_confidence: float = 1.0
    regime_is_stable: bool = True         # whether the confirm window stayed in one state
    prob_expansion: float = np.nan        # posterior probability on last day
    prob_transition: float = np.nan       # posterior probability on last day
    prob_contraction: float = np.nan      # posterior probability on last day
    kill_switch_active: bool = False      # True = kill switch was active this quarter

def apply_txn_cost(ret: float, bps: float) -> float:
    return ret - (bps / 1e4)


def _ibkr_pro_usd_margin_rate_annual(
    borrowed_balance_usd: float,
    benchmark_rate: float,
    floor_benchmark_at_zero: bool = True,
) -> float:
    """Approximate IBKR Pro USD blended annual margin rate from balance tiers."""
    borrowed_balance_usd = max(float(borrowed_balance_usd), 0.0)
    if borrowed_balance_usd <= 0.0:
        return 0.0

    bm = max(float(benchmark_rate), 0.0) if floor_benchmark_at_zero else float(benchmark_rate)
    tiers = [
        (100_000.0, 0.0150),
        (900_000.0, 0.0100),
        (49_000_000.0, 0.0075),
        (200_000_000.0, 0.0050),
        (float("inf"), 0.0050),
    ]

    remaining = borrowed_balance_usd
    weighted_rate = 0.0
    for tier_size, spread in tiers:
        if remaining <= 0.0:
            break
        alloc = min(remaining, tier_size)
        weighted_rate += alloc * (bm + spread)
        remaining -= alloc

    return weighted_rate / borrowed_balance_usd


def _financing_cost_return_for_period(
    cfg,
    equity_start: float,
    regime_exp: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> tuple[float, float, float]:
    """Return financing drag, annual financing rate, and borrowed notional in USD."""
    if not getattr(cfg, "use_ibkr_margin_financing", False):
        return 0.0, 0.0, 0.0

    borrowed_fraction = max(float(regime_exp) - 1.0, 0.0)
    if borrowed_fraction <= 0.0:
        return 0.0, 0.0, 0.0

    account_nav_usd = float(getattr(cfg, "ibkr_margin_account_nav_usd", 1_000_000.0))
    benchmark_rate = float(getattr(cfg, "ibkr_margin_benchmark_rate", 0.0))
    floor_bm = bool(getattr(cfg, "ibkr_margin_floor_benchmark_at_zero", True))
    day_count = int(getattr(cfg, "ibkr_margin_day_count", 360))

    borrowed_notional_usd = account_nav_usd * max(float(equity_start), 0.0) * borrowed_fraction
    annual_rate = _ibkr_pro_usd_margin_rate_annual(
        borrowed_balance_usd=borrowed_notional_usd,
        benchmark_rate=benchmark_rate,
        floor_benchmark_at_zero=floor_bm,
    )
    period_days = max(int((pd.Timestamp(end_dt) - pd.Timestamp(start_dt)).days), 1)
    financing_drag = borrowed_fraction * annual_rate * (period_days / day_count)
    return float(financing_drag), float(annual_rate), float(borrowed_notional_usd)


def _equal_weight_turnover(prev_tickers: list[str], curr_tickers: list[str]) -> float:
    prev = sorted(set(prev_tickers))
    curr = sorted(set(curr_tickers))
    if not prev and not curr:
        return 0.0
    if (not prev and curr) or (prev and not curr):
        return 1.0
    u = sorted(set(prev).union(curr))
    w_prev = 1.0 / len(prev)
    w_curr = 1.0 / len(curr)
    total_abs = 0.0
    for t in u:
        a = w_prev if t in prev else 0.0
        b = w_curr if t in curr else 0.0
        total_abs += abs(b - a)
    return float(0.5 * total_abs)


def _execution_cost_bps(cfg, turnover: float) -> float:
    if not getattr(cfg, "use_turnover_cost_model", False):
        return float(cfg.txn_cost_bps)
    one_way_bps = (
        float(cfg.commission_bps_oneway)
        + float(cfg.exchange_fee_bps_oneway)
        + float(cfg.spread_bps_oneway)
        + float(cfg.slippage_bps_oneway)
        + float(getattr(cfg, "impact_bps_oneway", 0.0))
    )
    return float(turnover * one_way_bps)

def build_equal_weight_portfolio_return(
    stock_rets_q: pd.DataFrame,
    tickers: list[str],
    dt: pd.Timestamp,
    max_abs_stock_quarter_return: float | None = None,
) -> float:
    if len(tickers) == 0:
        return 0.0
    r = stock_rets_q.loc[dt, tickers].dropna()
    if r.empty:
        return 0.0
    if max_abs_stock_quarter_return is not None:
        r = r.clip(lower=-float(max_abs_stock_quarter_return), upper=float(max_abs_stock_quarter_return))
    return float(r.mean())

def run_walkforward_backtest(
    cfg,
    sector_features_q: pd.DataFrame,
    sector_labels_q: pd.DataFrame,     # next quarter returns per sector ETF
    sector_returns_q: pd.DataFrame,    # realized quarter returns per sector ETF
    spy_returns_q: pd.Series,
    constituents: pd.DataFrame,        # Symbol, Sector
    stock_prices_m: pd.DataFrame,
    stock_returns_q: pd.DataFrame,
    stock_dollarvol_m: pd.DataFrame | None = None,
    sector_etf_to_gics: dict[str, str] | None = None,
    constituents_history: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[QuarterResult]]:

    results: list[QuarterResult] = []
    equity = 1.0
    spy_eq = 1.0

    q_dates = sector_features_q.index.intersection(sector_labels_q.index)
    q_dates = q_dates.sort_values()

    # ── Regime engine configuration (Cached implicitly) ──
    global _GLOBAL_REGIME_DF_CACHE
    if _GLOBAL_REGIME_DF_CACHE is None:
        try:
            print("\n[backtest] Building HMM regime classification exactly once for this session...")
            macro_feats = load_macro_features()

            q_dates_insample = [d for d in q_dates if d <= pd.Timestamp("2018-12-31")]
            regime_insample = run_walkforward_regimes(
                features_df=macro_feats, 
                rebalance_dates=pd.DatetimeIndex(q_dates_insample)
            )

            q_dates_oos = [d for d in q_dates if d > pd.Timestamp("2018-12-31")]
            regime_oos = classify_oos(
                features_df=macro_feats, 
                rebalance_dates=pd.DatetimeIndex(q_dates_oos)
            )
            _GLOBAL_REGIME_DF_CACHE = pd.concat([regime_insample, regime_oos]).sort_index()
        except Exception as e:
            cached_regimes = _load_cached_regime_history()
            if not cached_regimes.empty:
                print(
                    f"[backtest] WARNING: Regime engine failed to build ({e}). "
                    f"Using cached regime classifications from {cached_regimes.index.min().date()} "
                    f"to {cached_regimes.index.max().date()}."
                )
                _GLOBAL_REGIME_DF_CACHE = cached_regimes
            else:
                print(f"[backtest] WARNING: Regime engine failed to build ({e}). Defaulting to 1.0 exposure.")
                _GLOBAL_REGIME_DF_CACHE = pd.DataFrame()
    
    regime_df = _GLOBAL_REGIME_DF_CACHE

    # Start after we have enough training history
    min_train = max(20, cfg.lookback_months_sector // 3)
    prev_chosen: list[str] = []
    for i in range(min_train, len(q_dates)-1):
        dt = q_dates[i]  # feature date = end of quarter (T-1) predicting next quarter (T)
        next_dt = q_dates[i+1]  # quarter to trade
        assert next_dt > dt, "Quarter alignment error: next_dt must be strictly after dt."

        # Train on all data up to dt (inclusive)
        X_train = sector_features_q.loc[:dt]
        Y_train = sector_labels_q.loc[:dt]
        if not X_train.empty:
            assert X_train.index.max() <= dt, "Feature leakage detected in training window."
        if not Y_train.empty:
            assert Y_train.index.max() <= dt, "Label leakage detected in training window."
        assert next_dt not in X_train.index, "Feature leakage detected: test quarter present in training features."

        model = SectorReturnModel(alpha=10.0)
        model.fit(X_train, Y_train)

        X_last = sector_features_q.loc[[dt]]
        preds = model.predict_one(X_last)

        # Select top K sector ETFs by predicted return
        pred_sorted = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [s for s,_ in pred_sorted[:cfg.top_k_sectors]]

        # Map to GICS sector names
        if sector_etf_to_gics is None:
            # fallback: map XLK->Information Technology etc. (you can extend)
            sector_etf_to_gics = {
                "XLC":"Communication Services",
                "XLY":"Consumer Discretionary",
                "XLP":"Consumer Staples",
                "XLE":"Energy",
                "XLF":"Financials",
                "XLV":"Health Care",
                "XLI":"Industrials",
                "XLB":"Materials",
                "XLRE":"Real Estate",
                "XLK":"Information Technology",
                "XLU":"Utilities"
            }
        pred_sector_names = [sector_etf_to_gics.get(x, x) for x in top_sectors]

        # Select stocks within predicted sectors
        current_constituents = constituents_asof(constituents, dt)
        current_constituents = apply_history_events(current_constituents, constituents_history, dt)
        if current_constituents.empty:
            current_constituents = constituents
        stock_picks = select_stocks_within_sectors(
            constituents=current_constituents,
            sector_names_pred=pred_sector_names,
            stock_prices_m=stock_prices_m.loc[:dt],  # only information up to dt
            stock_volume_d_m=stock_dollarvol_m.loc[:dt] if stock_dollarvol_m is not None else None,
            min_avg_dollar_vol=cfg.min_avg_dollar_vol,
            top_n_per_sector=cfg.top_n_stocks_per_sector,
            min_history_months=cfg.min_history_months_for_stock_selection,
        )

        # Build portfolio tickers
        chosen = sorted(set([t for lst in stock_picks.values() for t in lst]))

        # Realized returns over next quarter
        gross_ret = build_equal_weight_portfolio_return(
            stock_returns_q,
            chosen,
            next_dt,
            max_abs_stock_quarter_return=cfg.max_abs_stock_quarter_return,
        )

        # Apply HMM regime exposure logic
        if not regime_df.empty and next_dt in regime_df.index:
            reg_row = regime_df.loc[next_dt]
            regime = str(reg_row['regime'])
            regime_conf = float(reg_row['confidence'])
            raw_regime = str(reg_row.get('raw_regime', regime))
            raw_regime_conf = float(reg_row.get('raw_confidence', regime_conf))
            smoothed_regime = str(reg_row.get('smoothed_regime', regime))
            smoothed_regime_conf = float(reg_row.get('smoothed_confidence', regime_conf))
            regime_is_stable = bool(reg_row.get('is_stable', True))
            prob_expansion = float(reg_row.get('prob_expansion', np.nan))
            prob_transition = float(reg_row.get('prob_transition', np.nan))
            prob_contraction = float(reg_row.get('prob_contraction', np.nan))

            base_exposure = float(cfg.resolved_regime_exposure(regime))

            if regime == "Contraction":
                regime_exp = base_exposure * regime_conf
            else:
                regime_exp = base_exposure
        else:
            regime = "Unknown"
            regime_conf = 1.0
            regime_exp = 1.0
            raw_regime = "Unknown"
            raw_regime_conf = 1.0
            smoothed_regime = "Unknown"
            smoothed_regime_conf = 1.0
            regime_is_stable = True
            prob_expansion = np.nan
            prob_transition = np.nan
            prob_contraction = np.nan

        # --- Check kill switch (intra-quarter) ---
        turnover = _equal_weight_turnover(prev_chosen, chosen)
        txn_cost_bps_applied = _execution_cost_bps(cfg, turnover)
        financing_cost_return, financing_rate_annual, borrowed_notional_usd = _financing_cost_return_for_period(
            cfg=cfg,
            equity_start=equity,
            regime_exp=regime_exp,
            start_dt=dt,
            end_dt=next_dt,
        )
        port_ret = apply_txn_cost(gross_ret * regime_exp, txn_cost_bps_applied) - financing_cost_return

        spy_ret = float(spy_returns_q.loc[next_dt]) if next_dt in spy_returns_q.index else 0.0

        equity *= (1 + port_ret)
        spy_eq *= (1 + spy_ret)

        # Diagnostics: realized sector ranks (top 3)
        realized = sector_returns_q.loc[next_dt].dropna().sort_values(ascending=False)
        realized_top = realized.head(cfg.top_k_sectors).index.tolist()

        # Diagnostics: top stocks of the quarter overall
        stock_q = stock_returns_q.loc[next_dt].dropna().sort_values(ascending=False)
        top_stocks = stock_q.head(20).index.tolist()

        # How many of those top stocks are in predicted sectors?
        top_in_pred = 0
        pred_set = set(pred_sector_names)
        # Map each stock to its sector
        sym_to_sector = dict(zip(current_constituents["Symbol"], current_constituents["Sector"]))
        for t in top_stocks:
            sec = sym_to_sector.get(t)
            if sec in pred_set:
                top_in_pred += 1

        results.append(QuarterResult(
            date=next_dt,
            predicted_sectors=top_sectors,
            predicted_sector_names=pred_sector_names,
            realized_sector_ranks=realized_top,
            selected_stocks=stock_picks,
            n_holdings=len(chosen),
            turnover=turnover,
            gross_return=gross_ret,
            txn_cost_bps_applied=txn_cost_bps_applied,
            financing_cost_return=financing_cost_return,
            financing_rate_annual=financing_rate_annual,
            borrowed_notional_usd=borrowed_notional_usd,
            portfolio_return=port_ret,
            spy_return=spy_ret,
            top_stocks_quarter=top_stocks,
            top_stocks_in_predicted_sectors=top_in_pred,
            regime=regime,
            regime_confidence=regime_conf,
            regime_exposure=regime_exp,
            raw_regime=raw_regime,
            raw_regime_confidence=raw_regime_conf,
            smoothed_regime=smoothed_regime,
            smoothed_regime_confidence=smoothed_regime_conf,
            regime_is_stable=regime_is_stable,
            prob_expansion=prob_expansion,
            prob_transition=prob_transition,
            prob_contraction=prob_contraction,
        ))
        prev_chosen = chosen

    # summary dataframe
    rows = [{
        "Date": r.date,
        "PortRet": r.portfolio_return,
        "SPYRet": r.spy_return,
        "GrossRet": r.gross_return,
        "Turnover": r.turnover,
        "TxnCostBps": r.txn_cost_bps_applied,
        "FinancingCostRet": r.financing_cost_return,
        "FinancingRateAnnual": r.financing_rate_annual,
        "BorrowedNotionalUSD": r.borrowed_notional_usd,
        "NHoldings": r.n_holdings,
        "Regime": r.regime,
        "RegimeConf": r.regime_confidence,
        "RegimeExp": r.regime_exposure,
        "RawRegime": r.raw_regime,
        "RawRegimeConf": r.raw_regime_confidence,
        "SmoothedRegime": r.smoothed_regime,
        "SmoothedRegimeConf": r.smoothed_regime_confidence,
        "RegimeStable": r.regime_is_stable,
        "ProbExpansion": r.prob_expansion,
        "ProbTransition": r.prob_transition,
        "ProbContraction": r.prob_contraction,
        "PredSectors": ",".join(r.predicted_sectors),
        "PredSectorNames": ",".join(r.predicted_sector_names),
        "RealTopSectors": ",".join(r.realized_sector_ranks),
        "TopStocksInPredSectors": r.top_stocks_in_predicted_sectors
    } for r in results]

    if rows:
        df = pd.DataFrame(rows).set_index("Date").sort_index()
    else:
        df = pd.DataFrame(
            columns=[
                "PortRet",
                "SPYRet",
                "GrossRet",
                "Turnover",
                "TxnCostBps",
                "FinancingCostRet",
                "FinancingRateAnnual",
                "BorrowedNotionalUSD",
                "NHoldings",
                "Regime",
                "RegimeConf",
                "RegimeExp",
                "RawRegime",
                "RawRegimeConf",
                "SmoothedRegime",
                "SmoothedRegimeConf",
                "RegimeStable",
                "ProbExpansion",
                "ProbTransition",
                "ProbContraction",
                "PredSectors",
                "PredSectorNames",
                "RealTopSectors",
                "TopStocksInPredSectors",
            ]
        )
        df.index.name = "Date"
        return df, results

    df["PortEq"] = (1 + df["PortRet"]).cumprod()
    df["SPYEq"]  = (1 + df["SPYRet"]).cumprod()
    df["AlphaEq"] = df["PortEq"] - df["SPYEq"]
    return df, results


def run_backtest_pipeline(
    output_dir: str | Path | None = None,
    cfg_override=None,
) -> tuple[pd.DataFrame, list[QuarterResult]]:
    try:
        from .config import Config
        from .features import sector_features, make_sector_training_set
        from .paths import ensure_layout, input_file, package_root, reports_dir
        from .prices import (
            download_monthly_adjclose,
            load_monthly_prices_csv,
            monthly_returns_from_prices,
            quarter_returns_from_monthly,
        )
        from .universe import load_sp500_constituents, load_constituents_history
        from .data_checks import run_data_checks
        from .validation import run_bias_diagnostics
        from .reliability import evaluate_reliability
        from .period_backtests import run_period_backtests
        from .train_test_report import run_train_test_split_report
        from .rolling_walkforward_report import run_rolling_walkforward_report
        from .bootstrap_report import run_bootstrap_significance_report
        from .universe_realism_report import run_universe_realism_report
        from .parameter_freeze import run_parameter_freeze_check
        from .delisting_returns import load_delisting_returns, apply_delisting_returns_to_quarterly
    except ImportError:  # pragma: no cover - supports direct script execution
        from config import Config
        from features import sector_features, make_sector_training_set
        from paths import ensure_layout, input_file, package_root, reports_dir
        from prices import (
            download_monthly_adjclose,
            load_monthly_prices_csv,
            monthly_returns_from_prices,
            quarter_returns_from_monthly,
        )
        from universe import load_sp500_constituents, load_constituents_history
        from data_checks import run_data_checks
        from validation import run_bias_diagnostics
        from reliability import evaluate_reliability
        from period_backtests import run_period_backtests
        from train_test_report import run_train_test_split_report
        from rolling_walkforward_report import run_rolling_walkforward_report
        from bootstrap_report import run_bootstrap_significance_report
        from universe_realism_report import run_universe_realism_report
        from parameter_freeze import run_parameter_freeze_check
        from delisting_returns import load_delisting_returns, apply_delisting_returns_to_quarterly

    cfg = cfg_override if cfg_override is not None else Config()
    base_dir = package_root()
    ensure_layout()
    out_dir = Path(output_dir) if output_dir is not None else reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    pf_csv = out_dir / "parameter_freeze_results.csv"
    pf_txt = out_dir / "parameter_freeze_report.txt"
    if cfg.enable_parameter_freeze_report:
        snapshot_path = cfg.resolved_snapshot_path(base_dir)
        pf_df, pf_text, pf_matched = run_parameter_freeze_check(
            cfg=cfg,
            snapshot_path=snapshot_path,
            auto_write_if_missing=cfg.auto_write_frozen_config_if_missing,
        )
        pf_df.to_csv(pf_csv, index=False)
        pf_txt.write_text(pf_text, encoding="utf-8")
        if cfg.enforce_frozen_config and not pf_matched:
            raise RuntimeError(
                "Parameter freeze gate blocked backtest: config differs from frozen snapshot. "
                f"See {pf_txt}."
            )

    constituents = load_sp500_constituents(str(input_file("sp500_constituents.csv")))
    history_path = input_file("constituents_history_template.csv")
    constituents_history = None
    if history_path.exists():
        try:
            constituents_history = load_constituents_history(str(history_path))
        except Exception:
            constituents_history = None
    local_prices_path = input_file("stock_prices.csv")
    benchmark_note = "Benchmark source: downloaded SPY."
    if constituents_history is not None and not constituents_history.empty:
        universe_note = "Universe source: point-in-time filter enabled using DateAdded plus Added/Removed history events."
    elif "DateAdded" in constituents.columns and constituents["DateAdded"].notna().any():
        universe_note = "Universe source: point-in-time filter enabled using DateAdded from constituents CSV."
    else:
        universe_note = "Universe source: static constituents (higher survivorship bias risk)."

    if local_prices_path.exists():
        dq_df, dq_text = run_data_checks(
            prices_path=local_prices_path,
            constituents=constituents,
            history=constituents_history,
            spy_ticker=cfg.spy_ticker,
            coverage_pass_threshold=cfg.coverage_pass_threshold,
            outlier_warn_threshold=cfg.outlier_warn_threshold,
            outlier_count_warn_max=cfg.outlier_count_warn_max,
        )
        dq_csv = out_dir / "data_checks_results.csv"
        dq_txt = out_dir / "data_checks_report.txt"
        dq_df.to_csv(dq_csv, index=False)
        dq_txt.write_text(dq_text, encoding="utf-8")

        fail_count = int((dq_df["status"] == "FAIL").sum())
        warn_count = int((dq_df["status"] == "WARN").sum())
        warn_checks = set(dq_df.loc[dq_df["status"] == "WARN", "check"].tolist())
        forced_warn_hits = sorted(warn_checks.intersection(set(cfg.data_checks_warn_fail_list)))

        block_reasons = []
        if cfg.data_checks_fail_on_fail and fail_count > 0:
            block_reasons.append(f"{fail_count} FAIL checks")
        if cfg.data_checks_max_warn is not None and warn_count > cfg.data_checks_max_warn:
            block_reasons.append(f"WARN count {warn_count} exceeds max {cfg.data_checks_max_warn}")
        if forced_warn_hits:
            block_reasons.append(f"warn checks treated as fatal triggered: {', '.join(forced_warn_hits)}")

        if block_reasons:
            raise RuntimeError(
                "Data quality gate blocked backtest: "
                + "; ".join(block_reasons)
                + f". See {dq_csv} and {dq_txt}."
            )
    else:
        dq_csv = out_dir / "data_checks_results.csv"
        dq_txt = out_dir / "data_checks_report.txt"
        pd.DataFrame(
            [{"check": "prices_file_exists", "status": "WARN", "detail": "No local stock_prices.csv; skipped local data checks."}]
        ).to_csv(dq_csv, index=False)
        dq_txt.write_text("Data checks skipped: no local stock_prices.csv found.", encoding="utf-8")

    if local_prices_path.exists():
        stock_px_m = load_monthly_prices_csv(local_prices_path)
        stock_px_m = stock_px_m.loc[cfg.start_date:cfg.end_date]
        stock_px_m = stock_px_m.loc[:, ~stock_px_m.columns.duplicated()].copy()
        has_local_spy = cfg.spy_ticker in stock_px_m.columns

        if has_local_spy:
            spy_px_m = stock_px_m[[cfg.spy_ticker]].copy()
            stock_px_m = stock_px_m.drop(columns=[cfg.spy_ticker])
            benchmark_note = "Benchmark source: SPY column from local stock_prices.csv."
        else:
            benchmark_note = "Benchmark source: synthetic market benchmark from equal-weight stock universe (SPY not found in CSV)."

        stock_rets_m = monthly_returns_from_prices(stock_px_m)
        sym_sector = constituents[["Symbol", "Sector"]].drop_duplicates()
        by_sector = sym_sector.groupby("Sector")["Symbol"].apply(list).to_dict()

        synthetic_sector_rets = {}
        for sector_name, symbols in by_sector.items():
            members = [s for s in symbols if s in stock_rets_m.columns]
            if not members:
                continue
            sec_rets = stock_rets_m[members].mean(axis=1, skipna=True)
            coverage = stock_rets_m[members].notna().sum(axis=1)
            sec_rets = sec_rets.where(coverage >= 3)
            synthetic_sector_rets[sector_name] = sec_rets

        sector_rets_m = pd.DataFrame(synthetic_sector_rets).dropna(how="all")
        if has_local_spy:
            spy_rets_m = monthly_returns_from_prices(spy_px_m)[cfg.spy_ticker]
        else:
            spy_rets_m = stock_rets_m.mean(axis=1, skipna=True).rename(cfg.spy_ticker)
            spy_px_m = (1 + spy_rets_m.fillna(0.0)).cumprod().to_frame(cfg.spy_ticker)

        common_idx = sector_rets_m.index.intersection(spy_rets_m.index)
        sector_rets_m = sector_rets_m.loc[common_idx].sort_index()
        spy_rets_m = spy_rets_m.loc[common_idx].sort_index()
        sector_px_m = (1 + sector_rets_m.fillna(0.0)).cumprod()
        spy_px_m = spy_px_m.loc[common_idx].sort_index()
    else:
        sector_px_m = download_monthly_adjclose(cfg.sector_tickers, cfg.start_date, cfg.end_date)
        spy_px_m = download_monthly_adjclose(cfg.spy_ticker, cfg.start_date, cfg.end_date)

        sector_rets_m = monthly_returns_from_prices(sector_px_m)
        spy_rets_m = monthly_returns_from_prices(spy_px_m)[cfg.spy_ticker]

        stock_list = sorted(constituents["Symbol"].unique().tolist())
        stock_px_m = download_monthly_adjclose(stock_list, cfg.start_date, cfg.end_date)
        stock_px_m = stock_px_m.loc[cfg.start_date:cfg.end_date]

    sector_rets_q = quarter_returns_from_monthly(sector_px_m)
    spy_rets_q = quarter_returns_from_monthly(spy_px_m)[cfg.spy_ticker]

    X_m = sector_features(sector_rets_m, spy_rets_m)
    X_q, Y_q_next = make_sector_training_set(X_m, sector_rets_q)

    stock_rets_q = quarter_returns_from_monthly(stock_px_m)

    delist_csv = out_dir / "delisting_integration_results.csv"
    delist_txt = out_dir / "delisting_integration_report.txt"
    delist_df = pd.DataFrame(columns=["Date", "Ticker", "DelistReturn"])
    delist_applied_count = 0
    if cfg.enable_delisting_returns_integration:
        delist_path = base_dir / cfg.delisting_returns_path if not Path(cfg.delisting_returns_path).is_absolute() else Path(cfg.delisting_returns_path)
        delist_df = load_delisting_returns(delist_path)
        stock_rets_q, delist_audit_df, delist_text = apply_delisting_returns_to_quarterly(stock_rets_q, delist_df)
        delist_audit_df.to_csv(delist_csv, index=False)
        delist_txt.write_text(delist_text, encoding="utf-8")
        delist_applied_count = int((delist_audit_df["status"] == "APPLIED").sum()) if not delist_audit_df.empty else 0
        if cfg.enforce_delisting_data_gate and delist_applied_count < int(cfg.min_applied_delisting_events):
            raise RuntimeError(
                "Delisting data gate blocked backtest: "
                f"applied_events={delist_applied_count} < min_required={int(cfg.min_applied_delisting_events)}. "
                f"See {delist_csv} and provide {delist_path}."
            )

    ur_csv = out_dir / "universe_realism_results.csv"
    ur_txt = out_dir / "universe_realism_report.txt"
    if cfg.enable_universe_realism_report:
        ur_df, ur_text = run_universe_realism_report(
            constituents=constituents,
            history=constituents_history,
            stock_prices_m=stock_px_m,
            delist_df=delist_df if cfg.enable_delisting_returns_integration else None,
            min_removed_events_for_realism=cfg.min_removed_events_for_realism,
        )
        ur_df.to_csv(ur_csv, index=False)
        ur_txt.write_text(ur_text, encoding="utf-8")
        ur_warn = int((ur_df["status"] == "WARN").sum())
        if cfg.enforce_universe_realism_gate and cfg.max_universe_realism_warn is not None and ur_warn > cfg.max_universe_realism_warn:
            raise RuntimeError(
                "Universe realism gate blocked backtest: "
                f"WARN={ur_warn} exceeds max {cfg.max_universe_realism_warn}. See {ur_txt}."
            )

    bt_df, bt_details = run_walkforward_backtest(
        cfg=cfg,
        sector_features_q=X_q,
        sector_labels_q=Y_q_next,
        sector_returns_q=sector_rets_q,
        spy_returns_q=spy_rets_q,
        constituents=constituents,
        stock_prices_m=stock_px_m,
        stock_returns_q=stock_rets_q,
        stock_dollarvol_m=None,
        sector_etf_to_gics=None,
        constituents_history=constituents_history,
    )

    if _GLOBAL_REGIME_DF_CACHE is not None and not _GLOBAL_REGIME_DF_CACHE.empty:
        regime_diag_path = out_dir / "regime_classification_diagnostics.csv"
        _GLOBAL_REGIME_DF_CACHE.reset_index().rename(columns={"date": "Date"}).to_csv(regime_diag_path, index=False)

    # ── Kill switch overlay ────────────────────────────────────────────────────
    # Pull HY OAS from FRED and apply kill switch exposure overlay.
    # Produces bt_df["PortRetKS"] — portfolio return scaled to 0.30 when active.
    # bt_df["PortRet"] is left unchanged (pre-kill-switch baseline is preserved).
    try:
        from macro_data import _get_fred_client  # noqa: PLC0415
        _fred_ks  = _get_fred_client()
        _hy_raw   = _fred_ks.get_series(
            getattr(cfg, "kill_switch_fred_series", "BAMLH0A0HYM2")
        )
        _ks_hist  = build_kill_switch_history(
            hy_oas_series   = _hy_raw,
            trigger_zscore  = float(getattr(cfg, "kill_switch_trigger_zscore", 3.0)),
            deact_zscore    = float(getattr(cfg, "kill_switch_deact_zscore",   2.0)),
            window          = int(getattr(cfg,   "kill_switch_window",          5)),
            min_obs         = int(getattr(cfg,   "kill_switch_min_obs",         252)),
            contraction_exp = float(getattr(cfg, "regime_exposure_contraction", 0.30)),
        )
        bt_df = apply_kill_switch_to_returns(
            returns_df      = bt_df,
            ks_history      = _ks_hist,
            return_col      = "PortRet",
            contraction_exp = float(getattr(cfg, "regime_exposure_contraction", 0.30)),
        )
        ks_csv = out_dir / "kill_switch_history.csv"
        _ks_hist.to_csv(ks_csv)
        print(f"[kill_switch] Overlay applied. Active quarters: "
              f"{int(bt_df['ks_active'].sum())}/{len(bt_df)}. "
              f"History saved: {ks_csv}")
    except Exception as _ks_err:
        _ks_hist = _load_cached_kill_switch_history()
        if not _ks_hist.empty:
            bt_df = apply_kill_switch_to_returns(
                returns_df=bt_df,
                ks_history=_ks_hist,
                return_col="PortRet",
                contraction_exp=float(getattr(cfg, "regime_exposure_contraction", 0.30)),
            )
            print(
                f"[kill_switch] WARNING: live HY OAS refresh failed ({_ks_err}). "
                f"Using cached history from {_ks_hist.index.min().date()} to {_ks_hist.index.max().date()}."
            )
        else:
            print(f"[kill_switch] WARNING: overlay skipped — {_ks_err}")
    # ── End kill switch overlay ────────────────────────────────────────────────

    bt_path = out_dir / "backtest_results.csv"
    bt_df.to_csv(bt_path)

    if not bt_df.empty and {"GrossRet", "Turnover", "TxnCostBps"}.issubset(bt_df.columns):
        turn_csv = out_dir / "turnover_results.csv"
        turn_txt = out_dir / "turnover_report.txt"
        turn_df = bt_df[[
            "GrossRet",
            "PortRet",
            "SPYRet",
            "Turnover",
            "TxnCostBps",
            "FinancingCostRet",
            "FinancingRateAnnual",
            "BorrowedNotionalUSD",
            "NHoldings",
        ]].copy()
        turn_df.to_csv(turn_csv)

        avg_turn = float(turn_df["Turnover"].mean())
        med_turn = float(turn_df["Turnover"].median())
        avg_cost_bps = float(turn_df["TxnCostBps"].mean())
        avg_financing_bps = float((turn_df["FinancingCostRet"].mean()) * 1e4)
        avg_financing_rate = float(turn_df["FinancingRateAnnual"].replace(0.0, np.nan).mean())
        avg_borrowed_usd = float(turn_df["BorrowedNotionalUSD"].mean())
        avg_n = float(turn_df["NHoldings"].mean())
        resolved_exposures = cfg.resolved_regime_exposures()
        one_way_bps = (
            float(cfg.commission_bps_oneway)
            + float(cfg.exchange_fee_bps_oneway)
            + float(cfg.spread_bps_oneway)
            + float(cfg.slippage_bps_oneway)
            + float(getattr(cfg, "impact_bps_oneway", 0.0))
        )
        turn_txt.write_text(
            "\n".join(
                [
                    "Turnover / Execution Cost Summary",
                    f"Exposure profile: {getattr(cfg, 'regime_exposure_profile', 'custom')}",
                    (
                        "Resolved regime exposures: "
                        f"Expansion={resolved_exposures['Expansion']:.2f}, "
                        f"Transition={resolved_exposures['Transition']:.2f}, "
                        f"Contraction={resolved_exposures['Contraction']:.2f}"
                    ),
                    f"Average holdings per quarter: {avg_n:.2f}",
                    f"Average turnover (one-way): {avg_turn:.2%}",
                    f"Median turnover (one-way): {med_turn:.2%}",
                    f"Average applied execution cost: {avg_cost_bps:.2f} bps per quarter",
                    f"Average financing drag: {avg_financing_bps:.2f} bps per quarter",
                    f"Average annualized financing rate on borrowed balance: {avg_financing_rate:.2%}" if pd.notna(avg_financing_rate) else "Average annualized financing rate on borrowed balance: n/a",
                    f"Average borrowed notional (USD, using configured account NAV): ${avg_borrowed_usd:,.0f}",
                    (
                        "IBKR-style one-way cost assumption (bps): "
                        f"commission={cfg.commission_bps_oneway:.2f} + "
                        f"exchange={cfg.exchange_fee_bps_oneway:.2f} + "
                        f"spread={cfg.spread_bps_oneway:.2f} + "
                        f"slippage={cfg.slippage_bps_oneway:.2f} + "
                        f"impact={getattr(cfg, 'impact_bps_oneway', 0.0):.2f} = {one_way_bps:.2f}"
                    ),
                    (
                        "IBKR Pro USD financing assumption: blended benchmark-plus-spread tiers "
                        "for 0-100k, 100k-1m, 1m-50m, 50m-250m, and 250m+ borrowed balances."
                    ),
                    (
                        "Benchmark proxy used in backtest: "
                        f"{float(getattr(cfg, 'ibkr_margin_benchmark_rate', 0.0)):.2%} annual."
                    ),
                    "Note: fixed per-order minimum ticket costs are approximated in commission bps.",
                ]
            ),
            encoding="utf-8",
        )

        # Execution-cost sensitivity using realized turnover path and gross returns.
        scen_rows = []
        for spec in cfg.execution_cost_scenarios:
            try:
                name, bps_txt = spec.split(":")
                bps = float(bps_txt)
            except Exception:
                continue
            net = turn_df["GrossRet"] - (turn_df["Turnover"] * (bps / 1e4))
            eq = (1 + net).cumprod()
            total = float(eq.iloc[-1] - 1.0) if len(eq) else np.nan
            ann_ret = cagr(net, periods=4)
            ann_vol = annualized_volatility(net, periods=4)
            sharpe = sharpe_ratio(net, periods=4)
            scen_rows.append(
                {
                    "Scenario": name,
                    "OneWayCostBps": bps,
                    "TotalReturn": total,
                    "AnnRet": ann_ret,
                    "AnnVol": ann_vol,
                    "Sharpe": sharpe,
                }
            )
        if scen_rows:
            pd.DataFrame(scen_rows).to_csv(out_dir / "execution_cost_scenarios.csv", index=False)

    diagnostics_df, diagnostics_text = run_bias_diagnostics(
        cfg=cfg,
        strategy_df=bt_df,
        sector_features_q=X_q,
        sector_labels_q=Y_q_next,
        sector_returns_q=sector_rets_q,
        spy_returns_q=spy_rets_q,
        constituents=constituents,
        stock_prices_m=stock_px_m,
        stock_returns_q=stock_rets_q,
        benchmark_note=benchmark_note,
        universe_note=universe_note,
        constituents_history=constituents_history,
    )
    diagnostics_path = out_dir / "diagnostics_results.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)
    diagnostics_txt_path = out_dir / "diagnostics_report.txt"
    diagnostics_txt_path.write_text(diagnostics_text, encoding="utf-8")

    reliability_score, reliability_text = evaluate_reliability(
        checks_df=dq_df if local_prices_path.exists() else pd.DataFrame(),
        diagnostics_df=diagnostics_df,
        constituents_history=constituents_history,
        benchmark_note=benchmark_note,
        universe_note=universe_note,
        min_target=cfg.min_reliability_score,
        stale_feature_sharpe_ratio_max=cfg.stale_feature_sharpe_ratio_max,
        shuffled_label_sharpe_ratio_max=cfg.shuffled_label_sharpe_ratio_max,
        min_history_events_for_trust=cfg.min_history_events_for_trust,
    )
    reliability_txt_path = out_dir / "reliability_report.txt"
    reliability_txt_path.write_text(reliability_text, encoding="utf-8")

    period_csv = out_dir / "period_backtests_results.csv"
    period_txt = out_dir / "period_backtests_report.txt"
    if cfg.enable_period_backtests:
        period_df, period_text = run_period_backtests(bt_df, cfg.period_windows)
        period_df.to_csv(period_csv, index=False)
        period_txt.write_text(period_text, encoding="utf-8")

    tt_csv = out_dir / "train_test_split_results.csv"
    tt_txt = out_dir / "train_test_split_report.txt"
    if cfg.enable_train_test_split_report:
        tt_df, tt_text = run_train_test_split_report(
            backtest_df=bt_df,
            split_date=cfg.train_test_split_date,
            min_oos_quarters=cfg.min_oos_quarters,
        )
        tt_df.to_csv(tt_csv, index=False)
        tt_txt.write_text(tt_text, encoding="utf-8")

    rw_csv = out_dir / "rolling_walkforward_results.csv"
    rw_txt = out_dir / "rolling_walkforward_report.txt"
    if cfg.enable_rolling_walkforward_report:
        rw_df, rw_text = run_rolling_walkforward_report(
            backtest_df=bt_df,
            train_quarters=cfg.rolling_train_quarters,
            test_quarters=cfg.rolling_test_quarters,
            step_quarters=cfg.rolling_step_quarters,
        )
        rw_df.to_csv(rw_csv, index=False)
        rw_txt.write_text(rw_text, encoding="utf-8")

    bs_csv = out_dir / "bootstrap_significance_results.csv"
    bs_txt = out_dir / "bootstrap_significance_report.txt"
    if cfg.enable_bootstrap_report:
        bs_df, bs_text = run_bootstrap_significance_report(
            backtest_df=bt_df,
            n_boot=cfg.bootstrap_iterations,
            seed=cfg.bootstrap_seed,
        )
        bs_df.to_csv(bs_csv, index=False)
        bs_txt.write_text(bs_text, encoding="utf-8")

    if cfg.enforce_reliability_gate and reliability_score < cfg.min_reliability_score:
        raise RuntimeError(
            f"Reliability gate blocked backtest: score={reliability_score:.1f} < target={cfg.min_reliability_score:.1f}. "
            f"See {reliability_txt_path}."
        )

    qlog_path = out_dir / "quarter_log.txt"
    with qlog_path.open("w", encoding="utf-8") as f:
        for r in bt_details:
            f.write(f"\n=== {r.date.date()} ===\n")
            f.write(f"Predicted sectors (ETFs): {r.predicted_sectors}\n")
            f.write(f"Predicted sector names   : {r.predicted_sector_names}\n")
            f.write(f"Real top sectors         : {r.realized_sector_ranks}\n")
            f.write(f"Portfolio return         : {r.portfolio_return:.2%} | SPY {r.spy_return:.2%}\n")
            f.write(
                f"Gross return / execution / financing: "
                f"{r.gross_return:.2%} / {r.txn_cost_bps_applied:.2f} bps / {r.financing_cost_return:.2%}\n"
            )
            f.write(
                f"Borrowed notional / annual rate: "
                f"${r.borrowed_notional_usd:,.0f} / {r.financing_rate_annual:.2%}\n"
            )
            f.write(f"Holdings / turnover      : {r.n_holdings} / {r.turnover:.2%}\n")
            f.write(f"Top20 stocks in pred sectors: {r.top_stocks_in_predicted_sectors}/20\n")
            f.write("Stock picks:\n")
            for sec, picks in r.selected_stocks.items():
                f.write(f"  - {sec}: {picks}\n")

    print(f"\nSaved: {bt_path}, {qlog_path}, {diagnostics_path}, {diagnostics_txt_path}, {dq_csv}, {dq_txt}, {reliability_txt_path}")
    return bt_df, bt_details


if __name__ == "__main__":
    run_backtest_pipeline()
