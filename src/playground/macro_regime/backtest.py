import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

try:
    from .sector_model import SectorReturnModel
    from .stock_model import select_stocks_within_sectors
    from .universe import constituents_asof, apply_history_events
except ImportError:  # pragma: no cover - supports direct script execution
    from sector_model import SectorReturnModel
    from stock_model import select_stocks_within_sectors
    from universe import constituents_asof, apply_history_events

@dataclass
class QuarterResult:
    date: pd.Timestamp
    predicted_sectors: list[str]          # ETF tickers
    predicted_sector_names: list[str]     # GICS names
    realized_sector_ranks: list[str]      # top realized ETF tickers in that quarter
    selected_stocks: dict[str, list[str]] # sector_name -> stocks
    portfolio_return: float
    spy_return: float
    top_stocks_quarter: list[str]         # top performing stocks overall
    top_stocks_in_predicted_sectors: int  # count

def apply_txn_cost(ret: float, bps: float) -> float:
    return ret - (bps / 1e4)

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

    # Start after we have enough training history
    min_train = max(20, cfg.lookback_months_sector // 3)
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
        port_ret = build_equal_weight_portfolio_return(
            stock_returns_q,
            chosen,
            next_dt,
            max_abs_stock_quarter_return=cfg.max_abs_stock_quarter_return,
        )
        port_ret = apply_txn_cost(port_ret, cfg.txn_cost_bps)

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
            portfolio_return=port_ret,
            spy_return=spy_ret,
            top_stocks_quarter=top_stocks,
            top_stocks_in_predicted_sectors=top_in_pred
        ))

    # summary dataframe
    rows = [{
        "Date": r.date,
        "PortRet": r.portfolio_return,
        "SPYRet": r.spy_return,
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


def run_backtest_pipeline(output_dir: str | Path | None = None) -> tuple[pd.DataFrame, list[QuarterResult]]:
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

    cfg = Config()
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

    bt_path = out_dir / "backtest_results.csv"
    bt_df.to_csv(bt_path)

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
            f.write(f"Top20 stocks in pred sectors: {r.top_stocks_in_predicted_sectors}/20\n")
            f.write("Stock picks:\n")
            for sec, picks in r.selected_stocks.items():
                f.write(f"  - {sec}: {picks}\n")

    print(f"\nSaved: {bt_path}, {qlog_path}, {diagnostics_path}, {diagnostics_txt_path}, {dq_csv}, {dq_txt}, {reliability_txt_path}")
    return bt_df, bt_details


if __name__ == "__main__":
    run_backtest_pipeline()
