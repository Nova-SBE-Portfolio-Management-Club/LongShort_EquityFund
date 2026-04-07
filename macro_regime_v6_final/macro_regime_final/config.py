from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class Config:
    start_date: str = "1999-12-31"
    end_date: str = "2026-02-13"

    # Sector ETFs (Select Sector SPDR)
    sector_tickers: List[str] = field(default_factory=lambda: [
        "XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"
    ])

    spy_ticker: str = "SPY"

    # Backtest params
    rebalance_freq: str = "Q"     # quarterly
    lookback_months_sector: int = 36   # for training features
    lookback_months_stock: int = 13    # to compute 12-1 momentum
    top_k_sectors: int = 3
    top_n_stocks_per_sector: int = 10

    # Simple costs (round trip per rebalance)
    txn_cost_bps: float = 10.0

    # Turnover-aware execution cost model (more realistic than flat bps)
    use_turnover_cost_model: bool = True
    commission_bps_oneway: float = 0.4
    exchange_fee_bps_oneway: float = 0.2
    spread_bps_oneway: float = 8.0
    slippage_bps_oneway: float = 10.0
    impact_bps_oneway: float = 15.0
    execution_cost_scenarios: List[str] = field(default_factory=lambda: [
        "Base:33.6",
        "Conservative:57.8",
        "Stressed:106.0",
    ])

    # Universe / liquidity
    min_avg_dollar_vol: float = 5_000_000  # average $ volume filter
    min_history_months_for_stock_selection: int = 24
    max_abs_stock_quarter_return: float = 0.80

    # Data-quality gate
    data_checks_fail_on_fail: bool = True
    data_checks_max_warn: int | None = None
    data_checks_warn_fail_list: List[str] = field(default_factory=list)

    # Reliability scoring gate
    min_reliability_score: float = 55.0
    enforce_reliability_gate: bool = True
    stale_feature_sharpe_ratio_max: float = 0.90
    shuffled_label_sharpe_ratio_max: float = 0.80
    min_history_events_for_trust: int = 200

    # Research-mode data thresholds (more realistic for imperfect historical feeds)
    coverage_pass_threshold: float = 0.88
    outlier_warn_threshold: float = 0.80
    outlier_count_warn_max: int = 80

    # Subperiod backtests (institutional robustness slices)
    enable_period_backtests: bool = True
    period_windows: List[str] = field(default_factory=lambda: [
        "2005-09-30:2025-12-31:Full",
        "2005-09-30:2009-12-31:GFC_Regime",
        "2010-01-01:2014-12-31:PostGFC_Early",
        "2015-01-01:2019-12-31:LateCycle_PreCovid",
        "2020-01-01:2022-12-31:Covid_And_Shock",
        "2023-01-01:2025-12-31:Recent_Regime",
    ])

    # Strict train/test holdout report
    enable_train_test_split_report: bool = True
    train_test_split_date: str = "2015-12-31"
    min_oos_quarters: int = 24

    # Rolling walk-forward decay report
    enable_rolling_walkforward_report: bool = True
    rolling_train_quarters: int = 32
    rolling_test_quarters: int = 8
    rolling_step_quarters: int = 4

    # Bootstrap significance report
    enable_bootstrap_report: bool = True
    bootstrap_iterations: int = 5000
    bootstrap_seed: int = 42

    # Universe realism / survivorship diagnostics
    enable_universe_realism_report: bool = True
    min_removed_events_for_realism: int = 200
    enforce_universe_realism_gate: bool = True
    max_universe_realism_warn: int | None = 1

    # Parameter freeze governance (to avoid re-tuning on final sample)
    enable_parameter_freeze_report: bool = True
    parameter_snapshot_path: str = "data/frozen_config.json"
    enforce_frozen_config: bool = False
    auto_write_frozen_config_if_missing: bool = False

    # Delisting return integration
    enable_delisting_returns_integration: bool = True
    delisting_returns_path: str = "data/delisting_returns.csv"
    enforce_delisting_data_gate: bool = False
    min_applied_delisting_events: int = 50

    # ── Regime engine (HMM) ──────────────────────────────────────────────────
    # ALL parameters below are frozen from first principles.
    # Do NOT tune these to improve backtest output — that is overfitting.

    # HMM structure
    regime_n_states        : int   = 3       # Expansion / Transition / Contraction
    regime_covariance_type : str   = "full"  # Full Gaussian per state
    regime_n_iter          : int   = 100     # Baum-Welch iterations
    regime_random_state    : int   = 42      # Reproducibility
    regime_confirm_window  : int   = 5       # Days regime must be stable
    regime_min_train_days  : int   = 504     # ~2 years before first training

    # Exposure multipliers — logic-derived, NOT optimized
    # Profiles:
    # - "levered": mild leverage in Expansion, full risk in Transition
    # - "unlevered": no leverage in Expansion, reduced risk in Transition
    # - "custom": use the explicit regime_exposure_* values below
    regime_exposure_profile    : str   = "levered"
    regime_exposure_expansion  : float = 1.20   # Used directly when profile == "custom"
    regime_exposure_transition : float = 1.00   # Used directly when profile == "custom"
    regime_exposure_contraction: float = 0.50   # Shared default for Contraction
    regime_exposure_expansion_unlevered  : float = 1.00
    regime_exposure_transition_unlevered : float = 0.80

    # IBKR-style leverage financing model
    # Methodology follows IBKR Pro USD margin loans: benchmark + tiered spread, blended by balance.
    # The benchmark itself is configurable because the repo does not yet carry a full historical IBKR
    # benchmark series; the default below is the current USD benchmark implied by IBKR's published table.
    use_ibkr_margin_financing      : bool = True
    ibkr_margin_account_nav_usd    : float = 1_000_000.0
    ibkr_margin_benchmark_rate     : float = 0.0364
    ibkr_margin_floor_benchmark_at_zero: bool = True
    ibkr_margin_day_count          : int = 360

    # Period boundaries
    regime_burn_in_end   : str = "2006-12-31"
    regime_insample_end  : str = "2018-12-31"
    regime_oos_start     : str = "2019-01-01"
    regime_oos_end       : str = "2022-12-31"
    regime_robust_start  : str = "2023-01-01"

    # ── Kill switch (intra-quarter de-risk) ──────────────────────────────────
    # ALL parameters below are frozen from first principles.
    # Do NOT tune these to improve backtest output — that is overfitting.

    # Trigger: HY OAS expanding z-score >= this in any day of the rolling window
    kill_switch_trigger_zscore : float = 3.0

    # Deactivation: z-score must stay BELOW this for `kill_switch_window` consecutive days.
    # Lower than trigger intentionally — hysteresis prevents whipsawing.
    kill_switch_deact_zscore   : float = 2.0

    # Rolling window length for both trigger detection and deactivation confirmation
    kill_switch_window         : int   = 5

    # Minimum z-score observations required before kill switch can ever fire
    kill_switch_min_obs        : int   = 252

    # FRED series for HY OAS — public ICE BofA US High Yield Index OAS
    kill_switch_fred_series    : str   = "BAMLH0A0HYM2"

    def resolved_snapshot_path(self, base_dir: Path) -> Path:
        p = Path(self.parameter_snapshot_path)
        if p.is_absolute():
            return p
        return base_dir / p

    def resolved_regime_exposures(self) -> dict[str, float]:
        profile = str(getattr(self, "regime_exposure_profile", "levered")).strip().lower()
        if profile == "unlevered":
            return {
                "Expansion": float(self.regime_exposure_expansion_unlevered),
                "Transition": float(self.regime_exposure_transition_unlevered),
                "Contraction": float(self.regime_exposure_contraction),
            }
        if profile == "levered":
            return {
                "Expansion": float(self.regime_exposure_expansion),
                "Transition": float(self.regime_exposure_transition),
                "Contraction": float(self.regime_exposure_contraction),
            }
        if profile == "custom":
            return {
                "Expansion": float(self.regime_exposure_expansion),
                "Transition": float(self.regime_exposure_transition),
                "Contraction": float(self.regime_exposure_contraction),
            }
        raise ValueError(
            "regime_exposure_profile must be one of: levered, unlevered, custom"
        )

    def resolved_regime_exposure(self, regime: str) -> float:
        exposures = self.resolved_regime_exposures()
        return float(exposures.get(regime, 1.0))
