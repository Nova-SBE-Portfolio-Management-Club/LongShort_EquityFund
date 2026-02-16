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

    # Universe / liquidity
    min_avg_dollar_vol: float = 5_000_000  # average $ volume filter
    min_history_months_for_stock_selection: int = 24
    max_abs_stock_quarter_return: float = 0.80

    # Data-quality gate
    data_checks_fail_on_fail: bool = True
    data_checks_max_warn: int | None = None
    data_checks_warn_fail_list: List[str] = field(default_factory=list)

    # Reliability scoring gate
    min_reliability_score: float = 65.0
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
    max_universe_realism_warn: int | None = 0

    # Parameter freeze governance (to avoid re-tuning on final sample)
    enable_parameter_freeze_report: bool = True
    parameter_snapshot_path: str = "data/frozen_config.json"
    enforce_frozen_config: bool = True
    auto_write_frozen_config_if_missing: bool = False

    # Delisting return integration
    enable_delisting_returns_integration: bool = True
    delisting_returns_path: str = "data/delisting_returns.csv"
    enforce_delisting_data_gate: bool = True
    min_applied_delisting_events: int = 50

    def resolved_snapshot_path(self, base_dir: Path) -> Path:
        p = Path(self.parameter_snapshot_path)
        if p.is_absolute():
            return p
        return base_dir / p
