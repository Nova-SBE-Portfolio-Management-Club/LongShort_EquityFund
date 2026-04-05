from __future__ import annotations

import pandas as pd


def evaluate_reliability(
    checks_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    constituents_history: pd.DataFrame | None,
    benchmark_note: str,
    universe_note: str,
    min_target: float = 85.0,
    stale_feature_sharpe_ratio_max: float = 0.80,
    shuffled_label_sharpe_ratio_max: float = 0.70,
    min_history_events_for_trust: int = 500,
) -> tuple[float, str]:
    score = 100.0
    reasons: list[str] = []

    if checks_df is not None and not checks_df.empty:
        fail_count = int((checks_df["status"] == "FAIL").sum())
        warn_count = int((checks_df["status"] == "WARN").sum())
        score -= 30.0 * fail_count
        score -= 6.0 * warn_count
        if fail_count:
            reasons.append(f"Data checks FAIL count: {fail_count}")
        if warn_count:
            reasons.append(f"Data checks WARN count: {warn_count}")

        warn_set = set(checks_df.loc[checks_df["status"] == "WARN", "check"].tolist())
        if "overall_data_coverage" in warn_set:
            score -= 8.0
            reasons.append("Coverage warning present.")
        if "history_ticker_in_prices" in warn_set:
            score -= 5.0
            reasons.append("History has tickers not in prices.")
        if "return_outliers" in warn_set:
            score -= 8.0
            reasons.append("Extreme monthly return outliers present.")

    if diagnostics_df is not None and not diagnostics_df.empty:
        stale = diagnostics_df.loc[diagnostics_df["Test"] == "StaleFeatures_ShiftPlus1Q", "Sharpe"]
        main = diagnostics_df.loc[diagnostics_df["Test"] == "Strategy_Main", "Sharpe"]
        shuffled = diagnostics_df.loc[diagnostics_df["Test"] == "ShuffledLabels_Sanity", "Sharpe"]
        if len(stale) and len(main):
            stale_sh = float(stale.iloc[0])
            main_sh = float(main.iloc[0])
            if stale_sh >= main_sh * stale_feature_sharpe_ratio_max:
                score -= 20.0
                reasons.append("Stale-feature test does not degrade enough.")
            if len(shuffled):
                shuf = float(shuffled.iloc[0])
                if shuf >= main_sh * shuffled_label_sharpe_ratio_max:
                    score -= 10.0
                    reasons.append("Shuffled-label sanity still too strong.")

    if "synthetic" in (benchmark_note or "").lower():
        score -= 15.0
        reasons.append("Synthetic benchmark in use.")

    hist_rows = 0 if constituents_history is None else len(constituents_history)
    if hist_rows < min_history_events_for_trust:
        score -= 15.0
        reasons.append(f"History events sparse ({hist_rows} rows).")

    if "dateadded from constituents csv" in (universe_note or "").lower():
        score -= 10.0
        reasons.append("Universe uses DateAdded-only logic.")

    score = max(0.0, min(100.0, score))
    status = "PASS" if score >= min_target else "FAIL"

    lines = [
        "Reliability Score Report",
        f"Score: {score:.1f}/100",
        f"Target: {min_target:.1f}",
        f"Status: {status}",
    ]
    if reasons:
        lines.append("Drivers:")
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append("Drivers: none.")

    if status == "FAIL":
        lines.append("Action: do not present as production-grade strategy yet.")
    else:
        lines.append("Action: acceptable to present with disclosed assumptions.")

    return score, "\n".join(lines)
