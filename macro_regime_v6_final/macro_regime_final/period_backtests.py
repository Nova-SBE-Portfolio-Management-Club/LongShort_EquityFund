from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .metrics import annualized_downside_volatility, annualized_volatility, cagr, sharpe_ratio, sortino_ratio, total_return
except ImportError:  # pragma: no cover - supports direct script execution
    from metrics import annualized_downside_volatility, annualized_volatility, cagr, sharpe_ratio, sortino_ratio, total_return


def _perf(series: pd.Series, freq: int = 4) -> dict:
    if series.empty:
        return {"TotalReturn": np.nan, "CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "MaxDD": np.nan}
    total = total_return(series)
    growth = cagr(series, periods=freq)
    vol = annualized_volatility(series, periods=freq)
    dvol = annualized_downside_volatility(series, periods=freq)
    sharpe = sharpe_ratio(series, periods=freq)
    sortino = sortino_ratio(series, periods=freq)
    eq = (1 + series).cumprod()
    dd = (eq - eq.cummax()) / (eq.cummax() + 1e-12)
    mdd = float(dd.min())
    return {"TotalReturn": total, "CAGR": growth, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": mdd}


def run_period_backtests(backtest_df: pd.DataFrame, period_windows: list[str]) -> tuple[pd.DataFrame, str]:
    rows = []
    for spec in period_windows:
        try:
            start, end, label = spec.split(":")
        except ValueError:
            continue
        sub = backtest_df.loc[start:end].copy()
        if sub.empty:
            rows.append({"Period": label, "Start": start, "End": end, "NQuarters": 0})
            continue
        p = _perf(sub["PortRet"])
        s = _perf(sub["SPYRet"])
        rows.append(
            {
                "Period": label,
                "Start": str(sub.index.min().date()),
                "End": str(sub.index.max().date()),
                "NQuarters": int(len(sub)),
                "Port_TotalReturn": p["TotalReturn"],
                "SPY_TotalReturn": s["TotalReturn"],
                "Port_CAGR": p["CAGR"],
                "SPY_CAGR": s["CAGR"],
                "Port_Sharpe": p["Sharpe"],
                "SPY_Sharpe": s["Sharpe"],
                "Port_Sortino": p["Sortino"],
                "SPY_Sortino": s["Sortino"],
                "Port_MaxDD": p["MaxDD"],
                "SPY_MaxDD": s["MaxDD"],
                "AvgAlphaQ": float((sub["PortRet"] - sub["SPYRet"]).mean()),
                "HitRate": float((sub["PortRet"] > sub["SPYRet"]).mean()),
            }
        )

    df = pd.DataFrame(rows)
    lines = ["Period Backtest Summary"]
    if df.empty:
        lines.append("No valid period windows.")
    else:
        lines.append(f"Generated {len(df)} period slices.")
    return df, "\n".join(lines)
