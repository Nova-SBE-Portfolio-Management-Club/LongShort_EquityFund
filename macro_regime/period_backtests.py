from __future__ import annotations

import numpy as np
import pandas as pd


def _perf(series: pd.Series, freq: int = 4) -> dict:
    if series.empty:
        return {"TotalReturn": np.nan, "CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "MaxDD": np.nan}
    total = float((1 + series).prod() - 1)
    years = len(series) / freq
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(series.std() * np.sqrt(freq))
    down = series[series < 0]
    dvol = float(down.std() * np.sqrt(freq)) if not down.empty else 0.0
    sharpe = cagr / (vol + 1e-12)
    sortino = cagr / (dvol + 1e-12)
    eq = (1 + series).cumprod()
    dd = (eq - eq.cummax()) / (eq.cummax() + 1e-12)
    mdd = float(dd.min())
    return {"TotalReturn": total, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": mdd}


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
