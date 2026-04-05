from __future__ import annotations

import numpy as np
import pandas as pd


def _perf(series: pd.Series, freq: int = 4) -> dict:
    if series.empty:
        return {
            "TotalReturn": np.nan,
            "CAGR": np.nan,
            "Vol": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "MaxDD": np.nan,
        }
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
    return {
        "TotalReturn": total,
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": mdd,
    }


def _row(label: str, sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"Sample": label, "Start": None, "End": None, "NQuarters": 0}
    p = _perf(sub["PortRet"])
    s = _perf(sub["SPYRet"])
    return {
        "Sample": label,
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
        "HitRateVsSPY": float((sub["PortRet"] > sub["SPYRet"]).mean()),
    }


def run_train_test_split_report(
    backtest_df: pd.DataFrame,
    split_date: str,
    min_oos_quarters: int = 24,
) -> tuple[pd.DataFrame, str]:
    if backtest_df.empty:
        df = pd.DataFrame([{"Sample": "All", "NQuarters": 0}])
        return df, "Train/Test Split Summary\nNo rows available."

    bt = backtest_df.copy()
    bt.index = pd.to_datetime(bt.index, errors="coerce")
    bt = bt[~bt.index.isna()].sort_index()

    split_ts = pd.Timestamp(split_date)
    train = bt.loc[bt.index <= split_ts]
    test = bt.loc[bt.index > split_ts]

    rows = [_row("Train", train), _row("Test", test), _row("All", bt)]
    out = pd.DataFrame(rows)

    train_sh = float(rows[0].get("Port_Sharpe", np.nan))
    test_sh = float(rows[1].get("Port_Sharpe", np.nan))
    decay = test_sh / (train_sh + 1e-12) if np.isfinite(train_sh) and abs(train_sh) > 1e-12 else np.nan
    oos_n = int(rows[1].get("NQuarters", 0) or 0)

    lines = [
        "Train/Test Split Summary",
        f"Split date: {split_ts.date()} (Train <= split, Test > split)",
        f"Train quarters: {int(rows[0].get('NQuarters', 0) or 0)}",
        f"Test quarters: {oos_n}",
    ]
    if np.isfinite(decay):
        lines.append(f"Sharpe decay (Test/Train): {decay:.3f}")
    if oos_n < int(min_oos_quarters):
        lines.append(
            f"Warning: OOS sample is small ({oos_n} < {int(min_oos_quarters)} quarters)."
        )
    return out, "\n".join(lines)
