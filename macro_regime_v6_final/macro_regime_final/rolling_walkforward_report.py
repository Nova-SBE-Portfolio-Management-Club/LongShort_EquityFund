from __future__ import annotations

import numpy as np
import pandas as pd


def _perf(series: pd.Series, freq: int = 4) -> dict:
    if series.empty:
        return {"TotalReturn": np.nan, "CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan}
    total = float((1 + series).prod() - 1)
    years = len(series) / freq
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(series.std() * np.sqrt(freq))
    sharpe = cagr / (vol + 1e-12)
    return {"TotalReturn": total, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe}


def run_rolling_walkforward_report(
    backtest_df: pd.DataFrame,
    train_quarters: int = 32,
    test_quarters: int = 8,
    step_quarters: int = 4,
) -> tuple[pd.DataFrame, str]:
    if backtest_df.empty:
        out = pd.DataFrame([{"Window": 0, "TrainN": 0, "TestN": 0}])
        return out, "Rolling Walk-Forward Summary\nNo rows available."

    bt = backtest_df.copy()
    bt.index = pd.to_datetime(bt.index, errors="coerce")
    bt = bt[~bt.index.isna()].sort_index()
    n = len(bt)
    rows: list[dict] = []

    w = 0
    start = 0
    while start + train_quarters + test_quarters <= n:
        w += 1
        train = bt.iloc[start : start + train_quarters]
        test = bt.iloc[start + train_quarters : start + train_quarters + test_quarters]

        p_train = _perf(train["PortRet"])
        p_test = _perf(test["PortRet"])
        s_train = _perf(train["SPYRet"])
        s_test = _perf(test["SPYRet"])

        tr_sh = float(p_train["Sharpe"])
        te_sh = float(p_test["Sharpe"])
        decay = te_sh / (tr_sh + 1e-12) if np.isfinite(tr_sh) else np.nan

        rows.append(
            {
                "Window": w,
                "TrainStart": str(train.index.min().date()),
                "TrainEnd": str(train.index.max().date()),
                "TestStart": str(test.index.min().date()),
                "TestEnd": str(test.index.max().date()),
                "TrainN": int(len(train)),
                "TestN": int(len(test)),
                "Port_Train_Sharpe": p_train["Sharpe"],
                "Port_Test_Sharpe": p_test["Sharpe"],
                "SPY_Train_Sharpe": s_train["Sharpe"],
                "SPY_Test_Sharpe": s_test["Sharpe"],
                "Sharpe_Decay_TestOverTrain": decay,
                "Port_Test_CAGR": p_test["CAGR"],
                "SPY_Test_CAGR": s_test["CAGR"],
                "Port_Test_Alpha_CAGR": float(p_test["CAGR"] - s_test["CAGR"]),
                "Port_Test_HitRateVsSPY": float((test["PortRet"] > test["SPYRet"]).mean()),
            }
        )
        start += step_quarters

    out = pd.DataFrame(rows)
    if out.empty:
        return (
            out,
            (
                "Rolling Walk-Forward Summary\n"
                "No valid windows. Increase data length or reduce train/test window sizes."
            ),
        )

    mean_decay = float(out["Sharpe_Decay_TestOverTrain"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    median_decay = float(out["Sharpe_Decay_TestOverTrain"].replace([np.inf, -np.inf], np.nan).dropna().median())
    mean_alpha = float(out["Port_Test_Alpha_CAGR"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    beat_spy_windows = int((out["Port_Test_CAGR"] > out["SPY_Test_CAGR"]).sum())
    total_windows = int(len(out))

    lines = [
        "Rolling Walk-Forward Summary",
        f"Windows: {total_windows}",
        f"Train/Test/Step quarters: {train_quarters}/{test_quarters}/{step_quarters}",
        f"Mean Sharpe decay (Test/Train): {mean_decay:.3f}",
        f"Median Sharpe decay (Test/Train): {median_decay:.3f}",
        f"Mean Test CAGR alpha vs SPY: {mean_alpha:.2%}",
        f"Windows beating SPY on Test CAGR: {beat_spy_windows}/{total_windows}",
    ]
    return out, "\n".join(lines)
