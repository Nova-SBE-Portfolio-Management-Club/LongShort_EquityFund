import numpy as np
import pandas as pd

try:
    from .metrics import annualized_volatility, cagr, sharpe_ratio
except ImportError:  # pragma: no cover - supports direct script execution
    from metrics import annualized_volatility, cagr, sharpe_ratio

def perf_stats(rets: pd.Series) -> dict:
    if rets.empty:
        return {}
    ann_ret = cagr(rets)
    ann_vol = annualized_volatility(rets)
    sharpe = sharpe_ratio(rets)
    mdd = max_drawdown((1 + rets).cumprod().values)
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": mdd}

def max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())

def summarize(backtest_df: pd.DataFrame) -> str:
    if backtest_df.empty:
        return "=== PERFORMANCE SUMMARY ===\nNo backtest periods were generated."

    p = perf_stats(backtest_df["PortRet"])
    s = perf_stats(backtest_df["SPYRet"])

    hit_rate = float((backtest_df["PortRet"] > backtest_df["SPYRet"]).mean())
    avg_alpha = float((backtest_df["PortRet"] - backtest_df["SPYRet"]).mean())

    lines = []
    lines.append("=== PERFORMANCE SUMMARY ===")
    lines.append(f"Port: ann_ret={p['ann_return']:.2%} ann_vol={p['ann_vol']:.2%} sharpe={p['sharpe']:.2f} maxDD={p['max_dd']:.2%}")
    lines.append(f"SPY : ann_ret={s['ann_return']:.2%} ann_vol={s['ann_vol']:.2%} sharpe={s['sharpe']:.2f} maxDD={s['max_dd']:.2%}")
    lines.append("")
    lines.append("=== ALPHA DIAGNOSTICS ===")
    lines.append(f"Hit-rate (Port > SPY per quarter): {hit_rate:.1%}")
    lines.append(f"Avg alpha per quarter: {avg_alpha:.2%}")
    lines.append(f"Avg TopStocksInPredSectors (out of top20): {backtest_df['TopStocksInPredSectors'].mean():.2f}")
    return "\n".join(lines)
