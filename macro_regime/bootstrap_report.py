from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_sharpe(quarterly_rets: np.ndarray) -> float:
    if quarterly_rets.size == 0:
        return float("nan")
    mean_q = float(np.mean(quarterly_rets))
    vol_q = float(np.std(quarterly_rets, ddof=1)) if quarterly_rets.size > 1 else 0.0
    ann_ret = (1.0 + mean_q) ** 4 - 1.0
    ann_vol = vol_q * np.sqrt(4.0)
    return ann_ret / (ann_vol + 1e-12)


def _ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(x, alpha / 2))
    hi = float(np.quantile(x, 1 - alpha / 2))
    return lo, hi


def run_bootstrap_significance_report(
    backtest_df: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    if backtest_df.empty:
        out = pd.DataFrame([{"Metric": "NQuarters", "PointEstimate": 0}])
        return out, "Bootstrap Significance Summary\nNo rows available."

    bt = backtest_df.copy()
    bt.index = pd.to_datetime(bt.index, errors="coerce")
    bt = bt[~bt.index.isna()].sort_index()
    bt = bt[["PortRet", "SPYRet"]].dropna()
    if bt.empty:
        out = pd.DataFrame([{"Metric": "NQuarters", "PointEstimate": 0}])
        return out, "Bootstrap Significance Summary\nNo valid PortRet/SPYRet pairs available."

    port = bt["PortRet"].to_numpy(dtype=float)
    spy = bt["SPYRet"].to_numpy(dtype=float)
    alpha_q = port - spy
    n = len(port)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    port_b = port[idx]
    spy_b = spy[idx]
    alpha_b = alpha_q[idx]

    # Vectorized bootstrap stats
    port_mean_q_b = np.mean(port_b, axis=1)
    spy_mean_q_b = np.mean(spy_b, axis=1)
    alpha_mean_q_b = np.mean(alpha_b, axis=1)
    alpha_ann_b = alpha_mean_q_b * 4.0

    port_sh_b = np.apply_along_axis(_ann_sharpe, 1, port_b)
    spy_sh_b = np.apply_along_axis(_ann_sharpe, 1, spy_b)

    # Point estimates from realized sample
    port_sh = _ann_sharpe(port)
    spy_sh = _ann_sharpe(spy)
    alpha_mean_q = float(np.mean(alpha_q))
    alpha_ann = float(alpha_mean_q * 4.0)

    port_sh_ci = _ci(port_sh_b)
    spy_sh_ci = _ci(spy_sh_b)
    alpha_q_ci = _ci(alpha_mean_q_b)
    alpha_ann_ci = _ci(alpha_ann_b)

    p_alpha_le_0 = float(np.mean(alpha_ann_b <= 0.0))
    p_port_sh_le_0 = float(np.mean(port_sh_b <= 0.0))

    out = pd.DataFrame(
        [
            {
                "Metric": "NQuarters",
                "PointEstimate": float(n),
                "CI95_Low": np.nan,
                "CI95_High": np.nan,
                "PValueLike": np.nan,
            },
            {
                "Metric": "Port_AnnSharpe",
                "PointEstimate": float(port_sh),
                "CI95_Low": port_sh_ci[0],
                "CI95_High": port_sh_ci[1],
                "PValueLike": p_port_sh_le_0,
            },
            {
                "Metric": "SPY_AnnSharpe",
                "PointEstimate": float(spy_sh),
                "CI95_Low": spy_sh_ci[0],
                "CI95_High": spy_sh_ci[1],
                "PValueLike": np.nan,
            },
            {
                "Metric": "Alpha_MeanQuarterly",
                "PointEstimate": float(alpha_mean_q),
                "CI95_Low": alpha_q_ci[0],
                "CI95_High": alpha_q_ci[1],
                "PValueLike": np.nan,
            },
            {
                "Metric": "Alpha_AnnualizedSimple",
                "PointEstimate": float(alpha_ann),
                "CI95_Low": alpha_ann_ci[0],
                "CI95_High": alpha_ann_ci[1],
                "PValueLike": p_alpha_le_0,
            },
        ]
    )

    lines = [
        "Bootstrap Significance Summary",
        f"Iterations: {int(n_boot)}",
        f"N quarters: {n}",
        (
            f"Portfolio annualized Sharpe: {port_sh:.3f} "
            f"[95% CI: {port_sh_ci[0]:.3f}, {port_sh_ci[1]:.3f}]"
        ),
        (
            f"Alpha annualized (simple): {alpha_ann:.2%} "
            f"[95% CI: {alpha_ann_ci[0]:.2%}, {alpha_ann_ci[1]:.2%}]"
        ),
        f"P(alpha <= 0): {p_alpha_le_0:.3f}",
        f"P(portfolio Sharpe <= 0): {p_port_sh_le_0:.3f}",
    ]
    return out, "\n".join(lines)
