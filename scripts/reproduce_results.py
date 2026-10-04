"""Reproduce synthetic and bundled earnings results without external data services."""

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.playground.backtest_demo import generate_sample_data  # noqa: E402
from src.playground.earnings_surprise import best_earnsurp as earnings  # noqa: E402
from utils.backtest import backtest, drawdown_curve, summary_stats  # noqa: E402

CODE_FILES = [
    "src/utils/backtest.py",
    "src/playground/backtest_demo.py",
    "src/playground/earnings_surprise/best_earnsurp.py",
    "src/playground/earnings_surprise/backtest.py",
    "scripts/reproduce_results.py",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_results():
    longs, shorts = generate_sample_data()
    simulation = backtest(longs, shorts)
    frame = pd.DataFrame({"Synthetic 1x gross": simulation["returns"], "Flat cash": 0.0})
    config = {"seed": 42, "periods": 252, "initial_capital": 100_000, "gross_leverage": 1}
    return frame, config, {}, {"kind": "synthetic demonstration", "discarded_periods": 0}


def earnings_results():
    config = earnings.CONFIG.copy()
    raw_earnings = earnings.load_earnings(config)
    prices = earnings.load_prices(config)
    if not prices.index.is_unique or prices.index.hasnans or not prices.columns.is_unique:
        raise ValueError("Price dates and ticker columns must be unique and non-missing.")
    observed = prices.to_numpy(dtype=float)
    observed = observed[~np.isnan(observed)]
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Observed prices must be finite and positive.")
    signals = earnings.compute_sue(raw_earnings, config)
    signals = earnings.compute_zscore(signals, config)
    signals = earnings.compute_signals_strategy_A(signals, config)
    events = earnings.add_entry_exit_asym(signals, config)
    signal_matrix = earnings.build_signal_matrix(events, config)
    results = earnings.compute_portfolio_returns_long_anchored_50_50(signal_matrix, prices, config)
    capped, uncapped, long_ret, short_ret, benchmark = results[:5]
    frame = pd.DataFrame(
        {
            "L/S capped": capped,
            "L/S uncapped": uncapped,
            "Equal-weight price-file universe": benchmark,
        }
    )

    # Independently rebuild the scalar weight per active name for the audit.
    # This checks the published portfolio against its stated sizing rules.
    sig = signal_matrix.reindex(
        index=frame.index, columns=prices.columns.intersection(signal_matrix.columns)
    )
    long_mask = sig.eq(1)
    long_count = long_mask.sum(axis=1)
    short_mask = sig.eq(-1).mul(long_count.ge(config["min_longs_for_shorts"]), axis=0)
    short_mask = short_mask.mul(long_count.gt(0), axis=0)
    short_count = short_mask.sum(axis=1)
    long_size = (0.5 / long_count.replace(0, np.nan)).fillna(0)
    short_size = (-0.5 / short_count.replace(0, np.nan)).fillna(0)
    weights = long_mask.mul(long_size.clip(upper=config["max_weight_long"]), axis=0)
    weights += short_mask.mul(short_size.clip(lower=-config["max_weight_short"]), axis=0)
    used_returns = prices.ffill().pct_change(fill_method=None).reindex_like(weights)
    independently_weighted = (used_returns * weights).sum(axis=1)
    np.testing.assert_allclose(capped, independently_weighted, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(capped, long_ret + short_ret, atol=1e-13, rtol=1e-12)
    if (results[6][long_count.eq(0)] != 0).any():
        raise ValueError("Short positions exist without active longs.")
    exposure = weights.abs().sum(axis=1)
    if (exposure > 1 + 1e-12).any():
        raise ValueError("Portfolio exceeds its 100% gross exposure cap.")
    raw_missing = (prices.isna() | prices.shift(1).isna()).reindex_like(weights)
    active = weights.ne(0)
    inputs = {
        str(Path(config[key]).relative_to(ROOT)): sha256(Path(config[key]))
        for key in ("earnings_path", "prices_path")
    }
    config["earnings_path"] = str(Path(config["earnings_path"]).relative_to(ROOT))
    config["prices_path"] = str(Path(config["prices_path"]).relative_to(ROOT))
    audit = {
        "kind": "reproduction of the existing earnings implementation",
        "earnings_rows": len(raw_earnings),
        "price_rows": len(prices),
        "price_tickers": len(prices.columns),
        "input_missing_price_cells": int(prices.isna().sum().sum()),
        "signal_tickers_missing_from_prices": signal_matrix.columns.difference(
            prices.columns
        ).tolist(),
        "active_missing_return_cells": int((active & used_returns.isna()).sum().sum()),
        "active_forward_filled_price_cells": int(
            (active & raw_missing & used_returns.notna()).sum().sum()
        ),
        "average_gross_exposure": float(exposure.mean()),
        "maximum_gross_exposure": float(exposure.max()),
        "flat_periods": int(exposure.eq(0).sum()),
        "checks_passed": [
            "unique ordered price dates and unique ticker columns",
            "finite positive observed prices",
            "independent capped-weight reconstruction",
            "portfolio return equals long plus short contributions",
            "no short-only positions and gross exposure at most 100%",
        ],
    }
    return frame, config, inputs, audit


def manifest_for(frame, config, inputs, audit):
    if frame.empty or not frame.index.is_unique or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Published returns must have a nonempty, unique, finite common sample.")
    stats = {name: summary_stats(frame[name], risk_free_rate=0) for name in frame}
    stats = {
        name: {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
        for name, values in stats.items()
    }
    return {
        "sample_start": frame.index.min().date().isoformat(),
        "sample_end": frame.index.max().date().isoformat(),
        "observed_periods": len(frame),
        "periods_per_year": 252,
        "risk_free_rate": 0,
        "trading_cost_bps": 0,
        "config": config,
        "input_sha256": inputs,
        "source_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "matplotlib", "openpyxl")},
        },
        "data_audit": audit,
        "metrics": stats,
    }


def plot_results(frame, title, path):
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    colors = ["#0f766e", "#64748b", "#2563eb"]
    for (name, returns), color in zip(frame.items(), colors, strict=False):
        axes[0].plot(frame.index, (1 + returns).cumprod(), label=name, color=color, linewidth=1.4)
        axes[1].plot(frame.index, drawdown_curve(returns) * 100, color=color, linewidth=1)
    axes[0].set_title(title, loc="left", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Growth of 1 (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "LongShort_EquityFund"})
    plt.close(figure)


def report_text(name, manifest):
    synthetic = name == "synthetic_demo"
    title = (
        "Synthetic backtester demonstration"
        if synthetic
        else "Bundled earnings-surprise reproduction"
    )
    lines = [
        f"# {title}",
        "",
        f"Sample: **{manifest['sample_start']} to {manifest['sample_end']}**, "
        f"with **{manifest['observed_periods']:,} observed periods** and 252 periods per year.",
        "",
        "The figures below are computed by the reproduction script and checked against the saved "
        "return series. All Sharpe ratios use a 0% cash rate; trading and financing costs are excluded.",
        "",
        "| Series | Total return | CAGR | Annualized volatility | Sharpe | Max drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for series, values in manifest["metrics"].items():
        sharpe = values["Sharpe Ratio"]
        formatted_sharpe = "undefined" if sharpe is None else f"{sharpe:.2f}"
        lines.append(
            f"| {series} | {values['Total Return']:.2%} | {values['Annualized Return']:.2%} | "
            f"{values['Annualized Volatility']:.2%} | {formatted_sharpe} | "
            f"{values['Max Drawdown']:.2%} |"
        )
    lines += ["", f"![{title}](figures/{name}.png)", "", "## Interpretation", ""]
    if synthetic:
        lines += [
            "Returns are generated with seed 42. Gross leverage 1 means 50% long and 50% short "
            "exposure, rebalanced each period. Flat cash earns 0% in this example. "
            "These numbers demonstrate the software and are not historical strategy performance.",
        ]
    else:
        audit = manifest["data_audit"]
        missing = ", ".join(audit["signal_tickers_missing_from_prices"])
        lines += [
            "This reproduces the default script configuration on the two bundled input files. "
            "The benchmark is the daily mean of available returns across the price-file universe, "
            "rather than a downloaded index. Uncapped results are a sizing comparison, not a holdout.",
            "",
            f"The input contains {audit['earnings_rows']:,} earnings rows and "
            f"{audit['price_tickers']} price tickers. Average gross exposure is "
            f"{audit['average_gross_exposure']:.1%}; {audit['flat_periods']:,} periods are flat.",
            "",
            "**Data and methodology limitations:**",
            "",
            f"- {len(audit['signal_tickers_missing_from_prices'])} signal tickers have no mapped price "
            f"column and are excluded: {missing or 'none'}.",
            f"- The input has {audit['input_missing_price_cells']:,} missing price cells. The existing "
            f"implementation forward-fills quotes: {audit['active_forward_filled_price_cells']:,} "
            "active ticker-period cells depend on that assumption.",
            f"- {audit['active_missing_return_cells']:,} active ticker-period cells have no return "
            "even after quote filling; the existing portfolio sum contributes zero for those cells.",
            "- Historical constituent coverage, delisting returns, and adjusted-price provenance "
            "have not been independently established for the bundled data.",
            "- Entry dates use business-day offsets and close-to-close returns. Announcement "
            "timestamps and exchange-calendar execution realism remain unverified.",
            "- Transaction costs, cash interest, borrow fees, and financing are excluded. The "
            "full-sample settings have no demonstrated untouched out-of-sample evaluation.",
            "- The checks establish computational reproduction and sizing consistency. They "
            "do not establish investable performance.",
            "",
            "## Verification checks",
            "",
            *[f"- {check}." for check in audit["checks_passed"]],
        ]
    lines += [
        "",
        "## Reproduction",
        "",
        "From the repository root:",
        "",
        "```bash",
        "python -m pip install -r requirements-results.txt",
        "python scripts/reproduce_results.py",
        "python scripts/reproduce_results.py --check",
        "```",
        "",
        f"Saved artifacts: [returns]({name}.returns.csv), "
        f"[configuration, input hashes, code hashes, environment, and metrics]({name}.manifest.json).",
        "",
        "Metric definitions and compatibility changes are documented in "
        "[BACKTESTING.md](../BACKTESTING.md).",
        "",
    ]
    return "\n".join(lines)


def check_saved(output, name, frame, manifest):
    saved = json.loads((output / f"{name}.manifest.json").read_text())
    recorded = pd.read_csv(output / f"{name}.returns.csv", index_col=0, parse_dates=True)
    pd.testing.assert_index_equal(recorded.index, frame.index, check_names=False)
    pd.testing.assert_index_equal(recorded.columns, frame.columns)
    np.testing.assert_allclose(recorded.to_numpy(), frame.to_numpy(), rtol=1e-11, atol=1e-13)
    for field in (
        "input_sha256",
        "source_sha256",
        "config",
        "sample_start",
        "sample_end",
    ):
        if saved[field] != manifest[field]:
            raise ValueError(
                f"{name}: published {field} differs; regenerate and review the report."
            )
    for key, value in manifest["data_audit"].items():
        recorded_value = saved["data_audit"][key]
        matches = (
            np.isclose(recorded_value, value, rtol=1e-12, atol=1e-14)
            if isinstance(value, float)
            else recorded_value == value
        )
        if not matches:
            raise ValueError(f"{name}: published data audit differs: {key}")
    for series, metrics in manifest["metrics"].items():
        for key, value in metrics.items():
            old = saved["metrics"][series][key]
            if value is None:
                if old is not None:
                    raise ValueError(f"{name}: undefined metric differs: {key}")
            elif not np.isclose(old, value, rtol=1e-10, atol=1e-12):
                raise ValueError(f"{name}: metric differs: {series}/{key}")
    expected_text = report_text(name, manifest)
    if (output / f"{name}.md").read_text() != expected_text:
        raise ValueError(f"{name}: displayed report differs from computed metrics.")
    if not (output / "figures" / f"{name}.png").is_file():
        raise ValueError(f"{name}: published figure is missing.")
    for relative_path, expected_hash in saved["artifact_sha256"].items():
        if sha256(output / relative_path) != expected_hash:
            raise ValueError(f"{name}: published artifact changed: {relative_path}")
    print(f"Verified {name}: inputs, code, sizing checks, returns, metrics, and displayed report.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "results")
    parser.add_argument(
        "--check", action="store_true", help="Compare with saved results without rewriting them."
    )
    args = parser.parse_args(argv)
    if not args.check:
        (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("synthetic_demo", synthetic_results),
        ("earnings_surprise", earnings_results),
    ):
        frame, config, inputs, audit = build()
        manifest = manifest_for(frame, config, inputs, audit)
        if args.check:
            check_saved(args.output_dir, name, frame, manifest)
            continue
        frame.to_csv(
            args.output_dir / f"{name}.returns.csv", index_label="date", float_format="%.17g"
        )
        (args.output_dir / f"{name}.md").write_text(report_text(name, manifest))
        title = (
            "Synthetic demo | seed 42"
            if name == "synthetic_demo"
            else "Earnings implementation reproduction | costs excluded"
        )
        plot_results(frame, title, args.output_dir / "figures" / f"{name}.png")
        manifest["artifact_sha256"] = {
            relative_path: sha256(args.output_dir / relative_path)
            for relative_path in (f"{name}.md", f"{name}.returns.csv", f"figures/{name}.png")
        }
        (args.output_dir / f"{name}.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(
            f"Saved {name}: {len(frame):,} observations, report, return series, manifest, and figure."
        )


if __name__ == "__main__":
    main()
