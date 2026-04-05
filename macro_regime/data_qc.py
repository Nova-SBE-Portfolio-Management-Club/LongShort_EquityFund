from pathlib import Path
import pandas as pd

try:
    from .config import Config
    from .data_checks import run_data_checks
    from .paths import ensure_layout, input_file, reports_dir
    from .universe import load_constituents_history, load_sp500_constituents
except ImportError:  # pragma: no cover - direct script mode
    from config import Config
    from data_checks import run_data_checks
    from paths import ensure_layout, input_file, reports_dir
    from universe import load_constituents_history, load_sp500_constituents


def generate_qc_report(base_dir: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    cfg = Config()
    _ = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    ensure_layout()
    prices_path = input_file("stock_prices.csv")
    constituents_path = input_file("sp500_constituents.csv")
    history_path = input_file("constituents_history_template.csv")

    constituents = load_sp500_constituents(str(constituents_path)) if constituents_path.exists() else pd.DataFrame()
    history = load_constituents_history(str(history_path)) if history_path.exists() else None

    checks_df, checks_summary = run_data_checks(
        prices_path=prices_path,
        constituents=constituents,
        history=history,
        spy_ticker=cfg.spy_ticker,
    )

    # Add extra descriptive summary lines for human review.
    extra = []
    if prices_path.exists():
        raw = pd.read_csv(prices_path)
        extra.append(f"Data Shape: {raw.shape}")
        if "Date" in raw.columns:
            dts = pd.to_datetime(raw["Date"], errors="coerce")
            if dts.notna().any():
                extra.append(f"Date Range: {dts.min()} to {dts.max()}")
    else:
        extra.append(f"Data file not found: {prices_path}")

    text = "\n".join(["=== Data QC Report ===", checks_summary, "", *extra])
    return checks_df, text


def main() -> None:
    base = Path(__file__).resolve().parent
    ensure_layout()
    out_csv = reports_dir() / "data_checks_results.csv"
    out_txt = reports_dir() / "data_checks_report.txt"
    qc_txt = reports_dir() / "qc_report.txt"

    df, report = generate_qc_report(base)
    df.to_csv(out_csv, index=False)
    out_txt.write_text(report, encoding="utf-8")
    qc_txt.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {out_csv}, {out_txt}, {qc_txt}")


if __name__ == "__main__":
    main()
