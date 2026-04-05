from pathlib import Path
import pandas as pd

try:
    from .config import Config
    from .data_qc import generate_qc_report
    from .paths import ensure_layout, input_file, reports_dir
    from .prices import download_monthly_adjclose
    from .universe import load_constituents_history, load_sp500_constituents
except ImportError:  # pragma: no cover - direct script mode
    from config import Config
    from data_qc import generate_qc_report
    from paths import ensure_layout, input_file, reports_dir
    from prices import download_monthly_adjclose
    from universe import load_constituents_history, load_sp500_constituents


def _build_universe(constituents: pd.DataFrame, history: pd.DataFrame | None) -> list[str]:
    tickers = set(constituents["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist())
    if history is not None and not history.empty:
        tickers.update(history["Ticker"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist())
    return sorted(tickers)


def main() -> None:
    cfg = Config()
    base = Path(__file__).resolve().parent
    ensure_layout()

    constituents_path = input_file("sp500_constituents.csv")
    history_path = input_file("constituents_history_template.csv")
    stock_csv = input_file("stock_prices.csv")

    if not constituents_path.exists():
        raise FileNotFoundError(f"Missing constituents file: {constituents_path}")

    constituents = load_sp500_constituents(str(constituents_path))
    history = load_constituents_history(str(history_path)) if history_path.exists() else None
    universe = _build_universe(constituents, history)
    print(f"Universe tickers: {len(universe)}")

    print("Downloading stock prices...")
    stock_px = download_monthly_adjclose(universe, cfg.start_date, cfg.end_date)

    print("Downloading SPY...")
    spy_px = download_monthly_adjclose(cfg.spy_ticker, cfg.start_date, cfg.end_date)
    if cfg.spy_ticker in spy_px.columns:
        stock_px[cfg.spy_ticker] = spy_px[cfg.spy_ticker]
    else:
        stock_px[cfg.spy_ticker] = spy_px.iloc[:, 0]

    stock_px = stock_px.sort_index()
    stock_px.to_csv(stock_csv, index_label="Date")
    print(f"Saved: {stock_csv}")

    checks_df, qc_text = generate_qc_report(base)
    checks_path = reports_dir() / "data_checks_results.csv"
    qctxt_path = reports_dir() / "qc_report.txt"
    checks_df.to_csv(checks_path, index=False)
    qctxt_path.write_text(qc_text, encoding="utf-8")
    print(qc_text)
    print(f"Saved: {checks_path}, {qctxt_path}")


if __name__ == "__main__":
    main()
