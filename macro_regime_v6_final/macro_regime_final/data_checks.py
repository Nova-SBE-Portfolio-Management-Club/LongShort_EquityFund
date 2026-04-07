from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

def _normalize_date_column(raw: pd.DataFrame) -> pd.DataFrame:
    if "Date" in raw.columns:
        return raw
    if len(raw.columns) > 0:
        first = str(raw.columns[0]).strip()
        if first == "" or first.lower().startswith("unnamed"):
            return raw.rename(columns={raw.columns[0]: "Date"})
    return raw


@dataclass
class CheckResult:
    check: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _fmt(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:.4f}"
    return str(n)


def _add(rows: list[CheckResult], check: str, status: str, detail: str) -> None:
    rows.append(CheckResult(check=check, status=status, detail=detail))


def _active_mask_for_ticker(
    dates: pd.DatetimeIndex,
    ticker: str,
    constituents: pd.DataFrame,
    history: pd.DataFrame | None,
) -> pd.Series:
    active = pd.Series(False, index=dates)

    if not constituents.empty:
        c = constituents.loc[constituents["Symbol"].astype(str).str.upper() == ticker]
        if not c.empty:
            added = pd.to_datetime(c["DateAdded"], errors="coerce").min() if "DateAdded" in c.columns else pd.NaT
            removed = pd.to_datetime(c["DateRemoved"], errors="coerce").min() if "DateRemoved" in c.columns else pd.NaT
            if pd.isna(added):
                added = dates.min()
            if pd.isna(removed):
                active.loc[dates >= added] = True
            else:
                active.loc[(dates >= added) & (dates < removed)] = True

    if history is not None and not history.empty:
        h = history.loc[history["Ticker"].astype(str).str.upper() == ticker].sort_values("Date")
        for _, ev in h.iterrows():
            d = pd.to_datetime(ev["Date"], errors="coerce")
            if pd.isna(d):
                continue
            action = str(ev["Action"]).strip().lower()
            if action == "added":
                active.loc[dates >= d] = True
            elif action == "removed":
                active.loc[dates >= d] = False

    return active


def run_data_checks(
    prices_path: str | Path,
    constituents: pd.DataFrame,
    history: pd.DataFrame | None = None,
    spy_ticker: str = "SPY",
    coverage_pass_threshold: float = 0.90,
    outlier_warn_threshold: float = 0.80,
    outlier_count_warn_max: int = 0,
) -> tuple[pd.DataFrame, str]:
    rows: list[CheckResult] = []
    p = Path(prices_path)
    if not p.exists():
        _add(rows, "prices_file_exists", "FAIL", f"Missing file: {p}")
        out = pd.DataFrame([r.__dict__ for r in rows])
        return out, "Data checks failed: prices file is missing."

    raw = _normalize_date_column(pd.read_csv(p))
    if "Date" not in raw.columns:
        _add(rows, "schema_date_column", "FAIL", "Required column 'Date' not found.")
        out = pd.DataFrame([r.__dict__ for r in rows])
        return out, "Data checks failed: missing Date column."
    _add(rows, "schema_date_column", "PASS", "Date column found.")

    if spy_ticker in raw.columns:
        _add(rows, "schema_spy_column", "PASS", f"{spy_ticker} column found.")
    else:
        _add(rows, "schema_spy_column", "WARN", f"{spy_ticker} column missing; benchmark may be synthetic.")

    dt = pd.to_datetime(raw["Date"], errors="coerce")
    bad_dt = int(dt.isna().sum())
    if bad_dt > 0:
        _add(rows, "date_parse", "FAIL", f"Unparseable Date rows: {bad_dt}")
    else:
        _add(rows, "date_parse", "PASS", "All Date rows parsed.")

    dupe_dates = int(dt.duplicated().sum())
    if dupe_dates > 0:
        _add(rows, "date_duplicates", "FAIL", f"Duplicate Date rows: {dupe_dates}")
    else:
        _add(rows, "date_duplicates", "PASS", "No duplicate Date rows.")

    monotonic = bool(pd.Series(dt.dropna()).is_monotonic_increasing)
    _add(rows, "date_monotonic", "PASS" if monotonic else "WARN", "Dates are increasing." if monotonic else "Dates are not sorted increasing.")

    px = raw.drop(columns=["Date"]).copy()
    px.columns = [str(c).upper().replace(".", "-") for c in px.columns]
    px = px.apply(pd.to_numeric, errors="coerce")

    nonpos = int((px <= 0).sum().sum())
    if nonpos > 0:
        _add(rows, "nonpositive_prices", "WARN", f"Non-positive price cells: {nonpos}")
    else:
        _add(rows, "nonpositive_prices", "PASS", "No non-positive prices detected.")

    if len(dt.dropna()) > 1:
        span_days = int((dt.max() - dt.min()).days)
        _add(rows, "date_span", "PASS", f"Rows={len(raw)} span_days={span_days} from {dt.min().date()} to {dt.max().date()}")
    else:
        _add(rows, "date_span", "WARN", "Insufficient rows for span check.")

    rets = px.pct_change(fill_method=None)
    outlier_count = int((rets.abs() > float(outlier_warn_threshold)).sum().sum())
    if outlier_count > int(outlier_count_warn_max):
        _add(
            rows,
            "return_outliers",
            "WARN",
            f"Monthly returns with |r|>{outlier_warn_threshold:.0%}: {outlier_count} (max allowed {outlier_count_warn_max})",
        )
    else:
        _add(
            rows,
            "return_outliers",
            "PASS",
            f"Outlier count within threshold: {outlier_count} (|r|>{outlier_warn_threshold:.0%}, max {outlier_count_warn_max})",
        )

    if isinstance(dt, pd.Series):
        dates = pd.DatetimeIndex(dt.dropna().sort_values().unique())
    else:
        dates = pd.DatetimeIndex([])
    if len(dates) > 0 and not px.empty:
        active_covs = []
        for t in px.columns:
            mask = _active_mask_for_ticker(dates, t, constituents, history)
            active_n = int(mask.sum())
            if active_n == 0:
                continue
            s = pd.Series(px[t].values, index=dates)
            cov_t = float(s[mask].notna().mean())
            active_covs.append(cov_t)
        active_cov = float(np.mean(active_covs)) if active_covs else 0.0
        cov_status = "PASS" if active_cov >= float(coverage_pass_threshold) else "WARN"
        _add(
            rows,
            "overall_data_coverage",
            cov_status,
            f"Average non-null coverage over active windows: {_fmt(active_cov)} (target {coverage_pass_threshold:.2f})",
        )
    else:
        _add(rows, "overall_data_coverage", "WARN", "Could not compute coverage (empty dates or price panel).")

    if not constituents.empty:
        cons_syms = set(constituents["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False))
        px_syms = set(px.columns)
        overlap = len(cons_syms & px_syms)
        overlap_ratio = overlap / max(len(cons_syms), 1)
        ov_status = "PASS" if overlap_ratio >= 0.95 else "WARN"
        _add(rows, "constituent_price_overlap", ov_status, f"Overlap symbols={overlap}/{len(cons_syms)} ({overlap_ratio:.2%})")
    else:
        _add(rows, "constituent_price_overlap", "WARN", "Constituent table empty.")

    if history is None or history.empty:
        _add(rows, "history_events", "WARN", "No history events provided.")
    else:
        bad_action = int((~history["Action"].astype(str).str.lower().isin(["added", "removed"])).sum())
        if bad_action > 0:
            _add(rows, "history_actions", "FAIL", f"Invalid history actions: {bad_action}")
        else:
            _add(rows, "history_actions", "PASS", "History actions valid (added/removed).")

        unknown_hist = int((~history["Ticker"].astype(str).str.upper().isin(px.columns)).sum())
        uh_status = "WARN" if unknown_hist > 0 else "PASS"
        _add(rows, "history_ticker_in_prices", uh_status, f"History tickers missing from prices: {unknown_hist}")

    out = pd.DataFrame([r.__dict__ for r in rows])
    n_fail = int((out["status"] == "FAIL").sum())
    n_warn = int((out["status"] == "WARN").sum())
    summary = [
        "Data Quality Summary",
        f"FAIL={n_fail} WARN={n_warn} PASS={int((out['status'] == 'PASS').sum())}",
    ]
    if n_fail > 0:
        summary.append("Recommendation: fix FAIL checks before trusting backtest output.")
    elif n_warn > 0:
        summary.append("Recommendation: review WARN checks; results may still carry data bias.")
    else:
        summary.append("All checks passed.")
    return out, "\n".join(summary)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    try:
        from .paths import ensure_layout, input_file, reports_dir
    except ImportError:  # pragma: no cover - direct script mode
        from paths import ensure_layout, input_file, reports_dir
    from universe import load_sp500_constituents, load_constituents_history  # direct script mode

    ensure_layout()
    cons = load_sp500_constituents(str(input_file("sp500_constituents.csv")))
    hist_path = input_file("constituents_history_template.csv")
    hist = load_constituents_history(str(hist_path)) if hist_path.exists() else None
    df, text = run_data_checks(input_file("stock_prices.csv"), cons, hist, spy_ticker="SPY")
    out_csv = reports_dir() / "data_checks_results.csv"
    out_txt = reports_dir() / "data_checks_report.txt"
    df.to_csv(out_csv, index=False)
    out_txt.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out_csv}, {out_txt}")
