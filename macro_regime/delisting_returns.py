from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_delisting_returns(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "DelistReturn"])

    df = pd.read_csv(p)
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["date", "ticker", "delistreturn"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Delisting returns CSV missing required columns: {missing}")

    out = df[[cols["date"], cols["ticker"], cols["delistreturn"]]].copy()
    out.columns = ["Date", "Ticker", "DelistReturn"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    out["DelistReturn"] = pd.to_numeric(out["DelistReturn"], errors="coerce")
    out = out.dropna(subset=["Date", "Ticker", "DelistReturn"]).sort_values("Date")
    return out


def apply_delisting_returns_to_quarterly(
    stock_rets_q: pd.DataFrame,
    delist_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if delist_df is None or delist_df.empty:
        audit = pd.DataFrame(
            [{"Date": None, "Ticker": None, "DelistReturn": None, "status": "NO_DATA", "detail": "No delisting rows provided."}]
        )
        return stock_rets_q, audit, "Delisting integration skipped: no delisting rows provided."

    out = stock_rets_q.copy()
    rows: list[dict] = []

    for _, ev in delist_df.iterrows():
        d = pd.to_datetime(ev["Date"], errors="coerce")
        t = str(ev["Ticker"]).upper()
        dr = float(ev["DelistReturn"])
        if pd.isna(d):
            rows.append({"Date": None, "Ticker": t, "DelistReturn": dr, "status": "SKIP", "detail": "Invalid event date"})
            continue

        q_end = d.to_period("Q").to_timestamp(how="end").normalize()
        if q_end not in out.index:
            rows.append(
                {
                    "Date": str(d.date()),
                    "Ticker": t,
                    "DelistReturn": dr,
                    "status": "SKIP",
                    "detail": f"Quarter {q_end.date()} not in returns index",
                }
            )
            continue
        if t not in out.columns:
            rows.append(
                {
                    "Date": str(d.date()),
                    "Ticker": t,
                    "DelistReturn": dr,
                    "status": "SKIP",
                    "detail": "Ticker not in returns panel",
                }
            )
            continue

        prev = out.at[q_end, t]
        if pd.isna(prev):
            new_ret = dr
        else:
            # Combine in-quarter return with terminal delisting return.
            new_ret = (1.0 + float(prev)) * (1.0 + dr) - 1.0
        out.at[q_end, t] = new_ret
        rows.append(
            {
                "Date": str(d.date()),
                "Ticker": t,
                "DelistReturn": dr,
                "status": "APPLIED",
                "detail": f"Quarter {q_end.date()} return overwritten/combined.",
            }
        )

    audit = pd.DataFrame(rows)
    n_applied = int((audit["status"] == "APPLIED").sum()) if not audit.empty else 0
    n_skip = int((audit["status"] == "SKIP").sum()) if not audit.empty else 0
    txt = f"Delisting integration: applied={n_applied} skipped={n_skip} total_events={len(audit)}"
    return out, audit, txt
