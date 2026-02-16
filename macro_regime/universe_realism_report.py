from __future__ import annotations

import pandas as pd


def run_universe_realism_report(
    constituents: pd.DataFrame,
    history: pd.DataFrame | None,
    stock_prices_m: pd.DataFrame,
    delist_df: pd.DataFrame | None = None,
    min_removed_events_for_realism: int = 200,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict] = []

    cons_syms = set(constituents.get("Symbol", pd.Series(dtype=str)).astype(str).str.upper().tolist())
    px_syms = set(stock_prices_m.columns.astype(str).str.upper().tolist())
    overlap = len(cons_syms & px_syms)
    overlap_ratio = overlap / max(len(cons_syms), 1)
    rows.append(
        {
            "check": "constituent_price_overlap",
            "status": "PASS" if overlap_ratio >= 0.95 else "WARN",
            "detail": f"Overlap symbols={overlap}/{len(cons_syms)} ({overlap_ratio:.2%})",
        }
    )

    has_added = "DateAdded" in constituents.columns and constituents["DateAdded"].notna().any()
    has_removed = "DateRemoved" in constituents.columns and constituents["DateRemoved"].notna().any()
    rows.append(
        {
            "check": "point_in_time_columns",
            "status": "PASS" if (has_added or has_removed) else "WARN",
            "detail": f"DateAdded present={bool(has_added)} DateRemoved present={bool(has_removed)}",
        }
    )

    if history is None or history.empty:
        n_hist = 0
        n_added = 0
        n_removed = 0
    else:
        n_hist = int(len(history))
        action = history["Action"].astype(str).str.lower()
        n_added = int((action == "added").sum())
        n_removed = int((action == "removed").sum())

    rows.append(
        {
            "check": "history_events_presence",
            "status": "PASS" if n_hist > 0 else "WARN",
            "detail": f"History rows={n_hist}",
        }
    )
    rows.append(
        {
            "check": "removed_events_count",
            "status": "PASS" if n_removed >= int(min_removed_events_for_realism) else "WARN",
            "detail": (
                f"Removed events={n_removed} "
                f"(target >= {int(min_removed_events_for_realism)} for stronger survivorship control)"
            ),
        }
    )

    n_delist = 0 if delist_df is None else int(len(delist_df))
    rows.append(
        {
            "check": "delisting_returns_rows",
            "status": "PASS" if n_delist > 0 else "WARN",
            "detail": f"Delisting return rows available: {n_delist}",
        }
    )

    if stock_prices_m.empty:
        disappearing = 0
        detail = "Empty price panel."
        status = "WARN"
    else:
        end_date = stock_prices_m.index.max()
        cutoff = end_date - pd.DateOffset(months=12)
        last_valid = stock_prices_m.apply(lambda c: c.dropna().index.max() if c.notna().any() else pd.NaT)
        disappearing = int(((last_valid.notna()) & (last_valid < cutoff)).sum())
        status = "PASS" if disappearing > 0 else "WARN"
        detail = (
            f"Symbols with no prices in final 12m: {disappearing} "
            f"(can indicate delisting/death coverage rather than survivor-only panel)"
        )
    rows.append({"check": "symbols_disappearing_before_end", "status": status, "detail": detail})

    out = pd.DataFrame(rows)
    n_warn = int((out["status"] == "WARN").sum())
    n_pass = int((out["status"] == "PASS").sum())

    lines = [
        "Universe Realism Summary",
        f"PASS={n_pass} WARN={n_warn}",
        "Note: WARN does not invalidate results by itself, but indicates survivorship/delisting realism gaps.",
    ]
    return out, "\n".join(lines)
