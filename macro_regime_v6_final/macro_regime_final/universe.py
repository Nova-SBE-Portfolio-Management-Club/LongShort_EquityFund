import pandas as pd

def load_sp500_constituents(path: str) -> pd.DataFrame:
    """
    Expected columns (minimum):
      - Symbol
      - Security (optional)
      - GICS Sector  (or Sector)
    """
    df = pd.read_csv(path)

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    if "gics sector" in cols:
        sector_col = cols["gics sector"]
    elif "sector" in cols:
        sector_col = cols["sector"]
    else:
        raise ValueError("CSV must contain 'GICS Sector' or 'Sector' column.")

    if "symbol" in cols:
        sym_col = cols["symbol"]
    else:
        raise ValueError("CSV must contain 'Symbol' column.")

    out = df[[sym_col, sector_col]].copy()
    out.columns = ["Symbol", "Sector"]
    out["Symbol"] = out["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    out["Sector"] = out["Sector"].astype(str)

    # Optional point-in-time metadata
    lower_map = {c.lower().strip(): c for c in df.columns}
    added_col = None
    for key in ["date added", "date_added", "added"]:
        if key in lower_map:
            added_col = lower_map[key]
            break
    removed_col = None
    for key in ["date removed", "date_removed", "removed", "removal date"]:
        if key in lower_map:
            removed_col = lower_map[key]
            break

    if added_col is not None:
        out["DateAdded"] = pd.to_datetime(df[added_col], errors="coerce")
    else:
        out["DateAdded"] = pd.NaT
    if removed_col is not None:
        out["DateRemoved"] = pd.to_datetime(df[removed_col], errors="coerce")
    else:
        out["DateRemoved"] = pd.NaT

    # drop empty core fields only
    out = out.dropna(subset=["Symbol", "Sector"]).drop_duplicates(subset=["Symbol"])
    return out


def constituents_asof(constituents: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    """
    Point-in-time constituents filter:
      - include symbols with DateAdded <= asof_date (or missing DateAdded)
      - exclude symbols with DateRemoved <= asof_date when available
    """
    out = constituents.copy()
    if "DateAdded" in out.columns:
        out = out[(out["DateAdded"].isna()) | (out["DateAdded"] <= asof_date)]
    if "DateRemoved" in out.columns:
        out = out[(out["DateRemoved"].isna()) | (out["DateRemoved"] > asof_date)]
    return out


def load_constituents_history(path: str) -> pd.DataFrame:
    """
    Optional event file schema:
      - Date
      - Ticker
      - Action  (Added / Removed)
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["date", "ticker", "action"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"History CSV missing required columns: {missing}")

    out = df[[cols["date"], cols["ticker"], cols["action"]]].copy()
    out.columns = ["Date", "Ticker", "Action"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    out["Action"] = out["Action"].astype(str).str.strip().str.lower()
    out = out[out["Action"].isin(["added", "removed"])]
    out = out.dropna(subset=["Date", "Ticker"]).sort_values("Date")
    return out


def apply_history_events(base_members: pd.DataFrame, history: pd.DataFrame | None, asof_date: pd.Timestamp) -> pd.DataFrame:
    """
    Apply Added/Removed events through asof_date to point-in-time members.
    """
    if history is None or history.empty:
        return base_members

    events = history.loc[history["Date"] <= asof_date]
    if events.empty:
        return base_members

    out = base_members.copy()
    current_symbols = set(out["Symbol"].tolist())
    symbol_to_sector = dict(zip(out["Symbol"], out["Sector"]))
    fallback_sector = out["Sector"].mode().iloc[0] if not out.empty else "Unknown"

    for _, ev in events.iterrows():
        t = ev["Ticker"]
        a = ev["Action"]
        if a == "added":
            if t not in current_symbols:
                sec = symbol_to_sector.get(t, fallback_sector)
                out = pd.concat(
                    [
                        out,
                        pd.DataFrame(
                            [{"Symbol": t, "Sector": sec, "DateAdded": pd.NaT, "DateRemoved": pd.NaT}]
                        ),
                    ],
                    ignore_index=True,
                )
                current_symbols.add(t)
        elif a == "removed":
            if t in current_symbols:
                out = out[out["Symbol"] != t]
                current_symbols.remove(t)

    return out.drop_duplicates(subset=["Symbol"])
