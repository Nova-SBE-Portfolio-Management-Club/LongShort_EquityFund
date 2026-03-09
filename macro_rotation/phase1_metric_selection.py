from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from fredapi import Fred
except ImportError:
    Fred = None  # type: ignore


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
ROLLING_DIR = OUTPUT_DIR / "rolling"
REPORT_DIR = OUTPUT_DIR / "reports"

for folder in [DATA_DIR, OUTPUT_DIR, HEATMAP_DIR, ROLLING_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Put your Bloomberg export here later.
# Expected format:
# - one sheet per region, e.g. "US", "Europe", "Japan", "China", "Commodities"
# - first column = Date
# - subsequent columns = ticker names exactly matching the config below
BLOOMBERG_FILE = DATA_DIR / "bloomberg_phase1.xlsx"

# Use environment variable instead of hardcoding the key
# Example in terminal:
# export FRED_API_KEY="your_key_here"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

START_DATE = "2000-01-01"
END_DATE = None  # None = latest available
ROLLING_WINDOW = 36

# Redundancy thresholds for Phase 1
PEARSON_THRESHOLD = 0.70
SPEARMAN_THRESHOLD = 0.70
ROLLING_MEDIAN_THRESHOLD = 0.60


# ============================================================
# METRIC CONFIG
# ============================================================


@dataclass
class Metric:
    region: str
    name: str
    source: str  # "fred", "bloomberg", "derived", "manual"
    ticker: str
    transform: str
    role: str
    active: bool = True


METRICS: List[Metric] = [
    # ---------------- US ----------------
    Metric("US", "ISM Manufacturing PMI", "bloomberg", "NAPMPMI Index", "level_z_input", "growth"),
    Metric("US", "10Y TIPS Real Yield", "fred", "DFII10", "level_z_input", "real_rates"),
    Metric("US", "HY OAS", "fred", "BAMLH0A0HYM2", "level_z_input", "credit_stress"),
    Metric("US", "Nonfarm Payrolls", "fred", "PAYEMS", "yoy_pct", "labor_growth"),
    Metric("US", "VIX", "fred", "VIXCLS", "level_z_input", "risk_stress"),
    Metric("US", "NFCI", "fred", "NFCI", "level_z_input", "financial_conditions"),
    # ---------------- Europe ----------------
    Metric("Europe", "Eurozone Composite PMI", "bloomberg", "PMITCEZ Index", "level_z_input", "growth"),
    Metric("Europe", "EUR/USD", "bloomberg", "EURUSD Curncy", "pct_change_12m", "fx"),
    Metric("Europe", "Italy 10Y BTP Yield", "bloomberg", "GBTPGR10 Index", "level_z_input", "rates"),
    Metric("Europe", "Germany 10Y Bund Yield", "bloomberg", "GDBR10 Index", "level_z_input", "rates"),
    Metric("Europe", "ECB Deposit Rate", "bloomberg", "EURR002W Index", "mom_diff", "policy"),
    Metric("Europe", "TTF Natural Gas", "bloomberg", "TZ1 Comdty", "pct_change_12m", "energy"),
    Metric("Europe", "STOXX 600", "bloomberg", "SXXP Index", "pct_change_12m", "market"),
    Metric("Europe", "Eurozone CPI YoY", "bloomberg", "ECCPEMUY Index", "level_z_input", "inflation"),
    # ---------------- Japan ----------------
    Metric("Japan", "USD/JPY", "bloomberg", "USDJPY Curncy", "pct_change_12m", "fx"),
    Metric("Japan", "BOJ Policy Rate", "bloomberg", "BOJDTR Index", "mom_diff", "policy"),
    Metric("Japan", "Japan Mfg PMI", "bloomberg", "SEASPMI Index", "level_z_input", "growth"),
    Metric("Japan", "Japan Wage Growth", "bloomberg", "JNLCWAGE Index", "yoy_pct", "wages"),
    Metric("Japan", "JGB 10Y Yield", "bloomberg", "GJGB10 Index", "level_z_input", "rates"),
    Metric("Japan", "Nikkei 225", "bloomberg", "NKY Index", "pct_change_12m", "market"),
    Metric("Japan", "Japan CPI YoY", "bloomberg", "JNCPIYOY Index", "level_z_input", "inflation"),
    # ---------------- China ----------------
    Metric("China", "China Credit Impulse", "bloomberg", "CHBGCRIM Index", "level_z_input", "credit"),
    Metric("China", "Total Social Financing", "bloomberg", "CNTSF Index", "yoy_pct", "credit"),
    Metric("China", "Caixin Mfg PMI", "bloomberg", "CHPMINDX Index", "level_z_input", "growth"),
    Metric("China", "1Y LPR", "bloomberg", "CHLR1YPR Index", "mom_diff", "policy"),
    Metric("China", "USD/CNH", "bloomberg", "USDCNH Curncy", "pct_change_12m", "fx"),
    Metric("China", "CSI 300", "bloomberg", "SHSZ300 Index", "pct_change_12m", "market"),
    # ---------------- Commodities ----------------
    Metric("Commodities", "WTI Front Month", "bloomberg", "CL1 Comdty", "pct_change_12m", "oil_price"),
    Metric("Commodities", "WTI 2nd Month", "bloomberg", "CL2 Comdty", "pct_change_12m", "oil_curve"),
    Metric("Commodities", "Gold Spot", "bloomberg", "XAUUSD Curncy", "pct_change_12m", "gold"),
    Metric("Commodities", "Copper LME 3m", "bloomberg", "LMCADS Index", "pct_change_12m", "copper"),
    Metric("Commodities", "Baker Hughes Rig Count", "bloomberg", "BHUSCOIL Index", "yoy_pct", "oil_supply"),
    Metric("Commodities", "Global Mfg PMI", "bloomberg", "PMITMGL Index", "level_z_input", "global_demand"),
    Metric("Commodities", "DXY Broad USD", "fred", "DTWEXBGS", "pct_change_12m", "usd"),
    Metric("Commodities", "US Crude Inventories", "fred", "WCRFPUS2", "yoy_pct", "oil_inventory"),
]


# ============================================================
# TRANSFORMATIONS
# ============================================================

def winsorize(series: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def level_z_input(series: pd.Series) -> pd.Series:
    # For Phase 1 correlations, we do not z-score yet.
    # We keep transformed raw inputs clean and comparable later.
    return series.copy()


def yoy_pct(series: pd.Series) -> pd.Series:
    return series.pct_change(12) * 100.0


def mom_diff(series: pd.Series) -> pd.Series:
    return series.diff(1)


def pct_change_12m(series: pd.Series) -> pd.Series:
    return series.pct_change(12) * 100.0


TRANSFORM_MAP: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "level_z_input": level_z_input,
    "yoy_pct": yoy_pct,
    "mom_diff": mom_diff,
    "pct_change_12m": pct_change_12m,
}


# ============================================================
# DATA LOADERS
# ============================================================

def init_fred() -> Optional["Fred"]:
    if Fred is None:
        warnings.warn("fredapi is not installed. FRED data will be skipped.")
        return None
    if not FRED_API_KEY:
        warnings.warn("FRED_API_KEY is not set. FRED data will be skipped.")
        return None
    return Fred(api_key=FRED_API_KEY)


def load_fred_series(fred: Optional["Fred"], series_id: str) -> pd.Series:
    if fred is None:
        return pd.Series(dtype=float, name=series_id)

    try:
        s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
        s = pd.Series(s, name=series_id)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        return s
    except Exception as exc:
        warnings.warn(f"Could not load FRED series {series_id}: {exc}")
        return pd.Series(dtype=float, name=series_id)


def load_bloomberg_excel(filepath: Path) -> Dict[str, pd.DataFrame]:
    if not filepath.exists():
        warnings.warn(f"Bloomberg file not found at {filepath}. Placeholder mode enabled.")
        return {}

    try:
        xls = pd.ExcelFile(filepath)
        sheets: Dict[str, pd.DataFrame] = {}
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            if df.empty:
                sheets[sheet_name] = pd.DataFrame()
                continue

            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.rename(columns={date_col: "Date"}).dropna(subset=["Date"])
            df = df.set_index("Date").sort_index()
            sheets[sheet_name] = df
        return sheets
    except Exception as exc:
        warnings.warn(f"Could not read Bloomberg Excel file: {exc}")
        return {}


def extract_bloomberg_series(
    bloomberg_sheets: Dict[str, pd.DataFrame],
    region: str,
    ticker: str,
) -> pd.Series:
    df = bloomberg_sheets.get(region, pd.DataFrame())
    if df.empty or ticker not in df.columns:
        warnings.warn(f"Missing Bloomberg series: region={region}, ticker={ticker}")
        return pd.Series(dtype=float, name=ticker)

    s = pd.to_numeric(df[ticker], errors="coerce")
    s.name = ticker
    return s


# ============================================================
# DERIVED FEATURES
# ============================================================

def build_derived_features(raw_region_df: pd.DataFrame, region: str) -> pd.DataFrame:
    df = raw_region_df.copy()

    if region == "Europe":
        italy = "GBTPGR10 Index"
        germany = "GDBR10 Index"
        if italy in df.columns and germany in df.columns:
            df["BTP_Bund_Spread"] = df[italy] - df[germany]

    if region == "Commodities":
        cl1 = "CL1 Comdty"
        cl2 = "CL2 Comdty"
        if cl1 in df.columns and cl2 in df.columns:
            df["WTI_Curve_Spread"] = df[cl1] - df[cl2]

    return df


# ============================================================
# ALIGNMENT / CLEANING
# ============================================================

def to_month_end(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.resample("M").last()
    return s


def clean_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


# ============================================================
# PIPELINE BUILD
# ============================================================

def build_raw_dataset(metrics: List[Metric]) -> Dict[str, pd.DataFrame]:
    fred = init_fred()
    bloomberg_sheets = load_bloomberg_excel(BLOOMBERG_FILE)

    region_store: Dict[str, Dict[str, pd.Series]] = {}

    for metric in metrics:
        if not metric.active:
            continue

        region_store.setdefault(metric.region, {})

        if metric.source == "fred":
            s = load_fred_series(fred, metric.ticker)
        elif metric.source == "bloomberg":
            s = extract_bloomberg_series(bloomberg_sheets, metric.region, metric.ticker)
        else:
            s = pd.Series(dtype=float, name=metric.ticker)

        s = clean_series(s)
        s = to_month_end(s)
        region_store[metric.region][metric.ticker] = s

    raw_datasets: Dict[str, pd.DataFrame] = {}
    for region, series_dict in region_store.items():
        df = pd.concat(series_dict.values(), axis=1) if series_dict else pd.DataFrame()
        df = build_derived_features(df, region)
        raw_datasets[region] = df.sort_index()

    return raw_datasets


def build_transformed_dataset(
    metrics: List[Metric],
    raw_datasets: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    transformed_by_region: Dict[str, pd.DataFrame] = {}

    # Add derived metrics into the config dynamically
    full_metric_list = metrics.copy()
    full_metric_list.extend(
        [
            Metric("Europe", "BTP-Bund Spread", "derived", "BTP_Bund_Spread", "level_z_input", "fragmentation"),
            Metric("Commodities", "WTI Curve Spread", "derived", "WTI_Curve_Spread", "level_z_input", "curve_structure"),
        ]
    )

    for region in sorted(set(m.region for m in full_metric_list if m.active)):
        raw_df = raw_datasets.get(region, pd.DataFrame())
        out: Dict[str, pd.Series] = {}

        for metric in [m for m in full_metric_list if m.region == region and m.active]:
            if metric.ticker not in raw_df.columns:
                continue

            transform_fn = TRANSFORM_MAP.get(metric.transform)
            if transform_fn is None:
                warnings.warn(f"Unknown transform '{metric.transform}' for {metric.name}")
                continue

            transformed = transform_fn(raw_df[metric.ticker])
            transformed = clean_series(transformed)
            out[metric.name] = transformed

        region_df = pd.DataFrame(out).sort_index()
        transformed_by_region[region] = region_df

    return transformed_by_region


# ============================================================
# CORRELATIONS
# ============================================================

def rolling_spearman(a: pd.Series, b: pd.Series, window: int = 36) -> pd.Series:
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.shape[0] < window:
        return pd.Series(dtype=float)

    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]

    values = []
    dates = []

    for i in range(window, len(aligned) + 1):
        sub_x = x.iloc[i - window : i]
        sub_y = y.iloc[i - window : i]
        rho = sub_x.rank().corr(sub_y.rank(), method="pearson")
        values.append(rho)
        dates.append(aligned.index[i - 1])

    return pd.Series(values, index=dates, name=f"{a.name}__{b.name}")


def compute_region_correlations(region_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = region_df.dropna(axis=1, how="all")
    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    return pearson, spearman


def redundancy_report(region: str, region_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(region_df.columns)
    rows = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            a = region_df[columns[i]]
            b = region_df[columns[j]]

            aligned = pd.concat([a, b], axis=1).dropna()
            if len(aligned) < max(24, ROLLING_WINDOW):
                continue

            pearson = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="pearson")
            spearman = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
            rolling = rolling_spearman(aligned.iloc[:, 0], aligned.iloc[:, 1], window=ROLLING_WINDOW)

            rolling_median_abs = np.nan
            if not rolling.empty:
                rolling_median_abs = float(np.nanmedian(np.abs(rolling)))

            redundant = (
                abs(pearson) >= PEARSON_THRESHOLD
                and abs(spearman) >= SPEARMAN_THRESHOLD
                and rolling_median_abs >= ROLLING_MEDIAN_THRESHOLD
            )

            rows.append(
                {
                    "region": region,
                    "metric_1": columns[i],
                    "metric_2": columns[j],
                    "n_obs": len(aligned),
                    "pearson": pearson,
                    "spearman": spearman,
                    "rolling_spearman_median_abs_36m": rolling_median_abs,
                    "flag_redundant": redundant,
                }
            )

    report = pd.DataFrame(rows).sort_values(
        by=["flag_redundant", "rolling_spearman_median_abs_36m", "spearman"],
        ascending=[False, False, False],
    )
    return report


# ============================================================
# PLOTTING
# ============================================================

def save_heatmap(matrix: pd.DataFrame, title: str, filepath: Path) -> None:
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(filepath, dpi=180)
    plt.close(fig)


def save_rolling_plots(region: str, region_df: pd.DataFrame, report: pd.DataFrame, top_n: int = 5) -> None:
    flagged = report[report["flag_redundant"]].head(top_n)
    if flagged.empty:
        return

    for _, row in flagged.iterrows():
        m1 = row["metric_1"]
        m2 = row["metric_2"]
        roll = rolling_spearman(region_df[m1], region_df[m2], window=ROLLING_WINDOW)
        if roll.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(roll.index, roll.values)
        ax.axhline(0.0, linestyle="--")
        ax.axhline(SPEARMAN_THRESHOLD, linestyle=":")
        ax.axhline(-SPEARMAN_THRESHOLD, linestyle=":")
        ax.set_title(f"{region} | 36M Rolling Spearman | {m1} vs {m2}")
        ax.set_ylabel("Spearman rho")
        fig.tight_layout()

        safe_name = f"{region}_{m1[:25]}__{m2[:25]}".replace("/", "_").replace(" ", "_")
        fig.savefig(ROLLING_DIR / f"{safe_name}.png", dpi=180)
        plt.close(fig)


# ============================================================
# OUTPUTS
# ============================================================

def save_metric_inventory(metrics: List[Metric]) -> None:
    inventory = pd.DataFrame([asdict(m) for m in metrics])
    inventory.to_csv(REPORT_DIR / "metric_inventory.csv", index=False)


def save_transformed_data(transformed_by_region: Dict[str, pd.DataFrame]) -> None:
    for region, df in transformed_by_region.items():
        df.to_csv(DATA_DIR / f"{region.lower()}_transformed_phase1.csv")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Starting Phase 1 metric-selection pipeline...")

    save_metric_inventory(METRICS)

    raw_datasets = build_raw_dataset(METRICS)
    transformed_by_region = build_transformed_dataset(METRICS, raw_datasets)
    save_transformed_data(transformed_by_region)

    summary_rows = []

    for region, region_df in transformed_by_region.items():
        print(f"\nProcessing region: {region}")

        non_empty_cols = region_df.dropna(axis=1, how="all").columns.tolist()
        region_df = region_df[non_empty_cols]

        if region_df.shape[1] < 2:
            print(f"Skipping {region}: not enough data.")
            continue

        pearson, spearman = compute_region_correlations(region_df)

        pearson.to_csv(REPORT_DIR / f"{region.lower()}_pearson.csv")
        spearman.to_csv(REPORT_DIR / f"{region.lower()}_spearman.csv")

        save_heatmap(pearson, f"{region} Pearson Correlation", HEATMAP_DIR / f"{region.lower()}_pearson.png")
        save_heatmap(spearman, f"{region} Spearman Correlation", HEATMAP_DIR / f"{region.lower()}_spearman.png")

        report = redundancy_report(region, region_df)
        report.to_csv(REPORT_DIR / f"{region.lower()}_redundancy_report.csv", index=False)

        save_rolling_plots(region, region_df, report, top_n=5)

        flagged_count = int(report["flag_redundant"].sum()) if not report.empty else 0
        summary_rows.append(
            {
                "region": region,
                "n_metrics_used": region_df.shape[1],
                "n_pairs_tested": len(report),
                "n_flagged_redundant": flagged_count,
            }
        )

        print(f"{region}: {region_df.shape[1]} metrics, {len(report)} pairs tested, {flagged_count} flagged.")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "phase1_summary.csv", index=False)

    print("\nDone. Check outputs/")
    print("- outputs/reports/")
    print("- outputs/heatmaps/")
    print("- outputs/rolling/")


if __name__ == "__main__":
    main()
