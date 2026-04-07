"""
experiment_hy_momentum.py
=========================
Self-contained experiment: replace copper/gold ratio with HY OAS momentum
as the fourth HMM feature.

PURPOSE
-------
Test whether HY OAS momentum (3-month rate of change, z-scored) produces
a better-calibrated Expansion state than copper/gold ratio, which is
anchored to the 2003-2007 commodity supercycle and rarely fires.

This file is STANDALONE — it does not import anything from the main
codebase. Copy it anywhere and run it. It will pull its own data,
train its own HMM, and produce a comparison report.

WHAT IT TESTS
-------------
    Feature set A (current):   hy_oas | yield_curve | vix | copper_gold
    Feature set B (candidate): hy_oas | yield_curve | vix | hy_oas_mom

OUTPUT
------
    - Side-by-side emission means for both models
    - Sanity checks for both (GFC, Post-GFC Bull, COVID, Inflation 2022)
    - Regime distribution comparison
    - Correlation check: hy_oas vs hy_oas_mom (must be < 0.70)
    - Recommendation: keep copper/gold OR switch to momentum

DECISION RULE
-------------
    Switch to hy_oas_mom if ALL of the following hold:
        1. Expansion fires in >= 15% of days (currently 14.2% with Cu/Au)
        2. GFC   >= 80% Contraction or Transition
        3. COVID  >= 60% Contraction or Transition
        4. Post-GFC Bull >= 60% Expansion or Transition (NOT mostly Contraction)
        5. Correlation(hy_oas_z, hy_oas_mom_z) < 0.70

DO NOT touch main codebase until all 5 criteria pass.

PARAMETERS (frozen — do not tune to improve results)
-----------------------------------------------------
    n_states          = 3
    covariance_type   = "full"
    n_restarts        = 50
    n_iter            = 200
    momentum_window   = 63    # ~3 months of trading days
    min_zscore_obs    = 252
    fred_api_key      = set FRED_API_KEY env var or hardcode below
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ─── Parameters (frozen) ──────────────────────────────────────────────────────
FRED_API_KEY     = os.environ.get("FRED_API_KEY", "dae2ef1813e221d9a3f2ba1764b2c8a9")
DATA_START       = "2000-01-01"
BURN_IN_END      = "2006-12-31"
INSAMPLE_END     = "2018-12-31"
OOS_START        = "2019-01-01"
OOS_END          = "2022-12-31"

N_STATES         = 3
COVARIANCE_TYPE  = "full"
N_RESTARTS       = 50
N_ITER           = 200
MOMENTUM_WINDOW  = 63      # trading days (~3 months)
MIN_ZSCORE_OBS   = 252

EXPOSURE_MAP = {
    "Expansion"   : 1.00,
    "Transition"  : 1.00,
    "Contraction" : 0.50,
}

# ─── Data pulling ─────────────────────────────────────────────────────────────

def _fred_client() -> Fred:
    if not FRED_API_KEY:
        raise EnvironmentError("Set FRED_API_KEY env var or hardcode in script.")
    return Fred(api_key=FRED_API_KEY)


def _expanding_zscore(series: pd.Series, min_obs: int = MIN_ZSCORE_OBS) -> pd.Series:
    """Expanding z-score with shift(1) lookahead guard."""
    mu    = series.expanding().mean().shift(1)
    sigma = series.expanding().std().shift(1)
    z     = (series - mu) / sigma
    obs   = series.expanding().count().shift(1)
    return z.where(obs >= min_obs, np.nan)


def pull_raw_data(start: str = DATA_START) -> pd.DataFrame:
    """
    Pull all raw series needed for both feature sets.
    Returns daily DataFrame with columns:
        hy_oas, tsy_10y, tsy_2y, vix, copper, gold
    """
    print("[data] Pulling FRED series...")
    fred = _fred_client()

    fred_series = {
        "hy_oas"  : "BAMLH0A0HYM2",
        "tsy_10y" : "DGS10",
        "tsy_2y"  : "DGS2",
        "vix"     : "VIXCLS",
    }
    raw = {}
    for name, ticker in fred_series.items():
        s = fred.get_series(ticker, observation_start=start)
        s.name = name
        raw[name] = s
        print(f"  {name}: {len(s)} obs ({s.index[0].date()} -> {s.index[-1].date()})")

    df = pd.DataFrame(raw)
    df["yield_curve"] = df["tsy_10y"] - df["tsy_2y"]
    df = df.drop(columns=["tsy_10y", "tsy_2y"])

    print("[data] Pulling copper/gold from yfinance...")
    cg_raw = yf.download(
        ["HG=F", "GC=F"], start=start,
        auto_adjust=True, progress=False
    )["Close"].ffill(limit=3)
    df["copper_gold"] = (cg_raw["HG=F"] / cg_raw["GC=F"])
    print(f"  copper_gold: {df['copper_gold'].notna().sum()} obs")

    # Align to business days, forward-fill short gaps
    bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df    = df.reindex(bdays).ffill(limit=3)

    return df


def build_feature_set_a(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set A — current production set.
    hy_oas | yield_curve | vix | copper_gold
    """
    cols = ["hy_oas", "yield_curve", "vix", "copper_gold"]
    z    = pd.DataFrame(index=raw.index)
    for c in cols:
        z[c] = _expanding_zscore(raw[c])
    z = z.dropna()
    print(f"[feature_A] {len(z)} clean obs  "
          f"({z.index[0].date()} -> {z.index[-1].date()})")
    return z


def build_feature_set_b(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set B — candidate set.
    hy_oas | yield_curve | vix | hy_oas_mom

    hy_oas_mom = 63-day (3-month) rate of change of raw HY OAS,
    then z-scored on an expanding window.

    Stationary by construction — measures whether spreads are
    tightening (negative, risk-on) or widening (positive, risk-off).
    No commodity supercycle calibration issue.
    """
    # 63-day momentum of raw HY OAS (before z-scoring)
    hy_mom_raw = raw["hy_oas"].pct_change(MOMENTUM_WINDOW)

    cols = {
        "hy_oas"      : raw["hy_oas"],
        "yield_curve" : raw["yield_curve"],
        "vix"         : raw["vix"],
        "hy_oas_mom"  : hy_mom_raw,
    }
    z = pd.DataFrame(index=raw.index)
    for name, series in cols.items():
        z[name] = _expanding_zscore(series)
    z = z.dropna()
    print(f"[feature_B] {len(z)} clean obs  "
          f"({z.index[0].date()} -> {z.index[-1].date()})")
    return z


# ─── HMM training ─────────────────────────────────────────────────────────────

def train_hmm(X: np.ndarray) -> GaussianHMM:
    """
    50-restart training with k-means++ initialisation.
    Identical to production regime_engine.py logic.
    """
    best_model = None
    best_score = -np.inf

    km       = KMeans(n_clusters=N_STATES, init="k-means++",
                      n_init=10, random_state=0)
    km.fit(X)
    km_means = km.cluster_centers_

    for seed in range(N_RESTARTS):
        model = GaussianHMM(
            n_components    = N_STATES,
            covariance_type = COVARIANCE_TYPE,
            n_iter          = N_ITER,
            random_state    = seed,
            tol             = 1e-5,
        )
        if seed >= 10:
            model.means_ = km_means.copy()
            model.init_params = "stc"
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    return best_model


def label_states(model: GaussianHMM, feature_names: list) -> dict:
    """
    Label states by risk score = hy_oas + vix - yield_curve - <risk_appetite>.
    Risk appetite = copper_gold (set A) or -hy_oas_mom (set B,
    because widening momentum = negative risk appetite).
    """
    means = model.means_

    hy_idx  = feature_names.index("hy_oas")
    vix_idx = feature_names.index("vix")
    yc_idx  = feature_names.index("yield_curve")

    # Risk appetite proxy — last feature in each set
    ra_idx  = len(feature_names) - 1
    ra_name = feature_names[ra_idx]

    # For momentum: widening (positive z) = risk-off, so flip sign
    ra_sign = -1 if ra_name == "hy_oas_mom" else 1

    risk_scores = (
        means[:, hy_idx]
        + means[:, vix_idx]
        - means[:, yc_idx]
        - ra_sign * means[:, ra_idx]
    )

    sorted_states              = np.argsort(risk_scores)
    labels                     = {}
    labels[sorted_states[0]]   = "Expansion"
    labels[sorted_states[1]]   = "Transition"
    labels[sorted_states[2]]   = "Contraction"
    return labels


def classify_history(
    model        : GaussianHMM,
    state_labels : dict,
    features_df  : pd.DataFrame,
) -> pd.DataFrame:
    """Run Viterbi over full history. Returns daily regime DataFrame."""
    X              = features_df.values
    hidden_states  = model.predict(X)
    posteriors     = model.predict_proba(X)

    regime     = [state_labels[s] for s in hidden_states]
    confidence = [float(posteriors[t, s]) for t, s in enumerate(hidden_states)]
    exposure   = [EXPOSURE_MAP[r] for r in regime]

    hist = features_df.copy()
    hist["regime"]     = regime
    hist["confidence"] = confidence
    hist["exposure"]   = exposure
    return hist


# ─── Analysis ─────────────────────────────────────────────────────────────────

def print_emission_means(model: GaussianHMM,
                         state_labels: dict,
                         feature_names: list,
                         label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label} — Emission means (z-scored)")
    print(f"{'='*60}")
    header = f"  {'State':<14}" + "".join(f"{f:>12}" for f in feature_names)
    print(header)
    print("  " + "-" * (14 + 12 * len(feature_names)))
    for idx, name in state_labels.items():
        means = model.means_[idx]
        row   = f"  {name:<14}" + "".join(f"{m:>12.3f}" for m in means)
        print(row)


def run_sanity_checks(history: pd.DataFrame, label: str) -> dict:
    """
    Check regime assignments against known macro events.
    Returns dict of {check_name: passed (bool)}.
    """
    print(f"\n--- Sanity checks: {label} ---")

    checks = [
        # (name, start, end, acceptable_regimes, min_pct, hard)
        ("GFC Peak",       "2008-09-01", "2009-06-30",
         ["Contraction", "Transition"], 0.80, True),
        ("Post-GFC Bull",  "2013-01-01", "2015-12-31",
         ["Expansion",    "Transition"], 0.60, True),
        ("COVID Crash",    "2020-03-01", "2020-05-31",
         ["Contraction",  "Transition"], 0.60, True),
        ("Inflation 2022", "2022-01-01", "2022-12-31",
         ["Contraction",  "Transition"], 0.40, True),
    ]

    results = {}
    for name, start, end, acceptable, min_pct, hard in checks:
        window  = history.loc[start:end, "regime"]
        if window.empty:
            print(f"  {name:<22}: NO DATA")
            results[name] = None
            continue
        pct     = window.isin(acceptable).mean()
        dominant = window.value_counts().idxmax()
        passed  = pct >= min_pct
        status  = "PASS" if passed else "WARN"
        acc_str = " / ".join(acceptable)
        print(f"  {name:<22}: {dominant:<12} "
              f"({pct:.0%} in [{acc_str}])  [{status}]")
        results[name] = passed

    return results


def regime_distribution(history: pd.DataFrame, label: str) -> None:
    print(f"\n--- Regime distribution: {label} ---")
    dist = history["regime"].value_counts(normalize=True)
    for regime in ["Expansion", "Transition", "Contraction"]:
        pct = dist.get(regime, 0.0)
        bar = "#" * int(pct * 40)
        print(f"  {regime:<12}: {pct:>5.1%}  {bar}")

    # Average confidence
    print()
    for regime in ["Expansion", "Transition", "Contraction"]:
        mask = history["regime"] == regime
        if mask.any():
            conf = history.loc[mask, "confidence"].mean()
            print(f"  Avg confidence in {regime:<12}: {conf:.3f}")


def correlation_check(z_a: pd.DataFrame, z_b: pd.DataFrame) -> float:
    """
    Check correlation between hy_oas (same in both sets) and hy_oas_mom.
    High correlation (> 0.70) means both features carry redundant information.
    """
    common = z_a.index.intersection(z_b.index)
    corr   = z_a.loc[common, "hy_oas"].corr(z_b.loc[common, "hy_oas_mom"])
    print(f"\n--- Correlation check ---")
    print(f"  corr(hy_oas_z, hy_oas_mom_z) = {corr:.3f}")
    if abs(corr) < 0.70:
        print(f"  [OK] Below 0.70 threshold — features are sufficiently independent")
    else:
        print(f"  [WARN] Above 0.70 — features may be redundant, "
              f"consider VIX term structure or equity breadth instead")
    return corr


def year_by_year(history: pd.DataFrame, label: str) -> None:
    """Print dominant regime per year."""
    print(f"\n--- Year-by-year dominant regime: {label} ---")
    history["year"] = history.index.year
    by_year = history.groupby("year")["regime"].agg(
        lambda x: x.value_counts().idxmax()
    )
    # Group into rows of 5 years for readability
    years = list(by_year.items())
    for i in range(0, len(years), 5):
        chunk = years[i:i+5]
        print("  " + "  ".join(f"{y}: {r:<11}" for y, r in chunk))


def decision(
    results_a : dict,
    results_b : dict,
    dist_b    : dict,
    corr      : float,
) -> None:
    """Print final recommendation based on decision criteria."""
    print(f"\n{'='*60}")
    print("  DECISION")
    print(f"{'='*60}")

    criteria = {
        "Expansion fires >= 15% of days" :
            dist_b.get("Expansion", 0.0) >= 0.15,
        "GFC correctly defensive" :
            results_b.get("GFC Peak", False),
        "COVID correctly defensive" :
            results_b.get("COVID Crash", False),
        "Post-GFC Bull not over-defensive" :
            results_b.get("Post-GFC Bull", False),
        "Correlation(hy_oas, hy_oas_mom) < 0.70" :
            abs(corr) < 0.70,
    }

    all_pass = all(criteria.values())
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}")

    print()
    if all_pass:
        print("  RECOMMENDATION: Switch to hy_oas_mom.")
        print("  All 5 criteria passed. Update macro_data.py and regime_engine.py.")
    else:
        print("  RECOMMENDATION: Keep copper/gold for now.")
        print("  Not all criteria passed. Consider VIX term structure or equity breadth.")

    print()
    # Always remind what to do next
    print("  Next steps if switching:")
    print("  1. In macro_data.py: remove _pull_copper_gold(), add hy_oas_mom calc")
    print("  2. In regime_engine.py: update feature_cols list")
    print("  3. Update label_states() risk appetite sign flip (ra_sign = -1)")
    print("  4. Re-run regime_engine.py sanity checks on main codebase")
    print("  5. Re-run backtest.py and compare results")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EXPERIMENT: HY OAS Momentum vs Copper/Gold")
    print("  Replacing Feature Set A with Feature Set B")
    print("=" * 60)

    # 1. Pull data
    raw = pull_raw_data()

    # 2. Build both feature sets
    print()
    z_a = build_feature_set_a(raw)   # current: copper/gold
    z_b = build_feature_set_b(raw)   # candidate: hy_oas_mom

    # 3. Restrict to in-sample for training (no lookahead)
    X_a = z_a.loc[:INSAMPLE_END].values
    X_b = z_b.loc[:INSAMPLE_END].values

    # 4. Train both HMMs
    print("\n[hmm] Training model A (copper/gold)...")
    model_a = train_hmm(X_a)
    labels_a = label_states(model_a, list(z_a.columns))
    print(f"  Best log-likelihood: {model_a.score(X_a):.2f}")

    print("\n[hmm] Training model B (hy_oas_mom)...")
    model_b = train_hmm(X_b)
    labels_b = label_states(model_b, list(z_b.columns))
    print(f"  Best log-likelihood: {model_b.score(X_b):.2f}")

    # 5. Classify full history with each model
    hist_a = classify_history(model_a, labels_a, z_a)
    hist_b = classify_history(model_b, labels_b, z_b)

    # 6. Emission means
    print_emission_means(model_a, labels_a, list(z_a.columns), "Model A — copper/gold")
    print_emission_means(model_b, labels_b, list(z_b.columns), "Model B — hy_oas_mom")

    # 7. Regime distributions
    regime_distribution(hist_a, "Model A — copper/gold")
    regime_distribution(hist_b, "Model B — hy_oas_mom")

    # 8. Sanity checks
    results_a = run_sanity_checks(hist_a, "Model A — copper/gold")
    results_b = run_sanity_checks(hist_b, "Model B — hy_oas_mom")

    # 9. Year-by-year
    year_by_year(hist_a, "Model A — copper/gold")
    year_by_year(hist_b, "Model B — hy_oas_mom")

    # 10. Correlation check
    corr = correlation_check(z_a, z_b)

    # 11. Decision
    dist_b_raw = hist_b["regime"].value_counts(normalize=True).to_dict()
    decision(results_a, results_b, dist_b_raw, corr)

    # 12. Save detailed output for inspection
    out_a = hist_a[["regime", "confidence", "exposure"]]
    out_b = hist_b[["regime", "confidence", "exposure"]]
    out_a.to_csv("experiment_model_a_history.csv")
    out_b.to_csv("experiment_model_b_history.csv")
    print("\n[saved] experiment_model_a_history.csv")
    print("[saved] experiment_model_b_history.csv")
    print("\nDone. Review the DECISION section above before touching main codebase.")


if __name__ == "__main__":
    main()
