"""
regime_engine.py
================
HMM-based macro regime classifier.

Architecture
------------
    3 hidden states  → Expansion | Transition | Contraction
    4 observed vars  → hy_oas, yield_curve, vix, hy_oas_mom (all z-scored)
    Emissions        → Full Gaussian (each state has its own mean + covariance)
    Inference        → Forward algorithm → P(state | data up to T)
    Hard label       → argmax of state probabilities
    Confidence       → winning probability (used as exposure multiplier)

Exposure mapping (frozen before any backtest):
    Expansion    → 1.00  (full momentum)
    Transition   → 0.65  (partial)
    Contraction  → 0.30  (defensive — momentum crashes here)

Walk-forward protocol
---------------------
    At each quarterly rebalance date T:
        1. Train HMM on ALL data from DATA_START through T
           (expanding window — never re-use future data)
        2. Classify the current regime using last CONFIRM_WINDOW days
        3. Average posterior probabilities across the window to smooth
           one-day flips without collapsing everything to Transition
        4. Output: (regime_label, confidence_score, exposure_multiplier)

Period discipline (enforced internally):
    - HMM is NEVER trained before BURN_IN_END (insufficient history)
    - Walk-forward ONLY runs within in_sample period for tuning
    - OOS and robustness periods are classified using model trained on
      all in-sample data — NO retraining after INSAMPLE_END
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# ─── Frozen hyperparameters ───────────────────────────────────────────────────
# These are set from first principles BEFORE any backtest output is seen.
# Do NOT change these to improve backtest results — that is overfitting.

N_STATES          = 3           # Expansion / Transition / Contraction
COVARIANCE_TYPE   = "full"      # "diag" is safer for stability, "full" captures correlations
N_ITER            = 100         # Baum-Welch EM iterations
RANDOM_STATE      = 42          # Reproducibility
CONFIRM_WINDOW    = 5           # Days used to average posteriors for final regime confirmation
MIN_TRAIN_DAYS    = 504         # ~2 years minimum before first HMM training

# Exposure multipliers — logic-derived, not optimized
EXPOSURE_MAP = {
    "Expansion"   : 1.00,
    "Transition"  : 0.65,
    "Contraction" : 0.30,
}

# Period boundaries (must match macro_data.py)
BURN_IN_END  = "2006-12-31"
INSAMPLE_END = "2018-12-31"
OOS_END      = "2022-12-31"


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RegimeReading:
    """Single regime reading at one point in time."""
    date            : pd.Timestamp
    regime          : str           # "Expansion" | "Transition" | "Contraction"
    confidence      : float         # winning state probability [0, 1]
    exposure        : float         # position size multiplier
    state_probs     : dict          # full probability vector


# ─── Core HMM functions ───────────────────────────────────────────────────────

import warnings
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

def _fit_single_hmm(seed: int, X: np.ndarray, km_means: np.ndarray = None) -> tuple:
    model = GaussianHMM(
        n_components    = N_STATES,
        covariance_type = COVARIANCE_TYPE,
        n_iter          = 200,          # was 100
        random_state    = seed,
        tol             = 1e-5,         # tighter stopping tolerance
    )
    if seed >= 10 and km_means is not None:
        model.means_ = km_means
        model.init_params = "stc" # Only initialize starts and covars
        
    try:
        model.fit(X)
        score = model.score(X)
        return score, model
    except Exception as e:
        return -np.inf, None

def _train_hmm(X: np.ndarray) -> GaussianHMM:
    """
    Train with 50 restarts. First 10 use random init (current behaviour).
    Next 40 use k-means++ init — gives the EM a smarter starting point
    and dramatically improves the chance of finding the global optimum.
    """
    # K-means++ initial means — gives EM a structured starting point
    km = KMeans(n_clusters=N_STATES, init="k-means++", n_init=10, random_state=0)
    km.fit(X)
    km_means = km.cluster_centers_

    # Local disable for warnings as joblib threads might bubble them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=-1)(
            delayed(_fit_single_hmm)(seed, X, km_means) for seed in range(50)
        )

    best_model = None
    best_score = -np.inf

    for score, model in results:
        if score > best_score and model is not None:
            best_score = score
            best_model = model

    return best_model


def _label_states(model: GaussianHMM, feature_names: list[str]) -> dict[int, str]:
    """
    Map HMM state indices to economic regime labels.

    Labeling rule (purely from emission means, no chart-gazing):
        - HY OAS mean     : lower = less stress
        - VIX mean        : lower = less fear
        - Yield curve mean: higher = steeper (healthier)
        - HY OAS Mom mean : higher (widening) = more stress → ADD to risk score
                            (opposite sign to copper/gold, which was risk-ON when positive)

    We compute a composite "risk score" for each state:
        risk_score = mean(hy_oas) + mean(vix) - mean(yield_curve) + mean(hy_oas_mom)

    hy_oas_mom sign logic:
        Positive momentum = spreads widening = risk-off = more stress → ADD
        (copper/gold was risk-on when high → SUBTRACT; hy_oas_mom is risk-off when high → ADD)

    Higher risk_score → more stressed → closer to Contraction.

    State with lowest risk_score  → Expansion
    State with highest risk_score → Contraction
    Middle state                  → Transition
    """
    means = model.means_  # shape (N_STATES, N_FEATURES)

    hy_idx  = feature_names.index("hy_oas")
    vix_idx = feature_names.index("vix")
    yc_idx  = feature_names.index("yield_curve")
    # hy_oas_mom: positive = spreads widening = risk-off -> ADD to risk score
    # (opposite sign to copper_gold which was risk-on when positive -> SUBTRACT)
    ra_idx  = feature_names.index("hy_oas_mom")

    risk_scores = (
        means[:, hy_idx]
        + means[:, vix_idx]
        - means[:, yc_idx]
        + means[:, ra_idx]   # ADD hy_oas_mom (positive = stress)
    )

    # Sort states by risk score ascending
    sorted_states = np.argsort(risk_scores)  # [lowest, mid, highest]

    state_labels = {}
    state_labels[sorted_states[0]] = "Expansion"
    state_labels[sorted_states[1]] = "Transition"
    state_labels[sorted_states[2]] = "Contraction"

    return state_labels


def _classify_current(
    model        : GaussianHMM,
    X_recent     : np.ndarray,
    state_labels : dict[int, str],
) -> tuple[str, float, dict]:
    """
    Run the forward algorithm on X_recent and return the current regime.

    X_recent: last CONFIRM_WINDOW rows of z-scored features.

    Returns:
        regime     — label of the most probable state
        confidence — probability of winning state
        probs_dict — full probability vector {label: prob}
    """
    # Posterior state probabilities at each time step
    # posteriors shape: (T, N_STATES)
    posteriors = model.predict_proba(X_recent)

    # Use the last row — most recent observation
    latest_probs = posteriors[-1]

    # Most probable state
    winning_state = int(np.argmax(latest_probs))
    confidence    = float(latest_probs[winning_state])
    regime        = state_labels[winning_state]

    probs_dict = {
        state_labels[i]: float(latest_probs[i])
        for i in range(N_STATES)
    }

    return regime, confidence, probs_dict


def _classify_avg_posterior(
    model        : GaussianHMM,
    X_window     : np.ndarray,
    state_labels : dict[int, str],
) -> tuple[str, float]:
    """
    Classify the window by averaging posterior probabilities across the
    confirmation window, then taking the argmax of that average vector.
    """
    posteriors = model.predict_proba(X_window)
    avg_probs = posteriors.mean(axis=0)
    winning_state = int(np.argmax(avg_probs))
    regime = state_labels[winning_state]
    confidence = float(avg_probs[winning_state])
    return regime, confidence


# ─── Walk-forward engine ──────────────────────────────────────────────────────

def run_walkforward_regimes(
    features_df     : pd.DataFrame,
    rebalance_dates : pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Walk-forward HMM regime classification.

    Protocol:
        For each rebalance date T in rebalance_dates:
            - Train HMM on all data from start through T (expanding window)
            - Classify regime using last CONFIRM_WINDOW days through T
            - Final regime is the raw last-day HMM winner
            - Keep a smoothed average-posterior regime as a diagnostic
            - Keep the stability flag only as a diagnostic
            - Output regime label + confidence for that date

    CRITICAL: rebalance_dates must all fall within the in_sample period.
    OOS and robustness periods use a single frozen model (see classify_oos).

    Parameters
    ----------
    features_df     : DataFrame with z-scored features + 'period' column
    rebalance_dates : quarterly dates at which the strategy rebalances

    Returns
    -------
    DataFrame indexed by rebalance date with columns:
        regime, confidence, exposure, is_stable, state_probs_*
    """
    feature_cols = ["hy_oas", "yield_curve", "vix", "hy_oas_mom"]
    X_full       = features_df[feature_cols]

    records = []

    for dt in rebalance_dates:
        if dt > pd.Timestamp(INSAMPLE_END):
            raise ValueError(
                f"walk-forward called with OOS date {dt.date()}. "
                "Use classify_oos() for out-of-sample periods."
            )

        # All feature data up to and including rebalance date
        X_train = X_full.loc[:dt].dropna()

        if len(X_train) < MIN_TRAIN_DAYS:
            print(f"  [regime] Skipping {dt.date()} — insufficient history "
                  f"({len(X_train)} < {MIN_TRAIN_DAYS} days)")
            continue

        # Train HMM on expanding window
        model        = _train_hmm(X_train.values)
        state_labels = _label_states(model, feature_cols)

        # Classify using last CONFIRM_WINDOW days
        X_window = X_train.values[-CONFIRM_WINDOW:]
        raw_regime, raw_confidence, _ = _classify_current(model, X_window, state_labels)
        smoothed_regime, smoothed_confidence = _classify_avg_posterior(model, X_window, state_labels)
        posteriors = model.predict_proba(X_window)
        daily_regimes = [state_labels[int(np.argmax(posteriors[t]))] for t in range(len(posteriors))]
        is_stable = len(set(daily_regimes)) == 1

        regime = raw_regime
        confidence = raw_confidence

        exposure = EXPOSURE_MAP[regime]

        # Full probability vector for the last observation
        last_probs  = posteriors[-1]
        probs_dict  = {state_labels[i]: float(last_probs[i]) for i in range(N_STATES)}

        records.append({
            "date"               : dt,
            "regime"             : regime,
            "confidence"         : confidence,
            "exposure"           : exposure,
            "is_stable"          : is_stable,
            "raw_regime"         : raw_regime,
            "raw_confidence"     : raw_confidence,
            "smoothed_regime"    : smoothed_regime,
            "smoothed_confidence": smoothed_confidence,
            "prob_expansion"     : probs_dict.get("Expansion", 0.0),
            "prob_transition"    : probs_dict.get("Transition", 0.0),
            "prob_contraction"   : probs_dict.get("Contraction", 0.0),
        })

        print(f"  [regime] {dt.date()} -> final={regime:<12} "
              f"raw={raw_regime:<12} smooth={smoothed_regime:<12} "
              f"conf={confidence:.2f} exp={exposure:.2f} stable={is_stable}")

    return pd.DataFrame(records).set_index("date")


def classify_oos(
    features_df     : pd.DataFrame,
    rebalance_dates : pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Classify OOS and robustness periods using a model trained on ALL
    in-sample data. No retraining after INSAMPLE_END.

    This is the correct protocol for true out-of-sample testing:
        - Model is frozen at end of in-sample period
        - It is NEVER updated when OOS data arrives
        - This reflects what you'd actually have done in real time

    Parameters
    ----------
    features_df     : full feature DataFrame (all periods)
    rebalance_dates : quarterly dates in OOS or robustness period

    Returns
    -------
    Same schema as run_walkforward_regimes output.
    """
    feature_cols = ["hy_oas", "yield_curve", "vix", "hy_oas_mom"]
    X_full       = features_df[feature_cols]

    # Train once on all in-sample data
    X_insample   = X_full.loc[:INSAMPLE_END].dropna()
    print(f"[regime] Training frozen OOS model on {len(X_insample)} in-sample days...")
    model        = _train_hmm(X_insample.values)
    state_labels = _label_states(model, feature_cols)

    print(f"[regime] State labels from emission means:")
    for state_idx, label in state_labels.items():
        means = model.means_[state_idx]
        print(f"  State {state_idx} -> {label}: "
              f"hy_oas={means[0]:.2f}, yield_curve={means[1]:.2f}, "
              f"vix={means[2]:.2f}, hy_oas_mom={means[3]:.2f}")

    records = []

    for dt in rebalance_dates:
        # All data up to T — model is frozen but we still need history for
        # the forward algorithm (posteriors use full sequence up to T)
        X_to_date = X_full.loc[:dt].dropna()
        if len(X_to_date) < CONFIRM_WINDOW:
            continue

        X_window = X_to_date.values[-CONFIRM_WINDOW:]
        raw_regime, raw_confidence, _ = _classify_current(model, X_window, state_labels)
        smoothed_regime, smoothed_confidence = _classify_avg_posterior(model, X_window, state_labels)
        posteriors = model.predict_proba(X_window)
        daily_regimes = [state_labels[int(np.argmax(posteriors[t]))] for t in range(len(posteriors))]
        is_stable = len(set(daily_regimes)) == 1

        regime = raw_regime
        confidence = raw_confidence

        exposure = EXPOSURE_MAP[regime]

        last_probs = posteriors[-1]
        probs_dict = {state_labels[i]: float(last_probs[i]) for i in range(N_STATES)}

        records.append({
            "date"               : dt,
            "regime"             : regime,
            "confidence"         : confidence,
            "exposure"           : exposure,
            "is_stable"          : is_stable,
            "raw_regime"         : raw_regime,
            "raw_confidence"     : raw_confidence,
            "smoothed_regime"    : smoothed_regime,
            "smoothed_confidence": smoothed_confidence,
            "prob_expansion"     : probs_dict.get("Expansion", 0.0),
            "prob_transition"    : probs_dict.get("Transition", 0.0),
            "prob_contraction"   : probs_dict.get("Contraction", 0.0),
        })

        print(f"  [regime-oos] {dt.date()} -> final={regime:<12} "
              f"raw={raw_regime:<12} smooth={smoothed_regime:<12} "
              f"conf={confidence:.2f} exp={exposure:.2f}")

    return pd.DataFrame(records).set_index("date")


# ─── Full regime history (daily, for analysis) ────────────────────────────────

def build_full_regime_history(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily regime series over the FULL date range.
    Used for visualization and regime analysis — NOT for backtest signals.

    Uses the same frozen model (trained on in-sample only) to classify
    every day including OOS and robustness. This is correct for analysis
    since we're just labeling history, not trading on it.
    """
    feature_cols = ["hy_oas", "yield_curve", "vix", "hy_oas_mom"]
    X_full       = features_df[feature_cols].dropna()

    X_insample   = X_full.loc[:INSAMPLE_END]
    model        = _train_hmm(X_insample.values)
    state_labels = _label_states(model, feature_cols)

    print("\n[regime] Emission means per state (z-scored):")
    print(f"{'State':<14} {'HY_OAS':>8} {'Yld_Crv':>8} {'VIX':>8} {'HY_Mom':>8}")
    for idx, label in state_labels.items():
        m = model.means_[idx]
        print(f"  {label:<12} {m[0]:>8.2f} {m[1]:>8.2f} {m[2]:>8.2f} {m[3]:>8.2f}")

    # Predict all states in one pass (Viterbi — most likely state sequence)
    hidden_states = model.predict(X_full.values)
    posteriors    = model.predict_proba(X_full.values)

    regime_series = pd.Series(
        [state_labels[s] for s in hidden_states],
        index=X_full.index,
        name="regime",
    )
    confidence_series = pd.Series(
        [float(posteriors[t, s]) for t, s in enumerate(hidden_states)],
        index=X_full.index,
        name="confidence",
    )
    exposure_series = regime_series.map(EXPOSURE_MAP).rename("exposure")

    history = pd.concat(
        [features_df[feature_cols], regime_series, confidence_series, exposure_series],
        axis=1,
    ).dropna(subset=["regime"])

    # Attach period labels
    period = features_df["period"].reindex(history.index)
    history["period"] = period

    return history


# ─── Validation helpers ───────────────────────────────────────────────────────

def validate_regime_history(history: pd.DataFrame) -> None:
    """
    Sanity-check the regime history against known macro events.

    These checks are observational — they tell you if the model is
    behaving sensibly. They are NOT tuning exercises. If the model
    misclassifies 2008, that is a problem with the model design,
    not a reason to change the exposure multipliers.

    Known ground truth:
        2008-09 to 2009-06 → should be Contraction (GFC)
        2010-01 to 2015-12 → should be mostly Expansion (post-GFC bull)
        2020-03 to 2020-05 → should be Contraction (COVID crash)
        2022-01 to 2022-12 → should be Contraction or Transition (inflation shock)
    """
    print("\n[regime] Sanity checks against known macro events:")
    print("=" * 60)

    checks = [
        # Hard checks: these MUST be defensive (Transition or Contraction)
        ("GFC Peak",       "2008-09-01", "2009-06-30", ["Contraction", "Transition"], 0.80, True),
        ("COVID Crash",    "2020-03-01", "2020-05-31", ["Contraction", "Transition"], 0.60, True),
        ("Inflation 2022", "2022-01-01", "2022-12-31", ["Contraction", "Transition"], 0.40, True),

        # Soft checks: these should NOT be mostly Contraction (would mean over-defensive)
        ("Post-GFC Bull",  "2013-01-01", "2015-12-31", ["Expansion", "Transition"],   0.60, False),
    ]

    all_pass = True
    for name, start, end, acceptable, min_pct, is_hard in checks:
        window = history.loc[start:end, "regime"]
        if window.empty:
            print(f"  {name:<20}: NO DATA")
            continue

        pct      = window.isin(acceptable).mean()
        dominant = window.value_counts().idxmax()
        status   = "PASS" if pct >= min_pct else "WARN"
        
        if status == "WARN" and is_hard:
            all_pass = False

        acc_str = " / ".join(acceptable)
        print(f"  {name:<20}: {dominant:<12} ({pct:.0%} {acc_str}) [{status}]")

    if all_pass:
        print("\n  [OK] All sanity checks passed -- regime model is behaving sensibly.")
    else:
        print("\n  [WARN] Some checks failed. Review HMM initialization or feature quality.")

    # Regime distribution
    print("\n[regime] Overall distribution:")
    dist = history["regime"].value_counts(normalize=True)
    for regime, pct in dist.items():
        print(f"  {regime:<12}: {pct:.1%}")

    # Average confidence per regime
    print("\n[regime] Average confidence per regime:")
    for regime in ["Expansion", "Transition", "Contraction"]:
        mask = history["regime"] == regime
        if mask.any():
            avg_conf = history.loc[mask, "confidence"].mean()
            print(f"  {regime:<12}: {avg_conf:.2f}")


def stress_test_regime_sensitivity(
    history     : pd.DataFrame,
    flip_pct    : float = 0.20,
    n_trials    : int   = 100,
    seed        : int   = 42,
) -> None:
    """
    Stress test: randomly flip flip_pct of regime labels and check
    how much the exposure series changes.

    If the exposure multiplier series is highly sensitive to random
    label flips, the regime signal is doing real work. If it barely
    changes, the regime is noise.

    This is a diagnostic only — it does not change any parameters.
    """
    rng = np.random.default_rng(seed)
    base_exposure = history["regime"].map(EXPOSURE_MAP)

    correlations = []
    for _ in range(n_trials):
        perturbed = history["regime"].copy()
        flip_idx  = rng.choice(len(perturbed), size=int(len(perturbed) * flip_pct), replace=False)
        all_regimes = list(EXPOSURE_MAP.keys())
        perturbed.iloc[flip_idx] = rng.choice(all_regimes, size=len(flip_idx))
        perturbed_exposure = perturbed.map(EXPOSURE_MAP)
        corr = base_exposure.corr(perturbed_exposure)
        correlations.append(corr)

    mean_corr = np.mean(correlations)
    print(f"\n[regime] Stress test — {flip_pct:.0%} random label flips over {n_trials} trials:")
    print(f"  Mean correlation of exposure series: {mean_corr:.3f}")
    if mean_corr < 0.85:
        print("  [OK] Regime signal is doing meaningful work (sensitive to label changes).")
    else:
        print("  [WARN] Exposure series barely changes under random flips -- regime may be near-constant.")


if __name__ == "__main__":
    from macro_data import load_macro_features, get_feature_cols

    # 1. Load features
    features_df = load_macro_features()

    # 2. Build full daily regime history (for analysis only)
    print("\n[regime] Building full regime history...")
    history = build_full_regime_history(features_df)

    # 3. Validate against known events
    validate_regime_history(history)

    # 4. Stress test
    stress_test_regime_sensitivity(history)

    # 5. Preview regime table
    print("\n[regime] Recent regime readings:")
    print(history[["regime", "confidence", "exposure"]].tail(20).to_string())
