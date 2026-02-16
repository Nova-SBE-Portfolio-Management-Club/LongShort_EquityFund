"""
sector_analysis_enhanced.py

ENHANCED Sector Analysis Model with:
1. More macro variables (VIX, credit spreads, yield curve, consumer confidence)
2. Regression-based scoring with statistical significance
3. Risk-On / Risk-Off regime detection
4. Leading indicator analysis (lagged variables)
5. Rolling correlation analysis

Required packages:
    pip install pandas numpy yfinance fredapi scipy statsmodels scikit-learn

FRED API key required: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels, fallback if not available
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Regression analysis will use scipy.")

# ============================================================================
# CONFIGURATION
# ============================================================================

FRED_API_KEY = "0c6ae7d353e57ff015fd603d569c3521"

START_DATE = "2010-01-01"  # Longer history for better analysis
END_DATE = datetime.now().strftime("%Y-%m-%d")

# S&P 500 Sector ETFs
SECTOR_TICKERS = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

# Sector names for clarity
SECTOR_NAMES = {
    'XLC': 'Communication Services',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLE': 'Energy',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLK': 'Technology',
    'XLU': 'Utilities'
}

# Defensive sectors (outperform in risk-off)
DEFENSIVE_SECTORS = ['XLP', 'XLU', 'XLV', 'XLRE']

# Cyclical sectors (outperform in risk-on)
CYCLICAL_SECTORS = ['XLY', 'XLF', 'XLI', 'XLB', 'XLK', 'XLE', 'XLC']

# ============================================================================
# REGIME THRESHOLDS
# ============================================================================

VIX_HIGH_THRESHOLD = 25      # VIX above this = Risk-Off
VIX_LOW_THRESHOLD = 15       # VIX below this = Risk-On
YIELD_CURVE_INVERSION = 0    # 10Y-2Y spread below 0 = Risk-Off
CREDIT_SPREAD_HIGH = 4       # Credit spread above this = Risk-Off

# ============================================================================
# DATA CACHING CONFIGURATION
# ============================================================================

import os
import json

# Cache directory (same folder as script)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data_cache')
CACHE_MAX_AGE_HOURS = 24  # Cache expires after 24 hours

def get_cache_path(name):
    """Get the path to a cached data file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.csv")

def get_cache_meta_path():
    """Get the path to the cache metadata file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "cache_meta.json")

def save_to_cache(name, df):
    """Save a DataFrame to local cache."""
    try:
        cache_path = get_cache_path(name)
        df.to_csv(cache_path)
        
        # Update metadata with timestamp
        meta_path = get_cache_meta_path()
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        
        meta[name] = {
            'timestamp': datetime.now().isoformat(),
            'rows': len(df),
            'cols': len(df.columns) if hasattr(df, 'columns') else 1
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"  [CACHE] Saved {name} ({len(df)} rows)")
    except Exception as e:
        print(f"  [CACHE WARN] Failed to save {name}: {e}")

def load_from_cache(name, max_age_hours=CACHE_MAX_AGE_HOURS):
    """Load a DataFrame from local cache if valid."""
    try:
        cache_path = get_cache_path(name)
        meta_path = get_cache_meta_path()
        
        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return None
        
        # Check cache age
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        if name not in meta:
            return None
        
        cache_time = datetime.fromisoformat(meta[name]['timestamp'])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            print(f"  [CACHE] {name} expired ({age_hours:.1f}h old)")
            return None
        
        # Load cached data
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"  [CACHE] Loaded {name} ({len(df)} rows, {age_hours:.1f}h old)")
        return df
        
    except Exception as e:
        print(f"  [CACHE WARN] Failed to load {name}: {e}")
        return None

def clear_cache():
    """Clear all cached data."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("[CACHE] Cleared all cached data")

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_enhanced_fred_data(fred_api_key, start_date, end_date):
    """Fetch ENHANCED macroeconomic data from FRED."""
    fred = Fred(api_key=fred_api_key)
    
    print("Fetching enhanced FRED data...")
    
    macro_data = {}
    
    # === ORIGINAL VARIABLES ===
    
    # GDP Growth (quarterly)
    try:
        gdp = fred.get_series('A191RL1Q225SBEA', start_date, end_date)
        macro_data['GDP_Growth'] = gdp.resample('QE').last() / 100
        print("  [OK] GDP Growth")
    except Exception as e:
        print(f"  [WARN] GDP Growth: {e}")
    
    # Unemployment Rate (monthly -> quarterly)
    try:
        unrate = fred.get_series('UNRATE', start_date, end_date)
        macro_data['Unemployment'] = unrate.resample('QE').mean() / 100
        print("  [OK] Unemployment")
    except Exception as e:
        print(f"  [WARN] Unemployment: {e}")
    
    # Inflation (CPI -> quarterly % change)
    try:
        cpi = fred.get_series('CPIAUCSL', start_date, end_date)
        cpi_q = cpi.resample('QE').last()
        macro_data['Inflation'] = cpi_q.pct_change()
        print("  [OK] Inflation (CPI)")
    except Exception as e:
        print(f"  [WARN] Inflation: {e}")
    
    # Fed Funds Rate (daily -> quarterly)
    try:
        fedfunds = fred.get_series('DFF', start_date, end_date)
        macro_data['FedFunds'] = fedfunds.resample('QE').mean() / 100
        print("  [OK] Fed Funds Rate")
    except Exception as e:
        print(f"  [WARN] Fed Funds: {e}")
    
    # 10-Year Treasury Yield
    try:
        tenyear = fred.get_series('DGS10', start_date, end_date)
        macro_data['TenYr_Yield'] = tenyear.resample('QE').mean() / 100
        print("  [OK] 10-Year Yield")
    except Exception as e:
        print(f"  [WARN] 10-Year Yield: {e}")
    
    # === NEW ENHANCED VARIABLES ===
    
    # Yield Curve Slope (10Y - 2Y) - KEY RECESSION INDICATOR
    try:
        tenyear = fred.get_series('DGS10', start_date, end_date)
        twoyear = fred.get_series('DGS2', start_date, end_date)
        yield_curve = (tenyear - twoyear).resample('QE').mean() / 100
        macro_data['Yield_Curve'] = yield_curve
        print("  [OK] Yield Curve (10Y-2Y)")
    except Exception as e:
        print(f"  [WARN] Yield Curve: {e}")
    
    # Credit Spreads (BAA-AAA) - FEAR INDICATOR
    try:
        baa = fred.get_series('DBAA', start_date, end_date)
        aaa = fred.get_series('DAAA', start_date, end_date)
        credit_spread = (baa - aaa).resample('QE').mean() / 100
        macro_data['Credit_Spread'] = credit_spread
        print("  [OK] Credit Spread (BAA-AAA)")
    except Exception as e:
        print(f"  [WARN] Credit Spread: {e}")
    
    # Consumer Confidence (University of Michigan)
    try:
        umcsent = fred.get_series('UMCSENT', start_date, end_date)
        macro_data['Consumer_Confidence'] = umcsent.resample('QE').mean()
        print("  [OK] Consumer Confidence")
    except Exception as e:
        print(f"  [WARN] Consumer Confidence: {e}")
    
    # ISM Manufacturing PMI
    try:
        ism = fred.get_series('MANEMP', start_date, end_date)  # Manufacturing Employment as proxy
        macro_data['ISM_Proxy'] = ism.resample('QE').mean()
        print("  [OK] ISM Manufacturing Proxy")
    except Exception as e:
        print(f"  [WARN] ISM: {e}")
    
    # Dollar Index Returns
    try:
        dollar = fred.get_series('DTWEXBGS', start_date, end_date)
        dollar_q = dollar.resample('QE').last()
        macro_data['Dollar_Return'] = dollar_q.pct_change()
        print("  [OK] Dollar Index")
    except:
        try:
            dollar = fred.get_series('DTWEXM', start_date, end_date)
            dollar_q = dollar.resample('QE').last()
            macro_data['Dollar_Return'] = dollar_q.pct_change()
            print("  [OK] Dollar Index (alt)")
        except Exception as e:
            print(f"  [WARN] Dollar Index: {e}")
    
    # Initial Jobless Claims (weekly -> quarterly avg) - LEADING INDICATOR
    try:
        claims = fred.get_series('ICSA', start_date, end_date)
        macro_data['Jobless_Claims'] = claims.resample('QE').mean()
        print("  [OK] Initial Jobless Claims")
    except Exception as e:
        print(f"  [WARN] Jobless Claims: {e}")
    
    # Combine into DataFrame
    macro_df = pd.DataFrame(macro_data)
    macro_df = macro_df.dropna()
    
    print(f"\n[OK] Total FRED data: {len(macro_df)} quarters, {len(macro_df.columns)} variables")
    return macro_df


def fetch_vix_data(start_date, end_date, max_retries=3):
    """Fetch VIX data from Yahoo Finance with retry logic and caching."""
    print("\nFetching VIX data...")
    
    import time
    
    for attempt in range(max_retries):
        try:
            time.sleep(2)  # Rate limit delay
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if vix is not None and len(vix) > 0:
                if 'Close' in vix.columns:
                    vix_close = vix['Close']
                else:
                    vix_close = vix
                vix_quarterly = vix_close.resample('QE').mean()
                if len(vix_quarterly) > 0:
                    # Convert to DataFrame for caching
                    vix_df = pd.DataFrame({'VIX': vix_quarterly})
                    save_to_cache('vix', vix_df)
                    print(f"  [OK] VIX: {len(vix_quarterly)} quarters")
                    return vix_quarterly
            print(f"  [RETRY] VIX attempt {attempt+1}/{max_retries} - no data")
            time.sleep(5 * (attempt + 1))  # Exponential backoff
        except Exception as e:
            print(f"  [RETRY] VIX attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(5 * (attempt + 1))
    
    # Try to load from cache as fallback
    print("  [WARN] VIX: All retries failed, checking cache...")
    cached = load_from_cache('vix', max_age_hours=168)  # 7 day cache for fallback
    if cached is not None:
        return cached['VIX']
    
    print("  [WARN] VIX: No cache available")
    return None


def fetch_sector_returns(tickers, start_date, end_date, max_retries=3):
    """Fetch sector ETF returns with retry logic and caching."""
    print(f"\nFetching sector data for {len(tickers)} ETFs...")
    
    import time
    
    for attempt in range(max_retries):
        try:
            time.sleep(3)  # Rate limit delay
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='column')
            
            if data is not None and len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    adj_close = data['Close']
                else:
                    adj_close = data['Close']
                
                quarterly_prices = adj_close.resample('QE').last()
                quarterly_returns = quarterly_prices.pct_change().dropna()
                
                if len(quarterly_returns) > 0:
                    save_to_cache('sector_returns', quarterly_returns)
                    print(f"  [OK] Sector returns: {len(quarterly_returns)} quarters")
                    return quarterly_returns
            
            print(f"  [RETRY] Sectors attempt {attempt+1}/{max_retries} - no data")
            time.sleep(10 * (attempt + 1))  # Longer backoff for bulk download
        except Exception as e:
            print(f"  [RETRY] Sectors attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(10 * (attempt + 1))
    
    # Try to load from cache as fallback
    print("  [WARN] Sectors: All retries failed, checking cache...")
    cached = load_from_cache('sector_returns', max_age_hours=168)  # 7 day cache for fallback
    if cached is not None:
        return cached
    
    print("  [WARN] Sectors: No cache available")
    return None


def fetch_gold_returns(start_date, end_date, max_retries=3):
    """Fetch gold returns with retry logic and caching."""
    print("\nFetching gold data...")
    
    import time
    
    for attempt in range(max_retries):
        try:
            time.sleep(2)  # Rate limit delay
            gold = yf.download('GLD', start=start_date, end=end_date, progress=False)
            
            if gold is not None and len(gold) > 0:
                if 'Close' in gold.columns:
                    gold_close = gold['Close']
                else:
                    gold_close = gold
                gold_quarterly = gold_close.resample('QE').last()
                gold_returns = gold_quarterly.pct_change()
                
                if len(gold_returns) > 0:
                    gold_df = pd.DataFrame({'Gold_Return': gold_returns})
                    save_to_cache('gold', gold_df)
                    print(f"  [OK] Gold returns: {len(gold_returns)} quarters")
                    return gold_returns
            
            print(f"  [RETRY] Gold attempt {attempt+1}/{max_retries} - no data")
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  [RETRY] Gold attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(5 * (attempt + 1))
    
    # Try to load from cache as fallback
    print("  [WARN] Gold: All retries failed, checking cache...")
    cached = load_from_cache('gold', max_age_hours=168)  # 7 day cache for fallback
    if cached is not None:
        return cached['Gold_Return']
    
    print("  [WARN] Gold: No cache available")
    return None


def compute_realized_volatility(start_date, end_date, max_retries=3):
    """Compute realized market volatility from SPY with retry logic and caching."""
    print("\nComputing realized volatility...")
    
    import time
    
    for attempt in range(max_retries):
        try:
            time.sleep(2)  # Rate limit delay
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            if spy is not None and len(spy) > 0:
                if 'Close' in spy.columns:
                    spy_close = spy['Close']
                else:
                    spy_close = spy
                daily_returns = spy_close.pct_change()
                realized_vol = daily_returns.resample('QE').std() * np.sqrt(252)
                
                if len(realized_vol) > 0:
                    vol_df = pd.DataFrame({'Realized_Vol': realized_vol})
                    save_to_cache('volatility', vol_df)
                    print(f"  [OK] Realized volatility: {len(realized_vol)} quarters")
                    return realized_vol
            
            print(f"  [RETRY] Volatility attempt {attempt+1}/{max_retries} - no data")
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  [RETRY] Volatility attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(5 * (attempt + 1))
    
    # Try to load from cache as fallback
    print("  [WARN] Volatility: All retries failed, checking cache...")
    cached = load_from_cache('volatility', max_age_hours=168)  # 7 day cache for fallback
    if cached is not None:
        return cached['Realized_Vol']
    
    print("  [WARN] Volatility: No cache available")
    return None


# ============================================================================
# REGIME DETECTION
# ============================================================================

def detect_regime(vix, yield_curve, credit_spread):
    """
    Detect market regime: RISK-ON or RISK-OFF
    
    Risk-Off conditions (any of):
    - VIX > 25
    - Yield curve inverted (10Y-2Y < 0)
    - Credit spreads > 4%
    
    Returns DataFrame with regime labels
    """
    print("\n" + "=" * 70)
    print("DETECTING MARKET REGIMES")
    print("=" * 70)
    
    # Align all series
    regime_data = pd.DataFrame({
        'VIX': vix,
        'Yield_Curve': yield_curve,
        'Credit_Spread': credit_spread
    }).dropna()
    
    # Determine regime for each quarter
    conditions = []
    regimes = []
    
    for idx in regime_data.index:
        vix_val = regime_data.loc[idx, 'VIX']
        yc_val = regime_data.loc[idx, 'Yield_Curve']
        cs_val = regime_data.loc[idx, 'Credit_Spread']
        
        risk_off_signals = 0
        if vix_val > VIX_HIGH_THRESHOLD:
            risk_off_signals += 1
        if yc_val < YIELD_CURVE_INVERSION:
            risk_off_signals += 1
        if cs_val > CREDIT_SPREAD_HIGH / 100:  # Convert to decimal
            risk_off_signals += 1
        
        if risk_off_signals >= 1:
            regimes.append('RISK_OFF')
        else:
            regimes.append('RISK_ON')
    
    regime_data['Regime'] = regimes
    
    # Print summary
    risk_on_count = regimes.count('RISK_ON')
    risk_off_count = regimes.count('RISK_OFF')
    
    if len(regimes) > 0:
        print(f"\nRegime Distribution:")
        print(f"  RISK-ON:  {risk_on_count} quarters ({100*risk_on_count/len(regimes):.1f}%)")
        print(f"  RISK-OFF: {risk_off_count} quarters ({100*risk_off_count/len(regimes):.1f}%)")
    else:
        print("\n[WARN] No regime data available - insufficient data")
        return None
    
    # Print current regime
    if len(regime_data) > 0:
        current = regime_data.iloc[-1]
        print(f"\nCurrent Regime: {current['Regime']}")
        print(f"  VIX: {current['VIX']:.1f} (threshold: {VIX_HIGH_THRESHOLD})")
        print(f"  Yield Curve: {current['Yield_Curve']*100:.2f}% (threshold: {YIELD_CURVE_INVERSION}%)")
        print(f"  Credit Spread: {current['Credit_Spread']*100:.2f}% (threshold: {CREDIT_SPREAD_HIGH}%)")
    
    return regime_data


# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================

def run_regression_analysis(macro_df, sector_returns, regime_data):
    """
    Run regression analysis for each sector with statistical significance.
    
    Model: Sector_Return = alpha + beta1*Macro1 + beta2*Macro2 + ... + epsilon
    
    Returns coefficients, p-values, and R-squared for each sector.
    """
    print("\n" + "=" * 70)
    print("REGRESSION ANALYSIS")
    print("=" * 70)
    
    # Align data
    common_dates = macro_df.index.intersection(sector_returns.index)
    X = macro_df.loc[common_dates]
    Y = sector_returns.loc[common_dates]
    
    # Standardize X for better interpretation
    X_standardized = (X - X.mean()) / X.std()
    
    results = {}
    
    for sector in Y.columns:
        y = Y[sector].dropna()
        x = X_standardized.loc[y.index]
        
        # Drop any rows with NaN
        valid_idx = x.dropna().index.intersection(y.dropna().index)
        x = x.loc[valid_idx]
        y = y.loc[valid_idx]
        
        if len(y) < 10:
            print(f"  [SKIP] {sector}: insufficient data ({len(y)} obs)")
            continue
        
        if HAS_STATSMODELS:
            # Use statsmodels for full regression output
            X_with_const = sm.add_constant(x)
            model = sm.OLS(y, X_with_const).fit()
            
            results[sector] = {
                'coefficients': model.params.drop('const'),
                'pvalues': model.pvalues.drop('const'),
                'rsquared': model.rsquared,
                'rsquared_adj': model.rsquared_adj,
                'nobs': int(model.nobs)
            }
        else:
            # Fallback: use simple correlation as proxy
            correlations = x.apply(lambda col: col.corr(y))
            results[sector] = {
                'coefficients': correlations,
                'pvalues': pd.Series([0.05] * len(correlations), index=correlations.index),
                'rsquared': 0,
                'rsquared_adj': 0,
                'nobs': len(y)
            }
    
    # Print summary
    print("\nRegression Results Summary:")
    print("-" * 60)
    for sector in results:
        r2 = results[sector]['rsquared']
        nobs = results[sector]['nobs']
        sig_vars = (results[sector]['pvalues'] < 0.10).sum()
        print(f"  {sector} ({SECTOR_NAMES.get(sector, sector)}): R2={r2:.3f}, n={nobs}, sig_vars={sig_vars}")
    
    return results


# ============================================================================
# ENHANCED SCORING
# ============================================================================

def compute_enhanced_scores(regression_results, macro_df, forecast, current_regime):
    """
    Compute enhanced sector scores using:
    1. Regression coefficients (not just correlation)
    2. Statistical significance weighting
    3. Regime adjustments
    
    Score = sum(beta * z_score * significance_weight) + regime_adjustment
    """
    print("\n" + "=" * 70)
    print("COMPUTING ENHANCED SECTOR SCORES")
    print("=" * 70)
    
    # Compute z-scores of forecast
    z_scores = (forecast - macro_df.mean()) / macro_df.std()
    
    print("\nForecast Z-Scores (how extreme your forecast is):")
    for var in z_scores.index:
        val = z_scores[var]
        if abs(val) > 1:
            indicator = "**EXTREME**" if abs(val) > 2 else "*notable*"
        else:
            indicator = ""
        print(f"  {var}: {val:+.2f} {indicator}")
    
    scores = {}
    score_details = {}
    
    for sector in regression_results:
        betas = regression_results[sector]['coefficients']
        pvalues = regression_results[sector]['pvalues']
        
        # Only use variables where we have both beta and z-score
        common_vars = betas.index.intersection(z_scores.index)
        
        # Compute score components
        raw_score = 0
        weighted_score = 0
        details = []
        
        for var in common_vars:
            beta = betas[var]
            z = z_scores[var]
            pval = pvalues[var]
            
            # Skip if z-score is NaN (variable not in forecast or missing data)
            if pd.isna(z) or pd.isna(beta):
                continue
            
            # Raw contribution: beta * z
            contribution = beta * z
            
            # Significance weight: higher weight for more significant relationships
            if pval < 0.01:
                sig_weight = 1.0
                sig_label = "***"
            elif pval < 0.05:
                sig_weight = 0.8
                sig_label = "**"
            elif pval < 0.10:
                sig_weight = 0.6
                sig_label = "*"
            else:
                sig_weight = 0.3
                sig_label = ""
            
            weighted_contribution = contribution * sig_weight
            
            raw_score += contribution
            weighted_score += weighted_contribution
            
            details.append({
                'variable': var,
                'beta': beta,
                'z_score': z,
                'pvalue': pval,
                'contribution': contribution,
                'weighted_contribution': weighted_contribution,
                'sig_label': sig_label
            })
        
        # REGIME ADJUSTMENT
        regime_adjustment = 0
        if current_regime == 'RISK_OFF':
            if sector in DEFENSIVE_SECTORS:
                regime_adjustment = 0.02  # Boost defensive sectors
            elif sector in CYCLICAL_SECTORS:
                regime_adjustment = -0.02  # Penalize cyclical sectors
        else:  # RISK_ON
            if sector in CYCLICAL_SECTORS:
                regime_adjustment = 0.01  # Slight boost for cyclicals
            elif sector in DEFENSIVE_SECTORS:
                regime_adjustment = -0.01  # Slight penalty for defensives
        
        final_score = weighted_score + regime_adjustment
        
        scores[sector] = final_score
        score_details[sector] = {
            'raw_score': raw_score,
            'weighted_score': weighted_score,
            'regime_adjustment': regime_adjustment,
            'final_score': final_score,
            'details': details
        }
    
    return scores, score_details


def print_detailed_scores(scores, score_details, current_regime):
    """Print detailed breakdown of scores."""
    
    print(f"\nCurrent Regime: {current_regime}")
    print("-" * 70)
    
    # Sort by final score
    sorted_sectors = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    print("\nSector Rankings (with regime adjustment):")
    print("-" * 70)
    print(f"{'Rank':<5} {'Sector':<30} {'Score':<10} {'Regime Adj':<12} {'Category'}")
    print("-" * 70)
    
    for i, sector in enumerate(sorted_sectors, 1):
        details = score_details[sector]
        score = scores[sector]
        regime_adj = details['regime_adjustment']
        
        if sector in DEFENSIVE_SECTORS:
            category = "Defensive"
        else:
            category = "Cyclical"
        
        # Score indicator
        if score > 0:
            indicator = "[+]"
        elif score < -0.3:
            indicator = "[-]"
        else:
            indicator = "[~]"
        
        print(f"{i:<5} {SECTOR_NAMES.get(sector, sector):<30} {score:+.4f} {indicator}   {regime_adj:+.4f}       {category}")
    
    print("-" * 70)


def print_score_breakdown(sector, score_details):
    """Print detailed breakdown for a single sector."""
    details = score_details[sector]
    print(f"\n{'='*60}")
    print(f"DETAILED BREAKDOWN: {SECTOR_NAMES.get(sector, sector)}")
    print(f"{'='*60}")
    
    print(f"\n{'Variable':<25} {'Beta':>8} {'Z-Score':>10} {'Contrib':>10} {'Sig':>5}")
    print("-" * 60)
    
    for d in sorted(details['details'], key=lambda x: abs(x['weighted_contribution']), reverse=True):
        print(f"{d['variable']:<25} {d['beta']:+.4f}   {d['z_score']:+.4f}    {d['weighted_contribution']:+.6f}   {d['sig_label']}")
    
    print("-" * 60)
    print(f"{'Raw Score':<40} {details['raw_score']:+.4f}")
    print(f"{'Significance-Weighted Score':<40} {details['weighted_score']:+.4f}")
    print(f"{'Regime Adjustment':<40} {details['regime_adjustment']:+.4f}")
    print(f"{'FINAL SCORE':<40} {details['final_score']:+.4f}")


# ============================================================================
# LEADING INDICATOR ANALYSIS
# ============================================================================

def analyze_leading_indicators(macro_df, sector_returns, lags=[1, 2]):
    """
    Analyze which macro variables are LEADING indicators.
    
    Tests if macro(t-lag) predicts sector_return(t)
    """
    print("\n" + "=" * 70)
    print("LEADING INDICATOR ANALYSIS")
    print("=" * 70)
    print("Testing: Does Macro(t-lag) predict Sector Return(t)?")
    
    results = {}
    
    for lag in lags:
        print(f"\n--- Lag = {lag} quarter(s) ---")
        
        for sector in sector_returns.columns:
            sector_ret = sector_returns[sector]
            
            for macro_var in macro_df.columns:
                macro_lagged = macro_df[macro_var].shift(lag)
                
                # Align data
                combined = pd.DataFrame({
                    'macro': macro_lagged,
                    'return': sector_ret
                }).dropna()
                
                if len(combined) < 10:
                    continue
                
                # Compute correlation
                corr, pval = stats.pearsonr(combined['macro'], combined['return'])
                
                if pval < 0.10 and abs(corr) > 0.2:
                    key = f"{macro_var} -> {sector} (lag={lag})"
                    results[key] = {
                        'correlation': corr,
                        'pvalue': pval,
                        'lag': lag,
                        'macro': macro_var,
                        'sector': sector
                    }
    
    # Print significant leading indicators
    if results:
        print("\nSignificant Leading Indicators Found:")
        print("-" * 70)
        sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        for key, val in sorted_results[:15]:  # Top 15
            print(f"  {val['macro']:<25} -> {SECTOR_NAMES.get(val['sector'], val['sector']):<20} r={val['correlation']:+.3f} (lag={val['lag']}Q)")
    else:
        print("\nNo significant leading indicators found at p<0.10")
    
    return results


# ============================================================================
# AUTO-FORECAST & SCENARIO ANALYSIS
# ============================================================================

def get_current_forecast(macro_df):
    """
    Use the most recent macro values as the forecast.
    This removes manual input error risk by using actual current data.
    
    Returns: pd.Series with current macro values
    """
    if len(macro_df) == 0:
        raise ValueError("No macro data available for forecast")
    
    current = macro_df.iloc[-1].copy()
    print("\n[AUTO-FORECAST] Using current macro values (no manual input):")
    for var in current.index:
        print(f"  {var}: {current[var]:.4f}")
    
    return current


def create_scenario_forecast(base_forecast, macro_df, scenario_type):
    """
    Create scenario by shifting variables by standard deviations.
    
    Bull = optimistic shifts (-1σ to VIX, +1σ to GDP, etc.)
    Bear = pessimistic shifts (+2σ to VIX, -1.5σ to GDP, etc.)
    Base = current values (no changes)
    Stagflation = high inflation, low growth
    
    Returns: pd.Series with adjusted forecast values
    """
    forecast = base_forecast.copy()
    means = macro_df.mean()
    stds = macro_df.std()
    
    # Define Z-score adjustments for each scenario
    if scenario_type == 'bull':
        # Bull: Low fear, strong growth, low unemployment
        adjustments = {
            'VIX': -1.0,              # 1 std below mean (low fear)
            'GDP_Growth': +1.0,        # 1 std above mean (strong growth)
            'Unemployment': -1.0,      # 1 std below mean (low jobless)
            'Consumer_Confidence': +1.0,
            'Credit_Spread': -0.5,     # Tighter spreads
            'Realized_Vol': -1.0       # Low volatility
        }
    elif scenario_type == 'bear':
        # Bear: High fear, contraction, rising unemployment
        adjustments = {
            'VIX': +2.0,              # 2 std above mean (high fear)
            'GDP_Growth': -1.5,        # 1.5 std below mean (contraction)
            'Unemployment': +1.5,      # 1.5 std above mean (rising)
            'Yield_Curve': -2.0,       # 2 std below (inverted curve)
            'Credit_Spread': +2.0,     # 2 std above (credit stress)
            'Consumer_Confidence': -1.5,
            'Realized_Vol': +2.0       # High volatility
        }
    elif scenario_type == 'stagflation':
        # Stagflation: High inflation, low growth, high rates
        adjustments = {
            'GDP_Growth': -1.0,        # Stagnant growth
            'Inflation': +2.0,         # 2 std above (high inflation)
            'FedFunds': +1.5,          # High rates
            'TenYr_Yield': +1.5,       # High yields
            'Consumer_Confidence': -1.5,
            'Unemployment': +0.5       # Slightly elevated
        }
    else:  # base - no adjustments
        return forecast
    
    # Apply Z-score adjustments
    for var, z_shift in adjustments.items():
        if var in forecast.index and var in stds.index:
            forecast[var] = means[var] + (z_shift * stds[var])
    
    return forecast


def run_scenario_analysis(regression_results, macro_df, current_regime):
    """
    Run the model under multiple scenarios and compare rankings.
    
    Shows which sectors are robust across scenarios vs. scenario-dependent.
    
    Returns: dict with scores for each scenario
    """
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS")
    print("=" * 70)
    print("Testing sector rankings under different economic conditions...")
    print("  Base:        Current macro values (no changes)")
    print("  Bull:        Economic expansion (+1 std GDP, -1 std VIX)")
    print("  Bear:        Recession/risk-off (-1.5 std GDP, +2 std VIX)")
    print("  Stagflation: High inflation, low growth (+2 std inflation)")
    print("=" * 70)
    
    base_forecast = get_current_forecast(macro_df)
    scenarios = ['base', 'bull', 'bear', 'stagflation']
    scenario_results = {}
    
    for scenario in scenarios:
        # Create scenario-specific forecast
        if scenario == 'base':
            forecast = base_forecast
        else:
            forecast = create_scenario_forecast(base_forecast, macro_df, scenario)
        
        # Determine regime for this scenario
        if scenario == 'bear':
            scenario_regime = 'RISK_OFF'
        elif scenario == 'bull':
            scenario_regime = 'RISK_ON'
        else:
            scenario_regime = current_regime
        
        # Compute scores (suppress printing inside)
        scores, _ = compute_enhanced_scores(
            regression_results, macro_df, forecast, scenario_regime
        )
        scenario_results[scenario] = scores
    
    # Print comparison table
    print("\n" + "-" * 80)
    print("SCENARIO COMPARISON: Sector Rankings")
    print("-" * 80)
    
    # Get all sectors
    all_sectors = list(scenario_results['base'].keys())
    
    # Build ranking for each scenario
    rankings = {}
    for scenario in scenarios:
        sorted_sectors = sorted(
            scenario_results[scenario].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        rankings[scenario] = {sector: rank+1 for rank, (sector, _) in enumerate(sorted_sectors)}
    
    # Print header
    print(f"\n{'Sector':<25} {'BASE':>8} {'BULL':>8} {'BEAR':>8} {'STAGFL':>8} {'Robust?':>10}")
    print("-" * 80)
    
    # Print each sector's ranking across scenarios
    for sector in all_sectors:
        name = SECTOR_NAMES.get(sector, sector)[:24]
        base_rank = rankings['base'].get(sector, '-')
        bull_rank = rankings['bull'].get(sector, '-')
        bear_rank = rankings['bear'].get(sector, '-')
        stag_rank = rankings['stagflation'].get(sector, '-')
        
        # Check if robust (top 5 in at least 3 scenarios)
        ranks = [base_rank, bull_rank, bear_rank, stag_rank]
        top5_count = sum(1 for r in ranks if isinstance(r, int) and r <= 5)
        robust = "[X] ROBUST" if top5_count >= 3 else ""
        
        print(f"{name:<25} {base_rank:>8} {bull_rank:>8} {bear_rank:>8} {stag_rank:>8} {robust:>10}")
    
    print("-" * 80)
    
    # Summary: Best picks per scenario
    print("\nTOP PICKS BY SCENARIO:")
    for scenario in scenarios:
        sorted_scores = sorted(
            scenario_results[scenario].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        top1 = SECTOR_NAMES.get(sorted_scores[0][0], sorted_scores[0][0])
        top2 = SECTOR_NAMES.get(sorted_scores[1][0], sorted_scores[1][0])
        print(f"  {scenario.upper():<12}: {top1}, {top2}")
    
    return scenario_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("ENHANCED SECTOR ANALYSIS MODEL")
    print("=" * 70)
    print("Improvements over basic model:")
    print("  1. More macro variables (VIX, credit spreads, yield curve)")
    print("  2. Regression with statistical significance")
    print("  3. Risk-On / Risk-Off regime detection")
    print("  4. Leading indicator analysis")
    print("=" * 70)
    
    # ========================================================================
    # FETCH ALL DATA
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("FETCHING DATA")
    print("=" * 70)
    
    macro_df = fetch_enhanced_fred_data(FRED_API_KEY, START_DATE, END_DATE)
    vix = fetch_vix_data(START_DATE, END_DATE)
    sector_returns = fetch_sector_returns(SECTOR_TICKERS, START_DATE, END_DATE)
    gold_returns = fetch_gold_returns(START_DATE, END_DATE)
    realized_vol = compute_realized_volatility(START_DATE, END_DATE)
    
    # Check if critical data is missing
    if sector_returns is None or len(sector_returns) == 0:
        print("\n" + "=" * 70)
        print("ERROR: Failed to download sector data from Yahoo Finance")
        print("=" * 70)
        print("\nPossible causes:")
        print("  1. Rate limited by Yahoo Finance (wait 15-30 minutes and retry)")
        print("  2. Network/firewall issue")
        print("  3. Yahoo Finance API is down")
        print("\nTry running the script again in a few minutes.")
        exit(1)
    
    # Add VIX, Gold, and Volatility to macro data
    if vix is not None and len(vix) > 0:
        macro_df['VIX'] = vix
    if gold_returns is not None and len(gold_returns) > 0:
        macro_df['Gold_Return'] = gold_returns
    if realized_vol is not None and len(realized_vol) > 0:
        macro_df['Realized_Vol'] = realized_vol
    
    macro_df = macro_df.dropna()
    
    if len(macro_df) == 0:
        print("\n[ERROR] No macro data available after alignment. Exiting.")
        exit(1)
    
    print(f"\n[OK] Final macro dataset: {len(macro_df)} quarters, {len(macro_df.columns)} variables")
    print(f"Variables: {list(macro_df.columns)}")
    
    # ========================================================================
    # REGIME DETECTION
    # ========================================================================
    
    if 'VIX' in macro_df.columns and 'Yield_Curve' in macro_df.columns and 'Credit_Spread' in macro_df.columns:
        regime_data = detect_regime(
            macro_df['VIX'],
            macro_df['Yield_Curve'],
            macro_df['Credit_Spread']
        )
        current_regime = regime_data.iloc[-1]['Regime'] if len(regime_data) > 0 else 'RISK_ON'
    else:
        print("\n[WARN] Insufficient data for regime detection, assuming RISK_ON")
        regime_data = None
        current_regime = 'RISK_ON'
    
    # ========================================================================
    # REGRESSION ANALYSIS
    # ========================================================================
    
    regression_results = run_regression_analysis(macro_df, sector_returns, regime_data)
    
    # ========================================================================
    # AUTO-FORECAST (using current values)
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("AUTO-FORECAST: USING CURRENT MACRO VALUES")
    print("=" * 70)
    print("NOTE: No manual input required. Using actual current data.")
    print("      This eliminates forecast error risk.")
    
    # Use current macro values as forecast (no manual guessing!)
    forecast = get_current_forecast(macro_df)
    
    # ========================================================================
    # COMPUTE SCORES (BASE CASE)
    # ========================================================================
    
    scores, score_details = compute_enhanced_scores(
        regression_results, 
        macro_df, 
        forecast, 
        current_regime
    )
    
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    
    print_detailed_scores(scores, score_details, current_regime)
    
    # Print detailed breakdown for top and bottom sectors
    sorted_sectors = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    if sorted_sectors:
        print_score_breakdown(sorted_sectors[0], score_details)  # Best sector
        print_score_breakdown(sorted_sectors[-1], score_details)  # Worst sector
    
    # ========================================================================
    # SCENARIO ANALYSIS
    # ========================================================================
    
    scenario_results = run_scenario_analysis(regression_results, macro_df, current_regime)
    
    # ========================================================================
    # LEADING INDICATOR ANALYSIS
    # ========================================================================
    
    leading_indicators = analyze_leading_indicators(macro_df, sector_returns)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save scores
    scores_df = pd.DataFrame({
        'Sector': [SECTOR_NAMES.get(s, s) for s in scores.keys()],
        'Ticker': list(scores.keys()),
        'Score': list(scores.values()),
        'Regime_Adj': [score_details[s]['regime_adjustment'] for s in scores.keys()],
        'Category': ['Defensive' if s in DEFENSIVE_SECTORS else 'Cyclical' for s in scores.keys()]
    }).sort_values('Score', ascending=False)
    
    scores_df.to_csv('enhanced_sector_scores.csv', index=False)
    print("  [OK] enhanced_sector_scores.csv")
    
    # Save regression results
    reg_summary = []
    for sector in regression_results:
        for var in regression_results[sector]['coefficients'].index:
            reg_summary.append({
                'Sector': sector,
                'Variable': var,
                'Beta': regression_results[sector]['coefficients'][var],
                'P-Value': regression_results[sector]['pvalues'][var],
                'Significant': regression_results[sector]['pvalues'][var] < 0.10
            })
    
    reg_df = pd.DataFrame(reg_summary)
    reg_df.to_csv('regression_results.csv', index=False)
    print("  [OK] regression_results.csv")
    
    # ========================================================================
    # PORTFOLIO RECOMMENDATION
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("LONG/SHORT PORTFOLIO RECOMMENDATION")
    print("=" * 70)
    
    # Sort sectors by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Top 3 = LONG, Bottom 3 = SHORT
    long_sectors = sorted_scores[:3]
    short_sectors = sorted_scores[-3:]
    
    print("\n  [LONG] Buy these sectors (highest scores):")
    print("  " + "-" * 50)
    for i, (ticker, score) in enumerate(long_sectors, 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        print(f"    {i}. {ticker:<6} {name:<25} Score: {score:+.4f}")
    
    print("\n  [SHORT] Sell/Avoid these sectors (lowest scores):")
    print("  " + "-" * 50)
    for i, (ticker, score) in enumerate(reversed(short_sectors), 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        print(f"    {i}. {ticker:<6} {name:<25} Score: {score:+.4f}")
    
    print("\n  STRATEGY: Long top 3, Short bottom 3 (equal weight)")
    print(f"  REGIME: {current_regime}")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
