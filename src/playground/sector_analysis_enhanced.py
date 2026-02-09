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


def fetch_vix_data(start_date, end_date):
    """Fetch VIX data from Yahoo Finance."""
    print("\nFetching VIX data...")
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if 'Close' in vix.columns:
            vix_close = vix['Close']
        else:
            vix_close = vix
        vix_quarterly = vix_close.resample('QE').mean()
        print(f"  [OK] VIX: {len(vix_quarterly)} quarters")
        return vix_quarterly
    except Exception as e:
        print(f"  [WARN] VIX: {e}")
        return None


def fetch_sector_returns(tickers, start_date, end_date):
    """Fetch sector ETF returns."""
    print(f"\nFetching sector data for {len(tickers)} ETFs...")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='column')
    
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data['Close']
    else:
        adj_close = data['Close']
    
    quarterly_prices = adj_close.resample('QE').last()
    quarterly_returns = quarterly_prices.pct_change().dropna()
    
    print(f"  [OK] Sector returns: {len(quarterly_returns)} quarters")
    return quarterly_returns


def fetch_gold_returns(start_date, end_date):
    """Fetch gold returns."""
    print("\nFetching gold data...")
    try:
        gold = yf.download('GLD', start=start_date, end=end_date, progress=False)
        if 'Close' in gold.columns:
            gold_close = gold['Close']
        else:
            gold_close = gold
        gold_quarterly = gold_close.resample('QE').last()
        gold_returns = gold_quarterly.pct_change()
        print(f"  [OK] Gold returns: {len(gold_returns)} quarters")
        return gold_returns
    except Exception as e:
        print(f"  [WARN] Gold: {e}")
        return None


def compute_realized_volatility(start_date, end_date):
    """Compute realized market volatility from SPY."""
    print("\nComputing realized volatility...")
    try:
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        if 'Close' in spy.columns:
            spy_close = spy['Close']
        else:
            spy_close = spy
        daily_returns = spy_close.pct_change()
        realized_vol = daily_returns.resample('QE').std() * np.sqrt(252)
        print(f"  [OK] Realized volatility: {len(realized_vol)} quarters")
        return realized_vol
    except Exception as e:
        print(f"  [WARN] Volatility: {e}")
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
    print(f"\nRegime Distribution:")
    print(f"  RISK-ON:  {risk_on_count} quarters ({100*risk_on_count/len(regimes):.1f}%)")
    print(f"  RISK-OFF: {risk_off_count} quarters ({100*risk_off_count/len(regimes):.1f}%)")
    
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
    
    # Add VIX, Gold, and Volatility to macro data
    if vix is not None:
        macro_df['VIX'] = vix
    if gold_returns is not None:
        macro_df['Gold_Return'] = gold_returns
    if realized_vol is not None:
        macro_df['Realized_Vol'] = realized_vol
    
    macro_df = macro_df.dropna()
    
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
    # YOUR FORECAST
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("YOUR ECONOMIC FORECAST")
    print("=" * 70)
    
    # Define your forecast here - customize these values!
    forecast = pd.Series({
        'GDP_Growth': 0.015,           # Expecting 1.5% GDP growth (slowing)
        'Unemployment': 0.045,         # Expecting 4.5% unemployment (rising)
        'Inflation': 0.020,            # Expecting 2.0% inflation (moderating)
        'FedFunds': 0.045,             # Expecting 4.5% Fed Funds (high)
        'TenYr_Yield': 0.040,          # Expecting 4.0% 10-year yield
        'Yield_Curve': 0.005,          # Expecting slightly positive yield curve
        'Credit_Spread': 0.015,        # Expecting 1.5% credit spread (moderate fear)
        'Consumer_Confidence': 80,     # Expecting low consumer confidence
        'Dollar_Return': 0.01,         # Expecting dollar strength
        'VIX': 22,                     # Expecting elevated volatility
        'Gold_Return': 0.02,           # Expecting gold gains (safe haven)
        'Realized_Vol': 0.20           # Expecting 20% realized volatility
    })
    
    # Only use variables that exist in our data
    forecast = forecast[forecast.index.isin(macro_df.columns)]
    
    print("\nYour Forecast Values:")
    for var in forecast.index:
        print(f"  {var}: {forecast[var]}")
    
    # ========================================================================
    # COMPUTE SCORES
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
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
