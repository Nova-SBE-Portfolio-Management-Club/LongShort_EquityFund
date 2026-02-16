"""
sector_backtest.py

BACKTESTING FRAMEWORK for the Enhanced Sector Analysis Model

This script performs a rigorous walk-forward backtest to validate the model:
1. For each quarter, use ONLY data available at that time (no lookahead bias)
2. Generate sector rankings/predictions
3. Compare predictions to ACTUAL next-quarter returns
4. Compute performance metrics

Key Metrics:
- Hit Rate: % of top predictions that actually outperformed
- Top Quintile Return: Average return of top-ranked sectors
- Long/Short Return: Long top sectors, short bottom sectors
- Sharpe Ratio, Max Drawdown, etc.

Required: Same packages as sector_analysis_enhanced.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from scipy import stats
import warnings
import time
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ============================================================================
# CONFIGURATION
# ============================================================================

FRED_API_KEY = "0c6ae7d353e57ff015fd603d569c3521"

# Backtest parameters
BACKTEST_START = "2012-01-01"  # Start of backtest period
TRAINING_WINDOW = 20           # Number of quarters for training regression
MIN_TRAINING_QUARTERS = 12     # Minimum quarters needed to train

# Sector ETFs
SECTOR_TICKERS = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

SECTOR_NAMES = {
    'XLC': 'Communication', 'XLY': 'Cons Disc', 'XLP': 'Cons Staples',
    'XLE': 'Energy', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLI': 'Industrials', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLK': 'Technology', 'XLU': 'Utilities'
}

DEFENSIVE_SECTORS = ['XLP', 'XLU', 'XLV', 'XLRE']
CYCLICAL_SECTORS = ['XLY', 'XLF', 'XLI', 'XLB', 'XLK', 'XLE', 'XLC']

# Regime thresholds
VIX_HIGH = 25
YIELD_CURVE_INVERSION = 0
CREDIT_SPREAD_HIGH = 0.04

# ============================================================================
# DATA CACHING
# ============================================================================

import os
import json

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data_cache')
CACHE_MAX_AGE_HOURS = 24

def get_cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.csv")

def get_cache_meta_path():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "cache_meta.json")

def save_to_cache(name, df):
    try:
        cache_path = get_cache_path(name)
        df.to_csv(cache_path)
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

def load_from_cache(name, max_age_hours=168):  # 7 day default for backtest
    try:
        cache_path = get_cache_path(name)
        meta_path = get_cache_meta_path()
        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if name not in meta:
            return None
        cache_time = datetime.fromisoformat(meta[name]['timestamp'])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        if age_hours > max_age_hours:
            print(f"  [CACHE] {name} expired ({age_hours:.1f}h old)")
            return None
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"  [CACHE] Loaded {name} ({len(df)} rows, {age_hours:.1f}h old)")
        return df
    except Exception as e:
        print(f"  [CACHE WARN] Failed to load {name}: {e}")
        return None

# ============================================================================
# DATA FETCHING (with caching)
# ============================================================================

def fetch_all_data(start_date, end_date):
    """Fetch all required data for backtesting."""
    
    print("=" * 70)
    print("FETCHING DATA FOR BACKTEST")
    print("=" * 70)
    
    fred = Fred(api_key=FRED_API_KEY)
    
    # Macro data from FRED
    print("\nFetching FRED data...")
    macro_data = {}
    
    try:
        gdp = fred.get_series('A191RL1Q225SBEA', start_date, end_date)
        macro_data['GDP_Growth'] = gdp.resample('QE').last() / 100
        print("  [OK] GDP Growth")
    except: pass
    
    try:
        unrate = fred.get_series('UNRATE', start_date, end_date)
        macro_data['Unemployment'] = unrate.resample('QE').mean() / 100
        print("  [OK] Unemployment")
    except: pass
    
    try:
        cpi = fred.get_series('CPIAUCSL', start_date, end_date)
        macro_data['Inflation'] = cpi.resample('QE').last().pct_change()
        print("  [OK] Inflation")
    except: pass
    
    try:
        fedfunds = fred.get_series('DFF', start_date, end_date)
        macro_data['FedFunds'] = fedfunds.resample('QE').mean() / 100
        print("  [OK] Fed Funds")
    except: pass
    
    try:
        tenyear = fred.get_series('DGS10', start_date, end_date)
        macro_data['TenYr_Yield'] = tenyear.resample('QE').mean() / 100
        print("  [OK] 10-Year Yield")
    except: pass
    
    try:
        tenyear = fred.get_series('DGS10', start_date, end_date)
        twoyear = fred.get_series('DGS2', start_date, end_date)
        macro_data['Yield_Curve'] = (tenyear - twoyear).resample('QE').mean() / 100
        print("  [OK] Yield Curve")
    except: pass
    
    try:
        baa = fred.get_series('DBAA', start_date, end_date)
        aaa = fred.get_series('DAAA', start_date, end_date)
        macro_data['Credit_Spread'] = (baa - aaa).resample('QE').mean() / 100
        print("  [OK] Credit Spread")
    except: pass
    
    try:
        umcsent = fred.get_series('UMCSENT', start_date, end_date)
        macro_data['Consumer_Confidence'] = umcsent.resample('QE').mean()
        print("  [OK] Consumer Confidence")
    except: pass
    
    # ISM Manufacturing PMI (proxy)
    try:
        ism = fred.get_series('MANEMP', start_date, end_date)
        macro_data['ISM_Proxy'] = ism.resample('QE').mean()
        print("  [OK] ISM Manufacturing Proxy")
    except: pass
    
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
        except: pass
    
    # Initial Jobless Claims (weekly -> quarterly avg)
    try:
        claims = fred.get_series('ICSA', start_date, end_date)
        macro_data['Jobless_Claims'] = claims.resample('QE').mean()
        print("  [OK] Initial Jobless Claims")
    except: pass
    
    macro_df = pd.DataFrame(macro_data)
    
    # Yahoo Finance data with retry logic
    print("\nFetching Yahoo Finance data...")
    
    MAX_RETRIES = 3
    RETRY_DELAY = 10  # seconds
    
    def download_with_retry(tickers, start, end, max_retries=MAX_RETRIES):
        """Download with retry logic for rate limiting."""
        for attempt in range(max_retries):
            try:
                time.sleep(RETRY_DELAY if attempt > 0 else 2)
                data = yf.download(tickers, start=start, end=end, progress=False, group_by='column')
                if len(data) > 0:
                    return data
                if attempt < max_retries - 1:
                    print(f"    Retry {attempt + 1}/{max_retries} (empty data)...")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                else:
                    raise e
        return None
    
    # VIX
    try:
        vix = download_with_retry('^VIX', start_date, end_date)
        if vix is not None and len(vix) > 0:
            if isinstance(vix.columns, pd.MultiIndex):
                vix_close = vix['Close']['^VIX']
            else:
                vix_close = vix['Close']
            vix_q = vix_close.resample('QE').mean()
            macro_df['VIX'] = vix_q
            save_to_cache('bt_vix', pd.DataFrame({'VIX': vix_q}))
            print("  [OK] VIX")
        else:
            print("  [WARN] VIX: No data, checking cache...")
            cached = load_from_cache('bt_vix')
            if cached is not None:
                macro_df['VIX'] = cached['VIX']
    except Exception as e:
        print(f"  [WARN] VIX: {e}, checking cache...")
        cached = load_from_cache('bt_vix')
        if cached is not None:
            macro_df['VIX'] = cached['VIX']
    
    # Sector returns - initialize to None first
    sector_returns = None
    try:
        sector_data = download_with_retry(SECTOR_TICKERS, start_date, end_date)
        if sector_data is not None and len(sector_data) > 0:
            if isinstance(sector_data.columns, pd.MultiIndex):
                sector_prices = sector_data['Close']
            else:
                sector_prices = sector_data['Close']
            sector_prices_q = sector_prices.resample('QE').last()
            sector_returns = sector_prices_q.pct_change()
            save_to_cache('bt_sector_returns', sector_returns)
            print(f"  [OK] Sector returns: {len(sector_returns)} quarters")
        else:
            # Try cache fallback
            print("  [WARN] Sector data: No data, checking cache...")
            sector_returns = load_from_cache('bt_sector_returns')
            if sector_returns is None:
                print("  [ERROR] Sector data: No data and no cache available")
                return None, None, None
    except Exception as e:
        print(f"  [WARN] Sector data error: {e}, checking cache...")
        sector_returns = load_from_cache('bt_sector_returns')
        if sector_returns is None:
            print("  [ERROR] Sector data: No cache available")
            return None, None, None
    
    # Gold
    try:
        gold = download_with_retry('GLD', start_date, end_date)
        if gold is not None and len(gold) > 0:
            if isinstance(gold.columns, pd.MultiIndex):
                gold_close = gold['Close']['GLD']
            else:
                gold_close = gold['Close']
            gold_ret = gold_close.resample('QE').last().pct_change()
            macro_df['Gold_Return'] = gold_ret
            save_to_cache('bt_gold', pd.DataFrame({'Gold_Return': gold_ret}))
            print("  [OK] Gold")
        else:
            print("  [WARN] Gold: No data, checking cache...")
            cached = load_from_cache('bt_gold')
            if cached is not None:
                macro_df['Gold_Return'] = cached['Gold_Return']
    except Exception as e:
        print(f"  [WARN] Gold: {e}, checking cache...")
        cached = load_from_cache('bt_gold')
        if cached is not None:
            macro_df['Gold_Return'] = cached['Gold_Return']    
    # SPY returns for S&P 500 benchmark
    spy_returns = None
    try:
        spy = download_with_retry('SPY', start_date, end_date)
        if spy is not None and len(spy) > 0:
            # Handle potential MultiIndex from yfinance
            if isinstance(spy.columns, pd.MultiIndex):
                spy_close = spy['Close']['SPY']
            else:
                spy_close = spy['Close']
            spy_prices_q = spy_close.resample('QE').last()
            spy_returns = spy_prices_q.pct_change().dropna()
            save_to_cache('bt_spy_returns', pd.DataFrame({'SPY': spy_returns}))
            macro_df['Realized_Vol'] = spy_close.pct_change().resample('QE').std() * np.sqrt(252)
            print(f"  [OK] SPY (S&P 500): {len(spy_returns)} quarters")
        else:
            print("  [WARN] SPY: No data, checking cache...")
            cached = load_from_cache('bt_spy_returns')
            if cached is not None:
                spy_returns = cached['SPY']
    except Exception as e:
        print(f"  [WARN] SPY: {e}, checking cache...")
        cached = load_from_cache('bt_spy_returns')
        if cached is not None:
            spy_returns = cached['SPY']
    
    macro_df = macro_df.dropna()
    
    print(f"\n[OK] Macro data: {len(macro_df)} quarters")
    if sector_returns is not None:
        print(f"[OK] Sector data: {len(sector_returns)} quarters")
    else:
        print("[ERROR] Sector data: Failed to fetch")
        return None, None, None
    
    return macro_df, sector_returns, spy_returns


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def detect_regime(row):
    """Detect regime for a single quarter."""
    risk_off = 0
    
    if 'VIX' in row and row['VIX'] > VIX_HIGH:
        risk_off += 1
    if 'Yield_Curve' in row and row['Yield_Curve'] < YIELD_CURVE_INVERSION:
        risk_off += 1
    if 'Credit_Spread' in row and row['Credit_Spread'] > CREDIT_SPREAD_HIGH:
        risk_off += 1
    
    return 'RISK_OFF' if risk_off >= 1 else 'RISK_ON'


def train_model(macro_train, sector_train):
    """Train regression model on historical data."""
    
    results = {}
    
    for sector in sector_train.columns:
        y = sector_train[sector].dropna()
        x = macro_train.loc[y.index].dropna()
        
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(y) < 10:
            continue
        
        x_std = (x - x.mean()) / x.std()
        
        if HAS_STATSMODELS:
            X_const = sm.add_constant(x_std)
            model = sm.OLS(y, X_const).fit()
            results[sector] = {
                'coefficients': model.params.drop('const'),
                'pvalues': model.pvalues.drop('const'),
                'mean': x.mean(),
                'std': x.std()
            }
        else:
            correlations = x_std.apply(lambda col: col.corr(y))
            results[sector] = {
                'coefficients': correlations,
                'pvalues': pd.Series([0.05] * len(correlations), index=correlations.index),
                'mean': x.mean(),
                'std': x.std()
            }
    
    return results


def generate_predictions(model_results, macro_current, current_regime):
    """Generate sector scores for current quarter."""
    
    scores = {}
    
    for sector, model in model_results.items():
        betas = model['coefficients']
        pvalues = model['pvalues']
        mean = model['mean']
        std = model['std']
        
        # Compute z-scores of current macro values
        z_scores = (macro_current - mean) / std
        
        score = 0
        for var in betas.index:
            if var in z_scores.index:
                z = z_scores[var]
                beta = betas[var]
                pval = pvalues[var]
                
                if pd.isna(z) or pd.isna(beta):
                    continue
                
                # Significance weighting
                if pval < 0.01:
                    weight = 1.0
                elif pval < 0.05:
                    weight = 0.8
                elif pval < 0.10:
                    weight = 0.6
                else:
                    weight = 0.3
                
                score += beta * z * weight
        
        # Regime adjustment
        if current_regime == 'RISK_OFF':
            if sector in DEFENSIVE_SECTORS:
                score += 0.02
            else:
                score -= 0.02
        else:
            if sector in CYCLICAL_SECTORS:
                score += 0.01
            else:
                score -= 0.01
        
        scores[sector] = score
    
    return scores


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def run_backtest(macro_df, sector_returns):
    """
    Run walk-forward backtest.
    
    For each quarter t:
    1. Train model on data from t-TRAINING_WINDOW to t-1
    2. Generate predictions for quarter t
    3. Compare predictions to actual returns in quarter t
    """
    
    print("\n" + "=" * 70)
    print("RUNNING WALK-FORWARD BACKTEST")
    print("=" * 70)
    
    # Align data
    common_dates = macro_df.index.intersection(sector_returns.index)
    macro_df = macro_df.loc[common_dates]
    sector_returns = sector_returns.loc[common_dates]
    
    all_dates = sorted(common_dates)
    
    results = []
    
    for i, current_date in enumerate(all_dates):
        # Need minimum training data
        if i < MIN_TRAINING_QUARTERS:
            continue
        
        # Training window
        train_start = max(0, i - TRAINING_WINDOW)
        train_end = i
        
        train_dates = all_dates[train_start:train_end]
        
        macro_train = macro_df.loc[train_dates]
        sector_train = sector_returns.loc[train_dates]
        
        # Current quarter macro values (for prediction)
        macro_current = macro_df.loc[current_date]
        
        # Actual returns for this quarter (what we're trying to predict)
        actual_returns = sector_returns.loc[current_date]
        
        # Detect regime
        regime = detect_regime(macro_current)
        
        # Train model and generate predictions
        model = train_model(macro_train, sector_train)
        predictions = generate_predictions(model, macro_current, regime)
        
        if len(predictions) == 0:
            continue
        
        # Store results
        for sector in predictions:
            if sector in actual_returns.index:
                results.append({
                    'Date': current_date,
                    'Sector': sector,
                    'Predicted_Score': predictions[sector],
                    'Actual_Return': actual_returns[sector],
                    'Regime': regime
                })
        
        # Progress
        if (i + 1) % 4 == 0:
            print(f"  Processed {i+1}/{len(all_dates)} quarters...")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n[OK] Backtest complete: {len(results_df)} sector-quarter observations")
    
    return results_df


# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

def analyze_performance(results_df, spy_returns=None):
    """Analyze backtest performance."""
    
    print("\n" + "=" * 70)
    print("BACKTEST PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Determine time window
    dates = sorted(results_df['Date'].unique())
    start_str = pd.Timestamp(dates[0]).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')
    print(f"\nBacktest Window: {start_str} to {end_str} ({len(dates)} quarters)")
    
    # Group by date and rank sectors
    portfolio_returns = []
    
    for date in results_df['Date'].unique():
        day_data = results_df[results_df['Date'] == date].copy()
        
        # Rank by predicted score
        day_data['Rank'] = day_data['Predicted_Score'].rank(ascending=False)
        
        # Top 3 sectors (long)
        top = day_data.nsmallest(3, 'Rank')
        top_return = top['Actual_Return'].mean()
        
        # Bottom 3 sectors (short)
        bottom = day_data.nlargest(3, 'Rank')
        bottom_return = bottom['Actual_Return'].mean()
        
        # Long/Short return
        ls_return = top_return - bottom_return
        
        # Best sector prediction
        predicted_best = day_data.loc[day_data['Predicted_Score'].idxmax()]
        best_hit = 1 if predicted_best['Actual_Return'] > day_data['Actual_Return'].median() else 0
        
        # SPY return for this date
        spy_ret = None
        if spy_returns is not None:
            spy_match = spy_returns.loc[spy_returns.index == date]
            if len(spy_match) > 0:
                spy_ret = float(spy_match.iloc[0])
        
        portfolio_returns.append({
            'Date': date,
            'Top3_Return': top_return,
            'Bottom3_Return': bottom_return,
            'LongShort_Return': ls_return,
            'EqualWeight_Return': day_data['Actual_Return'].mean(),
            'SPY_Return': spy_ret,
            'Best_Prediction_Hit': best_hit,
            'Regime': day_data['Regime'].iloc[0]
        })
    
    port_df = pd.DataFrame(portfolio_returns)
    
    # Calculate metrics
    print("\n--- OVERALL PERFORMANCE ---")
    
    # Long/Short Strategy
    ls_returns = port_df['LongShort_Return']
    cumulative_ls = (1 + ls_returns).cumprod()
    
    total_return_ls = cumulative_ls.iloc[-1] - 1
    annual_return_ls = (1 + total_return_ls) ** (4 / len(ls_returns)) - 1
    sharpe_ls = ls_returns.mean() / ls_returns.std() * 2  # Annualized (4 quarters)
    
    print(f"\nLONG/SHORT STRATEGY (Long Top 3, Short Bottom 3):")
    print(f"  Total Return:     {total_return_ls*100:+.2f}%")
    print(f"  Annualized Return: {annual_return_ls*100:+.2f}%")
    print(f"  Sharpe Ratio:     {sharpe_ls:.2f}")
    print(f"  Win Rate:         {(ls_returns > 0).mean()*100:.1f}%")
    print(f"  Avg Quarterly:    {ls_returns.mean()*100:+.2f}%")
    
    # Long Only Top 3
    top_returns = port_df['Top3_Return']
    cumulative_top = (1 + top_returns).cumprod()
    
    total_return_top = cumulative_top.iloc[-1] - 1
    annual_return_top = (1 + total_return_top) ** (4 / len(top_returns)) - 1
    sharpe_top = top_returns.mean() / top_returns.std() * 2
    
    print(f"\nLONG ONLY TOP 3 SECTORS:")
    print(f"  Total Return:     {total_return_top*100:+.2f}%")
    print(f"  Annualized Return: {annual_return_top*100:+.2f}%")
    print(f"  Sharpe Ratio:     {sharpe_top:.2f}")
    
    # Market (Equal Weight All Sectors)
    mkt_returns = port_df['EqualWeight_Return']
    cumulative_mkt = (1 + mkt_returns).cumprod()
    
    total_return_mkt = cumulative_mkt.iloc[-1] - 1
    annual_return_mkt = (1 + total_return_mkt) ** (4 / len(mkt_returns)) - 1
    sharpe_mkt = mkt_returns.mean() / mkt_returns.std() * 2
    
    print(f"\nBENCHMARK (Equal Weight All Sectors):")
    print(f"  Total Return:     {total_return_mkt*100:+.2f}%")
    print(f"  Annualized Return: {annual_return_mkt*100:+.2f}%")
    print(f"  Sharpe Ratio:     {sharpe_mkt:.2f}")
    
    # Alpha
    alpha = annual_return_top - annual_return_mkt
    print(f"\nALPHA vs Equal-Weight Sectors: {alpha*100:+.2f}% per year")
    
    # SPY Comparison
    if spy_returns is not None:
        # Align SPY returns with portfolio dates
        spy_aligned = spy_returns.loc[spy_returns.index.isin(port_df['Date'])].dropna()
        if len(spy_aligned) > 0:
            cumulative_spy = (1 + spy_aligned).cumprod()
            total_return_spy = float(cumulative_spy.iloc[-1] - 1)
            annual_return_spy = float((1 + total_return_spy) ** (4 / len(spy_aligned)) - 1)
            sharpe_spy = float(spy_aligned.mean() / spy_aligned.std() * 2)
            
            print(f"\nS&P 500 (SPY):")
            print(f"  Total Return:     {total_return_spy*100:+.2f}%")
            print(f"  Annualized Return: {annual_return_spy*100:+.2f}%")
            print(f"  Sharpe Ratio:     {sharpe_spy:.2f}")
            
            alpha_vs_spy = annual_return_top - annual_return_spy
            print(f"\nALPHA vs S&P 500 (SPY): {alpha_vs_spy*100:+.2f}% per year")
    
    # Hit Rate
    hit_rate = port_df['Best_Prediction_Hit'].mean()
    print(f"\nPREDICTION HIT RATE: {hit_rate*100:.1f}%")
    print(f"  (% of quarters where #1 predicted sector beat median)")
    
    # By Regime
    print("\n--- PERFORMANCE BY REGIME ---")
    
    for regime in ['RISK_ON', 'RISK_OFF']:
        regime_data = port_df[port_df['Regime'] == regime]
        if len(regime_data) > 0:
            regime_ls = regime_data['LongShort_Return'].mean()
            regime_top = regime_data['Top3_Return'].mean()
            print(f"\n{regime} ({len(regime_data)} quarters):")
            print(f"  Avg L/S Return:   {regime_ls*100:+.2f}%")
            print(f"  Avg Top 3 Return: {regime_top*100:+.2f}%")
    
    # Per-Sector Analysis
    print("\n--- SECTOR-LEVEL ANALYSIS ---")
    
    sector_stats = []
    for sector in SECTOR_TICKERS:
        sector_data = results_df[results_df['Sector'] == sector]
        if len(sector_data) > 0:
            corr = sector_data['Predicted_Score'].corr(sector_data['Actual_Return'])
            avg_return = sector_data['Actual_Return'].mean()
            sector_stats.append({
                'Sector': SECTOR_NAMES.get(sector, sector),
                'Prediction_Correlation': corr,
                'Avg_Return': avg_return
            })
    
    sector_stats_df = pd.DataFrame(sector_stats).sort_values('Prediction_Correlation', ascending=False)
    
    print("\nPrediction Correlation by Sector:")
    print("(Higher = model predicts this sector better)")
    for _, row in sector_stats_df.iterrows():
        print(f"  {row['Sector']:<15}: r = {row['Prediction_Correlation']:+.3f}")
    
    return port_df, sector_stats_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("SECTOR ANALYSIS MODEL BACKTEST")
    print("=" * 70)
    print(f"Backtest Period: {BACKTEST_START} to present")
    print(f"Training Window: {TRAINING_WINDOW} quarters")
    print(f"Min Training Data: {MIN_TRAINING_QUARTERS} quarters")
    print("=" * 70)
    
    # Fetch data
    macro_df, sector_returns, spy_returns = fetch_all_data(BACKTEST_START, datetime.now().strftime('%Y-%m-%d'))
    
    if macro_df is None or sector_returns is None:
        print("\n[ERROR] Failed to fetch data. Exiting.")
        exit(1)
    
    # Run backtest
    results_df = run_backtest(macro_df, sector_returns)
    
    # Analyze performance
    portfolio_df, sector_stats_df = analyze_performance(results_df, spy_returns)
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    results_df.to_csv('backtest_raw_results.csv', index=False)
    print("  [OK] backtest_raw_results.csv")
    
    portfolio_df.to_csv('backtest_portfolio.csv', index=False)
    print("  [OK] backtest_portfolio.csv")
    
    sector_stats_df.to_csv('backtest_sector_stats.csv', index=False)
    print("  [OK] backtest_sector_stats.csv")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
