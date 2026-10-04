"""
corrcoint.py
This file provides functions to calculate the correlation and cointegration
"""
from statsmodels.tsa.stattools import coint

def correlation(series1, series2):
    """
    Calculate Pearson correlation rescaled from [-1, 1] to [0, 1].
    
    Parameters:
    series1 (pd.Series): First time series.
    series2 (pd.Series): Second time series.
    
    Returns:
    float: Rescaled correlation score.
    """
    return (series1.corr(series2)+1)/2

def coint_mapper(p_value):
    """
    Transform a score in [0, 1], emphasizing values near 1.
    The cointegration function supplies 1 - p_value, so low p-values score higher.
    """
    return (0.25*p_value)/(1.25 - p_value)

def cointegration(series1, series2):
    """
    Calculate the cointegration between two time series.
    
    Parameters:
    series1 (pd.Series): First time series.
    series2 (pd.Series): Second time series.
    
    Returns:
    float: Cointegration score.
    """
    _, p_value, _ = coint(series1, series2)
    
    return coint_mapper(1-p_value)
