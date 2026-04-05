"""
corrcoint.py
This file provides functions to calculate the correlation and cointegration
"""
from statsmodels.tsa.stattools import coint

def correlation(series1, series2):
    """
    Calculate the correlation between two time series.
    
    Parameters:
    series1 (pd.Series): First time series.
    series2 (pd.Series): Second time series.
    
    Returns:
    float: Correlation coefficient.
    """
    return (series1.corr(series2)+1)/2

def coint_mapper(p_value):
    """
    This functions maps the p-value of the cointegration test to a distribution [0,1] that
    gives more importance to the high values [0.9, 1], and low importance to the low values [0, 0.9].
    
    It will be the cointegration score that will be used to rank the pairs.
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