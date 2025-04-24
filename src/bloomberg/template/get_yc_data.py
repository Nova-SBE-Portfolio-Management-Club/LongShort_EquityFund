from xbbg import blp
import pandas as pd
from datetime import datetime, timedelta

"""
Assuming yc_tickers.txt in form:
YCGT0251 Index,2022-01-01
YCGT0014 Index,2023-01-01
"""


def get_yc_data():
    """
    Fetches yield curve data from bloomberg using tickers listed in file
    Inputs: 
        start_data (str): Start data in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing historical yield curve data
    """

    tickers_file = 'src/bloomberg/exports/yc_tickers.txt'
    ticker_to_start = {}
    with open(tickers_file) as file:
        for line in file:
            if line.strip():
                ticker, start_date = line.strip().split(',')
                ticker_to_start[ticker] = start_date

    tickers = list(ticker_to_start.keys())
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    data = blp.bds(tickers, 'CURVE_TENOR_RATES')

    history = blp.bdh(data['tenor_ticker'], 'px_last', start_date, end_date)

    return history, data

print(get_yc_data())