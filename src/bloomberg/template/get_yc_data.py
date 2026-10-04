from xbbg import blp
from pathlib import Path
from datetime import datetime, timedelta

"""
Assuming yc_tickers.txt in form:
YCGT0251 Index,2022-01-01
YCGT0014 Index,2023-01-01
"""


def get_yc_data():
    """
    Fetches yield curve data from bloomberg using tickers listed in file
    Reads ticker/start-date pairs from exports/yc_tickers.txt.
    Returns historical prices and the tenor metadata used in the request.
    """

    tickers_file = Path(__file__).resolve().parents[1] / 'exports' / 'yc_tickers.txt'
    ticker_to_start = {}
    with open(tickers_file) as file:
        for line in file:
            if line.strip():
                ticker, start_date = line.strip().split(',')
                ticker_to_start[ticker] = start_date

    if not ticker_to_start:
        raise ValueError("No yield-curve tickers supplied.")
    # Fetch a common history covering all requested start dates.
    start_date = min(ticker_to_start.values())
    tickers = list(ticker_to_start.keys())
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    data = blp.bds(tickers, 'CURVE_TENOR_RATES')

    history = blp.bdh(data['tenor_ticker'], 'px_last', start_date, end_date)

    return history, data

if __name__ == "__main__":
    print(get_yc_data())