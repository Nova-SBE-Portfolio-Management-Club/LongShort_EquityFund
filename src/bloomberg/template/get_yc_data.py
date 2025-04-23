from xbbg import blp


data = blp.bds('YCGT0251 Index', 'CURVE_TENOR_RATES')
history = blp.bdh(data['tenor_ticker'], 'px_last', '2022-01-01','2023-09-20')

print(data)
print('\n========================\n')
print(history)


def get_yc_data(start_date):
    # TODO: Implement the function to fetch yield curve data from Bloomberg
    # 1. Read the tickers from the file under src/bloomberg/exports
    # 2. Adjust end_date = yesterday
    # 3. Use the blp.bdh function to get the data
    pass