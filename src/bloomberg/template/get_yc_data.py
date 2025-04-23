from xbbg import blp


data = blp.bds('YCGT0251 Index', 'CURVE_TENOR_RATES')
history = blp.bdh(data['tenor_ticker'], 'px_last', '2022-01-01','2023-09-20')

print(data)
print('\n========================\n')
print(history)