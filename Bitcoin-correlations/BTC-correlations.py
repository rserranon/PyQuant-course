#!/usr/bin/env python3

import pandas as pd
import datetime
from openbb_terminal.sdk import openbb
import sys
sys.path.append('../pyfolio-analysis')
from loadBTC import load_BTC
import matplotlib.pyplot as plt

today = datetime.date.today().isoformat()

spx = openbb.economy.index(["^GSPC"], start_date="2009-01-01", end_date=today)
spx_returns = spx.pct_change()
spx_returns.index = spx_returns.index.tz_localize("UTC")
spx_returns.index.names = ['date']

print(spx_returns.head())

btc_df, btc_returns = load_BTC('2009-01-01',today)
btc_df.index = btc_df.index.tz_convert('UTC')
print(btc_returns)

all_returns = spx_returns.merge(btc_returns, how='left', on='date').dropna()
all_returns.rename(columns={"^GSPC": "GSPC Return", "Adj Close": "BTC Return"}, inplace=True)
print(all_returns)

roll_correlation_serie = all_returns['GSPC Return'].rolling(180).corr(all_returns['BTC Return'])
#roll_correlation_serie.index = ["Date"]
roll_correlation_serie.columns=["Corr"]
print(type(roll_correlation_serie))
roll_correlation_serie = roll_correlation_serie.dropna()
print(roll_correlation_serie)

plt.plot(roll_correlation_serie)
plt.axvline(x =pd.to_datetime('2012-11-28'), color='yellow',linestyle='--' )
plt.axvline(x =pd.to_datetime('2016-07-09'), color='yellow',linestyle='--' )
plt.axvline(x =pd.to_datetime('2020-05-11'), color='yellow',linestyle='--' )
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Rolling Correlation (6 meses) SP500/BTC", weight='bold', fontsize = 15)
plt.ylabel('Coeficientes de correlación (-1..0..1)', weight='bold', fontsize = 12)
plt.xlabel('Tiempo', weight='bold', fontsize = 12)
#ax.set_aspect(aspect=2)
plt.show()
