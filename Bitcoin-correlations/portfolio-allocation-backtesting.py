#!/usr/bin/env python3
from pandas.core.series import base
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

weightings1 = {"SPY":"100"}
weightings2 = {"SPY":"95","BTC-USD":"5"}


members = ["BTC-USD","SPY"]


def PortfolioCalc(weightings, data, name):
  data[name] = sum([  int(weightings[x])*data[x] for x in list(weightings.keys())   ])
  return data

basedata = yf.Ticker(members[0]).history(period="max").reset_index()[["Date","Open"]]
basedata["Date"] = pd.to_datetime(basedata["Date"])
basedata['Date'] = basedata['Date'].dt.tz_convert('UTC')
basedata = basedata.rename(columns = {"Open":members[0]})

basedata = basedata.set_index('Date')
basedata.index = basedata.index.floor('D')

if (len(members)>1):
  for x in range(1,len(members)):
    newdata = yf.Ticker(members[x]).history(period="max").reset_index()[["Date","Open"]]
    newdata["Date"] = pd.to_datetime(newdata["Date"])
    newdata['Date'] = newdata['Date'].dt.tz_convert('UTC')
    newdata = newdata.set_index('Date')
    newdata.index = newdata.index.floor('D')
    newdata = newdata.rename(columns = {"Open":members[x]})
    basedata = basedata.join(newdata).dropna() 

basedata = basedata[dt.datetime(2016, 1, 1):]


print(basedata)

for x in members:
  basedata[x] = basedata[x]/(basedata[x].iloc[0])

# CAGR = (basedata['SPY'].iloc[-1]/ basedata['SPY'].iloc[0])**(1/len(basedata.index)-1)

basedata = PortfolioCalc(weightings1, basedata, "portfolio1")
basedata = PortfolioCalc(weightings2, basedata, "portfolio2")

#for x in members:
  #plt.semilogy(basedata["Date"], basedata[x], label=x)


# plt.subplots(figsize=(6, 2))
plt.style.use("dark_background")

plt.plot( basedata["portfolio1"], label = "100% S&P500 (SPY)")
plt.plot( basedata["portfolio2"], label = " 95% S&P500 (SPY), 5% BTC")


# xmin, xmax, ymin, ymax = plt.axis()
print(basedata)
plt.legend(loc="upper left")
plt.axhline(y=basedata['portfolio1'].iloc[-1], color='paleturquoise', linestyle='--')
plt.axhline(y=basedata['portfolio2'].iloc[-1], color='lightyellow', linestyle='--')
plt.axhline(y=1, color='red', linestyle='--')
plt.show()
