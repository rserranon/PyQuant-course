import yfinance as yf
import datetime
from dateutil import rrule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download data from Yahoo API and clean data
today = datetime.date.today().isoformat()
start_date = '2009-09-17'


response_SP500 = yf.download(
    "^GSPC", 
    start=start_date, 
    end=today
)

# trimming the data
price_SP500 = response_SP500.drop(columns = ['High', 'Low', 'Adj Close', 'Volume'])
price_SP500 = price_SP500.dropna()
price_SP500.reset_index(inplace=True)
price_SP500["Open"] = price_SP500.Open.astype(float)
price_SP500["Close"] = price_SP500.Close.astype(float)
price_SP500['Date'] = price_SP500['Date'].dt.tz_localize(None)
# Calculate % change
#price_SP500['Pct Change'] = price_SP500['Close'].pct_change()
price_SP500['Pct Change'] = ((price_SP500['Close']/price_SP500['Open']) - 1) * 100
price_SP500 = price_SP500.drop(columns = ['Open', 'Close'])
price_SP500.to_csv('price_SP500.csv')

response_BTC  = pd.read_csv('Bitcoin-Historical-Data-Investing.com.csv')

# trimming the data
price_BTC = response_BTC.drop(columns = ['High', 'Low', 'Vol.', 'Change %'])
price_BTC = price_BTC.dropna()
price_BTC["Open"] = price_BTC["Open"].str.replace(',','')
price_BTC["Close"] = price_BTC["Close"].str.replace(',','')
price_BTC["Open"] = price_BTC.Open.astype(float)
price_BTC["Close"] = price_BTC.Close.astype(float)
# Calculate % change
#price_BTC['Pct Change'] = price_BTC['Close'].pct_change()
price_BTC['Pct Change'] = ((price_BTC['Close']/price_BTC['Open']) - 1) * 100
price_BTC = price_BTC.drop(columns = ['Open', 'Close'])

price_BTC['Date'] = pd.to_datetime(price_BTC['Date'])
price_BTC.to_csv('price_BTC.csv')

DB_df = merge=pd.merge(price_SP500,price_BTC, how='inner', on='Date')
DB_df.columns=["Date", "Pct_Change_SP500", "Pct_Change_BTC"]
DB_df.set_index('Date', inplace=True)
DB_df.to_csv('DB_df.csv')

test = DB_df.loc['2017-01-01':'2021-01-20']
correlation = test['Pct_Change_SP500'].corr(test['Pct_Change_BTC'])
print(f"Global correlation: {correlation}")

test = DB_df.loc['2017-01-01':'2021-01-20']
correlation = test['Pct_Change_SP500'].rolling(30).corr(test['Pct_Change_BTC'])
print(f"Rolling correlation: {correlation[-1]}")

start = datetime.datetime(2011,1,1)
end = datetime.datetime(2022,12,31)
columns = ['Year', 'Corr']
years = []
corr = []
for dt in rrule.rrule(rrule.YEARLY, dtstart=start, until=end):
    eoy = datetime.date(dt.year, 12, 30)
    data_df = DB_df[(DB_df.index > np.datetime64(dt)) & (DB_df.index <= np.datetime64(eoy))]
    correlation = data_df['Pct_Change_SP500'].corr(data_df['Pct_Change_BTC'])
    years.append(dt.year)
    corr.append(correlation)

correlation_df = pd.DataFrame()
correlation_df['Year'] = years
correlation_df['Corr'] = corr
correlation_df.set_index('Year', inplace=True)

print(correlation_df)

fig = plt.figure(figsize=(8,6))
sns.lineplot(data = correlation_df, x = 'Year', y = 'Corr')
plt.show()
