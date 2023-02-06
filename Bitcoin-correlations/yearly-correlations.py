from pandas.compat import F
import yfinance as yf
import datetime
from dateutil import rrule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlations(x, serie, title, halvingsLines = False):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,6))
    sns.lineplot(data = serie, x = x, y = serie.values)
    plt.axhline(y=0, color='red', linestyle='--')
    if halvingsLines:
        plt.axvline(x =pd.to_datetime('2012-11-28'), color='yellow',linestyle='--' )
        plt.axvline(x =pd.to_datetime('2016-07-09'), color='yellow',linestyle='--' )
        plt.axvline(x =pd.to_datetime('2020-05-11'), color='yellow',linestyle='--' )
    plt.title(title, weight='bold', fontsize = 15)
    plt.ylabel('Coeficientes de correlación (-1..0..1)', weight='bold', fontsize = 12)
    plt.xlabel('Tiempo', weight='bold', fontsize = 12)
    plt.show()


def main():
    # Download data from Yahoo API
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
    price_SP500.set_index('Date', inplace=True)
    price_SP500.reset_index(inplace=True)
    price_SP500["Open"] = price_SP500.Open.astype(float)
    price_SP500["Close"] = price_SP500.Close.astype(float)
    price_SP500['Date'] = price_SP500['Date'].dt.tz_localize(None)
    price_SP500_copy = price_SP500.copy()
    # Calculate % change based on open/close data
    price_SP500['Pct Change'] = ((price_SP500['Close']/price_SP500['Open']) - 1) * 100
    price_SP500 = price_SP500.drop(columns = ['Open', 'Close'])
    price_SP500.to_csv('price_SP500.csv')

    # Load BTC data from file
    response_BTC  = pd.read_csv('Bitcoin-Historical-Data-Investing.com.csv')
    # trimming the data
    price_BTC = response_BTC.drop(columns = ['High', 'Low', 'Vol.', 'Change %'])
    price_BTC = price_BTC.dropna()
    price_BTC["Open"] = price_BTC["Open"].str.replace(',','')
    price_BTC["Close"] = price_BTC["Close"].str.replace(',','')
    price_BTC["Open"] = price_BTC.Open.astype(float)
    price_BTC["Close"] = price_BTC.Close.astype(float)
    # Calculate % change based on open/close data
    price_BTC['Pct Change'] = ((price_BTC['Close']/price_BTC['Open']) - 1) * 100
    # Save a copy for later resample
    price_BTC_copy = price_BTC.copy()
    price_BTC = price_BTC.drop(columns = ['Open', 'Close'])

    price_BTC['Date'] = pd.to_datetime(price_BTC['Date'])
    price_BTC.set_index('Date', inplace=True)
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

    roll_correlation_serie = DB_df['Pct_Change_SP500'].rolling(180).corr(DB_df['Pct_Change_BTC'])
    roll_correlation_serie.columns=["Date", "Corr"]
    roll_correlation_serie = roll_correlation_serie.dropna()


    plot_correlations('Date', roll_correlation_serie,'Rolling Correlation (6 meses) SP500/BTC', True)

    M2_response  = pd.read_csv('Base-monetaria-M2.csv')
    M2_response = M2_response.dropna()
    M2_response['Date'] = pd.to_datetime(M2_response['Date'])
    M2_response['Pct Change_M2'] = M2_response['M2'].pct_change()
    M2_response.to_csv('M2_response.csv')
    print(M2_response)

    price_BTC_copy['Date'] = pd.to_datetime(price_BTC_copy['Date'])
    price_BTC_copy = price_BTC_copy.set_index('Date')
    weekly_BTC_prices = price_BTC_copy.resample('W-MON').agg({'Open': 'first', 'Close': 'last'})
    weekly_BTC_prices['Pct Change_BTC'] = ((weekly_BTC_prices['Close']/weekly_BTC_prices['Open']) - 1) * 100
    print(weekly_BTC_prices)
    weekly_BTC_prices.to_csv('weekly_BTC_prices.csv')


    price_SP500_copy['Date'] = pd.to_datetime(price_SP500_copy['Date'])
    price_SP500_copy = price_SP500_copy.set_index('Date')
    weekly_SP500_prices = price_SP500_copy.resample('W-MON').agg({'Open': 'first', 'Close': 'last'})
    weekly_SP500_prices['Pct Change_SP500'] = ((weekly_SP500_prices['Close']/weekly_SP500_prices['Open']) - 1) * 100
    print(weekly_SP500_prices)
    weekly_SP500_prices.to_csv('weekly_SP500_prices.csv')

    M2_vs_BTC = merge=pd.merge(M2_response,weekly_BTC_prices, how='inner', on='Date')
    M2_vs_BTC = M2_vs_BTC.drop(columns = ['M2', 'Open', 'Close'])
    M2_vs_BTC.columns=['Date','Pct_Change_M2', 'Pct_Change_BTC']
    M2_vs_BTC.set_index('Date', inplace=True)
    print(M2_vs_BTC)
    M2_vs_BTC.to_csv('M2_vs_BTC.csv')


    M2_vs_SP500 = merge=pd.merge(M2_response,weekly_SP500_prices, how='inner', on='Date')
    M2_vs_SP500 = M2_vs_SP500.drop(columns = ['M2', 'Open', 'Close'])
    M2_vs_SP500.columns=['Date','Pct_Change_M2', 'Pct_Change_SP500']
    M2_vs_SP500.set_index('Date', inplace=True)
    print(M2_vs_SP500)
    M2_vs_BTC.to_csv('M2_vs_SP500.csv')

    roll_correlation_serie = M2_vs_BTC['Pct_Change_M2'].rolling(52).corr(M2_vs_BTC['Pct_Change_BTC'])
    roll_correlation_serie = roll_correlation_serie.dropna()


    plot_correlations('Date', roll_correlation_serie,'Rolling Correlation (1 año) M2/BTC', True)

    roll_correlation_serie = M2_vs_SP500['Pct_Change_M2'].rolling(52).corr(M2_vs_SP500['Pct_Change_SP500'])
    roll_correlation_serie = roll_correlation_serie.dropna()

    plot_correlations('Date', roll_correlation_serie,'Rolling Correlation (1 año) M2/SP500', False)

if __name__ == "__main__":
    main()
