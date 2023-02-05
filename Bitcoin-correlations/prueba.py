import pandas as pd
import json
from urllib.request import urlopen
import yfinance as yf
from datetime import datetime

##Pulling Bitcoin's price history
btc_url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&api_key=d1875a3943f6f2ee83a90ac2e05d5fa018618e00724e9018f9bd08c0ac932cc6"
btc_data = urlopen(btc_url).read() #Open the API contents 
btc_json = json.loads(btc_data) #Transform the contents of our response into a manageable JSON format

##Transform Bitcoin data so we can run analysis
btc_price = btc_json['Data']['Data'] ##Extract only the relevant data from the JSON variable we created earlier
btc_df = pd.DataFrame(btc_price) ##Convert the json format into a Pandas dataframe so we can make it easier to work with 
btc_df['btc_returns'] = ((btc_df['close']/btc_df['open']) - 1) * 100 #We create a coloumn for daily returns of Bitcoin that we'll need for later when we calculate the correlation. 
btc_df['Date'] = btc_df['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d')) #Formatting the date into a human-readable format
btc_returns = btc_df[['Date', 'btc_returns']] #In this line, we select the only 2 columns we'll need for our correlation calculations namely the Date and the Return

##Pulling S&P500's price history
spy = yf.Ticker("SPY")
spy_df = spy.history(period = 'max')

##Transform S&P500 data so we can run analysis
spy_df = spy_df.reset_index() #In the original dataframe, the date is part of the index which means we can't select it later. reset_index shifts the date into a normal column
spy_df['Date'] = spy_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d')) 
spy_df['spy_returns'] = ((spy_df['Close']/spy_df['Open']) - 1) * 100
spy_returns = spy_df[['Date', 'spy_returns']]


##Pulling gold's price history
gold = yf.Ticker("GC=F")
gold_df = gold.history(period = 'max')

##Transform gold data so we can run analysis
gold_df = gold_df.reset_index()
gold_df['Date'] = gold_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
gold_df['gold_returns'] = ((gold_df['Close']/gold_df['Open']) - 1) * 100 
gold_returns = gold_df[['Date', 'gold_returns']]

def calculate_correlation(assetA_df,assetB_df):
    joint_df = pd.merge(assetA_df,assetB_df) #pd.merge combines the two datafames into a single df.
    correlation = joint_df.iloc[:,1].rolling(30 ).corr(joint_df.iloc[:,2])
    return correlation

correlation_btc_spy = calculate_correlation(btc_returns,spy_returns) #We call the calculate_correlation function on the bitcoin and S&P500 dataframes that we created earlier.
correlation_btc_gold = calculate_correlation(btc_returns,gold_returns) #ditto but for Bitcoin & gold
correlation_gold_spy = calculate_correlation(spy_returns, gold_returns) #You get the point

print ("The correlation between Bitcoin and stocks is " + str(correlation_btc_spy.iloc[-1]))
print ("The correlation between Bitcoin and gold is " + str(correlation_btc_gold.iloc[-1]))
print ("The correlation between stocks and gold is" + str(correlation_gold_spy.iloc[-1]))
