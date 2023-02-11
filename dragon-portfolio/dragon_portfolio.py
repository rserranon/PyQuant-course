import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import math

### PORTFOLIO
# 24% EQUITY
# 18% FIXED INCOME
# 19% GOLD
# 18% COMDTY TREND/GLOBAL MACRO
# 21% LONG VOL

### EQUITY
# 80% GLOBAL, 19.2% of tot
# 20% EM, 4.8% of tot

### FIXED INCOME
# US TSY 50%, 9% of tot
# Corp bonds, 25% 4.5% of tot
# EM BONDS, 25%, 4.5% of tot

### GOLD
# GLD 90%, 17.1% of tot
# GDX 10%, 1.9 of tot

### COMDTY TREND+GLOBAL MACRO
# LYNX 75%, 13.5% of tot
# Lynx Asset Management AB
# SEB ASSET SELECTION C LUX 25%, 4.5% of tot
# IPM SYSTEMATIC MACRO UCITS 0%
# NORDKINN 0%

### LONG VOL
# AMUNDI 100%, 21% of tot
# AMUNDI FUNDS VOLATILITY WORLD - A USD
# amundi.com
# LATEST DATE (START MONTH) 2007-11, AMUNDI

### GLOBAL VARIABLES / DATA ###
start_date = '2007-11'
years = 13.167
global_data_raw = pd.read_csv('MSCI_World_SEK.csv')
global_data_raw = global_data_raw.set_index('Date')
global_data = pd.DataFrame(global_data_raw.loc[start_date:])
# us_data_raw = pd.read_csv('SPP_aktiefond_USA.csv')
# us_data_raw = us_data_raw.set_index('Date')
# us_data = pd.DataFrame(us_data_raw.loc[start_date:])
# avanza_zero_raw = pd.read_csv('Avanza_zero.csv')
# avanza_zero_raw = avanza_zero_raw.set_index('Date')
# avanza_zero = pd.DataFrame(avanza_zero_raw.loc[start_date:])
em_data = pd.read_csv('MSCI_EM_SEK.csv')
em_data = em_data.set_index('Date')
em_stock = pd.DataFrame(em_data.loc[start_date:])
tlt_raw = pd.read_csv('TLT_SEK.csv')
tlt_raw = tlt_raw.set_index('Date')
tlt = pd.DataFrame(tlt_raw.loc[start_date:])
# SEB_kortranta_raw = pd.read_csv('SEB_kortrantefond_USD_sek.csv')
# SEB_kortranta_raw = SEB_kortranta_raw.set_index('Date')
# SEB_kortranta = pd.DataFrame(SEB_kortranta_raw.loc[start_date:])
# AMF_rantefond_raw = pd.read_csv('AMF_rantefond_long.csv')
# AMF_rantefond_raw = AMF_rantefond_raw.set_index('Date')
# AMF_rantefond = pd.DataFrame(AMF_rantefond_raw.loc[start_date:])
corp_bond_data = pd.read_csv('Barclays_global_corp_SEK.csv')
corp_bond_data = corp_bond_data.set_index('Date')
corp_bond = pd.DataFrame(corp_bond_data.loc[start_date:])
em_bond_data = pd.read_csv('ishares_JPMorgan_EM_bond_sek.csv')
em_bond_data = em_bond_data.set_index('Date')
em_bond = pd.DataFrame(em_bond_data.loc[start_date:])
gld_raw = pd.read_csv('GLD_SEK.csv')
gld_raw = gld_raw.set_index('Date')
gld = pd.DataFrame(gld_raw.loc[start_date:])
# gdx_raw = pd.read_csv('GDX_SEK.csv')
# gdx_raw = gdx_raw.set_index('Date')
# gdx = pd.DataFrame(gdx_raw.loc[start_date:])
lynx_raw = pd.read_csv('Lynx.csv')
lynx_raw = lynx_raw.set_index('Date')
lynx = pd.DataFrame(lynx_raw.loc[start_date:])
SEB_selection_raw = pd.read_csv('SEB_asset_selection_c.csv')
SEB_selection_raw = SEB_selection_raw.set_index('Date')
SEB_selection = pd.DataFrame(SEB_selection_raw.loc[start_date:])
amundi_raw = pd.read_csv('Amundi_long_SEK.csv')
amundi_raw = amundi_raw.set_index('Date')
amundi = pd.DataFrame(amundi_raw.loc[start_date:])
risk_free_rate_raw = pd.read_csv('risk_free_rate.csv')
risk_free_rate_raw = risk_free_rate_raw.set_index('Date')
risk_free_rate = pd.DataFrame(risk_free_rate_raw.loc[start_date:])
# dfs = [global_data, us_data, avanza_zero, tlt, SEB_kortranta, AMF_rantefond, gld, \
#         gdx, lynx, SEB_selection, amundi]
dfs = [global_data, em_stock, tlt, corp_bond, em_bond, gld, \
    lynx, SEB_selection, amundi]
# cols = ['Global', 'US', 'Ava_Z', 'TLT', 'SEB_kort', 'AMF_ranta', 'GLD',\
#     'GDX', 'Lynx', 'SEB_select', 'Amundi']
cols = ['Global', 'EM_eq', 'TLT', 'Corp_bond', 'EM_bond', 'GLD',\
        'Lynx', 'SEB_select', 'Amundi']
data = pd.concat(dfs, axis=1).reset_index()
data = data.set_index('Date')
data.columns = cols
initial_weight = np.array([0.192,0.048,0.09,0.045,\
            0.045, 0.19, 0.135, 0.045, 0.21])
# 60/40
# initial_weight = np.array([0.6, 0, 0.4, 0, 0, 0, 0,\
#         0, 0])
### GLOBAL VARIABLES / DATA ###

def get_data_weight_every_month():
    # Balansera varje mån
    monthly_returns = data.pct_change()
    
    # returns = data/data.shift(1)
    monthly_returns_portfolio_mean = monthly_returns.mean()
    allocated_monthly_returns = (initial_weight * monthly_returns_portfolio_mean)
    portfolio_return = np.sum(allocated_monthly_returns)
    # calculate portfolio monthly returns, rebalanserar varje månad (ggr monthly_returns med matris)
    monthly_returns['portfolio_monthly_returns'] = monthly_returns.dot(initial_weight)
    monthly_returns.to_csv('monthly_returns.csv')
    monthly_returns['portfolio_monthly_returns_cum'] = (1+monthly_returns['portfolio_monthly_returns']).cumprod()
    monthly_returns['portfolio_monthly_returns_cum'] = monthly_returns['portfolio_monthly_returns_cum'].fillna(1)
    monthly_returns['portfolio_monthly_returns_cum'] *= 100
    calc_risk(monthly_returns)

    # Cumulative_returns_monthly = (1+monthly_returns).cumprod()
    # Cumulative_returns_monthly.to_csv('initial_weight.csv')
    # Cumulative_returns_monthly['portfolio_monthly_returns'].plot()
    # matrix_covariance_portfolio = monthly_returns.iloc[:,:-1]
    # matrix_covariance_portfolio = (matrix_covariance_portfolio.cov())*12
    # portfolio_variance = np.dot(initial_weight.T,np.dot(matrix_covariance_portfolio, initial_weight))

    #standard deviation (risk of portfolio)
    # portfolio_standard_deviation = np.sqrt(portfolio_variance)
    # portfolio_risk = []
    # sharpe_ratio_port = []
    # portfolio_risk.append(portfolio_standard_deviation)
    # #sharpe_ratio (risk free rate = 0)
    # RF = 0
    # sharpe_ratio = (((portfolio_return)- RF)/portfolio_standard_deviation)
    # print(sharpe_ratio*12)
    # sharpe_ratio_port.append(sharpe_ratio)
    # portfolio_risk = np.array(portfolio_risk)
    # print(portfolio_standard_deviation)
    # sharpe_ratio_port = np.array(sharpe_ratio_port)
    # print(Cumulative_returns_monthly.tail(5))
    # cagr = (data.iloc[-1]/data.iloc[0])**(1/years) - 1
    # cov = returns.cov()

    # exp_return = []
    # sigma = []
    # for _ in range(20000):
    #   w = random_weights(len(cols))
    #   exp_return.append(np.dot(w, cagr.T))
    #   sigma.append(np.sqrt(np.dot(np.dot(w.T, cov), w)))

    # plt.plot(sigma, exp_return, 'ro', alpha=0.1)
    # plt.show()
    return monthly_returns

def get_data_weight_every_x_month(rebalance_every):
    # Rebalance every X months
    monthly_returns = data.pct_change()
    # monthly_returns = monthly_returns.drop('portfolio_monthly_returns', axis=1)
    weighted_portfolio = pd.DataFrame(monthly_returns.iloc[0])
    # Starta med ett visst värde representerat av vikten, tänk i pengar
    weighted_portfolio = weighted_portfolio.fillna(100).transpose()*initial_weight
    # X months ahead
    for i in range(len(monthly_returns.index)):
        if i == 0:
            weighted_portfolio['portfolio_monthly_returns_cum'] = weighted_portfolio.sum(axis='columns')
        elif i % (rebalance_every) == 0:
            # Rebalance then multiply
            # First, figure out how much weight the position has now
            new_weight = weighted_portfolio.iloc[-1:].copy(deep=True)
            for name in cols:
                last_val = new_weight[name].iat[0]
                # Weighting of totals:
                new_weight[name] = new_weight[name].iat[0] / new_weight['portfolio_monthly_returns_cum'].iat[0]
                # Difference to adjust:
                new_weight[name] = weighted_portfolio[name].iat[0] - new_weight[name].iat[0]*100
                # Adjust value:
                new_weight[name] = last_val + (new_weight['portfolio_monthly_returns_cum'].iat[0] * new_weight[name].iat[0]) / 100
                # % change in new value
                new_weight[name] = new_weight[name].iat[0] * (monthly_returns[name].iat[i]+1)
            new_weight = new_weight.drop('portfolio_monthly_returns_cum', axis=1)
            new_weight['portfolio_monthly_returns_cum'] = new_weight.sum(axis=1)
            weighted_portfolio = pd.concat([weighted_portfolio, new_weight], ignore_index=True)
        else:
            to_append = weighted_portfolio.drop('portfolio_monthly_returns_cum', axis=1)
            to_append = to_append.iloc[-1]
            to_append = to_append * (monthly_returns.iloc[i]+1)
            to_append['portfolio_monthly_returns_cum'] = to_append.sum()
            weighted_portfolio = pd.concat([weighted_portfolio, to_append.to_frame().T], ignore_index=True)
    weighted_portfolio = weighted_portfolio.set_index(monthly_returns.index)
    weighted_portfolio.to_csv('weighted.csv')
    # plt.plot(weighted_portfolio['portfolio_monthly_returns_cum'])
    # plt.show()
    weighted_portfolio['portfolio_monthly_returns'] = weighted_portfolio['portfolio_monthly_returns_cum'].pct_change()
    calc_risk(weighted_portfolio)
    return weighted_portfolio

def get_data_every_pct(pct):
    ### Reweight at +-X%
    monthly_returns = data.pct_change()
    dynamic_portfolio = pd.DataFrame(monthly_returns.iloc[0])
    # Start with a certain value represented by the weight, think in terms of money
    dynamic_portfolio = dynamic_portfolio.fillna(100).transpose()*initial_weight
    # weight_dict = {'Global':0.18, 'US':0.00, 'Ava_Z':0.06, 'TLT':0.09, 'SEB_kort':0.00, \
    #     'AMF_ranta':0.09, 'GLD':0.19, 'GDX':0.0, 'Lynx':0.135, 'SEB_select':0.045, 'Amundi':0.21}
    weight_dict = {}
    for i in range(len(cols)):
        weight_dict[cols[i]] = initial_weight[i]
    # Check weighting
    transactions = 0
    for i in range(len(monthly_returns.index)):
        if i == 0:
            dynamic_portfolio['portfolio_monthly_returns_cum'] = dynamic_portfolio.sum(axis=1)
        else:
            # Check weight
            old_weight = dynamic_portfolio.iloc[-1:].copy(deep=True)
            new_weight_all = dynamic_portfolio.iloc[-1:].copy(deep=True)
            tickers_down = []
            tickers_up = []
            stop = False
            for name in cols:
                last_val = old_weight[name].iat[0]
                # Weighting of total:
                old_weight[name] = (old_weight[name].iat[0] / 100)
                # New weights of the total (have increased from 100)
                # Difference to adjust:
                new_weight_all[name] = (weight_dict[name]*old_weight['portfolio_monthly_returns_cum'].iat[0]/100) 
                # Check if someone is +-X% of their own weighting
                if old_weight[name].iat[0] >= new_weight_all[name].iat[0]*(1+pct/100):
                    tickers_down.append(name)
                    stop = True
                # elif new_weight[name].iat[0] <= weight_dict[name]*0.9:
                elif old_weight[name].iat[0] <= new_weight_all[name].iat[0]*(1-pct/100):
                    tickers_up.append(name)
                    stop = True
                # Change new_weight to what is to be adjusted instead of the total value
                new_weight_all[name] = new_weight_all[name] - old_weight[name]
            if stop:
                df_copy = old_weight.copy(deep=True)
                df_copy = df_copy.drop('portfolio_monthly_returns_cum', axis=1)
                df_copy_all = new_weight_all.copy(deep=True)
                df_copy_all = df_copy_all.drop('portfolio_monthly_returns_cum', axis=1)
                # Start with those to be adjusted down and adjust with the largest value in the sorted list
                # If it gets less than 0 remove it. Order should not matter then.
                to_change_up = df_copy_all.sort_values(by=i-1, axis = 1, ascending=False)
                tickers_up_sort = list(to_change_up.columns)
                while len(tickers_down) > 0:
                    left_to_change = df_copy_all[tickers_down[0]].iat[0]
                    # Check what remains if the largest value from the second list is taken > 0
                    # Take it within 1% in case of rounding errors
                    if left_to_change + df_copy_all[tickers_up_sort[0]].iat[0] < -0.01:
                        # Adjust
                        # Change the large value
                        df_copy[tickers_down[0]] = df_copy[tickers_down[0]].iat[0] - df_copy_all[tickers_up_sort[0]].iat[0]
                        # Adjust the value that changed
                        df_copy[tickers_up_sort[0]] = df_copy[tickers_up_sort[0]].iat[0] + df_copy_all[tickers_up_sort[0]].iat[0]
                        # Calculate what is left to adjust and adjust
                        df_copy_all[tickers_down[0]] = df_copy_all[tickers_down[0]].iat[0] + df_copy_all[tickers_up_sort[0]].iat[0]
                        # Reset the one used to adjust
                        df_copy_all[tickers_up_sort[0]] = df_copy_all[tickers_up_sort[0]].iat[0] - df_copy_all[tickers_up_sort[0]].iat[0]
                        # If this should also be adjusted up, remove from there
                        if tickers_up_sort[0] in tickers_up:
                            tickers_up.remove(tickers_up_sort[0])
                        # It no longer needs to be adjusted, so can be removed
                        tickers_up_sort.pop(0)
                        # Count a transaction for the one being removed
                        transactions += 1
                    # If they are the same size, remove both
                    else:
                        # Adjust what is left from the largest value in the second list
                        # Change the large value with its own remainder
                        df_copy[tickers_down[0]] = df_copy[tickers_down[0]].iat[0] + df_copy_all[tickers_down[0]].iat[0]
                        # Adjust the largest value in the second list by the same amount
                        df_copy[tickers_up_sort[0]] = df_copy[tickers_up_sort[0]].iat[0] - df_copy_all[tickers_down[0]].iat[0]
                        # Reset the one used to adjust
                        df_copy_all[tickers_up_sort[0]] = df_copy_all[tickers_up_sort[0]].iat[0] + df_copy_all[tickers_down[0]].iat[0]
                        # Calculate what is left to adjust and adjust, take the change value first
                        df_copy_all[tickers_down[0]] = df_copy_all[tickers_down[0]].iat[0] - df_copy_all[tickers_down[0]].iat[0]
                        # The rest is now 0 and can be removed and go on the next value
                        tickers_down.pop(0)
                        # Count two transactions for the one removing & the one being adjusted with
                        transactions += 2
                # Now take the ones to be adjusted up (if they exist)
                to_change_down = df_copy_all.sort_values(by=i-1, axis = 1)
                tickers_down_sort = list(to_change_down.columns)
                while len(tickers_up) > 0:
                    left_to_change = df_copy_all[tickers_up[0]].iat[0]
                    if left_to_change + df_copy_all[tickers_down_sort[0]].iat[0] > 0.01:
                        df_copy[tickers_up[0]] = df_copy[tickers_up[0]].iat[0] - df_copy_all[tickers_down_sort[0]].iat[0]
                        df_copy[tickers_down_sort[0]] = df_copy[tickers_down_sort[0]].iat[0] + df_copy_all[tickers_down_sort[0]].iat[0]
                        df_copy_all[tickers_up[0]] = df_copy_all[tickers_up[0]].iat[0] + df_copy_all[tickers_down_sort[0]].iat[0]
                        df_copy_all[tickers_down_sort[0]] = df_copy_all[tickers_down_sort[0]].iat[0] - df_copy_all[tickers_down_sort[0]].iat[0]
                        tickers_down_sort.pop(0)
                        transactions += 1
                    else:
                        df_copy[tickers_up[0]] = df_copy[tickers_up[0]].iat[0] + df_copy_all[tickers_up[0]].iat[0]
                        df_copy[tickers_down_sort[0]] = df_copy[tickers_down_sort[0]].iat[0] - df_copy_all[tickers_up[0]].iat[0]
                        df_copy_all[tickers_down_sort[0]] = df_copy_all[tickers_down_sort[0]].iat[0] + df_copy_all[tickers_up[0]].iat[0]
                        df_copy_all[tickers_up[0]] = df_copy_all[tickers_up[0]].iat[0] - df_copy_all[tickers_up[0]].iat[0]
                        tickers_up.pop(0)
                        transactions += 2
                # We now have the new weights (df_copy) and can multiply them by the monthly change
                to_append = df_copy * (monthly_returns.iloc[i]+1) *100
                to_append['portfolio_monthly_returns_cum'] = to_append.iloc[0].sum()
                dynamic_portfolio = pd.concat([dynamic_portfolio, to_append], ignore_index=True)
            else:
                to_append = dynamic_portfolio.drop('portfolio_monthly_returns_cum', axis=1)
                to_append = to_append.iloc[-1]
                to_append = to_append * (monthly_returns.iloc[i]+1)
                to_append['portfolio_monthly_returns_cum'] = to_append.sum()
                dynamic_portfolio = pd.concat([dynamic_portfolio, to_append.to_frame().T], ignore_index=True)
                
    dynamic_portfolio['portfolio_monthly_returns'] = dynamic_portfolio['portfolio_monthly_returns_cum'].pct_change()
    print('Number of transactions:')
    print(transactions)
    dynamic_portfolio = dynamic_portfolio.set_index(data.index)
    calc_risk(dynamic_portfolio, transactions)
    # print(dynamic_portfolio)
    return dynamic_portfolio
        
def random_weights(n):
    k = np.random.rand(n)
    return k / sum(k)

def calc_risk(df, transactions=False):
    # Average annual return
    # print(df)
    # CAGR = (vf/vi)^(1/years)-1
    cagr = (df['portfolio_monthly_returns_cum'].iloc[-1]/df['portfolio_monthly_returns_cum'].iat[0])**(1/years) - 1
    print('ANNUAL RETURN (CAGR): ')
    print(cagr)
    # Std dev & sharpe
    std_dev = df['portfolio_monthly_returns'].std()
    # std dev of negative returns
    std_neg = df['portfolio_monthly_returns'][df['portfolio_monthly_returns']<0].std()
    print(df['portfolio_monthly_returns'][df['portfolio_monthly_returns']<0])
    # data2 = data.index[data['valuecol1'] > 0]
    # print('STD DEV:')
    # print(std_dev*12)
    print('VOLATILITY:')
    vol = (std_dev*math.sqrt(12))*100
    print(vol)
    mean_return = df['portfolio_monthly_returns'].mean()
    mean_return_rate = risk_free_rate['Rate'].mean()
    sharpe_ratio = (mean_return-mean_return_rate) / std_dev
    sharpe_ratio_yearly = sharpe_ratio*math.sqrt(12)
    print('SHARPE RATIO:')
    print(sharpe_ratio_yearly)
    sortiono = (mean_return-mean_return_rate) / std_neg
    sortiono_yearly = sortiono*math.sqrt(12)
    print('SORTINO RATIO: ')
    print(sortiono_yearly)
    # Max yearly drawdown
    # Calculates the value that has been from the start 
    roll_max = df['portfolio_monthly_returns_cum'].cummax()
    # Calculates the maximum drop from the maximum value that day.
    monthly_drawdown = df['portfolio_monthly_returns_cum']/roll_max - 1
    max_monthly_drawdown = monthly_drawdown.cummin()
    df['monthly_drawdown'] = monthly_drawdown
    print('MAX DRAWDOWN:')
    print(max_monthly_drawdown.min())
    calmars = (mean_return*12)/abs(max_monthly_drawdown.min())
    print('CALMARS RATIO: ')
    print(calmars)
    if transactions:
        create_figure(df, 'portfolio_monthly_returns_cum', 'Dragon Portfolio (Rebalance every 20%)', cagr, vol, sharpe_ratio_yearly, \
            sortiono_yearly, max_monthly_drawdown.min(), calmars, transactions)
    else:
        create_figure(df, 'portfolio_monthly_returns_cum', 'Dragon Portfolio (Rebalance every 12th month)', cagr, vol, sharpe_ratio_yearly, \
            sortiono_yearly, max_monthly_drawdown.min(), calmars)
    create_figure(df, 'monthly_drawdown', 'Dragon Portfolio (Rebalance every 20%), drawdowns')
    max_monthly_drawdown.plot()
    plt.show()

def fire(df):
    print(df)
    start_val = 5000000
    withdrawal_per_year = start_val/25
    # CPI start 2007-11
    cpi = [1.00, 1.025, 1.018, 1.037, 1.062, 1.061, 1.062, 1.060, 1.061, 1.076, 1.096, 1.117, 1.137, 1.139]
    money_left = [start_val]
    num = 0
    for row in df.iterrows():
        if num == 0:
            num += 1
        elif num % 12 == 0:
            new_withdrawal_sum = withdrawal_per_year*cpi[int(num/12)]
            money_left.append((money_left[num-1]-new_withdrawal_sum) * (1+row[-1][-2]))
            num += 1
        else:
            money_left.append(money_left[num-1] * (1+row[-1][-2]))
            num += 1
    df['money_left'] = money_left
    # Calculate expenses per year
    expenses = [withdrawal_per_year]
    num1 = 0
    for i in range(len(money_left)):
        if i == 0:
            pass
        elif i % 12 == 0:
            to_add = expenses[num1] * cpi[num1+1]
            expenses.append(to_add)
            num1 += 1
        else:
            to_add = expenses[num1] * cpi[num1]
            expenses.append(to_add)
    print(money_left[-1]/money_left[0])
    df['expenses'] = expenses
    df['expenses'] *= 25
    create_figure2(df, 'money_left', 'Money left (start with 25X yearly expenses) - 60/40', 'expenses', '25X yearly expenses')
    return


def create_figure(df, series_name, label, cagr=False, vol=False, sharpe=False, sortino=False, max_drawdown=False, calmars=False, transactions=False):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    x_start = len(df.index) % 79 - 1
    if x_start < 0:
        x_start = 0
        # x_start = len(df.index) // 53 - 1
    # ax1.set_ylabel('Andel över sitt 200 Day Moving Average', color=color)
    ax1.plot(df.index[x_start:], df[series_name].iloc[x_start:], label=label)
    # ax1.tick_params(axis='y')
    every_nth = len(df.index) // 79
    num = 2
    for n, lab in enumerate(ax1.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            lab.set_visible(False)
            num += 1
        else:
            if num % 3 == 0:
                lab.set_visible(False)
                num += 1
    for n, tick in enumerate(ax1.axes.get_xticklines()):
        if n % every_nth != 0:
            tick.set_visible(False)
    #ax1.set_xticklabels(df.index[x_start:], rotation=40, ha='right')
    ax1.set_xlim(df.index[x_start], df.index[-1])
    ax1.tick_params(axis='x', rotation=40)
    ax1.legend()
    if cagr:
        plt.annotate('Annualized return: ' + str(round(cagr*100, 1)) + ' %', xy=(0.05, 0.88), xycoords='axes fraction', size='medium')
        plt.annotate('Volatility: ' + str(round(vol, 1)), xy=(0.05, 0.84), xycoords='axes fraction', size='medium')
        plt.annotate('Sharpe ratio: ' + str(round(sharpe, 2)), xy=(0.05, 0.8), xycoords='axes fraction', size='medium')
        plt.annotate('Sortino ratio: ' + str(round(sortino, 2)), xy=(0.05, 0.76), xycoords='axes fraction', size='medium')
        plt.annotate('Max drawdown: ' + str(round(max_drawdown, 2)) + ' %', xy=(0.05, 0.72), xycoords='axes fraction', size='medium')
        plt.annotate('Calmar ratio: ' + str(round(calmars, 2)), xy=(0.05, 0.68), xycoords='axes fraction', size='medium')
    if transactions:
        plt.annotate('Number of transactions: ' + str(transactions), xy=(0.05, 0.64), xycoords='axes fraction', size='medium')
    plt.title(label)
    if not cagr:
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax1.xaxis_date()
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax1.xaxis.set_major_locator(mtick.MaxNLocator(30))

    plt.show()
    # name = str(datetime.now().strftime('%Y-%m-%d') + '_' + label + '.jpg')
    # fig.savefig(name, dpi=250)
    plt.close(fig)

def create_figure2(df, series_name, label, series_name2=False, label2=False):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    x_start = len(df.index) % 79 - 1
    if x_start < 0:
        x_start = 0
    ax1.plot(df.index[x_start:], df[series_name].iloc[x_start:], label=label)
    if series_name2:
        ax1.plot(df.index[x_start:], df[series_name2].iloc[x_start:], label=label2)
    every_nth = len(df.index) // 79
    num = 2
    for n, lab in enumerate(ax1.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            lab.set_visible(False)
            num += 1
        else:
            if num % 3 == 0:
                lab.set_visible(False)
                num += 1
    for n, tick in enumerate(ax1.axes.get_xticklines()):
        if n % every_nth != 0:
            tick.set_visible(False)
    ax1.set_xticklabels(df.index[x_start:], rotation=40, ha='right')
    ax1.set_xlim(df.index[x_start], df.index[-1])
    ax1.tick_params(axis='x', rotation=40)
    ax1.legend()
    plt.title(label)
    plt.show()
    plt.close(fig)


def main():



    # fire(get_data_weight_every_month())
    # get_data_weight_every_month()
    get_data_weight_every_x_month(12)
    get_data_every_pct(20)


if __name__ == "__main__":
    main()
