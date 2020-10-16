import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')

tickers = ['HD', 'DIS','MSFT', 'BA', 'MMM', 'PFE', 'NKE', 'JNJ', 'MCD', 'INTC', 'XOM', 'GS', 'JPM', 'AXP', 'V', 'IBM', 'UNH', 'PG', 'GE', 'KO', 'CSCO', 'CVX', 'CAT', 'MRK', 'WMT', 'VZ', 'RTX', 'TRV', 'AAPL', 'ADBE', 'EBAY', 'QCOM', 'HPQ', 'JNPR', 'AMD']

def download_stocks_data():
    
    pd.core.common.is_list_like = pd.api.types.is_list_like
    from pandas_datareader import data as pdr
    import datetime
    import fix_yahoo_finance as yf
    
    start = datetime.datetime(2013, 1, 1)
    end = datetime.datetime(2018, 1, 1) 
    
    #5 years EOD data for 36 volatile US companies
    df = pdr.get_data_yahoo(tickers, start, end)['Close']
    
    #This downloads the data in home directory
    df.to_csv('pairs_data.csv')
    
    """
                       HD        DIS       MSFT  ...       HPQ       JNPR   AMD
    Date                                         ...                           
    2012-12-31  61.849998  49.790001  26.709999  ...  6.471390  19.670000  2.40
    2013-01-02  63.480000  51.099998  27.620001  ...  6.821072  20.549999  2.53
    2013-01-03  63.299999  51.209999  27.250000  ...  6.875567  20.170000  2.49
    2013-01-04  63.180000  52.189999  26.740000  ...  6.875567  20.379999  2.59
    2013-01-07  62.840000  50.970001  26.690001  ...  6.889192  20.150000  2.67

    [5 rows x 35 columns]
    """

data = pd.read_csv('pairs_data.csv')
data.set_index('Date', inplace = True)

# print (data.head())

# Finding pairs of companies for which stock price movements are cointegrated
 
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
                
    return score_matrix, pvalue_matrix, pairs
    """
    (array([[ 0.        ,  0.01923121, -2.76331235, ..., -0.76320512,
        -1.43098299, -0.64297937],
       [ 0.        ,  0.        , -1.75607   , ..., -1.56613894,
        -3.11076918, -2.22214599],
       [ 0.        ,  0.        ,  0.        , ..., -1.49655346,
        -1.33900035, -1.31791459],
       ...,
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
        -1.87720205, -2.47895264],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        , -2.4857934 ],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ]]), array([[1.        , 0.98633715, 0.17763738, ..., 0.93937044, 0.78669597,
        0.95244905],
       [1.        , 1.        , 0.6508046 , ..., 0.73496788, 0.08616026,
        0.4129744 ],
       [1.        , 1.        , 1.        , ..., 0.7625113 , 0.81765691,
        0.82426453],
       ...,
       [1.        , 1.        , 1.        , ..., 1.        , 0.59152336,
        0.28842961],
       [1.        , 1.        , 1.        , ..., 1.        , 1.        ,
        0.28537026],
       [1.        , 1.        , 1.        , ..., 1.        , 1.        ,
        1.        ]]), [('HD', 'V'), ('HD', 'TRV'), ('MSFT', 'MMM'), ('MSFT', 'JNJ'), ('MSFT', 'UNH'), ('MSFT', 'ADBE'), ('MMM', 'UNH'), ('MMM', 'CSCO'), ('MMM', 'AAPL'), ('MMM', 'ADBE'), ('PFE', 'INTC'), ('PFE', 'XOM'), ('PFE', 'JPM'), ('PFE', 'V'), ('PFE', 'IBM'), ('PFE', 'UNH'), ('PFE', 'CSCO'), ('PFE', 'TRV'), ('PFE', 'AAPL'), ('PFE', 'ADBE'), ('PFE', 'EBAY'), ('PFE', 'JNPR'), ('NKE', 'IBM'), ('JNJ', 'KO'), ('JNJ', 'ADBE'), ('XOM', 'QCOM'), ('JPM', 'EBAY'), ('V', 'UNH'), ('V', 'TRV'), ('UNH', 'CSCO'), ('UNH', 'TRV'), ('PG', 'CSCO'), ('PG', 'CAT'), ('PG', 'AAPL'), ('PG', 'ADBE'), ('PG', 'EBAY'), ('PG', 'HPQ'), ('PG', 'AMD'), ('GE', 'WMT'), ('KO', 'CSCO'), ('KO', 'TRV'), ('KO', 'AAPL'), ('KO', 'ADBE'), ('KO', 'EBAY'), ('KO', 'QCOM'), ('KO', 'HPQ'), ('KO', 'AMD'), ('CSCO', 'TRV'), ('VZ', 'RTX'), ('VZ', 'TRV'), ('VZ', 'AAPL'), ('VZ', 'ADBE'), ('VZ', 'EBAY'), ('VZ', 'QCOM'), ('VZ', 'HPQ'), ('VZ', 'JNPR'), ('VZ', 'AMD')])
    """

# Heatmap to show p-value of cointegration between pairs of stocks

def heatmap(data):
    scores, pvalues, pairs = find_cointegrated_pairs(data)
    import seaborn
    fig, ax = plt.subplots(figsize = (10, 10))
    seaborn.heatmap(pvalues, xticklabels = tickers, yticklabels = tickers, cmap = 'RdYlGn_r', mask = (pvalues >= 0.01) )
    print (pairs)

# Lets select from the pairs to create a trading strategy based on those stocks

S1 = data['HD'].copy()    # Home Depot Inc - largest home improvement retailer in the United States, supplying tools, construction products, and services. 
S2 = data['TRV'].copy()    # The Travelers Companies - second-largest writer of U.S. commercial property casualty insurance, and the sixth-largest writer of U.S. personal insurance through independent agents

score, pvalue, _ = coint(S1, S2)
# print (pvalue)    # 0.004232376153277136

# Calculating the Spread

def spread(S1, S2):
    
    # We use a linear regression to get the coefficient for the linear combination to construct between our two securities
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    S1 = data['HD'].copy()
    b = results.params[1]
    
    spread = S2 - b * S1
    spread.plot(figsize=(12,6))
    plt.axhline(spread.mean(), color='black')
    plt.legend(['Spread']);
    
    # We can clearly see that the spread plot moves around mean


def ratio(S1, S2):
    ratio = S1 / S2
    ratio.plot(figsize = (12, 6))
    a = ratio.mean()
    plt.axhline(a, color = 'black')
    plt.legend(['Price Ratio'])
 
    # We now need to standardize this ratio (using z-score) because the absolute ratio might not be the most ideal way of analyzing this trend.

def zscore(S1, S2):
    ratio = S1 / S2
    b = ratio.mean()
    score =  (ratio - b) / np.std(ratio)
    a = score.mean()
    score.plot(figsize = (12, 6))
    plt.axhline(a)
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    plt.show()
    
    # Using indicator lines(green, red) we clearly see that big divergences come back to mean

# Trading Signals

ratios = S1 / S2
a = int (len(ratios) * 0.70)
train = ratios[:a]
test = ratios[a:]
    
def rolling_ratio_zscore(S1, S2):
    
    ratios_mavg5 = train.rolling(window=5, center=False).mean()
    ratios_mavg60 = train.rolling(window=60, center=False).mean()
    std_60 = train.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    plt.figure(figsize=(12, 6))
    train.plot()
    ratios_mavg5.plot()
    ratios_mavg60.plot()
    plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])
    plt.ylabel('Ratio')
    plt.show()

def trade_signals(S1, S2):
    ratios_mavg5 = train.rolling(window=5, center=False).mean()
    ratios_mavg60 = train.rolling(window=60, center=False).mean()
    std_60 = train.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    
    plt.figure(figsize=(12,6))
    zscore_60_5.plot()
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
    plt.show()


# When you buy the ratio, you buy stock S1 and sell S2 , When you sell the ratio, you sell stock S1 and buy S2

def ratio_with_signal(S1, S2):
    ratios_mavg5 = train.rolling(window=5, center=False).mean()
    ratios_mavg60 = train.rolling(window=60, center=False).mean()
    std_60 = train.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    
    plt.figure(figsize=(12,6))

    train[160:].plot()
    buy = train.copy()
    sell = train.copy()
    buy[zscore_60_5>-1] = 0
    sell[zscore_60_5<1] = 0
    buy[160:].plot(color='g', linestyle='None', marker='^')
    sell[160:].plot(color='r', linestyle='None', marker='^')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, ratios.min(), ratios.max()))
    plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
    plt.show()

# Trading using a simple strategy

def trade(S1, S2, window1, window2):
    
    if (window1 == 0) or (window2 == 0):
        return 0
    
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    
    # Simulate trading
    # Start with no money and no positions
    money = 1000
    countS1 = 0
    countS2 = 0
    max_positions = 5
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] > 1:
            if countS1 >= 0 and countS2 >= 0:
                money += S1[i] - S2[i] * ratios[i]
                countS1 -= 1
                countS2 += ratios[i]
                print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] < -1:
            if countS1 <= max_positions and countS2 <= max_positions :
                money -= S1[i] - S2[i] * ratios[i]
                countS1 += 1
                countS2 -= ratios[i]
                print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.75 and .75
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
    print (money)

trade(data.HD[a:], data.TRV[a:], 5, 60)
# ratio_with_signal(S1, S2)
# rolling_ratio_zscore(S1, S2)
# trade_signals(S1, S2)
# rolling_ratio_zscore(S1, S2)
# zscore(S1, S2)
# ratio(S1, S2)
# spread(S1, S2)
# heatmap(data)
# print (find_cointegrated_pairs(data))