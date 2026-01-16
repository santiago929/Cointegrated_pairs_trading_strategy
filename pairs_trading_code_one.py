# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:14:45 2023

@author: Santiago
"""
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate
import datetime

import statsmodels.api as stat
from statsmodels.tsa.api import VAR
import statsmodels.tsa.stattools as ts

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

symbol_list = ['PL=F','HG=F']

data = pd.DataFrame(columns=symbol_list)  

for ticker in symbol_list:
    data[ticker] = yf.download(ticker, '2019-07-08', '2023-07-07')['Adj Close']

data = data.dropna()

scaler = StandardScaler()

scaled = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled, columns=[symbol_list[0],symbol_list[1]], index=data.index)


def unit_root(ticker_list):
    
    for i in ticker_list:
        u_r = ts.adfuller(data[i])
        if u_r[0] <= u_r[4]['5%'] and u_r[1]<= 0.05:
            print('Time series is statiorary')
        else:
            print("Time series are non-stationary")
            
    return u_r

unit_root(symbol_list)


def plot_price_series(df, ts1, ts2):
    
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1],label=ts1)
    ax.plot(df.index, df[ts2],label=ts2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2019, 7, 8), datetime.datetime(2023,7,7))
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.figure(figsize=(30,10))
    plt.show()

def plot_residual(df, high_bound=None, low_bound=None):
    
    
    fig, ax= plt.subplots()
    ax.plot(df.index, df['e_hat'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2019, 7, 8), datetime.datetime(2023,7,7))
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.title('Residual plot')
    
    if high_bound and low_bound is not None: 
        plt.axhline(y = high_bound, color = 'b', linestyle = 'dashed')
        plt.axhline(y = low_bound, color = 'b', linestyle = 'dashed')
    else:  
        plt.plot(df['e_hat'])
        plt.show()
    
def engle_gragner(df, tkt1, tkt2):
    
    lr_model = LinearRegression(copy_X=True, fit_intercept=True)
    lr_model.fit(df[tkt2].values.reshape(-1,1), df[tkt1].values) 
    beta_hr = lr_model.coef_
    constant = lr_model.intercept_
    print('parameters: %.7f, %.7f' %(lr_model.intercept_, lr_model.coef_))
    df['e_hat'] = df[tkt1] - beta_hr*df[tkt2] - constant 
    cadf = ts.adfuller(df["e_hat"])
    print(cadf[1], cadf[0], cadf[4])
    delta_y = pd.DataFrame(df[tkt1]).diff().dropna()
    delta_x = pd.DataFrame(df[tkt2]).diff().dropna()
    e_hat_df = pd.DataFrame(df['e_hat'])
    rhs = delta_x.join(e_hat_df.shift(1).dropna().add_prefix('(Lag 1, ').add_suffix(')'))
    regression = stat.OLS(delta_y, rhs).fit()
    print(regression.summary())
    
    return beta_hr
    
def OU_process(res_df,data_frequency):
    tau = 1/data_frequency
    OU = VAR(res_df)
    result = OU.fit(1)
    print(result.summary())
    beta_1 = result.coefs[0,0][0] ## B
    theta = -np.log(beta_1)/tau
    H = np.log(2)/theta
    working_days = H/tau 
    beta_0 = result.coefs_exog[0][0] ## C
    mu = beta_0 / (1-beta_1)
    se = (result.resid)**2
    sse = np.sum(se['e_hat'])
    sigma_eq = np.sqrt((sse*tau)/(1-(np.exp(-2*theta*tau))))
    params = {'Beta_1':beta_1,'Theta':theta,'Half life': H,
                         'Working days':working_days,'Beta_0':beta_0,
                        'μ':mu, 'sigma_eq':sigma_eq}
    print(tabulate(pd.DataFrame(list(params.items()))))
    return [mu, sigma_eq]
    
if __name__ == "__main__":
    
    plot_price_series(data, symbol_list[0], symbol_list[1])
    engle_gragner(scaled_df, symbol_list[0], symbol_list[1])
    plot_residual(scaled_df)
    e = pd.DataFrame(scaled_df['e_hat'])
    e_t1 = e.join(e.shift(1).dropna().add_suffix('_t-1'))
    e = e_t1[1:]
    OUtput = OU_process(e,252)
    
    print(pd.DataFrame({'mu_eq': OUtput[0], 'sigma_eq': OUtput[1],
                                 'beta_hr': engle_gragner(scaled_df, symbol_list[0], 
                                                          symbol_list[1])}))





