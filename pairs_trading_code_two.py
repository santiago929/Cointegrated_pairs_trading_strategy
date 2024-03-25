# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:17:40 2023

@author: Santiago

"""

""" 
    CONSIDERATIONS:
    1. CHANGE THE csv_dir VARIABLE IN LINE #710.
    2. RUN THE CODE FROM HERE TO THE BOTTOM.
    3. csv FILES 'PL=F, HG=F, spx_data' SHOULD BE IN THE SAME FOLDER
    4. THIS CODE WILL REPRODUCE TWO csv FILES FOR RESULT's ANALYSIS NAMED:
        results.csv AND opt.csv'
    5. VARIABLES IN LINE #625 IS EXTRACTED FROM file: pairs_trading_code_one
    6. COMMENTS ARE FOUND THROUGH THE CODE FOR FURTHER GUIDANCE
"""

from __future__ import print_function

from abc import ABCMeta
import datetime
import numpy as np
import pandas as pd
import os, os.path

import time
import pprint

from scipy.stats import norm

from sklearn.linear_model import LinearRegression    
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Event(object):
    pass

class MarketEvent(Event):
    
    def __init__(self):
        self.type = 'MARKET'
        
class SignalEvent(Event):

    def __init__(self, strategy_id, symbol, datetime, signal_type, beta):

        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.beta = beta 

class OrderEvent(Event):
    
    def __init__(self, symbol, order_type, quantity, direction):

        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        
    def print_order(self):
        print(
            "Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s" %
            (self.symbol, self.order_type, self.quantity, self.direction)
            )
        
class FillEvent(Event):
    
    def __init__(self, timeindex, symbol, exchange, quantity,
                 direction, fill_cost, commission=None):

        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def calculate_ib_commission(self):
        
        pass

class DataHandler(object):
    
    __metaclass__ = ABCMeta

    def get_latest_bar(self, symbol):

        raise NotImplementedError("Should implement get_latest_bar()")
    
    def get_latest_bars(self, symbol, N=1):

        raise NotImplementedError("Should implement get_latest_bars()")

    def get_latest_bar_datetime(self, symbol):
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    def get_latest_bar_value(self, symbol, val_type):

        raise NotImplementedError("Should implement get_latest_bar_value()")

    def get_latest_bars_values(self, symbol, val_type, N=1):

        raise NotImplementedError("Should implement get_latest_bars_values()")

    def update_bars(self):

        raise NotImplementedError("Should implement update_bars()")



class HistoricCSVDataHandler(DataHandler):
       
    def __init__(self, events, csv_dir, symbol_list):
        
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True 
        
        self._open_convert_csv_files()
        
    def _open_convert_csv_files(self):
    
        comb_index = None
        for s in self.symbol_list:
            self.symbol_data[s]= pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s), 
                header=0, index_col=0, parse_dates=True,
                names=[
                    'datetime', 'open', 'high',
                    'low', 'close', 'vol', 
                    'adj_close', 'scaled_adj_close'
                    ])
        
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)
            
            self.latest_symbol_data[s] = []

        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].\
                reindex(index=comb_index, method='pad').iterrows()


    def _get_new_bar(self,symbol):
        
        for b in self.symbol_data[symbol]:
            yield b
            
    def get_latest_bar(self, symbol):
    
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]
    
    def get_latest_bars(self, symbol, N=1):
   
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]
        
    def get_latest_bar_datetime(self, symbol):

        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        
        try:
           bars_list = self.latest_symbol_data[symbol]
        except KeyError:
           print("That symbol is not available in the historical data set.")
           raise
        else:
           return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):

        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])
   
    def update_bars(self):
        
        for s in symbol_list:
            try:
                bar= next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())
        
            
try:
    import Queue as queue
except ImportError:
    import queue
    
class Strategy(object):
    
    __metaclass__ = ABCMeta
    
    def calculate_signals(self):
        raise NotImplementedError("Should implement calculate_signals()")


def create_sharpe_ratio(returns, periods=252):
    
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)

def create_drawdowns(pnl):
   
    hwm = [0]  
    
    idx = pnl.index
    drawdown = pd.Series(index = idx)
    duration = pd.Series(index = idx)
    
    for t in range(1,len(idx)):
        hwm.append(max(hwm[t-1], pnl[t]))
        drawdown[t] = (hwm[t]-pnl[t])
        duration[t] = (0 if drawdown[t] == 0 else duration[t-1]+1)
    return drawdown, drawdown.max(), duration.max()

def var_cov_var(cash, strat_ret, c= 0.95):
    
    ret_mean = np.mean(strat_ret)
    ret_sigma = np.std(strat_ret)
    
    alpha = norm.ppf(1-c, ret_mean, ret_sigma) 
    
    return cash - cash*(alpha+1)

def market_beta(x, y, N):
    
    obs = len(x)
    betas = np.full(obs, np.nan)
    alphas = np.full(obs, np.nan)

    
    for i in range((obs-N)):
        regressor = LinearRegression()
        regressor.fit(x.to_numpy()[i : i + N+1].reshape(-1,1), 
                                 y.to_numpy()[i : i + N+1])
        
        betas[i+N]  = regressor.coef_[0]
        alphas[i+N]  = regressor.intercept_       
        
    return (alphas, betas) 

def rolling_beta(benchmark_df, strategy_df, csv_dir, rolling_period):

    analytics = market_beta(benchmark_df['rtns-r_f'], 
                            strategy_df['returns'].fillna(0), rolling_period)
    analytics = pd.DataFrame(list(zip(*analytics)), columns = ['alpha', 'beta'])
    analytics.index = strat_df.index
    plt.figure(figsize=(12,8))
    plt.title('Rolling Beta')
    analytics.beta.plot.line()
    plt.grid(True)

def plot_results(strategy_df):
    
    fig = plt.figure(figsize= (30,20))
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(311)
    ax1.set_ylabel('Portfolio value, %', fontsize = 22.0)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.set_xlabel('Date', fontsize = 22.0)
    strategy_df['equity_curve'].plot(ax=ax1, color="blue", lw=2.)
    plt.grid(True)
    plt.xticks(fontsize = 20.0)
    plt.yticks(fontsize = 20.0)

    ax2 = fig.add_subplot(312)
    ax2.set_ylabel('Period returns, %', fontsize = 22.0)
    ax2.set_xlabel('Date', fontsize = 22.0)
    strategy_df['returns'].plot(ax=ax2, color="black", lw=2.)
    plt.grid(True)
    plt.xticks(fontsize = 20.0)
    plt.yticks(fontsize = 20.0)
   
    ax3 = fig.add_subplot(313)
    ax3.set_ylabel('Drawdowns, %', fontsize = 22.0)
    ax3.set_xlabel('Date', fontsize = 22.0)
    strategy_df['drawdown'].plot(ax=ax3, color="red", lw=2.)
    plt.grid(True)
    plt.xticks(fontsize = 20.0)
    plt.yticks(fontsize = 20.0)  
    plt.show()
    
    

class Portfolio(object):
    
    def __init__(self, bars, events, start_date, csv_dir, initial_capital = 100000.0):
        
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        
        self.all_positions = self.construct_all_positions()
        self.current_positions = dict( (k,v) for k, v in \
                                      [(s,0) for s in self.symbol_list])
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()
        self.csv_dir = csv_dir
            
    def construct_all_positions(self):
        
        d = dict( (k,v) for k, v in [(s,0) for s in self.symbol_list] )
        d['datetime'] = self.start_date
        return [d]
    
    def construct_all_holdings(self):
        
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]
            
    def construct_current_holdings(self):

        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d             
            
    def update_timeindex(self, event):
        
        latest_datetime = self.bars.get_latest_bar_datetime(
            self.symbol_list[0])
        
        dp = dict( (k,v) for k, v in [(s,0) for s in self.symbol_list])
        dp['datetime'] = latest_datetime
        
        for s in self.symbol_list:
            dp[s] = self.current_positions[s]
        
        self.all_positions.append(dp)
        
        dh = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']
        
        for s in self.symbol_list:
            market_value = self.current_positions[s] * \
                self.bars.get_latest_bar_value(s, "adj_close")
            dh[s] = market_value
            dh['total'] += market_value
            
        self.all_holdings.append(dh)
        
        
    def update_positions_from_fill(self, fill):
        
        fill_dir = 0
        
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1
        self.current_positions[fill.symbol] += fill_dir*fill.quantity
        
        
    def update_holdings_from_fill(self, fill):
        
        fill_dir = 0
        
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1
        
        fill_cost = self.bars.get_latest_bar_value(fill.symbol, "adj_close")
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= (cost + fill.commission)
        
    def update_fill(self, event):

        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)
            
    def generate_naive_order(self, signal):
        
        order = None
        
        symbol = signal.symbol
        direction = signal.signal_type
        beta = signal.beta
        
        mkt_quantity = 100
        cur_quantity = self.current_positions[symbol]
        order_type = 'MKT'
        
        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity*round(beta,2), 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity*round(beta,2), 'SELL')
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')

        return order

    def update_signal(self, event):
        
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)
            
    def create_equity_curve_dataframe(self):

        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        curve['VaR'] = var_cov_var(curve['total'], curve['returns'])
        
        self.equity_curve = curve
        
    def output_summary_stats(self, csv_dir, MADD_pct = 0.25):

        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        sharpe_ratio = create_sharpe_ratio(returns, periods=252)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown
        self.equity_curve['dd_$'] = drawdown*self.equity_curve['total']
        self.equity_curve['MADD'] = MADD_pct*self.equity_curve['total']
        self.equity_curve['RHS'] = self.equity_curve['MADD'] - self.equity_curve['dd_$']
        

        stats = [("Total Return", "%0.2f%%" % \
        ((total_return - 1.0) * 100.0)),
        ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
        ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
        ("Drawdown Duration", "%d" % dd_duration)]
        self.equity_curve.to_csv(csv_dir+'/results.csv')
        return stats

               
class ExecutionHandler(object):

    __metaclass__ = ABCMeta

    def execute_order(self, event):
        
        raise NotImplementedError("Should implement execute_order()")

class SimulatedExecutionHandler(ExecutionHandler):
    
    def __init__(self, events):
        
        self.events = events
        
    def execute_order(self, event):
        
        if event.type == 'ORDER':
            fill_event = FillEvent(
            datetime.datetime.utcnow(), event.symbol,
            'COMEX', event.quantity, event.direction, fill_cost=0, commission= 0
            )
            self.events.put(fill_event)


class Backtest(object):
    
    def __init__(
            self, csv_dir, symbol_list, initial_capital,
            heartbeat, start_date, data_handler,
            execution_handler, portfolio, strategy, z_value):
        
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.z_value = z_value
        self._generate_trading_instances(z_value)
        
        

    def _generate_trading_instances(self, z_value):

        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        
        self.data_handler = self.data_handler_cls(self.events, self.csv_dir,
                                                  self.symbol_list)
        
        self.portfolio = self.portfolio_cls(self.data_handler, self.events,
                                            self.start_date, self.csv_dir,
                                            self.initial_capital)
        
        self.strategy = self.strategy_cls(self.data_handler, self.events, self.portfolio, z_value)
        self.execution_handler = self.execution_handler_cls(self.events)
        
    def _run_backtest(self):

        i = 0
        while True:
            i += 1
            
       
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
                break
        
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)
                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)
                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)
                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)
            time.sleep(self.heartbeat)
            
    
    def _output_performance(self):
        
        self.portfolio.create_equity_curve_dataframe()
        stats = self.portfolio.output_summary_stats(csv_dir)
        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)
        
        return stats
        

    def simulate_trading(self, z_list, csv_dir):

        out = open(csv_dir+"/opt.csv", "w")
        
        
        for i, z in enumerate(z_list):
            self._generate_trading_instances(z)
            self._run_backtest()
            stats = self._output_performance()
            pprint.pprint(stats)
            
            tot_ret = stats[0][1].replace("%","")
            cagr = stats[1][1].replace("%","")
            sharpe = stats[2][1]
            max_dd = stats[3][1].replace("%","")
            
            out.write(
                "%s,%s,%s,%s,%s\n" % (z,
                    tot_ret, cagr, sharpe, max_dd))
        out.close()
        
 
class PairsStrategy(Strategy):
    
    def __init__(
        self, bars, events, portfolio, z_value, 
         beta_eg= 0.6708, mu_eq = -0.0035, sigma_eq = 1.4078,
         trading_window= 1):  ### THIS DATA IS EXTRACTED FROM THE CODE ONE. 
                                # trade_window IS FIXED TO "1" TO GRAB THE WHOLE DATA
        low_bound = mu_eq - z_value * sigma_eq
        high_bound = mu_eq + z_value * sigma_eq
        
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.trading_window = trading_window
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.beta_eg = beta_eg
        self.mu_eq = mu_eq
        self.z_value = z_value
        
        self.pair = ('PL=F','HG=F')
        self.datetime = datetime.datetime.utcnow()
        self.long_market = False
        self.short_market = False
        
        self.portfolio_cls = portfolio
        
        
    def calculate_xy_signals(self, resid):
        
        
        y_signal = None
        x_signal = None
        p0 = self.pair[0]
        p1 = self.pair[1]
        dt = self.datetime
        hr = abs(self.beta_eg) 
        
    
        if resid <= self.low_bound and not self.long_market:
            self.long_market = True
            y_signal = SignalEvent(1, p0, dt, 'LONG', 1.0)
            x_signal = SignalEvent(1, p1, dt, 'SHORT', hr)
            
        if resid >= self.mu_eq and self.long_market:
            self.long_market = False
            y_signal = SignalEvent(1, p0, dt, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt, 'EXIT', 1.0)
 
        if resid >= self.high_bound and not self.short_market:
            self.short_market = True
            y_signal = SignalEvent(1, p0, dt, 'SHORT', 1.0)
            x_signal = SignalEvent(1, p1, dt, 'LONG', hr)

        if resid <= self.mu_eq and self.short_market:
            self.short_market = False
            y_signal = SignalEvent(1, p0, dt, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt, 'EXIT', 1.0)
            
        return y_signal, x_signal
    
            
    def calculate_signals_for_pairs(self):
        
        y = self.bars.get_latest_bars_values(
                 self.pair[0], "scaled_adj_close", N=self.trading_window)
             
        x = self.bars.get_latest_bars_values(
                self.pair[1], "scaled_adj_close", N=self.trading_window)
        
        if y is not None and x is not None:

            if len(y) >= self.trading_window and len(x) >= self.trading_window:

                resid = (y - self.beta_eg * x)[-1]

                y_signal, x_signal = self.calculate_xy_signals(resid)
                if y_signal is not None and x_signal is not None:
                    self.events.put(y_signal)
                    self.events.put(x_signal)                    
                            
                    
    def calculate_signals(self, event):

        if event.type == 'MARKET':
            self.calculate_signals_for_pairs()
            
    
if __name__ == "__main__":
    csv_dir = '' ### MUST BE CHANGED!
    symbol_list = ['PL=F','HG=F']                                         
    initial_capital = 100000.0
    heartbeat = 0.0
    start_date = datetime.datetime(2019, 7, 8)
    
    ## ACTIVATE THE TWO LINES BELOW TO RUN THE STRATEGY WITH DIFFERENT Z ##
    
    # z_list = np.arange(0.5,1.6,0.1)         
    # z_list = np.round(z_list, decimals = 1).tolist()
    
    ## ACTIVATE THE BELOW TO RUN THE STRATEGY FOR A SINGLE Z ##
    
    z_list = [0.6] 
    
    for z in z_list:
        z_value = z
    
        backtest = Backtest(
            csv_dir, symbol_list, initial_capital, heartbeat,start_date, 
            HistoricCSVDataHandler, SimulatedExecutionHandler, 
            Portfolio, PairsStrategy, z_value)
        
        backtest.simulate_trading(z_list, csv_dir)
    
    strat_df = pd.read_csv(csv_dir+"/results.csv", header=0,
        parse_dates=True, index_col=0)
    strat_df = strat_df[1:-1]
    
    spx_df = pd.read_csv(csv_dir+"/spx_data.csv", header=0,
        parse_dates=True, index_col=0) # THIS FILE IS PROVIDED ALONG WITH THE CODE
               
    rolling_beta(spx_df, strat_df, csv_dir, 30)
    plot_results(strat_df)
    
    
    fig = plt.figure(figsize= (30,20))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(311)
    ax.set_ylabel('Portfolio value, %', fontsize = 22.0)
    ax.set_title('SPX investment vs Z=0.7 Strategy', fontsize = 22.0)
    ax.set_xlabel('datetime', fontsize = 22.0)
    strat_df['equity_curve'].plot(ax=ax, color="red", lw=2., label = 'Z=0.7 Strategy')
    spx_df['equity_curve'].plot(ax=ax, color="blue", lw=2., label='SPX Index')
    ax.legend(loc='upper left', fontsize = 20.0)
    plt.grid(True)
    plt.xticks(fontsize = 20.0)
    plt.yticks(fontsize = 20.0)    
    
    fig = plt.figure(figsize= (30,20))
    fig.patch.set_facecolor('white')
    drawdown, max_dd, dd_duration = create_drawdowns(spx_df['equity_curve'])
    spx_df['drawdown'] = drawdown
    ax2 = fig.add_subplot(313)
    ax2.set_title('SPX investment vs Z=0.7 Strategy', fontsize = 22.0)
    ax2.set_ylabel('Drawdowns, %', fontsize = 22.0)
    ax2.set_xlabel('datetime', fontsize = 22.0)
    strat_df['drawdown'].plot(ax=ax2, color="red", lw=2., label = 'Z=0.7 Strategy')
    spx_df['drawdown'].plot(ax=ax2, color="blue", lw=2., label='SPX Index')
    ax2.legend(loc='upper left', fontsize = 20.0)
    plt.grid(True)
    plt.xticks(fontsize = 20.0)
    plt.yticks(fontsize = 20.0)
    
    