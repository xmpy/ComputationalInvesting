'''
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on January, 24, 2013

@author: Sourabh Bajaj
@contact: sourabhbajaj@gatech.edu
@summary: Example tutorial code.
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from operator import itemgetter


print "Pandas Version", pd.__version__


def simulate(startdate, enddate, ls_symbols, allocation):
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo',cachestalltime=0)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # You need to normalize
    na_price = d_data['close'].values / d_data['close'].values[0]
    na_price *= allocation
    na_price = na_price.sum(axis=1)

    r = tsu.returnize0(na_price.copy())
    avg_daily_return = np.average(r)
    std = np.std(r)

    sharpe_ratio = avg_daily_return/std *math.sqrt(252)

    cul_return = na_price[-1] / na_price[0]
    return std,avg_daily_return,sharpe_ratio,cul_return

def gen_pum(n, item_n):
    possible_result = []
    if item_n == 1:
        possible_result = [ [n,] ]
    else:
        possible_result = [ [i,] + t_r for i in range(n+1) for t_r in gen_pum(n-i, item_n - 1)]
    return possible_result
    

def get_optimize(startdate, enddate, ls_symbols):
    #for
    r = gen_pum(10, len(ls_symbols))
    vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, ls_symbols, r[0])

    a = sharpe
    #print sharpe
    result = r[0]
    for i in r:
        allocation = [j/10.0 for j in i]
        vol, daily_ret, sharpe, cum_ret = simulate(startdate, enddate, ls_symbols, allocation)
        if sharpe > a:
            a = sharpe
            result = i
    #print a
    result = [j/10.0 for j in result]

    return result
def get_data(startdate,enddate,ls_symbols):
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(startdate, enddate + dt.timedelta(days=1), dt_timeofday)
    print ldt_timestamps
    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo',cachestalltime=0)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    # You need to normalize
    #na_price = d_data['close'].values / d_data['close'].values[0]
    #ldt_timestamps = [ dt.datetime(t) for t in ldt_timestamps]
    ldt_timestamps = [ dt.datetime.strptime(str(i),'%Y-%m-%d %H:%M:%S') for i in ldt_timestamps]
    na_price = d_data['close'].values
    print ldt_timestamps
    return na_price,ldt_timestamps

class simulator:
    def __init__(self, file_name, cash):
        self.cash = cash
        self.all_value = cash
        dt_start = dt.datetime(9999,1,1)
        dt_end = dt.datetime(1,1,1)

        ls_symbols = set()

        with open(file_name,'rb') as f:
            for line in f:
                cols = line.strip().split(',')
                if len(cols) < 7:
                    continue
                year = int(cols[0])
                month = int(cols[1])
                day = int(cols[2])
                symbol = cols[3]
                dt_temp = dt.datetime(year, month, day)
                if dt_temp < dt_start:
                    dt_start = dt_temp
                if dt_temp > dt_end:
                    dt_end = dt_temp
                ls_symbols.add(symbol)
        self.ls_symbols = list(ls_symbols)
        self.dt_start = dt_start
        self.dt_end = dt_end
        self.data, self.ldt_timestamps = get_data(self.dt_start,self.dt_end,self.ls_symbols)
        self.equities = dict([(s,0) for s in ls_symbols])

    def buy(self,dt_date,symbol,share_number):
        row_number = self.ldt_timestamps.index(dt_date)
        col_number = self.ls_symbols.index(symbol)

        price = self.data[row_number][col_number]
        self.cash -= share_number*price
        self.equities[symbol] += share_number 

    def sell(self,dt_date,symbol,share_number):
        row_number = self.ldt_timestamps.index(dt_date)
        col_number = self.ls_symbols.index(symbol)

        price = self.data[row_number][col_number]
        self.cash += share_number*price
        self.equities[symbol] -= share_number

    def now_we_have(self,dt_date):
        temp = []
        result = 0
        print dt_date
        print self.equities
        if dt_date not in self.ldt_timestamps:
            return self.all_value
        for symbol in self.ls_symbols:
            row_number = self.ldt_timestamps.index(dt_date)
            col_number = self.ls_symbols.index(symbol)
            price = self.data[row_number][col_number]
            result += price * self.equities[symbol]
        result += self.cash
        self.all_value = result
        return result


def get_score(na_price):
    r = tsu.returnize0(na_price.copy())
    avg_daily_return = np.average(r)
    std = np.std(r)

    sharpe_ratio = avg_daily_return/std *math.sqrt(252)

    cul_return = na_price[-1] / na_price[0]
    return std,avg_daily_return,sharpe_ratio,cul_return

if __name__ == '__main__':

    input_file = 'orders.csv'
    cash = 1000000

    simu = simulator(input_file,cash)
    order_sequence = {}
    out = open('result.csv','w')

    with open(input_file,'rb') as f:
        for line in f:
            cols = line.strip().split(',')
            if len(cols) < 7:
                continue
            year = int(cols[0])
            month = int(cols[1])
            day = int(cols[2])
            symbol = cols[3]
            dt_temp = dt.datetime(year, month, day,16)
            buy_sell_indicator = cols[4]
            share_number = int(cols[5])
            order_sequence.setdefault(dt_temp,[]).append((buy_sell_indicator,symbol,share_number))
    print order_sequence
    print (simu.dt_end - simu.dt_start).days
    funds = []
    for d in simu.ldt_timestamps:
        #interval = dt.timedelta(days = i,hours=16)
        if d in order_sequence:
            for o in order_sequence[d]:
                print o[0],o[1],o[2]
                if o[0] == "Buy":
                    simu.buy(d,o[1],o[2])
                else:
                    simu.sell(d,o[1],o[2])
        r =  [str(d.year),str(d.month),str(d.day),str(simu.now_we_have(d))]
        funds.append(simu.now_we_have(d))
        out.write(','.join(r)+'\n')
    funds = np.array(funds)

    vol, daily_ret, sharpe, cum_ret = get_score(funds)
    print vol
    print daily_ret
    print sharpe
    print cum_ret
    

