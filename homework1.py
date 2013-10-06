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


if __name__ == '__main__':
    ls_symbols = ['AAPL', 'GLD', 'GOOG', 'XOM']
    #ls_symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    #ls_symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

    # Start and End date of the charts
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    allocation = [0.4, 0.4, 0.0, 0.2]


    vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ls_symbols, allocation)
    print vol
    print daily_ret
    print sharpe
    print cum_ret
    '''
    #print gen_pum(10,4)
    allocation = get_optimize(dt_start, dt_end, ls_symbols)
    print allocation
    vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ls_symbols, allocation)
    print "----------------------------------"
    print vol
    print daily_ret
    print sharpe
    print cum_ret
    '''
   # main()
