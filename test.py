from stock import DataBase,Stock,Portfolio,sync_stocks
import pandas as pd
import json
import datetime
#

mydb = DataBase()
db = DataBase(address='hui-razer.lan',port=27017)

db.copyFrom(mydb)

db.listAllStock('DAILY')

db = DataBase()
#

s = Stock('A')
s.readDB(db)
s.lastUpdate

#
# with open('sp500.json','r') as f:
#     symbols = json.load(f)
#
# dy = db['DAILY']
#
# ass = db.listAllStock('INTRADAY')
# data = db.read('MSFT','INTRADAY')
#
# next(iter(data['2020-03-11 10:55:00'].values()))
# data.pop('metadata')
# res={k:{l:float(m) for l,m in i.items()} for k,i in data.items()
#        if isinstance(next(iter(i.values())),str)}
# res
#
# for s in ass:
#     data = db.read(s,'INTRADAY')
#     meta = data.pop('metadata')
#     res = {k:{l:float(m) for l,m in i.items()} for k,i in data.items()
#             if isinstance(next(iter(i.values())),str)}
#     res.update(metadata = meta)
#     db.save(s,'INTRADAY',res)
#
#
#
#
#
#
# s = Stock('A','Daily')
# s.get_series()
#
#
# s.readDB(db)
# s.data
#
# #
# #
# # db = DataBase()
# # # db['DAILY'].drop()
# #
# #

my = {
"MSFT": 53,
"NVDA": 34,
"FB": 16,
"AMZN": 2,
"AMD": 2,
"SNAP": 33,
"ISEE": 3,
"STRO": 15,
}

p = Portfolio(my,)
p.readDB(db)
p.lastUpdate




#
# # p.plot_series(toplot=['close'],period="2020-02-28")
#
#
# # r=s.get_series()
# # s.savetoDB(db)
# # f=s.plot_series(toplot=['close','adjustedclose'])
# #
# # f=s.plot_series(toplot=['adjustedclose'])
# # print(s.url)
# #
# # # data = s.call_API()
# # #
# # s.readDB(db)
# # s.metadata
# # s.df.head()
# # s.data['2016-03-28']
#
#
# # s.df.loc['2000-03-15',:]
# #
# # data['Meta Data']
# #
# # test = Stock('X','daily')
# #
# # test.parse_data(data)
# # test.df.loc['2012-03-09']
# #
# # f=s.plot_series()
# # s.df.head()
# #
# #
# #
# # # this is the Yahool data
# # df = pd.read_csv('/Users/hui/Downloads/A (1).csv')
# # df=df.set_index('Date')
# # df.index = pd.to_datetime(df.index)
# # df.loc['2012-03-09',:]
# # df.head()


import sys

arg = sys.argv
print(arg)

def a():
    """
    hal
    """
    pass

a.__doc__


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
