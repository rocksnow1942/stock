import requests
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from collections import deque
import json
from pymongo import MongoClient
import datetime
# import pprint
from dateutil.parser import parse
from mymodule import mprint

mprint.printToScreen = True


"""
API from:
https://www.alphavantage.co/documentation/
"""

client = MongoClient('localhost',27017)
db = client.STOCK

load_dotenv('api.env')

apikey = os.environ.get('key1')



class Stock():
    _OLDTIME = datetime.datetime(1000,1,1)
    def __init__(self,symbol,shares=1,mode='intraday'):
        self.symbol = symbol
        self.shares = shares
        self.df = None
        self.data = None
        self.metadata =None
        self.mode=mode.upper()

    def readDB(self,db):
        """
        read data of the stock from database.
        """
        col = db[self.mode]
        data = col.find_one({'metadata.Symbol':self.symbol},projection={'_id':False})
        if data:
            self.metadata = data.pop('metadata')
            self.data = data
            self.parse_df()
            return self
        else:
            raise ValueError(f'No data for {self} in database.')

    def savetoDB(self,db):
        """
        save this stock data to db.
        only update data with dates newer than the last update time.
        """
        col = db[self.mode]
        lastupdate = self._getRefreshDate(db)
        toupdate = {k:i for k,i in self.data.items() if (parse(k)>lastupdate)}
        toupdate.update(metadata=self.metadata)
        col.find_one_and_update(
          {'metadata.Symbol': self.symbol},
          {'$set':toupdate},
          upsert=True)

    def _getRefreshDate(self,db):
        """
        get refresh date of a stock in the database, return veryl old time if not found.
        """
        col = db[self.mode]
        res =  col.find_one({'metadata.Symbol': self.symbol},projection={'metadata':True})
        if res:
            return parse(res.pop('metadata')['Last Refreshed'])
        else:
            return self._OLDTIME

    def lastRefresh(self,db):
        """
        return time difference in days of current to last refresh time.
        """
        t = self._getRefreshDate(db)
        timedelta = datetime.datetime.now() - t
        return timedelta.days

    def save(self):
        """
        save as plain json file.
        """
        if 'Daily' in self.metadata['Information']:
            with open(f"{self.symbol}-Daily.json",'w') as f:
                json.dump(self.data,f,separators=(',',':'))
        elif 'Intraday' in self.metadata['Information']:
            with open(f"{self.symbol}-Intraday-{self.metadata['Last Refreshed'][0:10]}.json",'w') as f:
                json.dump(self.data,f,separators=(',',':'))

    def __repr__(self):
        return f"{self.shares} shares {self.symbol}"

    def parse_df(self):
        """
        parse self.data to self.df
        """
        reg = re.compile('[^a-zA-Z]')
        self.df = pd.DataFrame.from_dict(self.data,orient='index').astype(float)
        self.df.columns = [reg.sub("",i) for i in self.df.columns]
        self.df.index = pd.to_datetime(self.df.index)
        self.df.sort_index(ascending=True,inplace=True)

    @staticmethod
    def cleanDict(d):
        if isinstance(d,dict):
            p = re.compile('\d\. ')
            return {p.sub('',k):Stock.cleanDict(i) for k,i in d.items()}
        else:
            return d


    def get_series(self,outputsize='full',interval='5min',):
        """
        mode can be intraday,daily,weekly or monthly,
        """
        mode = self.mode
        func = f"TIME_SERIES_{mode}"
        url = f"https://www.alphavantage.co/query?function={func}&outputsize={outputsize}&symbol={self.symbol}&interval={interval}&apikey={apikey}"
        maxAttempt = 4
        while True:
            maxAttempt -=1
            res = requests.get(url)
            if res.ok:
                data = res.json()
            else:
                raise ValueError ('No response.')
            if data.get('Meta Data',None): # break if find correct response
                time.sleep(12) # wait for 12 seconds before next attempt.
                break
            elif data.get("Error Message",None):
                return 'Skip'
            elif maxAttempt == 0:
                raise ValueError (f"Max Attempt Reached: {data}")
            else:
                mprint(f'Wait 60seconds. Attempt left {maxAttempt}.')
                time.sleep(60)
        self.metadata = self.cleanDict(data.get('Meta Data',{}))
        keys = list(data.keys())
        keys.remove('Meta Data')
        self.data = self.cleanDict(data[keys[0]])
        self.parse_df()
        return self

    def plot_series(self,toplot=['open','high','low','volume'],length=None,period=[],format='auto'):
        if format == 'auto':
            if self.mode == 'INTRADAY':
                format= '%m/%d %H:%M'
            elif self.mode == 'DAILY':
                format='%y/%m/%d'
        df = self.df
        if length:
            df = df.iloc[0:length,:]
        elif isinstance(period,str):
            df = df[period:]
        elif period:
            df = df[period[0]:period[1]]
        else:
            df = df

        df = df.sort_index(ascending=True)
        # df = df
        df.loc[:,f'Time Period {self.mode}'] = df.index.map(lambda x:x.strftime(format))
        fig,ax = plt.subplots()
        df.plot(x=f'Time Period {self.mode}',y=toplot,secondary_y=['volume'],title=str(self),rot=25,ax=ax)
        plt.tight_layout()
        return fig

    @property
    def currentValue(self):
        return self.df.iloc[-1,:].close * self.shares

class Portfolio():
    def __init__(self,stocks,mode='intraday'):
        self.mode = mode.upper()
        if isinstance(stocks,list):
            if isinstance(stocks[0],str):
                self.stocks = [ Stock(i,mode=mode) for i in stocks]
            else:
                self.stocks = stocks
        elif isinstance(stocks,dict):
            self.stocks = []
            for k,i in stocks.items():
                self.stocks.append(Stock(k,i,mode=mode))

    def __getitem__(self,i):
        if isinstance(i,int):
            return self.stocks[i]
        elif isinstance(i,str):
            for s in self.stocks:
                if s.symbol == i:
                    return s
        elif isinstance(i,list):
            res = []
            for s in self.stocks:
                if s.symbol == i:
                    res.append(s)
            return res
        else:
            return self.stocks[i]

    def save(self,):
        for s in self.stocks:
            s.save()

    def savetoDB(self):
        for s in self.stocks:
            s.savetoDB()

    def readDB(self,db):
        for s in self.stocks:
            s.readDB(db)
        return self

    def __repr__(self):
        return "Portfolio:\n"+"\n".join(str(i) for i in self.stocks)

    def get_series(self,**kwargs):
        pool = deque(self.stocks)
        while pool:
            s = pool.popleft()
            try:
                s.get_series(**kwargs)
            except Exception as e:
                print(e,f'Current Deque {pool}')
                time.sleep(60)
            if getattr(s,'df',None) is None:
                pool.append(s)
        return self

    @property
    def currentValue(self):
        return sum(i.currentValue for i in self.stocks)

    def plot_series(self,target='all',toplot=['open','high','low'],length=None,period=[],format='auto'):
        if format == 'auto':
            if self.mode == 'INTRADAY':
                format= '%m/%d %H:%M'
            elif self.mode == 'DAILY':
                format='%y/%m/%d'

        if target == 'all':
            target = self.stocks
        elif isinstance(target,list):
            if isinstance(target[0],int):
                target = [self.stocks[i] for i in target]
            elif isinstance(target[0],str):
                target = [i for i in self.stocks if (i.symbol in target)]
        dfs = [s.df * s.shares for s in target]
        index = dfs[0].index
        for i in dfs: index.union(i.index)
        dfs = [ i.reindex(index).fillna(method='bfill') for i in dfs]
        df = sum(dfs)
        if length:
            df = df.iloc[0:length,:]
        elif isinstance(period,str):
            df = df[period:]
        elif period:
            df = df[period[0]:period[1]]
        else:
            df = df
        df = df.sort_index(ascending=True)
        df.loc[:,f'Time Period {self.mode}'] = df.index.map(lambda x:x.strftime(format))
        fig, ax = plt.subplots()
        df.plot(x=f'Time Period {self.mode}',y=toplot,
                    secondary_y=['volume'],title="Portfolio",rot=40,ax=ax)
        plt.tight_layout()
        return fig




def sync_stocks(stocks,client,days=0,mode='intraday',):
    for s in stocks:
        stock = Stock(s,mode=mode)
        daysSince = stock.lastRefresh(client)
        print(f"{s} synced {daysSince} days ago.")
        if daysSince > days:
            print(f"Start syncing {s}.")
            res = stock.get_series(outputsize='full',interval='5min')
            if res=='Skip':
                print(f'Skipped {s}')
                continue
            else:
                stock.savetoDB(client)
            print(f"Synced stock {s}.")



if __name__=="__main__":
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
    with open('SP500.json') as f:
        sp500_symbols = json.load(f)
    sync_stocks(sp500_symbols,db,days=3,mode='daily')



#
# my = {
# "MSFT": 53,
# "NVDA": 34,
# "FB": 16,
# "AMZN": 2,
# "AMD": 2,
# "SNAP": 33,
# "ISEE": 3,
# "STRO": 15,
# }
#
# port = Portfolio(my)
# port
# port.readDB(db)
#
#
#
# port.plot_series(toplot=['open','close'],period='2020/02/28')


s = Stock('ACE','intraday')

s.readDB(db)
