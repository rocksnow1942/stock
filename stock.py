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
from dateutil.parser import parse
from mymodule import mprint

mprint.printToScreen = True

"""
API from:
https://www.alphavantage.co/documentation/
"""


load_dotenv('api.env')

apikey = os.environ.get('key1')

class DataBase():
    _OLDTIME = datetime.datetime(1000,1,1)

    def __init__(self, address='localhost', port=27017,database='STOCK'):
        client = MongoClient(address,port)
        self.db = client[database]

    def __getitem__(self,i):
        return self.db[i]

    def copyFrom(self,db):
        cols = db.db.list_collection_names()
        for col in cols:
            data = db[col].find({})
            self.db[col].insert_many(data)
        return self

    def read(self,symbol,mode):
        """
        return data of a symbol in dictionary, it is almost the same as returned from API.
        {'2020-03-10 00:00:00': {'open': 73.05,
          'high': 73.62,
          'low': 69.66,
          'close': 72.91,
          'volume': 3537258.0},
         'metadata':{'Information': 'Daily Prices (open, high, low, close) and Volumes',
                     'Last Refreshed': '2020-03-10',
                     'Output Size': 'Full size',
                     'Symbol': 'A',
                     'Time Zone': 'US/Eastern'}
                    }
        """
        col = self[mode]
        return col.find_one({'metadata.Symbol':symbol},projection={'_id':False})

    def save(self,symbol,mode,data):
        """
        save the data to stock [symbol] in [mode] collection
        """
        col = self[mode]
        col.find_one_and_update(
          {'metadata.Symbol': symbol},
          {'$set':data},
          upsert=True)

    def lastUpdate(self,symbol,mode):
        """
        return the last update time of stock [symbol] in [mode] collection
        """
        col = self[mode]
        res =  col.find_one({'metadata.Symbol': symbol},projection={'metadata':True})
        if res:
            return parse(res.pop('metadata')['Last Refreshed'])
        else:
            return self._OLDTIME

    def deleteStockNotIn(self,mode,symbols):
        """
        delete a list of stocks not in [Symbols] from [mode] collection.
        """
        col = self[mode]
        res=col.delete_many({'metadata.Symbol':{"$nin":symbols}})
        return res.raw_result

    def deleteStock(self,mode,symbols):
        """
        delete a list of stocks in [Symbols] from [mode] collection.
        """
        col = self[mode]
        res=col.delete_many({'metadata.Symbol':{"$in":symbols}})
        return res.raw_result

    def listAllStock(self,mode):
        """
        list all stocks in [mode] collection
        """
        metas = self[mode].find(projection={'metadata':True})
        return sorted([i['metadata']['Symbol'] for i in metas])

    def changeAllfieldToFloat(self,mode):
        """
        change fileds to float
        """
        ass = self.listAllStock(mode)
        for s in ass:
            data = self.read(s,mode)
            data.pop('metadata')
            res = {k:{l:float(m) for l,m in i.items()} for k,i in data.items()
                    if isinstance(next(iter(i.values())),str)}
            self.save(s,mode,res)

    def updateStatus(self,mode):
        """
        return the refreshed dates of all symbols
        """
        col = self[mode]
        res = col.find(projection={'metadata':True,'_id':False})
        return [{i['metadata']['Symbol']: i['metadata']['Last Refreshed']} for i in res]


class Stock():
    def __init__(self,symbol,mode='intraday',shares=1,db=None):
        self.symbol = symbol.upper().replace('.','-')
        self.shares = shares
        self.df = None
        self.data = None
        self.metadata = None
        self.mode=mode.upper()
        self.db = db
        if db:
            self.readDB()

    def readDB(self,db=None):
        """
        read data of the stock from database.
        """
        db = db or self.db
        data = db.read(self.symbol,self.mode)
        if data:
            self.metadata = data.pop('metadata')
            self.data = data
            self.parse_df()
            return self
        else:
            raise ValueError(f'No data for {self} in database.')

    def savetoDB(self,db=None,onlyUpdate = True):
        """
        save this stock data to db.
        only update data with dates newer than the last update time.
        """
        db = db or self.db
        lastupdate = db.lastUpdate(self.symbol,self.mode)
        if onlyUpdate:
            toupdate = {k:i for k,i in self.data.items() if (parse(k)>lastupdate)}
        else:
            toupdate = {k:i for k,i in self.data.items()}
        toupdate.update(metadata=self.metadata)
        db.save(self.symbol,self.mode,toupdate)

    @property
    def lastUpdate(self):
        return self.metadata and parse(self.metadata['Last Refreshed'])

    def lastRefresh(self,db=None):
        """
        return time difference in days of current to last refresh time.
        """
        db = db or self.db
        t = db.lastUpdate(self.symbol,self.mode)
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

    @property
    def url(self,):
        """
        determine the API url to call.
        """
        size = 'full'
        now = datetime.datetime.utcnow()
        if self.mode == 'DAILY':
            if self.lastUpdate and  ((now - self.lastUpdate).days < 90):
                size = 'compact'
        elif self.mode == 'INTRADAY':
            if self.lastUpdate and (((now - self.lastUpdate).total_seconds() - 3600*4)/300 < 90):
                size = 'compact'
        if self.mode == 'INTRADAY':
            func = "TIME_SERIES_INTRADAY"
            return f"https://www.alphavantage.co/query?function={func}&outputsize={size}&symbol={self.symbol}&interval=5min&apikey={apikey}"
        elif self.mode == 'DAILY':
            func="TIME_SERIES_DAILY_ADJUSTED"
            return f"https://www.alphavantage.co/query?function={func}&outputsize={size}&symbol={self.symbol}&apikey={apikey}"

    def call_API(self):
        """
        call API from
        """
        url = self.url
        maxAttempt = 30
        data = None
        while True:
            maxAttempt -=1
            res = requests.get(url)
            if res.ok:
                data = res.json()
            else:
                raise ValueError ('No response.')
            if data.get('Meta Data',None): # break if find correct response
                time.sleep(9) # wait for 10 seconds before next attempt.
                break
            elif data.get("Error Message",None):
                print(data)
                return 'Skip'
            elif maxAttempt == 0:
                print(data)
                return "Maxattempt"
            else:
                mprint(f'Wait 5 seconds. Attempt left {maxAttempt}.')
                time.sleep(5)
        return data

    def parse_data(self,data):
        """
        parse the data from API call
        """
        self.metadata = self.cleanDict(data.get('Meta Data',{}))
        keys = list(data.keys())
        keys.remove('Meta Data')
        res =  self.cleanDict(data[keys[0]])
        res = {k:{l:float(m) for l,m in i.items()} for k,i in res.items()}
        self.data = res
        self.parse_df()
        return self

    def sync(self):
        """
        mode can be intraday,daily,weekly or monthly,
        """
        data = self.call_API()
        if isinstance(data,str): return data # just to be compatible with original behavior.
        return self.parse_data(data)


    def plot_series(self,toplot=['open','close'],length=None,period=[],format='auto'):
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
    @property
    def currentPrice(self):
        return self.df.iloc[-1,:].close

class Portfolio():
    def __init__(self,stocks,mode='intraday',db=None):
        self.mode = mode.upper()
        self.db = db
        if isinstance(stocks,list):
            if isinstance(stocks[0],str):
                self.stocks = [ Stock(i,mode=mode,db=db) for i in stocks]
            else:
                self.stocks = stocks
        elif isinstance(stocks,dict):
            self.stocks = []
            for k,i in stocks.items():
                self.stocks.append(Stock(k,shares=i,mode=mode,db=db))

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

    def savetoDB(self,db=None):
        db = db or self.db
        for s in self.stocks:
            s.savetoDB(db)

    def readDB(self,db=None):
        db = db or self.db
        for s in self.stocks:
            s.readDB(db)
        return self

    def __repr__(self):
        return "Portfolio:\n"+"\n".join(str(i) for i in self.stocks)

    def sync(self,):
        pool = deque(self.stocks)
        while pool:
            s = pool.popleft()
            try:
                s.sync()
            except Exception as e:
                print(e,f'Current Deque {pool}')
                time.sleep(10)
            if getattr(s,'df',None) is None:
                pool.append(s)
        return self

    @property
    def lastUpdate(self):
        return [{i.symbol:i.lastUpdate} for i in self.stocks ]

    @property
    def currentValue(self):
        return sum(i.currentValue for i in self.stocks)

    def plot_series(self,target='all',toplot=['open','close'],length=None,period=[],format='auto'):
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

        df.plot(x=f'Time Period {self.mode}',y=toplot,
                    secondary_y=['volume'],title="Portfolio",rot=40)


def sync_stocks(stocks,db,days=0,mode='intraday',onlyUpdate=True):
    for s in stocks:
        stock = Stock(s,mode=mode)
        daysSince = stock.lastRefresh(db)
        if daysSince > days:
            print(f"{s} synced {daysSince} days ago.")
            res = stock.sync()
            if res=='Skip':
                print(f'Skipped [{s}].')
                continue
            elif res == 'Maxattempt':
                print(f'MaxAttempt reached during [{s}]. Switch key.')
                switchapikey()
                continue
            else:
                stock.savetoDB(db,onlyUpdate)
            print(f"Synced stock {s}.")

def switchapikey():
    global apikey
    apikey = os.environ.get('key2')

def main(d=0,id=0):
    """
    -d [int]: days threhold to synchronize daily data.
    default: 0
    -id [int]:days threhold to synchronize intraday data.
    """
    d = int(d)
    id = int(id)
    db = DataBase()
    print("+"*15+'Syncing DAILY data.'+'+'*15+'\n')
    symbols = db.listAllStock('DAILY')
    sync_stocks(symbols,db,days=d,mode='Daily')
    print("="*15+'Done DAILY data.'+'='*15+'\n\n\n')
    print("+"*15+'Syncing INTRADAY data.'+'+'*15+'\n')
    symbols = db.listAllStock('INTRADAY')
    sync_stocks(symbols,db,days=id,mode='intraday')
    print("="*15+'Done INTRADAY data.'+'='*15+'\n\n')

if __name__=='__main__':
    import sys
    arg = sys.argv
    if '-h' in arg:
        print(main.__doc__)
    else:
        kwargs = {}
        for i,k in enumerate(arg):
            if k.startswith('-'):
                kwargs.update({k[1:]:arg[i+1]})

        main(**kwargs)
