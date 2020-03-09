import requests
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from collections import deque
"""
API from:
https://www.alphavantage.co/documentation/
"""


load_dotenv('api.env')

apikey = os.environ.get('key')

class Stock():
    def __init__(self,symbol,shares=1):
        self.symbol = symbol
        self.shares = shares

    def __repr__(self):
        return f"{self.shares} shares {self.symbol}"

    def get_series(self,mode="intraday",outputsize='compact',interval='1min'):
        """
        mode can be intraday,daily,weekly or monthly,
        """
        func = f"TIME_SERIES_{mode.upper()}"
        url = f"https://www.alphavantage.co/query?function={func}&outputsize={outputsize}&symbol={self.symbol}&interval={interval}&apikey={apikey}"
        res = requests.get(url)
        if res.ok:
            data = res.json()
        else:
            raise ValueError ('No response.')
        self.data = data
        self.metadata = data.pop('Meta Data')
        reg = re.compile('[^a-zA-Z]')
        self.df = pd.DataFrame.from_dict(data[list(data.keys())[0]],orient='index').astype(float)
        self.df.columns = [reg.sub("",i) for i in self.df.columns]
        self.df.index = pd.to_datetime(self.df.index)

    def plot_series(self,toplot=['open','high','low','volume'],length=None):
        self.df.iloc[0:length,:].plot(y=toplot,
                    secondary_y=['volume'],title=str(self),)

    @property
    def currentValue(self):
        return self.df.iloc[0,:].close * self.shares



class Portfolio():
    def __init__(self,stocks):
        if isinstance(stocks,list):
            self.stocks = stocks
        elif isinstance(stocks,dict):
            self.stocks = []
            for k,i in stocks.items():
                self.stocks.append(Stock(k,i))

    def __repr__(self):
        return "My Portfolio:\n"+"\n".join(str(i) for i in self.stocks)

    def get_series(self,**kwargs):
        pool = deque(self.stocks)
        while pool:
            s = pool.popleft()
            try:
                s.get_series(**kwargs)
            except Exception as e:
                print(e,f'Current Deque {pool}')
                time.sleep(3)
            if getattr(s,'df',None) is None:
                pool.append(s)
        return self

    @property
    def currentValue(self):
        return sum(i.currentValue for i in self.stocks)

    def plot_series(self,target='all',toplot=['open','high','low','volume'],length=None):
        if target == 'all':
            target = self.stocks
        elif isinstance(target,list):
            if isinstance(target[0],int):
                target = [self.stocks[i] for i in target]
            elif isinstance(target[0],str):
                target = [i for i in self.stocks if (i.symbol in target)]
        df = sum([s.df * s.shares for s in target])
        df.iloc[0:length,:].plot(y=toplot,
                    secondary_y=['volume'],title="Portfolio",)

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

myPortfolio = Portfolio(my)

myPortfolio.get_series()


myPortfolio.currentValue()

df = sum([s.df * s.shares for s in myPortfolio.stocks])
df
df.iloc[0:length,:].plot(y=toplot,
            secondary_y=['volume'],title="Portfolio",)

myPortfolio.plot_series()
