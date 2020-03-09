import requests
import os
from dotenv import load_dotenv

"""
API from:
https://www.alphavantage.co/documentation/
"""


load_dotenv('api.env')

apikey = os.environ.get('key')
apikey

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=5min&apikey={apikey}"
res = requests.get(url)

res.json()

class Stock():
    def __init__(self,symbol):
        self.symbol = symbol

    def get_series(self,)


class Portfolio():
    def __init__(self,stocks):
        self.stocks = stocks
