#!/usr/bin/env python
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd
from config import ticker_list

print("Pulling list of tickers from config.py")
files=[]
today = date.today()

def getData(ticker):
	data = pdr.get_data_yahoo(ticker, start="2001-01-01", end=(today.strftime("%Y-%m-%d")))
	dataname= ticker+'_'+str(today)
	files.append(dataname)
	SaveData(data, dataname)

def SaveData(df, filename):
	df.to_csv('./data/'+filename+'.csv')
	print(filename, "Has saved")


for tik in ticker_list:
	getData(tik)
for i in range(0,len(ticker_list)):
	df1= pd.read_csv('./data/'+str(files[i])+'.csv')

print("Complete")

