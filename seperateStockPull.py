from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd

ticker_list=['NVDA']

files=[]
today = date.today()

def getData(ticker):
	print (ticker)
	data = pdr.get_data_yahoo(ticker, start="2001-01-01", end=(today.strftime("%Y-%m-%d")))
	dataname= ticker+'_'+str(today)
	print("test")
	files.append(dataname)
	SaveData(data, dataname)

def SaveData(df, filename):
	df.to_csv('./data/'+filename+'.csv')


	for tik in ticker_list:
		getData(tik)
	for i in range(0,11):
		df1= pd.read_csv('./data/'+str(files[i])+'.csv')
		print("test")
		print (df1.head())


for i in range(0, len(ticker_list)):
	getData(ticker_list[i])
