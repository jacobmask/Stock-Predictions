#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:15:47 2021

@author: taleahbirkicht
"""
import yfinance as yf
import pandas as pd

stock_pop = ['NVDA', 'INTC', 'GOOGL', 'MSFT', 'AMZN', 
             'FB', 'T', 'VZ', 'CSCO', 'ORCL']

df1 = pd.DataFrame()
current = []

googl = yf.Ticker('GOOGL')
for stock in stock_pop:
    current.append(stock)
    #get ticker
    ticker = yf.Ticker(stock)
    #collect data
    hist = ticker.history(period="max")
    bs = ticker.balance_sheet
    earn = ticker.earnings
    rec = ticker.recommendations
    #transform
    bs = bs.transpose()
    hist.index = hist.index.date
    bs.index = bs.index.date
    earn = earn.rename_axis(index='Date')
    earn.index = earn.index.astype(str) + '-12-31'
    earn.index = pd.to_datetime(earn.index)
    #combine
    df = pd.concat([hist, bs, earn], axis=1)
    df = df.fillna(method='ffill')
    df = df.rename_axis(index='Date')
    df.to_csv('./data/'+stock+'.csv')
    
    

#df1.to_csv("test.csv")

    

















''' 
#collect data
googl_hist = googl.history(period="max")
googl_bs = googl.balance_sheet
googl_earn = googl.earnings
googl_rec = googl.recommendations

#transform
googl_bs = googl_bs.transpose()

#googl_earn['Year'] = googl_earn['Year'].astype(str) + '-12-31'
googl_hist.index = googl_hist.index.date
#googl_hist = googl_hist[~googl_hist.index.duplicated(keep='first')]

googl_bs.index = googl_bs.index.date
#googl_bs = googl_bs[~googl_bs.index.duplicated(keep='first')]

googl_earn = googl_earn.rename_axis(index='Date')
googl_earn.index = googl_earn.index.astype(str) + '-12-31'
googl_earn.index = pd.to_datetime(googl_earn.index)
print(googl_earn)

#print(googl_rec)


df = pd.concat([googl_hist, googl_bs, googl_earn], axis=1)
df = df.fillna(method='ffill')
df = df.rename_axis(index='Date')
print(df)
df.to_csv("Google.csv")
'''
