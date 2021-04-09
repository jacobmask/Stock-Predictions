#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:15:47 2021

@author: taleahbirkicht

Modified: 3/24/2021
Author: Jacob Mask
Notes: Implemented config ticker list.
"""
import yfinance as yf
import pandas as pd
from config import ticker_list
import numpy as np


df1 = pd.DataFrame()

for stock in ticker_list:
    #get ticker
    ticker = yf.Ticker(stock)
    #collect data
    hist = ticker.history(period="1d")
    bs = ticker.balance_sheet(period="1d")
    earn = ticker.earnings(period="1d")
    rec = ticker.recommendations(period="1d")
    #transform
    bs = bs.transpose()
    hist.index = hist.index.date
    bs.index = bs.index.date
    earn = earn.rename_axis(index='Date')
    earn.index = earn.index.astype(str) + '-12-31'
    earn.index = pd.to_datetime(earn.index)
    rec = rec.drop(columns=['Firm', 'From Grade', 'Action'])
    rec['To Grade'] = rec['To Grade'].str.lower().replace({'sell': 1.0,
                                               'strong sell': 1.0,
                                               'negative': 1.5,
                                               'reduce': 1.5,
                                               'underperform': 2.0,
                                               'market underperform': 2.0,
                                               'market underweight': 2.0,
                                               'underweight': 2.0,
                                               'sector underperform': 2.0,
                                               'hold': 3.0,
                                               'neutral': 3.0,
                                               'market perform': 3.0,
                                               'market weight': 3.0,
                                               'perform': 3.0,
                                               'equal-weight': 3.0,
                                               'sector weight': 3.0,
                                               'sector perform': 3.0,
                                               'in-line': 3.0,
                                               'average': 3.0,
                                               'fair value': 3.0,
                                               'peer perform': 3.0,
                                               'mixed': 3.0,
                                               'outperform': 4.0,
                                               'market outperform': 4.0,
                                               'overweight': 4.0,
                                               'sector outperform': 4.0,
                                               'positive': 4.5,
                                               'strong buy': 5.0,
                                               'buy': 5.0,
                                               'long-term buy': 5.0,
                                               '': np.nan})
    rec = rec.dropna()
    rec.index = rec.index.date
    rec = rec.rename_axis(index='Date')
    rec.to_csv('test.csv')
    rec = rec.groupby('Date').agg({'To Grade': 'mean'})
    #combine
    df = pd.concat([hist, rec, bs, earn], axis=1)
    df = df.fillna(method='ffill')
    df = df.rename_axis(index='Date')
    df.to_csv('./StockCSV/'+stock+'.csv')
