#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Taleah Birkicht, Tim Campbell, Jacob Mask
Last Modified: 4/28/21
Description: This file automatically adds or modifies a csv for each of the
stock population determined by the config file into a folder called StockCSVRecs.
The csv will contain historical market data for the stock price such as
High Low, Close, Open as well as analyst recommendation data. The analyst
recommendation data is one column named "Recommendation".
"""
import yfinance as yf
import pandas as pd
from config import ticker_list
import numpy as np
import os


for stock in ticker_list:
    #get ticker
    ticker = yf.Ticker(stock)
    #collect data
    hist = ticker.history(period="max")
    rec = ticker.recommendations
    #transform
    hist.index = hist.index.date
    hist = hist.rename_axis(index='Date')
    rec = rec.drop(columns=['Firm', 'From Grade', 'Action'])
    #applies quantitative value to categorical variables
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
    #averages the recommendations for a given day
    rec = rec.groupby('Date').agg({'To Grade': 'mean'})
    rec = rec.rename(columns={'To Grade': 'Recommendation'})
    hist = hist.rename_axis(index='Date')
    
    #only obtain dates larger than first analyst recommendation
    hist2 = hist[(hist.index >= rec.index[0])]

    #combine market data and analyst recommendations
    df = pd.concat([hist2, rec], axis=1, join='outer')
    #analyst recommendations are forward filled in until the next recommendation
    #is made
    df = df.fillna(method='ffill')
    df = df.rename_axis(index='Date')
    
    #creates new folder if it does not already exist
    if not os.path.exists('StockCSVRecs'):
        os.makedirs('StockCSVRecs')
    
    #writes to csv file
    df.to_csv('./StockCSVRecs/'+stock+'_rec'+'.csv')
