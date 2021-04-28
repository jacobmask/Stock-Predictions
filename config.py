"""
Author: Tim Campbell
Modified: 3/28/2021
Notes:
"""

# List of tickers to be used in data-clean.py
ticker_list = ['NVDA', 'INTC', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'T', 'VZ', 'CSCO', 'ORCL']

# Cols indicates the columns used from the CSV in the network.
# Range must stay from 1 to 8, inclusively, and 4 must always be present unless
# 'Close' is changed in lines 44 & 45 of neural-network.py to an existing column name.
cols = [1, 2, 3, 4, 5, 6, 7, 8]
