import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import yfinance as yf
import matplotlib.pyplot as plt
import pandas
from sklearn import preprocessing


def main():
    # Graph modeling test:
    # sin_wave = np.array([math.sin(x) for x in np.arange(200)])
    # plt.plot(sin_wave[:100])
    # plt.show()

    df = pandas.read_csv("Google.csv")
    dataset = df.values
    #print(dataset)

    X = dataset[:, 1:6]
    Y = dataset[:, 2]

    min_max_scaler = preprocessing.MinMaxScaler()
    XScale = min_max_scaler.fit_transform(X)

    print(XScale)


if __name__ == '__main__':
    main()
