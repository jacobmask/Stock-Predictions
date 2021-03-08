import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import yfinance as yf
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(XScale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


if __name__ == '__main__':
    main()
