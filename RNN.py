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

    X = dataset[:, 1:6]
    Y = dataset[:, 1]

    min_max_scaler = preprocessing.MinMaxScaler()
    XScale = min_max_scaler.fit_transform(X)

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(XScale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    X_train = np.asarray(X_train).astype(np.float32)
    Y_train = np.asarray(Y_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    Y_val = np.asarray(Y_val).astype(np.float32)

    # print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

    model = Sequential([Dense(32, activation='relu', input_shape=(5, )), Dense(32, activation='relu'),
                        Dense(1, activation='sigmoid'), ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_val, Y_val))


if __name__ == '__main__':
    main()
