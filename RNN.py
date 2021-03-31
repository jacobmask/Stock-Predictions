import os
from os import listdir
import numpy as np
import tensorflow as tf
from keras import models
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import math
import yfinance as yf
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import SimpleRNN
from keras.layers import Dropout
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from os.path import isfile, join

np.set_printoptions(precision=3, suppress=True)


def main():
    csv_files = [f for f in listdir('StockCSV') if isfile(join("StockCSV", f))]
    os.chdir('StockCSV')
    for csv_name in csv_files:
        df = pandas.read_csv(csv_name, usecols=[1, 2, 3, 4, 5])
        print(csv_name)
        rnn(df)
    os.chdir('..')


def rnn(data):
    train_data = data.copy()
    output_data = data.pop("Close")
    train_data = np.array(train_data)

    stock_model = tf.keras.Sequential([
        layers.Dense(5),
        layers.Dense(7),
        layers.Dense(1)])

    stock_model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam())

    stock_model.fit(train_data, output_data, epochs=10)

    prediction = stock_model.predict(train_data, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)

    # print(prediction)


if __name__ == '__main__':
    main()
