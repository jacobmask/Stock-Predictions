import os
from os import listdir
import numpy as np
import tensorflow as tf
from keras import models
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, preprocessing, experimental
import math
import yfinance as yf
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import SimpleRNN
from keras.layers import Dropout
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from os.path import isfile, join
from config import cols
np.set_printoptions(precision=3, suppress=True)


def main():
    csv_files = [f for f in listdir('StockCSV') if isfile(join("StockCSV", f))]
    os.chdir('StockCSV')
    for csv_name in csv_files:
        df = pandas.read_csv(csv_name, usecols=cols)
        print(csv_name)
        neural_network(df)
    os.chdir('..')


def neural_network(data):
    normalize = tf.keras.layers.experimental.preprocessing.Normalization()
    train_data = data.copy()

    # Single column output
    output_data = data.pop("Close")

    # Double column output
    # output_data = pandas.DataFrame()
    # output_data["Low"] = data.pop("Low")
    # output_data["High"] = data.pop("High")

    train_data = np.array(train_data)

    normalize.adapt(train_data)

    stock_model = tf.keras.Sequential([
        normalize,
        layers.Dense(32),
        layers.Dense(64),
        layers.Dense(1)])

    stock_model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=['accuracy'])

    # calculate num of epochs based on length of dataset
    weighted_epoch = round(len(data.index)/10)
    stock_model.fit(train_data, output_data, epochs=10)

    prediction = stock_model.predict(train_data, batch_size=None, verbose=0, steps=None, callbacks=None,
                                     max_queue_size=10, workers=1, use_multiprocessing=False)

    print(prediction)
    plt.plot(prediction)
    plt.show()


if __name__ == '__main__':
    main()

# high and low
# next day output
# layers