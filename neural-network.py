"""
Author: Tim Campbell
Modified: 3/27/2021
Description: This file uses data from csv files stored in the directory StockCSVRecs
to train a neural network implemented with keras. Output is a graph of actual stock price
and the predicted stock price along with printing the most recent actual and predicted price.
The actual price is the price of the stock at the time of running data-clean.py
The predicted price represents a prediction of the next close price. If the market is still open,
the prediction represents the close price for the same day, and if it is closed, it represents
the next day's close.
"""

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from config import cols
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def main():
    """Creates a Pandas data frame for each csv files and passes it to the neural_network
    function"""

    csv_files = [f for f in listdir('StockCSVRecs') if isfile(join("StockCSVRecs", f))]

    os.chdir('StockCSVRecs')

    # loop through each csv file and pass it to neural_network
    for csv_name in csv_files:
        df = pd.read_csv(csv_name, usecols=cols)
        stock_name = "$" + csv_name[:-8]
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        neural_network(stock_name, df)
    os.chdir('..')


def neural_network(stock_name, df):
    """Implements a sequential neural network using keras from tensorflow.
     Output will include printing text and a graph for each data frame passed in."""

    # split data into traning/test data and scale for better fitting
    X = df.drop('Close', axis=1)
    y = np.ravel(df['Close'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # initialize neural network model
    stock_model = tf.keras.Sequential([
        layers.Dense(32),
        layers.Dense(64),
        layers.Dense(1)])

    # compile network with default args
    stock_model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam())

    # calculate num of epochs based on length of dataset
    weighted_epoch = round(len(df) / 100)

    print("-----Beginning training for %s-----" % stock_name)
    # fit the model with traning data
    stock_model.fit(X_train, y_train, epochs=weighted_epoch, verbose=0)

    # create closing price preditions with test data using default parameters
    prediction = stock_model.predict(X_test, batch_size=None, verbose=0, steps=1, callbacks=None,
                                     max_queue_size=10, workers=1, use_multiprocessing=False)

    prediction_price = float(str(prediction[-1]).strip("[").strip("]"))

    print()
    print("Next market close price prediction for %s: " % stock_name, '$'+(str(round(prediction_price, 2))))
    print("Current price for %s: $" % stock_name + str(round(y_test[-1], 2)))
    print()

    # generate graph of predicted closing price and actual closing price
    plt.plot(prediction, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.title(stock_name)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

