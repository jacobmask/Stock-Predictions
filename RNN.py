import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from config import cols
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


def main():
    csv_files = [f for f in listdir('StockCSVRecs') if isfile(join("StockCSVRecs", f))]
    os.chdir('StockCSVRecs')
    for csv_name in csv_files:
        df = pd.read_csv(csv_name, usecols=cols)
        # date = pandas.read_csv(csv_name, usecols=[0])
        stock_name = "$" + csv_name[:-8]
        train_data = df.copy()

        # Single column output
        output_data = pd.DataFrame({"Close": df.pop("Close")})
        neural_network(stock_name, train_data, output_data)

        # Double column output
        # output_data = pandas.DataFrame()
        # output_data["Low"] = data.pop("Low")
        # output_data["High"] = data.pop("High")
    os.chdir('..')


def neural_network(stock_name, train_data, output_data):

    normalize = tf.keras.layers.experimental.preprocessing.Normalization()

    train_data = np.array(train_data)

    normalize.adapt(train_data)

    stock_model = tf.keras.Sequential([
        normalize,
        layers.Dense(32),
        layers.Dense(64),
        layers.Dense(1)])

    stock_model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=tf.keras.metrics.Accuracy())

    # calculate num of epochs based on length of dataset
<<<<<<< HEAD
    weighted_epoch = round(len(data.index)/100)
=======
    weighted_epoch = round(len(train_data)/100)
    print("-----Beginning training for %s-----" % stock_name)
>>>>>>> 88b0321eedbfae496be1e30ec97734355d7d67fa
    stock_model.fit(train_data, output_data, epochs=weighted_epoch)

    prediction = stock_model.predict(train_data, batch_size=None, verbose=0, steps=1, callbacks=None,
                                     max_queue_size=10, workers=1, use_multiprocessing=False)

    print()
    print("Same day price prediction for %s: " % stock_name, prediction[-1])
    print("Actual price for %s: $" % stock_name + str(round(output_data["Close"][output_data.index[-1]],2)))
    print()

    plt.plot(prediction)
    # plt.plot(output_data)
    plt.xlabel("Date")
    plt.ylabel("Price")
    # plt.show()


if __name__ == '__main__':
    main()

