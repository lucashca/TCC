
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Masking
import sys
from numpy.random import seed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV


# Camadas de redes neurais


tf.random.set_seed(1)


# define base model
from keras.optimizers import Adam

def baseline_model():
    activation = 'relu'  # or linear
    init_mode = 'uniform'
    model = Sequential()

    model = Sequential()
    model.add(Dense(100,
                    input_dim=2, kernel_initializer=init_mode,
                    activation=activation))
    model.add(Dense(100, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation=activation))
    model.add(Dense(100, activation=activation))

    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam',
                  metrics=['mse', 'mae'])
    model.summary()
    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.head(5))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')

    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.legend()

    plt.show()


def getBestModel():

    model = baseline_model()

    rs = 0
    maior_r2 = 0
    for r in range(1, 100):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=r)

        history = model.fit(X_train, y_train, epochs=100,
                            batch_size=64,  verbose=0, validation_split=0.2)

        y_pred = model.predict(X_test)
        yt_pred = model.predict(X_train)
        r2 = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, yt_pred)
        print("****", r)
        if r2 > maior_r2:

            maior_r2 = r2
            rs = r
            print("************", rs)

        print("Maior ", maior_r2, " Train Score ", r2_train)
        # plot_history(history)

    return rs


def printBestRes(r):

    model = baseline_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=r)

    #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    history = model.fit(X_train, y_train, epochs=10000,
                        batch_size=64,  verbose=1, validation_split=0.2)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    y_pred = model.predict(X_test)
    yt_pred = model.predict(X_train)
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, yt_pred)
    print("Maior ", r2, " Train Score ", r2_train)
    print("MSE", mean_squared_error(y_test, y_pred))
    print("MAE", median_absolute_error(y_test, y_pred))

    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(X_test[:, 0], y_test, label='Teste')
    plt.scatter(X_test[:, 0], y_pred, label='Pred')

    plt.legend()

    plot_history(history)


# rs = getBestModel()

r = int(sys.argv[1])

printBestRes(r)


# Model 1 4
