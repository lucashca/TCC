import sys
sys.path.insert(0, "../myTools")

from loadDataSet import loadMainDataSet, loadTesteDataSet,loadCompletDataSet,loadMainDataSetWithElevation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt


## Semente randomica do tensorflow
tf.random.set_seed(1)
####################

## Carregar o conjunto de dados
#dataSet = loadMainDataSet()
#dataSet = loadMainDataSet()
#dataSet = loadMainDataSet()
dataSet = loadMainDataSetWithElevation()
#############################

X = dataSet[:,:3]
y = dataSet[:,4]


r = int(sys.argv[1],10)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=r)

#X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2)





### Models Import #####
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam

#######################


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
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



def plot_results(model):

    y_pred = model.predict(X_test)
    yt_pred = model.predict(X_train)
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, yt_pred)
    print("R2 ", r2, " R2 Train  ", r2_train)
    print("MSE", mean_squared_error(y_test, y_pred))
    print("MAE", mean_absolute_error(y_test, y_pred))

    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(X_test[:, 0], y_test, label='Teste')
    plt.scatter(X_test[:, 0], y_pred, label='Pred')

    plt.legend()



def baseline_model():
    # create model

    activation = ['relu','sigmoid','tanh','elu','softmax','selu','softplus','softsign','hard_sigmoid','linear','exponential']  # or linear
    init_mode = ['uniform','random_uniform','zeros','ones','random_normal','identity']
    models = []
    for a in activation:
        for i in init_mode:
         # create model
            model = Sequential()
            model.add(Dense(15,input_dim=X_train.shape[1], kernel_initializer=i,activation=a))
            model.add(Dense(20, activation=a))
            model.add(Dense(20, activation=a))
            model.add(Dense(20, activation=a))
            
            model.add(Dense(1, kernel_initializer=i, activation=a))

            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
            model.summary()
            models.append([[i,a],model])
    return models


models = baseline_model()
batch_size = X_test.shape[0]
batch_size = int(batch_size/5)

for params,model in models:
    print(params)
    
    history = model.fit(X_train, y_train, epochs = 500, validation_split=0.2, verbose = 0,batch_size=batch_size)
    plot_results(model)
    plot_history(history)


plt.show()



