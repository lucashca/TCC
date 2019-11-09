
from sklearn.model_selection import train_test_split


from myTools.dataSetPreProcessing import train_validation_test_split
from myTools.loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet
from myTools.tools import verifyArgs,findBalancedDataSet,pltResults

from sklearn.metrics import r2_score,mean_squared_error


import matplotlib.pyplot as plt

import sys


from sklearn.neural_network import  MLPRegressor
from sklearn.model_selection import GridSearchCV

def GridSearchCVMLPRegrssor(X_train,y_train):
    d = [i for i in range(1,10)]
    d.append(None)

    param_grid = {
        'hidden_layer_sizes': [(5,5,5),(5,10,10),(10,10,10),(50,50,50),(15,25,50),(100,100,50)],
        'activation': ['relu'],
        'solver': ['adam','sgd','lbfgs'],
        'learning_rate': ['constant','adaptive'],
        'learning_rate_init': [0.001],
        'power_t': [0.5],
        'alpha': [0.0001],
        'max_iter': [1000],
        'early_stopping': [False],
        'warm_start': [False]}

    model = MLPRegressor(random_state=0,verbose=1)

    scores = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
    ]
    reg =GridSearchCV(model, cv=3,param_grid=param_grid,verbose=0,n_jobs=-1,scoring=scores,refit='r2',iid=True)

    reg.fit(X_train,y_train)

    return reg.best_estimator_,reg.best_params_,reg.best_score_



args = sys.argv[1:]

y_column,random_state = verifyArgs(args)

# Load data set
dataSet = loadMainDataSet()
#dataSet = loadTesteDataSet()
#Set features and target
X = dataSet[:,:2]
y = dataSet[:,y_column]



best_model , best_params, best_score,best_seed = findBalancedDataSet(range(1,10),X,y,GridSearchCVMLPRegrssor)

print("Best Params",best_params)
print("Best Score",best_score)
print("Best Seed",best_seed)

