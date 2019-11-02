
import sys
sys.path.insert(0, "../myTools")

from dataSetPreProcessing import train_validation_test_split
from loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet,loadMainDataSetWithElevation
from tools import verifyArgs,pltResults,findBalancedDataSet

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt




# Create train,validation and test data
#X_train,X_val,X_test,y_train,y_val,y_test = train_validation_test_split(X,y,random_state=0)
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=random_state)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



def GridSearchCVRandomForest(X_train,y_train):
    d = [i for i in range(1,20)]
    d.append(None)

    param_grid = {
        
        'bootstrap': [True],
        'max_depth': d,
        'criterion': ['mse'],
        'max_features': ['auto','log2'],
      #  'min_samples_leaf': [3, 4, 5],
      #  'min_samples_split': [8, 10, 12],
        'n_estimators':range(1,10),
        'random_state':range(1)
    }

    model = RandomForestRegressor()

    scores = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
    ]
    reg =GridSearchCV(model, cv=3,param_grid=param_grid,verbose=0,n_jobs=-1,scoring=scores,refit='r2',iid=False)

    reg.fit(X_train,y_train)

    return reg.best_estimator_,reg.best_params_,reg.best_score_



args = sys.argv[1:]

y_column,random_state = verifyArgs(args)

# Load data set
dataSet = loadMainDataSetWithElevation()
#dataSet = loadTesteDataSet()
#dataSet = loadMainDataSet()
#dataSet = loadCompletDataSet()
#Set features and target
y_column = 2 
X = dataSet[:,0:4]
y = dataSet[:,4]

best_model , best_params, best_score,best_seed = findBalancedDataSet(range(9,10),X,y,GridSearchCVRandomForest,False)

print("Best Params",best_params)
print("Best Score",best_score)
print("Best Seed",best_seed)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=best_seed)
pltResults(best_model,X.shape[1]-1,X_train,X_test,y_train,y_test)
  


def MELHOR_RESULTADO_1():
    param = {'bootstrap': [True], 'max_depth': [6], 'max_features': ['log2'], 'n_estimators': [9], 'random_state': [0]}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=8)
    model =RandomForestRegressor(random_state=0)
    grid = GridSearchCV(model,param, cv=10,verbose=1000,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    pltResults(best_model,1,X_train,X_test,y_train,y_test)
    #Best Params {'bootstrap': True, 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 9, 'random_state': 0}
    #Best Score 0.7532685842882771
    #Best Seed 8




