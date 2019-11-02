
from sklearn.model_selection import train_test_split


from myTools.dataSetPreProcessing import train_validation_test_split
from myTools.loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet,loadMainDataSetWithElevation
from myTools.tools import verifyArgs,findBalancedDataSet,pltResults

from sklearn.metrics import r2_score,mean_squared_error


import matplotlib.pyplot as plt

import sys


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def GridSearchCVGradientBoostingRegressor(X_train,y_train):
    d = [i for i in range(1,50)]
    d.append(None)

    param_grid={
        'n_estimators':[10,50,80], 
        'learning_rate': [0.05,  0.01], 
        'max_depth':[10,50],#d, 
        'min_samples_leaf':[3,5,9,17], 
        'max_features':[2],
        'random_state':range(1) 
    } 

    model = GradientBoostingRegressor(random_state=0,verbose=0)

    scores = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
    ]
    reg =GridSearchCV(model, cv=5,param_grid=param_grid,verbose=0,n_jobs=-1,scoring=scores,refit='r2',iid=True)

    reg.fit(X_train,y_train)

    return reg.best_estimator_,reg.best_params_,reg.best_score_



args = sys.argv[1:]

y_column,random_state = verifyArgs(args)

# Load data set
dataSet = loadMainDataSetWithElevation()
#dataSet = loadTesteDataSet()
#dataSet = loadMainDataSet()
#Set features and target
y_column = 2 
X = dataSet[:,0:4]
y = dataSet[:,4]


best_model , best_params, best_score,best_seed = findBalancedDataSet(range(1,10),X,y,GridSearchCVGradientBoostingRegressor)

print("Best Params",best_params)
print("Best Score",best_score)
print("Best Seed",best_seed)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=best_seed)
pltResults(best_model,X.shape[1]-1,X_train,X_test,y_train,y_test)
  

def MelhorResultado1():
    #Best Params {'learning_rate': 0.05, 'max_depth': 7, 'max_features': 2, 'min_samples_leaf': 17, 'n_estimators': 80, 'random_state': 0}
    #Best Score 0.7069801540890382
    #Best Seed 9
    pass
