
from sklearn.model_selection import train_test_split



from myTools.dataSetPreProcessing import train_validation_test_split
from myTools.loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet
from myTools.tools import verifyArgs,findBalancedDataSet,pltResults

from sklearn.metrics import r2_score,mean_squared_error


import matplotlib.pyplot as plt

import sys


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def GridSearchCVGradientBoostingRegressor(X_train,y_train):
    d = [i for i in range(1,10)]
    d.append(None)

    param_grid={
        'n_estimators':[100,10], 
        'learning_rate': [0.1, 0.05, 0.02, 0.01], 
        'max_depth':d, 
        'min_samples_leaf':[3,5,9,17], 
        'max_features':[1.0,0.3,0.1],
        'random_state':range(1) 
    } 

    model = GradientBoostingRegressor(random_state=0)

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



best_model , best_params, best_score,best_seed = findBalancedDataSet(range(1,10),X,y,GridSearchCVGradientBoostingRegressor)

print("Best Params",best_params)
print("Best Score",best_score)
print("Best Seed",best_seed)

