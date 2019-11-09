from sklearn.model_selection import train_test_split


from myTools.dataSetPreProcessing import train_validation_test_split
from myTools.loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet
from myTools.tools import verifyArgs,pltResults,findBalancedDataSet

from sklearn.metrics import r2_score,mean_squared_error


import matplotlib.pyplot as plt

import sys



# Create train,validation and test data
#X_train,X_val,X_test,y_train,y_val,y_test = train_validation_test_split(X,y,random_state=0)
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=random_state)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.svm  import SVR



def GridSearchCVRadiusNeighborsRegressor(X_train,y_train):
    d = [i for i in range(1,100)]
    d.append(None)

    param_grid = {
        'weights': ['uniform', 'distance'],
        'radius': range(1,100),
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], 
    }
    
   
    model = RadiusNeighborsRegressor()

    scores = [
        'r2',
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

best_model , best_params, best_score,best_seed = findBalancedDataSet(range(1,10),X,y,GridSearchCVRadiusNeighborsRegressor)

print("Best Params",best_params)
print("Best Score",best_score)
print("Best Seed",best_seed)


# 17 
# 234



