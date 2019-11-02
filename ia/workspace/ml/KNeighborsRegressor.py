from sklearn.model_selection import train_test_split


from myTools.dataSetPreProcessing import train_validation_test_split
from myTools.loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet,loadMainDataSetWithElevation
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm  import SVR



def GridSearchCVKNeighborsRegressor(X_train,y_train):
    d = [i for i in range(1,100)]
    d.append(None)

    param_grid = {
        'weights': ['uniform', 'distance'],
        'n_neighbors': range(2,100),
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], 
    }
    
   
    model = KNeighborsRegressor()

    scores = [
        'r2',
    ]

    reg =GridSearchCV(model, cv=3,param_grid=param_grid,verbose=0,n_jobs=-1,scoring=scores,refit='r2',iid=True)

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
X = dataSet[:,0:3]
y = dataSet[:,3]

best_model , best_params, best_score,best_seed = findBalancedDataSet(range(1,2),X,y,GridSearchCVKNeighborsRegressor)

print("Best Params",best_params)
print("Best Score",best_score)
print("Best Seed",best_seed)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=best_seed)
pltResults(best_model,X.shape[1]-1,X_train,X_test,y_train,y_test)
  

# 17 
# 234



