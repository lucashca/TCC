
import sys
sys.path.insert(0, "../myTools")

from sklearn.model_selection import train_test_split
from dataSetPreProcessing import train_validation_test_split
from loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet,loadMainDataSetWithElevation
from tools import verifyArgs,pltResults,findBalancedDataSet

from sklearn.metrics import r2_score,mean_squared_error


import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm  import SVR



def GridSearchCVKNeighborsRegressor(X_train,y_train):
   
    param_grid = {
        'weights': [ 'distance'],
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
X = dataSet[:,0:2]
y = dataSet[:,2]


best_model , best_params, best_score,best_seed = findBalancedDataSet(range(1,10),X,y,GridSearchCVKNeighborsRegressor)

print("#Best Params",best_params)
print("#Best Score",best_score)
print("#Best Seed",best_seed)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=best_seed)
pltResults(best_model,X.shape[1]-1,X_train,X_test,y_train,y_test)

########## MG #################

#Conluido - Best Score: 0.69719 Semente: 4 
#Best Params {'algorithm': 'auto', 'n_neighbors': 6, 'weights': 'distance'}
#Best Score 0.6971875382557432
#Best Seed 4
#R2 Test:  0.8477948730251221  MSE Test:  0.0026071842742281586
#R2 Train:  1.0  MSE Train:  0.0


def MELHOR_RESULTADO_MG():
    X = dataSet[:,0:4]
    y = dataSet[:,4]
    param ={'algorithm': ['auto'], 'n_neighbors': [6], 'weights': ['distance']}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=5)
    model = KNeighborsRegressor()
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,3,X_train,X_test,y_train,y_test)
    
    #Conluido - Best Score: 0.69719 Semente: 4 
    #Best Params {'algorithm': 'auto', 'n_neighbors': 6, 'weights': 'distance'}
    #Best Score 0.6971875382557432
    #Best Seed 4
    #R2 Test:  0.8477948730251221  MSE Test:  0.0026071842742281586
    #R2 Train:  1.0  MSE Train:  0.0

    MELHOR_RESULTADO_MG()

############ Na ####################
#Conluido - Best Score: 0.62096 Semente: 9 
#Best Params {'algorithm': 'brute', 'n_neighbors': 6, 'weights': 'distance'}
#Best Score 0.6209552829789358
#Best Seed 9
#R2 Test:  0.7066316533521843  MSE Test:  0.009787010608152074
#R2 Train:  0.9999999999999105  MSE Train:  1.9219260943726335e-15


def MELHOR_RESULTADO_NA():
    X = dataSet[:,0:4]
    y = dataSet[:,5]
    param ={'algorithm': ['brute'], 'n_neighbors': [6], 'weights': ['distance']}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=9)
    model = KNeighborsRegressor()
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,3,X_train,X_test,y_train,y_test)
    #Conluido - Best Score: 0.62096 Semente: 9 
    #Best Params {'algorithm': 'brute', 'n_neighbors': 6, 'weights': 'distance'}
    #Best Score 0.6209552829789358
    #Best Seed 9
    #R2 Test:  0.7066316533521843  MSE Test:  0.009787010608152074
    #R2 Train:  0.9999999999999105  MSE Train:  1.9219260943726335e-15

    MELHOR_RESULTADO_NA()



############ K #########################
#Conluido - Best Score: 0.67596 Semente: 4 
#Best Params {'algorithm': 'brute', 'n_neighbors': 8, 'weights': 'distance'}
#Best Score 0.6759638656720487
#Best Seed 4
#R2 Test:  0.8262489727826184  MSE Test:  0.0030635365836426814
#R2 Train:  0.9999999999999659  MSE Train:  8.640127461660114e-16

def MELHOR_RESULTADO_K():
    X = dataSet[:,0:4]
    y = dataSet[:,6]
    param ={'algorithm': ['brute'], 'n_neighbors': [8], 'weights': ['distance']}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=4)
    model = KNeighborsRegressor()
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,3,X_train,X_test,y_train,y_test)
    #Conluido - Best Score: 0.67596 Semente: 4 
    #Best Params {'algorithm': 'brute', 'n_neighbors': 8, 'weights': 'distance'}
    #Best Score 0.6759638656720487
    #Best Seed 4
    #R2 Test:  0.8262489727826184  MSE Test:  0.0030635365836426814
    #R2 Train:  0.9999999999999659  MSE Train:  8.640127461660114e-16
    MELHOR_RESULTADO_K()