
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
    d = [10,20,30,40,50,60,70,80,90,100]
    e = [10,20,30,40,50,60,70,80,90,100]
    #d = [10,50,70]
    #e = [10,50,70]
    l = [2,4,5,11]

    param_grid = {
        
        'bootstrap': [True],
        'max_depth': d,
        'max_features': ['auto','log2'],
        'min_samples_leaf': l,
      # 'min_samples_split': [8, 10, 12],
        'n_estimators':e,
    }

    model = RandomForestRegressor(random_state=0)

    scores = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
    ]
    reg =GridSearchCV(model, cv=10,param_grid=param_grid,verbose=0,n_jobs=-1,scoring=scores,refit='r2',iid=False)

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
y = dataSet[:,6]

'''
best_model , best_params, best_score,best_seed = findBalancedDataSet(range(3,4),X,y,GridSearchCVRandomForest,False)

print("#Best Params",best_params)
print("#Best Score",best_score)
print("#Best Seed",best_seed)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=best_seed)
pltResults(best_model,X.shape[1]-1,X_train,X_test,y_train,y_test)
'''
############## MG #################

#Conluido - Best Score: 0.74281 Semente: 4 
#Best Params {'bootstrap': True, 'max_depth': 50, 'max_features': 'auto', 'min_samples_leaf': 4, 'n_estimators': 50}
#Best Score 0.7428107276465451
#Best Seed 4
#R2 Test:  0.7986851229976195  MSE Test:  0.003448405398166473
#R2 Train:  0.8624797661125632  MSE Train:  0.003325206618962523

#Conluido - Best Score: 0.73269 Semente: 9 
#Best Params {'bootstrap': True, 'max_depth': 50, 'max_features': 'auto', 'min_samples_leaf': 4, 'n_estimators': 50}
#Best Score 0.7326863119501887
#Best Seed 9
#R2 Test:  0.7214386448736931  MSE Test:  0.004432785719753988
#R2 Train:  0.8712915912063472  MSE Train:  0.003163202578723336

#Conluido - Best Score: 0.73366 Semente: 9 
#Best Params {'bootstrap': True, 'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 4, 'n_estimators': 60}
#Best Score 0.7336590449066758
#Best Seed 9
#R2 Test:  0.7158820898963456  MSE Test:  0.00452120795457348
#R2 Train:  0.8717629515241512  MSE Train:  0.0031516181905177796

def MELHOR_RESULTADO_MG():
    X = dataSet[:,0:4]
    y = dataSet[:,4]
    param = {'bootstrap': [True], 'max_depth': [20], 'max_features': ['auto'], 'min_samples_leaf': [4], 'n_estimators': [60]}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=9)
    model = RandomForestRegressor(random_state=0)
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,3,X_train,X_test,y_train,y_test)
    #Conluido - Best Score: 0.73366 Semente: 9 
    #Best Params {'bootstrap': True, 'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 4, 'n_estimators': 60}
    #Best Score 0.7336590449066758
    #Best Seed 9
    #R2 Test:  0.7158820898963456  MSE Test:  0.00452120795457348
    #R2 Train:  0.8717629515241512  MSE Train:  0.0031516181905177796
    MELHOR_RESULTADO_MG()



########### NA ############################

#Conluido - Best Score: 0.61644 Semente: 2 
#Best Params {'bootstrap': True, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'n_estimators': 100}
#Best Score 0.6164423932855161
#Best Seed 2
#R2 Test:  0.6470743582563896  MSE Test:  0.011315230193482718
#R2 Train:  0.8844389141031836  MSE Train:  0.002520229138039719

def MELHOR_RESULTADO_NA():
    X = dataSet[:,0:4]
    y = dataSet[:,5]
    param = {'bootstrap': [True], 'max_depth': [10], 'max_features': ['log2'], 'min_samples_leaf': [2], 'n_estimators': [100]}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)
    model = RandomForestRegressor(random_state=0)
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,0,X_train,X_test,y_train,y_test)
    #Conluido - Best Score: 0.61644 Semente: 2 
    #Best Params {'bootstrap': True, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'n_estimators': 100}
    #Best Score 0.6164423932855161
    #Best Seed 2
    #R2 Test:  0.6470743582563896  MSE Test:  0.011315230193482718
    #R2 Train:  0.8844389141031836  MSE Train:  0.002520229138039719
    MELHOR_RESULTADO_NA()



######### K #################
#Conluido - Best Score: 0.62035 Semente: 3 
#Best Params {'bootstrap': True, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'n_estimators': 70}
#Best Score 0.6203458158780136
#Best Seed 3
#R2 Test:  0.7171670059625839  MSE Test:  0.006843307283046334
#R2 Train:  0.897215092835952  MSE Train:  0.0024435392891614407

#Conluido - Best Score: 0.62385 Semente: 3 
#Best Params {'bootstrap': True, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'n_estimators': 60}
#Best Score 0.6238517912793121
#Best Seed 3
#R2 Test:  0.713847840031108  MSE Test:  0.0069236164155427465
#R2 Train:  0.9016407306364382  MSE Train:  0.0023383271510814194

def MELHOR_RESULTADO_K():
    X = dataSet[:,0:4]
    y = dataSet[:,6]
    param = {'bootstrap': [True], 'max_depth': [10], 'max_features': ['log2'], 'min_samples_leaf': [2], 'n_estimators': [60]}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=3)
    model = RandomForestRegressor(random_state=0)
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,0,X_train,X_test,y_train,y_test)
    #Conluido - Best Score: 0.62385 Semente: 3 
    #Best Params {'bootstrap': True, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'n_estimators': 60}
    #Best Score 0.6238517912793121
    #Best Seed 3
    #R2 Test:  0.713847840031108  MSE Test:  0.0069236164155427465
    #R2 Train:  0.9016407306364382  MSE Train:  0.0023383271510814194
MELHOR_RESULTADO_K()
