
import sys
sys.path.insert(0, "../myTools")

import joblib
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from loadDataSet import loadMainDataSet, loadTesteDataSet, loadCompletDataSet, loadMainDataSetWithElevation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


dataSet = loadMainDataSetWithElevation()


# Melhores modelos


# Modelo Magnésio
#6, 2, 'log2', True, 6, 33

def MELHOR_RESULTADO_MG():
    X = dataSet[:, 0:4]
    y = dataSet[:, 4]
    param = {'learning_rate': [0.05], 'max_depth': [50], 'max_features': [
        'log2'], 'min_samples_leaf': [11], 'n_estimators': [60]}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=9)
    model = GradientBoostingRegressor(random_state=0)
    grid = GridSearchCV(model, param, cv=10, verbose=0,
                        n_jobs=-1, scoring='r2', iid=True)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("Cross Validation Mg  R2 Score :", grid.best_score_)
    
    return best_model


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
   
    
    return best_model


def MELHOR_RESULTADO_K():
    X = dataSet[:, 0:4]
    y = dataSet[:, 6]
    param = {'bootstrap': [True], 'max_depth': [10], 'max_features': [
        'log2'], 'min_samples_leaf': [2], 'n_estimators': [60]}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3)
    model = RandomForestRegressor(random_state=0)
    grid = GridSearchCV(model, param, cv=10, verbose=0,
                        n_jobs=-1, scoring='r2', iid=True)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("Cross Validation K R2 Score :", grid.best_score_)
    return best_model

joblib.dump(MELHOR_RESULTADO_MG(), './dumps/modelMg.joblib')

joblib.dump(MELHOR_RESULTADO_NA(), './dumps/modelNa.joblib')

joblib.dump(MELHOR_RESULTADO_K(), './dumps/modelK.joblib')
