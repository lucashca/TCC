
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


dataSet,_,_ = loadMainDataSetWithElevation()


# Melhores modelos


# Modelo Magnésio
#6, 2, 'log2', True, 6, 33

def MELHOR_RESULTADO_MG():
    X = dataSet[:, :4]
    y = dataSet[:, 4]
    params = {'random_state':0,'learning_rate': 0.05, 'loss': 'lad', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 5, 'n_estimators': 500}
    model = GradientBoostingRegressor(**params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2,random_state=9)

    model.fit(X_train, y_train)

    r2Teste = model.score(X_test,y_test)
    r2Train = model.score(X_train,y_train)

    
    print("R2 Treino",r2Train)
    print("R2 Teste",r2Teste)
    

    return model




def MELHOR_RESULTADO_NA():
    X = dataSet[:,:4]
    y = dataSet[:,5]

    params = {'random_state':0, 'bootstrap': True, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'n_estimators': 500}
    model = RandomForestRegressor(**params)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=3)
    model.fit(X_train,y_train)

    r2Teste = model.score(X_test,y_test)
    r2Train = model.score(X_train,y_train)

    
    print("R2 Treino",r2Train)
    print("R2 Teste",r2Teste)
    
     
    return model


def MELHOR_RESULTADO_K():
    X = dataSet[:, :4]
    y = dataSet[:, 6]

    params = {'random_state':0,'learning_rate': 0.05, 'loss': 'lad', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 3, 'n_estimators': 500}
    model = GradientBoostingRegressor(**params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=3)
    model.fit(X_train, y_train)
    
    r2Teste = model.score(X_test,y_test)
    r2Train = model.score(X_train,y_train)

    
    print("R2 Treino",r2Train)
    print("R2 Teste",r2Teste)
    
    return model

joblib.dump(MELHOR_RESULTADO_MG(), './dumps/modelMg.joblib')

joblib.dump(MELHOR_RESULTADO_NA(), './dumps/modelNa.joblib')

joblib.dump(MELHOR_RESULTADO_K(), './dumps/modelK.joblib')
