import dataSetUtils
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0,                         n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               random_state=False, verbose=False)
# Perform K-Fold CV
    scores = cross_val_score(rfr, X, y, cv=10, scoring='r2')
    print("Bests")
    print(best_params)
    print("Média")
    print(sum(scores/len(scores)))
    return scores

csvFile = pd.read_csv("./dataSet/mainDataSet1.csv",usecols=[1,2,3,4,5,6,7])
csvFile = dataSetUtils.convertCSV(csvFile.values)
dataSet = np.array(csvFile)

#csvFile = pd.read_csv("./dataSet/sela.csv",usecols=[0,1,2])
#dataSet = np.array(csvFile.values)



yColunm = 3

dataSetUtils.normalizeColumn(dataSet,0)
dataSetUtils.normalizeColumn(dataSet,1)
dataSetUtils.normalizeColumn(dataSet,yColunm)

xTrain,xTest,yTrain,yTest = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=5)

s = rfr_model(xTrain,yTrain)
print(s)