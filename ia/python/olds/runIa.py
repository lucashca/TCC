from gridData import GridData
from sklearn.ensemble import RandomForestRegressor
import dataSetUtils
import pandas as pd
import numpy as np



csvFile = pd.read_csv("./dataSet/mainDataSet1.csv",usecols=[1,2,3,4,5,6,7])
csvFile = dataSetUtils.convertCSV(csvFile.values)
dataSet = np.array(csvFile)

yColunm = 2

dataSetUtils.normalizeColumn(dataSet,0)
dataSetUtils.normalizeColumn(dataSet,1)
dataSetUtils.normalizeColumn(dataSet,yColunm)

d = [i for i in range(1,11)]
d.append(None)
params = {
    "max_depth":d,
    "n_estimator":range(1,11),
    "random_state":range(10),
    #"n_estimator":s
}

models = []
for d in params["max_depth"]:
    for s in params["n_estimator"]:
        for r in params["random_state"]:
            models.append([
                {
                    "max_depth":d,
                    "n_estimator":s,
                    "random_state":r
                },
                RandomForestRegressor(n_estimators=s,max_depth=d,random_state=r)
            ])
    

grid = GridData(dataSet,100,models,4,yColunm)