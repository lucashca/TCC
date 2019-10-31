from dataSetUtils import normalizeColumn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import os
import joblib
import pandas as pd
import numpy as np

csvFile = pd.read_csv("mainDataSet1.csv",usecols=[1,2,3,4,5,6])
dataSetOriginal = np.array(csvFile.values)

dataSet = np.array(csvFile.values)


normalizeColumn(dataSet,0)
normalizeColumn(dataSet,1)
normalizeColumn(dataSet,2)
normalizeColumn(dataSet,3)
normalizeColumn(dataSet,4)
normalizeColumn(dataSet,5)

#### Melhores modelos 

#### Modelo Cálcio
#None, 1, 'log2', False, 0, 950

x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,2],test_size=0.3,random_state=950)
modelCa = RandomForestRegressor(max_depth=None,n_estimators=1,max_features='log2',random_state=0,bootstrap=False,criterion="mse")
modelCa.fit(x_train,y_train)
print("Score Ca",modelCa.score(x_test,y_test))

joblib.dump(modelCa, './dumps/modelCa.joblib')  

#### Modelo Magnésio
#6, 2, 'log2', True, 6, 33

x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,3],test_size=0.3,random_state=33)
modelMg = RandomForestRegressor(max_depth=6,n_estimators=2,max_features='log2',random_state=6,bootstrap=True,criterion="mse")
modelMg.fit(x_train,y_train)
print("Score Mg",modelMg.score(x_test,y_test))


joblib.dump(modelMg, './dumps/modelMg.joblib')  

#### Modelo Sódio
#8, 1, 'log2', True, 5, 731

x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,4],test_size=0.3,random_state=731)
modelNa = RandomForestRegressor(max_depth=8,n_estimators=1,max_features='log2',random_state=5,bootstrap=True,criterion="mse")
modelNa.fit(x_train,y_train)
print("Score Na",modelNa.score(x_test,y_test))


joblib.dump(modelNa, './dumps/modelNa.joblib')  


#### Modelo Potássio
#7, 4, 'auto', True, 0, 575

x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,5],test_size=0.3,random_state=575)
modelK = RandomForestRegressor(max_depth=7,n_estimators=4,max_features='auto',random_state=0,bootstrap=True,criterion="mse")
modelK.fit(x_train,y_train)
print("Score K",modelK.score(x_test,y_test))



joblib.dump(modelK, './dumps/modelK.joblib')  
