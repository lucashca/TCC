import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys
import threading
import time
import createRegressor
import dataSetUtils
import machineLearning as ml

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score,max_error

from mpl_toolkits.mplot3d import Axes3D  
from sklearn.svm import SVR
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
from sklearn import preprocessing
from multiprocessing import Process, Lock



## Import DataSet
csvFile = pd.read_csv("./dataSet/novdataset.csv",usecols=[1,2,3,4,5,6,7])
csvFile = dataSetUtils.convertCSV(csvFile.values)
dataSet = np.array(csvFile)


yColunm = 2

dataSetUtils.normalizeColumn(dataSet,0)
dataSetUtils.normalizeColumn(dataSet,1)
dataSetUtils.normalizeColumn(dataSet,yColunm)

dataSetNormalizada = dataSet

regressor = createRegressor.createRandomForrestRegressor()


bests = []
totalRand = 1
offset = 0
for i in range(totalRand):
    dataSplited = dataSetUtils.splitDataForTrain(dataSetNormalizada,yColunm,i+offset)
    models = ml.processRegressorByR2(regressor,dataSplited)
    print(models)
    bests.append({models:models,rand:i+offset})
    print("%d/%d"%(i,totalRand-1))
print("Finish")

melhorScore = bests[0].model.score
rand = 0
model 
for b in bests:
    score = b.model.score
  
    if score > melhorScore:
        melhorScore = score
        rand = b.rand
        model = b


dataSplited = dataSetUtils.splitDataForTrain(dataSetNormalizada,yColunm,rand)

print(rand)
print(model)

ml.printChart(melhorScore[0][1],dataSplited,dataSetNormalizada,yColunm)

