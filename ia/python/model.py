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
from sklearn.ensemble import RandomForestRegressor



def work(iteracoes,index):
    bests = []
    totalRand = iteracoes[-1]
    for i in iteracoes:
        dataSplited = dataSetUtils.splitDataForTrain(dataSetNormalizada,yColunm,i)
        models = ml.processRegressorByR2(regressor,dataSplited)
        print("Thread - %s - %d/%d"%(index,i,totalRand))
        models.rand = i
        bests.append(models)
    print("Finish")
    
    melhorScore = bests[0].score
    melhorModel = bests[0]
     #print("Thread - %s : Score %s"%(index,melhorModel.regressor))
    
    for b in bests:
        if b.score > melhorScore:
            melhorScore = b.score
            melhorModel = b

    print("Thread - %s : Score %.5f"%(index,melhorScore))
   
    regConfig = melhorModel.regressor[0]
    regConfig.append(melhorModel.rand)
    
    dataSplited = dataSetUtils.splitDataForTrain(dataSetNormalizada,yColunm,melhorModel.rand)
    ml.printChart(melhorModel.regressor[1],dataSplited,dataSetNormalizada,yColunm,regConfig)
   


def slitArray(totalSize,workers):
    
    work = []
    if(totalSize%workers == 0):
        carga = int(totalSize/workers)
    else:
        carga = int(totalSize/workers) + 1
    for i in range(workers):
        inicio = i*carga
        fim = i*carga + carga
        if fim > totalSize:
            fim = totalSize
        r = range(inicio,fim)
        work.append(r)
    
    print(work)

    return work





def testeConfiguration(d,rR,s,rD):
    testRegressor = []
    testRegressor.append([["RandomForestRegressor",0,0,0],RandomForestRegressor(max_depth=d, random_state=rR,n_estimators=s)])
    dataSplited = dataSetUtils.splitDataForTrain(dataSetNormalizada,yColunm,rD)   
    ml.printChart(testRegressor[0][1],dataSplited,dataSetNormalizada,yColunm,[d,rR,s,rD])

def createWorkers(wokersSize,totalRand):
    workRand = slitArray(totalRand,wokersSize)
    for i in range(wokersSize):
        t = Process(target=work,args=(workRand[i],i))
        t.start()





## Import DataSet
csvFile = pd.read_csv("./dataSet/mainDataSet1.csv",usecols=[1,2,3,4,5,6,7])
csvFile = dataSetUtils.convertCSV(csvFile.values)
dataSet = np.array(csvFile)

#csvFile = pd.read_csv("./dataSet/sela.csv",usecols=[0,1,2])
#dataSet = np.array(csvFile.values)



yColunm = 3

dataSetUtils.normalizeColumn(dataSet,0)
dataSetUtils.normalizeColumn(dataSet,1)
dataSetUtils.normalizeColumn(dataSet,yColunm)

dataSetNormalizada = dataSet



regressor = createRegressor.createRandomForrestRegressor()
#regressor = createRegressor.createDecisionTreeRegressor()


#createWorkers(5,100)

bestesConfig = [] 

#******************** Random Forrest Regressor
#**** Criterio mae *************************

#CA coluna yColumn = 2
bestesConfig.append([None,5,10,54])#0.65561
bestesConfig.append([None,4,6,16])#0.64219
bestesConfig.append([None,9,9,87])#0.64690
bestesConfig.append([None,0,10,30])#0.63937

#CA coluna yColumn = 3
bestesConfig.append([6,5,1,37])#0.6372
bestesConfig.append([None,1,4,99])#0.60803
bestesConfig.append([None,0,9,16])#0.60522

#CA coluna yColumn = 4
bestesConfig.append([None,9,6,35])#0.58762
bestesConfig.append([None,4,4,8])#0.54880
bestesConfig.append([None,5,7,72])#0.55263

#CA coluna yColumn = 5
bestesConfig.append([7,7,3,20])#0.53670
bestesConfig.append([10,0,8,89])#0.52600
bestesConfig.append([None,1,9,64])#0.50223


#CA coluna yColumn = 6
bestesConfig.append([None,9,1,59])#0.78351
bestesConfig.append([7,9,1,59])#0.82634
bestesConfig.append([6,6,1,84])#0.0.61183




#**** Criterio MSE *************************
#CA coluna yColumn = 2
bestesConfig.append([7,1,7,64])#0.63937
bestesConfig.append([7,4,7,16])#0.6574
bestesConfig.append([7,7,9,33])#0.62391
bestesConfig.append([9,4,3,87])#0.62479
bestesConfig.append([None,3,9,56]) #0.598

#K coluna yColumn = 3
bestesConfig.append([6,8,1,45])#0.68223
bestesConfig.append([10,4,8,64])#0.621
bestesConfig.append([6,2,2,33])#0.661
bestesConfig.append([4,4,1,16])#0.6768
bestesConfig.append([6,6,1,84])#0.70446

#Mg coluna yColumn = 4
bestesConfig.append([None,7,4,8])#0.6397
bestesConfig.append([8,7,1,27])#0.5919
bestesConfig.append([6,0,4,72])#0.5824

#Mg coluna yColumn = 5
bestesConfig.append([6,8,2,89])#0.50190
bestesConfig.append([None,6,1,28])#0.486
bestesConfig.append([5,5,1,63])#0.49791

#Mg coluna yColumn = 6
bestesConfig.append([7,4,7,16])#0.64662
bestesConfig.append([8,0,1,82])#0.64632
bestesConfig.append([9,4,8,33])#0.68420
bestesConfig.append([8,0,2,64])#0.62593
bestesConfig.append([6,9,2,59])#0.58745




#******************** Mult Layer Perceptron
bestesConfig.append(['relu','lgfgs','0.001','constant',5,12])#0.474
bestesConfig.append(['relu','lgfgs','0.01','constant',0,56])#0.43
bestesConfig.append(['relu','lgfgs','0.001','constant',0,33])#0.53
bestesConfig.append(['relu','lgfgs','0.001','constant',0,93])#0.433
bestesConfig.append(['relu','lgfgs',1e-5,'constant',3,64])#0.432



reg = RandomForestRegressor(max_depth=6, random_state=6,n_estimators=1)
dataSplited = dataSetUtils.splitDataForTrain(dataSetNormalizada,yColunm,84)   

xTrain = dataSplited[0]
xTest = dataSplited[1]
yTrain = dataSplited[2]
yTest = dataSplited[3]

reg.fit(xTrain,yTrain)
yPred = reg.predict(xTest)
score = reg.score(xTest,yTest)
print(score)