import pandas as pd
import numpy as np
import csv

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
from sklearn import preprocessing
import sys
import threading
import time
from multiprocessing import Process, Lock



def fixCsv(arr):
    #Corrige o dataset, convertendo strig para float e também coverte a unidade dos nutriente de mg/l para cmolc/dm3
    for data in arr:
        data[0] = float(data[0])
        data[1] = float(data[1])
        data[2] = round(float(data[2])/200.4,2) #Ca
        data[3] = round(float(data[3])/121.56,2) #Mg
        data[4] = round(float(data[4])/230,2) #Na
        data[5] = round(float(data[5])/391,2) #K
        data[6] = round(float(data[6])/506.47,2) #Cl
        
    return arr

def getMaxValues(data):
    ''' Esta função é utilizada para obter os valores máximos de cada coluna do DataSet'''
    maxValues = []
    for i in range(data.shape[1]):
        maxValues.append(max(abs(data[:,i])))
    return maxValues


def getMinValues(data):
    ''' Esta função é utilizada para obter os valores mínimos de cada coluna do DataSet'''
    minValues = []
    for i in range(data.shape[1]):
        minValues.append(min(abs(data[:,i])))
    return minValues


def normalizeData(dataSet):
    ''' Esta função é utilizada para normalizar os dados em uma faixa entre 0 e 1 '''
    maximos = getMaxValues(dataSet)
    minimos = getMinValues(dataSet)
    data = np.zeros(dataSet.shape)
    
    for i in range(dataSet.shape[1]):
         data[:,i] = (abs(dataSet[:,i]) - minimos[i])/(maximos[i] - minimos[i])
        #data[:,i] = abs(dataSet[:,i]/maximos[i])
    return data

   


def run(regression,totalIter,name,dataSetNormalizada,yColunm,rand):
    global menorError,melhorModel,melhorR2
    xTrain,xTest,yTrain,yTest = train_test_split(dataSetNormalizada[:,:2],dataSetNormalizada[:,yColunm],test_size=0.3,random_state=rand)
        
          
    model = regression[0][1]
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)
    #score = r2_score(yTest,yPred)
    #maior = score
    #menorError = mean_squared_error(yTest,yPred)
    bestModel = regression[0]
    rscore = r2_score(yTest,yPred)
    mError =   metrics.max_error(yTest,yPred)
    atual = 0
    for i,reg in enumerate(regression):
        try:
            model = reg[1]
            model.fit(xTrain,yTrain)
            yPred = model.predict(xTest)
            #score = r2_score(yTest,yPred)
            erro =  mean_squared_error(yTest,yPred)
            maxError = metrics.max_error(yTest,yPred)
             
            if not totalIter % 2 == 0:
                totalIter = totalIter + 1

           
            
            atual = atual+1
            
            percent = (atual*100) / totalIter
            #print("%s %d/%d - %.2f / %.2f"%(name,atual, totalIter,percent,maior))
            print("%s %d/%d - %.2f - %.5f  %.5f / %.5f : %.5f"%(name,atual, totalIter,percent,maxError,erro,menorError,rscore))

            if erro < menorError:
                rscore = r2_score(yTest,yPred)
                mError =   metrics.max_error(yTest,yPred)
                menorError = erro
                bestModel = reg
                melhorModel = reg
                melhorR2 = rscore

            '''
            if score > maior:
                maior = score
                bestModel = reg
            '''
        except Exception as e:
            print()


    appendRow([name,bestModel[0],rscore,menorError,mError,bestModel[1]])
    return bestModel


def mainWork(regression,name,dataSetNormalizada,rand,yColunm):
   
   
    try:
        
        totalIter = len(regression)
        bestModel = run(regression,totalIter,name,dataSetNormalizada,yColunm,rand)
        model = bestModel[1]
        
        

        model.fit(xTrain,yTrain)
        
        yPred = model.predict(xTest)
        score = r2_score(yTest,yPred)

        mse = mean_squared_error(yTest,yPred)
        maxError = metrics.max_error(yTest,yPred)

          
        ax = plt.subplot(321,projection='3d')
        ax.scatter(dataSetNormalizada[:,0],dataSetNormalizada[:,1],dataSetNormalizada[:,yColunm] ,c=dataSetNormalizada[:,yColunm] ,cmap="coolwarm")
        m = str(bestModel[0])
        ax.set_title("%.5f : %.5f : %s : %s : %.5f"% (mse,maxError,name,m,score))
        
        
        ax1 = plt.subplot(322,projection='3d')
        ax1.scatter(xTrain[:,0], xTrain[:,1],yTrain,c=yTrain ,cmap="coolwarm")
        ax1.set_title("Train Data")

        ax2 = plt.subplot(323,projection='3d')
        ax2.scatter(xTest[:,0], xTest[:,1],yTest,c=yTest ,cmap="coolwarm")
        ax2.set_title("Test Data")


        ax3 = plt.subplot(324,projection='3d')
        ax3.scatter(xTest[:,0], xTest[:,1],yPred,c=yPred ,cmap="coolwarm")
        ax3.set_title("Pred Data")

        ax3 = plt.subplot(325)
        ax3.scatter(xTest[:,0],yTest,c=yTest ,cmap="coolwarm")
        ax3.set_title("Pred Data")

        ax3 = plt.subplot(326)
        ax3.scatter(xTest[:,0],yPred,c=yPred ,cmap="coolwarm")
        ax3.set_title("Pred Data")


        plt.show()
    
    except Exception as e:
        print(e)
    


def getMin(dataSet,column):
    minimo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d < minimo):
            minimo = d
    return minimo    

def getMax(dataSet,column):
    maximo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d > maximo):
            maximo = d
    return maximo    

def normalizeColumn(dataSet,column):
    minimo = getMin(dataSet,column)
    maximo = getMax(dataSet,column)

    for i in range(len(dataSet)):
        d = dataSet[i][column]
        dataSet[i][column] = (d - minimo)/(maximo - minimo)


def appendRow(data):
    global row
    row.append(data)

def saveDump(data,name):
    global row

    csv_file = open("saida.csv", "w")
    c = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    c.writerow(["rand","Config","R2-Score","MSE","Max Error"])
    for r in row:
        c.writerow(r)


## Import DataSet
csvFile = pd.read_csv("novdataset.csv",usecols=[1,2,3,4,5,6,7])
csvFile = fixCsv(csvFile.values)
dataSet = np.array(csvFile)

yColunm = 2

normalizeColumn(dataSet,0)
normalizeColumn(dataSet,1)
normalizeColumn(dataSet,yColunm)

  
dataSetNormalizada = dataSet

melhorModel = []
melhorR2 = -1000
menorError = 1000

regression = []

ACTIVATION_TYPES = ["relu",
#"logistic","identity"
]
SOLVER_TYPES = ["lbfgs"]
ALPHA = [0.00001,0.001,0.01,0.1,1]
LEARNING_RATE_TYPES = ["constant","invscaling","adaptative"]

for depth in range(10):
    for r in range(10):
        #regression.append([[depth,r],DecisionTreeRegressor(max_depth=depth+1,random_state=r)])   
        for s in range(10):
            regression.append([["RandomForestRegressor",depth+1,r,s+1],RandomForestRegressor(max_depth=depth+1, random_state=r,n_estimators=s+1)])


row = []




def work(inicio,carga):
    for j in range(carga):
        i = inicio + j
        mainWork(regression,"%d"%i,dataSetNormalizada,i,yColunm)
        saveDump([],"")



totalWorkers = 3


carga = 5
inicio = 0
for i in range(totalWorkers):
    inicio = i*carga
    t = threading.Thread(target=work,args=(inicio,carga))
    t.start()
    