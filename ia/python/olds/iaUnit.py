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

   


def run(regression,totalIter,name):

    model = regression[0][1]
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)
    #score = r2_score(yTest,yPred)
    #maior = score
    menorError = mean_squared_error(yTest,yPred)
    bestModel = regression[0]
    rscore = r2_score(yTest,yPred)
            
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
            
                menorError = erro
                bestModel = reg
                

            '''
            if score > maior:
                maior = score
                bestModel = reg
            '''
        except Exception as e:
            print()
   
    return bestModel


def mainWork(regression,totalIter,name):
    try:
        bestModel = run(regression,totalIter,name)
        model = bestModel[1]
        
        model.fit(xTrain,yTrain)
        
        yPred = model.predict(xTest)
        score = r2_score(yTest,yPred)

        mse = mean_squared_error(yTest,yPred)
        maxError = metrics.max_error(yTest,yPred)

        saveDump([bestModel[0],score,mse,maxError],name)
          
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

def saveDump(data,name):
    fname = "./dump/"+str(name)+".csv"
    csv_file = open(fname, "w")
    c = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    c.writerow(["Config","R2-Score","MSE","Max Error"])
  
    c.writerow(data)
  
## Import DataSet
csvFile = pd.read_csv("novdataset.csv",usecols=[1,2,3,4,5,6,7])
csvFile = fixCsv(csvFile.values)
dataSet = np.array(csvFile)


#csvFile = pd.read_csv("sela.csv",usecols=[0,1,2])
#dataSet = np.array(csvFile.values)


## Converte para uma matrix do numpy
## Normaliza o DataSet na faixa de 0 e 1
#dataSetNormalizada = normalizeData(dataSet)
#dataSetNormalizada = dataSet

#dataSet[:,0] = preprocessing.normalize([dataSet[:,0]])
#dataSet[:,1] = preprocessing.normalize([dataSet[:,1]])


yColunm = 3

normalizeColumn(dataSet,0)
normalizeColumn(dataSet,1)
normalizeColumn(dataSet,yColunm)


dataSetNormalizada = dataSet

xTrain,xTest,yTrain,yTest = train_test_split(dataSetNormalizada[:,:2],dataSetNormalizada[:,yColunm],test_size=0.3,random_state=1)

regression = []
regression.append([["RandomForestRegressor",10,0,9],RandomForestRegressor(max_depth=7, random_state=7,n_estimators=10)])

totalWorkers = 1

worksize = int(len(regression)/totalWorkers) + 1

for i in range(totalWorkers):
    inicio = i*worksize
    fim = i*worksize + worksize
    if fim > len(regression) :
        fim = len(regression) 
    
    t = Process(target=mainWork,args=(regression[inicio:fim],worksize,"Thread %d" %i))
    t.start()














